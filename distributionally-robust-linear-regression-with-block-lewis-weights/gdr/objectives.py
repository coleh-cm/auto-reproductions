"""Objectives for distributionally-robust linear regression (SPEC E1, E3, E4, E5).

Paper convention (`body.tex:27`): the per-group factors 1/sqrt(n_i) are *folded*
into the data at load time, so every formula below uses the folded (A, b) and
never carries n_i. After folding the per-group loss is exactly

    ell_i(x) = (1/n_i) * ||(orig A_{S_i}) x - (orig b_{S_i})||^2
             = ||(folded A_{S_i}) x - (folded b_{S_i})||^2

and the benchmarked worst-group objective is its square

    F(x) = max_i ell_i(x) = max_i ||r_i||_2^2 = ||A x - b||_{G_inf}^2

(experiments.tex:11-14). The *theory* smooths the unsquared square root
f(x) = ||A x - b||_{G_inf} = max_i ||r_i||_2 (`body.tex:27`); the smoothed
surrogate f_tilde_{beta,delta} approximates f, and the guarantee
|f_tilde - f| <= beta log m + delta is Lemma 6.1 (`body.tex:230-231`).

All arrays are np.float64. d is tiny (=10), so clarity beats micro-optimization
and we keep the full d x d Hessian everywhere.
"""
from __future__ import annotations

import numpy as np

from .problem import (
    Problem,
    residuals,
    group_losses,
    group_norms,
    max_loss,
    max_loss_unsquared,
)

# ---------------------------------------------------------------------------
# Frozen-interface re-exports (SPEC sec 4).
# Some downstream units import the canonical names from gdr.objectives, others
# from gdr.problem; both paths must resolve to the *same* callable, so we
# re-export the problem.py implementations here rather than redefine them.
# `residuals` (plural, the problem.py canonical) is re-exported so that
# `from gdr.objectives import *` and `from gdr.objectives import residuals`
# both resolve; the singular `residual` wrapper below is the documented name
# in this unit's interface.
# ---------------------------------------------------------------------------
__all__ = [
    # folded-loss helpers (re-exported from gdr.problem; SPEC sec 4 names)
    "max_loss",            # F(x) = max_i ||r_i||^2   (squared scale, E1^2)
    "group_losses",        # [m] of ||r_i||^2
    "residuals",           # r = A x - b  [n]  (problem.py canonical; plural)
    "residual",            # singular wrapper of residuals (this unit's name)
    "group_norms",         # [m] of ||r_i||_2  (theory inner norm)
    "max_loss_unsquared",  # f(x) = max_i ||r_i||_2  (E1, theory scale)
    "group_norm_inf",      # ||A x - b||_{G_inf} = max_i ||r_i||_2  (E1)
    # smoothed surrogate (E3, E4)
    "smoothed",
    "smoothed_grad_hess",
    # finite-p objective (E5)
    "p_objective",
    "p_grad_hess",
]


def residual(problem: Problem, x: np.ndarray) -> np.ndarray:
    """r = A x - b, shape [n].  (folded data => block r_i is rows S_i.)"""
    return residuals(problem, x)  # [n]


def group_norm_inf(problem: Problem, x: np.ndarray) -> float:
    """||A x - b||_{G_inf} = max_i ||r_i||_2  (unsquared theory scale, E1)."""
    return float(group_norms(problem, x).max())  # scalar


# ---------------------------------------------------------------------------
# Smoothed surrogate (E3, body.tex:54-56 / body.tex:177-181) and derivatives (E4)
# ---------------------------------------------------------------------------

def _group_norms_from_r(r: np.ndarray, off: np.ndarray, m: int) -> np.ndarray:
    """||r_i||_2 per group, shape [m], from a stacked residual r [n]."""
    gn = np.empty(m, dtype=np.float64)          # [m]
    for i in range(m):                            # over groups
        ri = r[off[i]:off[i + 1]]                 # [n_i]
        gn[i] = float(np.sqrt(ri @ ri))           # scalar
    return gn                                     # [m]


def smoothed(problem: Problem, x: np.ndarray, beta: float, delta: float) -> float:
    """f_tilde_{beta,delta}(x)  (E3, body.tex:177-181).

        f_tilde = beta * log sum_i exp( (sqrt(delta^2 + ||r_i||^2) - delta) / beta )

    Computed with a max-shift for numerical stability:
        log sum exp(a) = a_max + log sum exp(a - a_max).
    """
    off = problem["offsets"]                       # [m+1]
    m = problem["m"]                               # scalar
    r = residuals(problem, x)                      # [n]
    gn = _group_norms_from_r(r, off, m)             # [m]  ||r_i||_2
    h = np.sqrt(delta * delta + gn * gn)            # [m]  h_i = sqrt(delta^2 + ||r_i||^2)
    a = (h - delta) / beta                          # [m]  exponents a_i = (h_i - delta)/beta
    a_max = a.max()                                # scalar (max-shift)
    return float(beta * (a_max + np.log(np.sum(np.exp(a - a_max)))))  # scalar


def _softmax(a: np.ndarray) -> np.ndarray:
    """s = softmax(a) in the simplex Delta^m, computed with a max-shift.  [m] -> [m]."""
    a_shift = a - a.max()                           # [m]
    e = np.exp(a_shift)                             # [m]
    return e / e.sum()                              # [m]


def smoothed_grad_hess(problem: Problem, x: np.ndarray, beta: float, delta: float
                       ) -> tuple[float, np.ndarray, np.ndarray]:
    """(f_tilde, grad [d], hess [d,d]) of the LSE surrogate (E4, body.tex:285-288, 431-435).

    Let  h_i = sqrt(delta^2 + ||r_i||^2),  a_i = (h_i - delta)/beta,
         s = softmax(a) in Delta^m  (max-shift),  g_i = A_{S_i}^T r_i / h_i.  Then

        grad f_tilde       = sum_i s_i g_i
        hess f_tilde       = sum_i s_i A_{S_i}^T ( I/h_i - r_i r_i^T / h_i^3 ) A_{S_i}
                           + (1/beta) sum_i s_i g_i g_i^T
                           - (1/beta) grad grad^T

    The block sums are A^T B A with block-diagonal B; the last two terms combine
    into (1/beta)( sum_i s_i g_i g_i^T - grad grad^T ) = (1/beta) Cov_s(g)  (PSD).
    Full d x d Hessian is kept (d tiny). See U13 for the rank-1 negative term.
    """
    A = problem["A"]                                # [n, d]
    off = problem["offsets"]                         # [m+1]
    m = problem["m"]                                 # scalar
    d = problem["d"]                                 # scalar
    r = residuals(problem, x)                       # [n]
    gn = _group_norms_from_r(r, off, m)              # [m]
    h = np.sqrt(delta * delta + gn * gn)             # [m]  h_i
    a = (h - delta) / beta                           # [m]  a_i
    s = _softmax(a)                                  # [m]  softmax weights

    # value (max-shift, identical to smoothed())
    a_max = a.max()                                  # scalar
    val = float(beta * (a_max + np.log(np.sum(np.exp(a - a_max)))))  # scalar

    grad = np.zeros(d, dtype=np.float64)             # [d]
    hess = np.zeros((d, d), dtype=np.float64)        # [d, d]
    Gi = np.zeros((m, d), dtype=np.float64)           # [m, d]  row i = g_i = A_i^T r_i / h_i
    for i in range(m):                               # over groups
        ri = r[off[i]:off[i + 1]]                    # [n_i]
        Ai = A[off[i]:off[i + 1], :]                 # [n_i, d]
        hi = h[i]                                    # scalar
        Atr = Ai.T @ ri                              # [d]   A_{S_i}^T r_i
        gi = Atr / hi                                # [d]   g_i
        Gi[i] = gi                                   # [d]
        grad += s[i] * gi                            # [d]  += s_i g_i
        AtA = Ai.T @ Ai                              # [d, d]
        # A_i^T (I/h_i - r_i r_i^T / h_i^3) A_i  = (1/h_i) A_i^T A_i - (1/h_i^3)(A_i^T r_i)(A_i^T r_i)^T
        hess += s[i] * ((1.0 / hi) * AtA - (1.0 / (hi ** 3)) * np.outer(Atr, Atr))  # [d, d]
    # softmax-curvature term: (1/beta)( sum_i s_i g_i g_i^T - grad grad^T ) = (1/beta) Cov_s(g)
    hess += (1.0 / beta) * (Gi.T @ (s[:, None] * Gi) - np.outer(grad, grad))       # [d, d]
    return val, grad, hess


# ---------------------------------------------------------------------------
# finite-p objective (E5, interpolation.tex:24-29):  f(x) = ||A x - b||_{G_p}^p
# ---------------------------------------------------------------------------

def p_objective(problem: Problem, x: np.ndarray, p: float) -> float:
    """f(x) = sum_i ||r_i||_2^p  (the p-th power of the Thm-2 objective, body.tex:27).

    p = inf is guarded: the p-th power is not meaningful for p = inf, so we
    return the limiting robust objective max_i ||r_i||_2 (the unsquared f(x)).
    """
    gn = group_norms(problem, x)                    # [m]
    if np.isinf(p):                                 # p = inf guard
        return float(gn.max())                       # scalar = ||A x - b||_{G_inf}
    return float(np.sum(gn ** p))                    # scalar = sum_i ||r_i||^p


def p_grad_hess(problem: Problem, x: np.ndarray, p: float
                ) -> tuple[float, np.ndarray, np.ndarray]:
    """(f, grad [d], hess [d,d]) of f(x) = sum_i ||r_i||_2^p  (E5, interpolation.tex:24-29).

        grad f  = p * sum_i ||r_i||^{p-2} A_{S_i}^T r_i
        hess f  = p * sum_i ||r_i||^{p-2} A_{S_i}^T A_{S_i}
                + p(p-2) * sum_i ||r_i||^{p-4} (A_{S_i}^T r_i)(A_{S_i}^T r_i)^T

    Only finite p >= 2 is defined by the paper; p = inf raises (the formula's
    ||r||^{p-4} exponent is meaningless).  ||r_i|| == 0 is guarded by an
    explicit `continue`: on the surviving path nrm > 0, so the ||r||^{p-2}
    and ||r||^{p-4} factors are well-defined and no eps floor is needed.
    """
    if not np.isfinite(p):
        raise ValueError("p_grad_hess is defined only for finite p >= 2 (E5)")
    A = problem["A"]                                # [n, d]
    off = problem["offsets"]                         # [m+1]
    m = problem["m"]                                 # scalar
    d = problem["d"]                                 # scalar
    r = residuals(problem, x)                       # [n]
    grad = np.zeros(d, dtype=np.float64)              # [d]
    hess = np.zeros((d, d), dtype=np.float64)         # [d, d]
    val = 0.0                                        # scalar
    for i in range(m):                               # over groups
        ri = r[off[i]:off[i + 1]]                    # [n_i]
        Ai = A[off[i]:off[i + 1], :]                 # [n_i, d]
        sq = float(ri @ ri)                          # scalar  ||r_i||^2
        nrm = np.sqrt(sq)                             # scalar  ||r_i||
        Atr = Ai.T @ ri                              # [d]    A_{S_i}^T r_i
        AtA = Ai.T @ Ai                              # [d, d] A_{S_i}^T A_{S_i}
        if nrm <= 0.0:                               # ||r_i|| == 0 guard
            # f = sum ||r||^p contributes 0; for p=2 the Hessian is 2 A^T A
            # (which still acts on directions even at r=0); for p>2 both the
            # ||r||^{p-2} and ||r||^{p-4} factors vanish, so nothing is added.
            if p == 2.0:
                hess += 2.0 * AtA                     # [d, d]
                # grad += 2.0 * Atr is a no-op (Atr == 0 here)
            continue
        # nrm > 0 on this path, so the powers are well-defined.
        val += nrm ** p                              # scalar  += ||r_i||^p
        grad += p * (nrm ** (p - 2)) * Atr            # [d]     p ||r||^{p-2} A_i^T r_i
        hess += p * (nrm ** (p - 2)) * AtA            # [d, d]  p ||r||^{p-2} A_i^T A_i
        if p != 2.0:
            hess += p * (p - 2) * (nrm ** (p - 4)) * np.outer(Atr, Atr)  # [d, d]
    return float(val), grad, hess
