"""Objectives: robust loss, group losses, and the LSE smoothed surrogate (SPEC E1, E3, E4, E5).

The theory (`body.tex:27`) works on the *unsquared* scale  f(x) = ||Ax-b||_{G_inf}
and the smoothed surrogate

    f_tilde_{beta,delta}(x) = beta * log sum_i exp( (sqrt(delta^2 + ||r_i||^2) - delta) / beta )
                            = beta * log sum_i exp( h_i(x) / beta ) - delta ,   h_i = sqrt(delta^2+||r_i||^2)

with |f_tilde - f| <= beta log m + delta   (Lemma 6.1, body.tex:230-231).
The benchmarked objective is its square F(x) = max_i ||r_i||^2 = f(x)^2.
"""
from __future__ import annotations

import numpy as np

from .problem import Problem, group_norms, group_residuals, residuals


# ---------------------------------------------------------------------------
# robust objective (E1) -- already in problem.py; re-export the canonical names
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# smoothed surrogate (E3, body.tex:54-56) and its derivatives (E4)
# ---------------------------------------------------------------------------

def _inner_h(r_i: np.ndarray, delta: float) -> float:
    """h_i = sqrt(delta^2 + ||r_i||^2)."""
    return float(np.sqrt(delta * delta + float(r_i @ r_i)))


def smoothed(problem: Problem, x: np.ndarray, beta: float, delta: float) -> float:
    """f_tilde_{beta,delta}(x)  (E3).  Uses a max-shift for numerical stability."""
    gn = group_norms(problem, x)            # ||r_i||_2
    h = np.sqrt(delta * delta + gn * gn)     # h_i
    a = (h - delta) / beta                   # exponents
    amax = a.max()                           # logsumexp with max-shift (numerically stable)
    return float(beta * (amax + np.log(np.sum(np.exp(a - amax)))))


def _softmax_weights(h: np.ndarray, beta: float, delta: float) -> np.ndarray:
    a = (h - delta) / beta
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def smoothed_grad_hess(problem: Problem, x: np.ndarray, beta: float, delta: float
                       ) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (f_tilde, grad [d], hess [d,d]) of the LSE surrogate (E4).

    Let h_i = sqrt(delta^2 + ||r_i||^2), s = softmax((h_i-delta)/beta) in Delta^m,
    g_i = A_{S_i}^T r_i / h_i   (per-group gradient of h_i). Then

        grad f_tilde = sum_i s_i * g_i
        hess f_tilde = sum_i s_i * A_{S_i}^T ( I/h_i - r_i r_i^T / h_i^3 ) A_{S_i}
                     + (1/beta) * sum_i s_i * g_i g_i^T
                     - (1/beta) * grad f_tilde * grad f_tilde^T
    """
    A = problem["A"]
    off = problem["offsets"]
    m = problem["m"]
    d = problem["d"]
    r = residuals(problem, x)
    gn = np.array([np.sqrt(float(r[off[i]:off[i + 1]] @ r[off[i]:off[i + 1]]))
                   for i in range(m)])
    h = np.sqrt(delta * delta + gn * gn)
    s = _softmax_weights(h, beta, delta)

    # value (with max-shift, matches smoothed())
    a = (h - delta) / beta
    a_shift = a - a.max()
    val = float(beta * (a.max() + np.log(np.exp(a_shift).sum())))

    grad = np.zeros(d, dtype=np.float64)
    hess = np.zeros((d, d), dtype=np.float64)
    # accumulators for the (1/beta) sum s_i g_i g_i^T term
    Gi = np.zeros((m, d), dtype=np.float64)
    for i in range(m):
        ri = r[off[i]:off[i + 1]]
        Ai = A[off[i]:off[i + 1], :]
        hi = h[i]
        # g_i = A_{S_i}^T r_i / h_i
        gi = (Ai.T @ ri) / hi
        Gi[i] = gi
        grad += s[i] * gi
        # A_{S_i}^T (I/h_i - r_i r_i^T / h_i^3) A_{S_i}  =  (1/h_i) A_i^T A_i - (1/h_i^3)(A_i^T r_i)(A_i^T r_i)^T
        AtA = Ai.T @ Ai
        Atr = Ai.T @ ri  # == h_i * g_i
        hess += s[i] * ((1.0 / hi) * AtA - (1.0 / (hi ** 3)) * np.outer(Atr, Atr))
    # cross / softmax-curvature term: (1/beta)(sum_i s_i g_i g_i^T - grad grad^T)
    hess += (1.0 / beta) * (Gi.T @ (s[:, None] * Gi) - np.outer(grad, grad))
    return val, grad, hess


# ---------------------------------------------------------------------------
# finite-p objective (E5, interpolation.tex:24-29) -- f(x) = ||Ax-b||_{G_p}^p
# ---------------------------------------------------------------------------

def p_objective(problem: Problem, x: np.ndarray, p: float) -> float:
    """f(x) = sum_i ||r_i||_2^p  (the p-th power of the Thm-2 objective, body.tex:27)."""
    gn = group_norms(problem, x)
    if np.isinf(p):
        # max_i ||r_i||_2  (unsquared robust; p-th power convention -> itself)
        return float(gn.max())
    return float(np.sum(gn ** p))


def p_grad_hess(problem: Problem, x: np.ndarray, p: float
                ) -> tuple[float, np.ndarray, np.ndarray]:
    """(f, grad, hess) of f(x)=sum_i ||r_i||_2^p  (E5).

        grad = p * sum_i ||r_i||^{p-2} A_{S_i}^T r_i
        hess = p * sum_i ||r_i||^{p-2} A_{S_i}^T A_{S_i}
             + p(p-2) * sum_i ||r_i||^{p-4} (A_{S_i}^T r_i)(A_{S_i}^T r_i)^T
    """
    A = problem["A"]
    off = problem["offsets"]
    m = problem["m"]
    d = problem["d"]
    r = residuals(problem, x)
    grad = np.zeros(d, dtype=np.float64)
    hess = np.zeros((d, d), dtype=np.float64)
    val = 0.0
    for i in range(m):
        ri = r[off[i]:off[i + 1]]
        Ai = A[off[i]:off[i + 1], :]
        ni = float(ri @ ri)
        nrm = np.sqrt(ni)  # ||r_i||_2
        if nrm == 0.0:
            # contributions vanish for p>2 (||r||^{p-2}=0); p=2 leaves plain A^T A
            if p == 2.0:
                hess += 2.0 * (Ai.T @ Ai)  # p=2: f=sum ||r||^2, grad=2 A^T r, hess=2 A^T A
                grad += 2.0 * (Ai.T @ ri)
                val += ni
            continue
        Atr = Ai.T @ ri
        AtA = Ai.T @ Ai
        val += nrm ** p
        grad += p * (nrm ** (p - 2)) * Atr
        hess += p * (nrm ** (p - 2)) * AtA
        if p != 2.0:
            hess += p * (p - 2) * (nrm ** (p - 4)) * np.outer(Atr, Atr)
    return float(val), grad, hess
