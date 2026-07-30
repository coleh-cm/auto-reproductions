"""Smoothed surrogate objective and its calculus (E7, E8, E10, T2, T3).

The empirical objective is the worst-group *mean-squared error*

    F(x) = max_i  ell_i(x),   ell_i(x) = (1/n_i) ||A_{S_i} x - b_{S_i}||_2^2      (E1, E2)

The theory text folds the per-row normalisation 1/sqrt(n_i) into the data
(`paper/body.tex:27`), so its ``f(x) = max_i ||A_{S_i}x - b_{S_i}||_2`` is the
**square root** of the empirical ``F`` (SPEC section 2 "Notation trap").  The
argmins coincide because sqrt is monotone, so optimising the surrogate optimises
``F``.

To keep the calculus identical to the paper we work with the *folded* per-group
design / residual

    A~_{S_i} = A_{S_i} / sqrt(n_i) ,   r~_{S_i} = A~_{S_i} x - b_{S_i}/sqrt(n_i)

for which  ||r~_{S_i}||_2^2 = ell_i(x).  The smoothed surrogate (E7,
`paper/body.tex:55`, label ``eq:intro_smooth_fair_objective``) is then

    ftilde_{beta,delta}(x) = beta * log sum_i exp( (sqrt(delta^2 + ||r~_i||^2) - delta) / beta )

with derivatives (E10, `paper/body.tex:287-288,431-435`).  Writing
    s_i = sqrt(delta^2 + ||r~_i||^2) - delta ,   h_i = s_i + delta = sqrt(delta^2 + ||r~_i||^2)
    sigma = softmax(s / beta)                                  # [m], >=0, sums to 1
    grad_s_i = A~_i^T r~_i / h_i                               # [d]
    hess_s_i = (1/h_i) A~_i^T ( I - r~_i r~_i^T / h_i^2 ) A~_i   # [d,d]
    g = sum_i sigma_i grad_s_i                                 # = grad ftilde
we have
    grad ftilde = g
    hess ftilde = (1/beta)( sum_i sigma_i grad_s_i grad_s_i^T - g g^T ) + sum_i sigma_i hess_s_i

The approximation guarantee (E8, `paper/body.tex:231`, Lemma 6.1) is
    |ftilde_{beta,delta}(x) - sqrt(F(x))| <= beta log m + delta
(the inner functions bound the *norm*, hence sqrt(F)).

The regularised variant fhat (T2, `paper/body.tex:182`) adds
    reg * || W^{1/2} A (x - x0) ||_2^2   with   reg = eps / (1000 * min{rank(A), m})
(the Algorithm-1 coefficient; the analysis coefficient eps/(110 R^2) with
R=(2+eps)sqrt(2(d+1)) at `paper/body.tex:580,586` is recorded in SPEC section 6
item 8 and selectable via ``coef_kind``).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import GroupProblem

__all__ = [
    "SmoothObjective",
    "make_smoothed",
    "make_regularized",
    "smoothed_value",
    "smoothed_grad",
]


class SmoothObjective(Protocol):
    """Contract for the smoothed / regularised surrogate (SPEC section 7)."""

    def value(self, x: np.ndarray) -> float: ...
    def grad(self, x: np.ndarray) -> np.ndarray: ...
    def hess(self, x: np.ndarray) -> np.ndarray: ...


# ------------------------------------------------------------------------------------------------
# Numerically stable softmax / logsumexp helpers
# ------------------------------------------------------------------------------------------------
def _lse(m: np.ndarray, beta: float) -> float:
    """beta * log sum_i exp(m_i / beta), computed stably (lse_beta)."""
    m = np.asarray(m, dtype=np.float64)
    mx = float(np.max(m))
    # shift: lse_beta(s) = beta*(M/beta) + beta*log sum exp((s-M)/beta)
    return mx + beta * float(np.log(np.sum(np.exp((m - mx) / beta))))


def _softmax(m: np.ndarray, beta: float) -> np.ndarray:
    """sigma = softmax(s / beta), shape [m], sums to 1."""
    m = np.asarray(m, dtype=np.float64)
    mx = np.max(m)
    e = np.exp((m - mx) / beta)
    return e / float(np.sum(e))


# ------------------------------------------------------------------------------------------------
# Folded problem
# ------------------------------------------------------------------------------------------------
class _Folded:
    """Per-group folded design/residual where ||r~_i||^2 = ell_i(x).

    Built once from a ``GroupProblem`` so the smoothed calculus matches the
    paper's formulas verbatim (every 1/sqrt(n_i) absorbed into the rows).
    """

    __slots__ = ("A", "b", "slices", "m", "d", "sizes", "_At")

    def __init__(self, problem: GroupProblem) -> None:
        sizes = problem.sizes.astype(np.float64)
        inv_sqrt = 1.0 / np.sqrt(sizes)                 # per-group 1/sqrt(n_i)
        row_scale = np.repeat(inv_sqrt, problem.sizes.astype(np.int64))  # [n]
        # fold the normalisation into the rows: A~ = A / sqrt(n_i), b~ = b / sqrt(n_i)
        self.A = problem.A * row_scale[:, None]
        self.b = problem.b * row_scale
        self.slices = problem.slices
        self.m = problem.m
        self.d = problem.d
        self.sizes = sizes
        self._At = self.A.T.copy()

    # ---- per-group quantities -------------------------------------------------
    def residuals(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x - self.b                     # [n], folded

    def group_norm_sq(self, x: np.ndarray) -> np.ndarray:
        """||r~_i||_2^2 = ell_i(x), shape [m] (E1)."""
        r = self.residuals(x)
        out = np.empty(self.m, dtype=np.float64)
        for i, (s, e) in enumerate(self.slices):
            out[i] = float(r[s:e] @ r[s:e])
        return out


# ------------------------------------------------------------------------------------------------
# Smoothed surrogate ftilde_{beta,delta}
# ------------------------------------------------------------------------------------------------
class Smoothed(SmoothObjective):
    """f~_{beta,delta} with value/grad/hess (E7, E10)."""

    def __init__(self, folded: _Folded, beta: float, delta: float) -> None:
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")
        self.f = folded
        self.beta = float(beta)
        self.delta = float(delta)

    # -- inner per-group quantities --
    def _inner(self, x: np.ndarray):
        """Return (r, normsq, h, s, grad_s, hess_s).

        r       : [n]  folded residual
        normsq  : [m]  ||r~_i||^2 = ell_i
        h       : [m]  sqrt(delta^2 + ||r~_i||^2)
        s       : [m]  h - delta
        grad_s  : list[m] of [d]   grad_s_i = A~_i^T r~_i / h_i
        hess_s  : list[m] of [d,d] hess_s_i = (1/h_i) A~_i^T (I - r~_i r~_i^T/h_i^2) A~_i
        """
        f = self.f
        r = f.residuals(x)
        normsq = np.empty(f.m)
        grad_s = [None] * f.m
        hess_s = [None] * f.m
        h = np.empty(f.m)
        for i, (s, e) in enumerate(f.slices):
            ri = r[s:e]
            ni2 = float(ri @ ri)
            normsq[i] = ni2
            hi = float(np.sqrt(self.delta * self.delta + ni2))
            h[i] = hi
            Ai = f.A[s:e]                                  # [n_i, d]
            gsi = Ai.T @ ri                                # [d] = A~_i^T r~_i
            gsi = gsi / hi
            grad_s[i] = gsi
            # hess_s_i = (1/h_i) A~_i^T (I - r~_i r~_i^T / h_i^2) A~_i
            AtA = Ai.T @ Ai                               # [d,d]
            # outer correction: A~_i^T r~_i r~_i^T A~_i = (A~_i^T r~_i)(A~_i^T r~_i)^T
            outer = np.outer(gsi * hi, gsi * hi) / (hi * hi)  # = (A~_i^T r~_i)(...)^T / h_i^2
            hess_s[i] = (AtA - outer) / hi
        s = h - self.delta
        return r, normsq, h, s, grad_s, hess_s

    def value(self, x: np.ndarray) -> float:
        f = self.f
        x = np.asarray(x, dtype=np.float64).ravel()
        normsq = f.group_norm_sq(x)
        h = np.sqrt(self.delta * self.delta + normsq)
        s = h - self.delta
        return _lse(s, self.beta)

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        _, _, _, s, grad_s, _ = self._inner(x)
        sigma = _softmax(s, self.beta)
        g = np.zeros(self.f.d, dtype=np.float64)
        for i in range(self.f.m):
            g += sigma[i] * grad_s[i]
        return g

    def hess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        _, _, _, s, grad_s, hess_s = self._inner(x)
        sigma = _softmax(s, self.beta)
        g = np.zeros(self.f.d, dtype=np.float64)
        for i in range(self.f.m):
            g += sigma[i] * grad_s[i]
        H = np.zeros((self.f.d, self.f.d), dtype=np.float64)
        # (1/beta) ( sum_i sigma_i grad_s_i grad_s_i^T - g g^T )
        outer_sum = np.zeros((self.f.d, self.f.d), dtype=np.float64)
        for i in range(self.f.m):
            gsi = grad_s[i]
            outer_sum += sigma[i] * np.outer(gsi, gsi)
        H += (outer_sum - np.outer(g, g)) / self.beta
        # + sum_i sigma_i hess_s_i
        for i in range(self.f.m):
            H += sigma[i] * hess_s[i]
        return H


# ------------------------------------------------------------------------------------------------
# Regularised surrogate fhat = ftilde + reg * || W^{1/2} A (x - x0) ||^2   (T2)
# ------------------------------------------------------------------------------------------------
class Regularized(SmoothObjective):
    """fhat(x) = ftilde(x) + reg * || sqrt(W) A (x - x0) ||_2^2  (T2, `paper/body.tex:182`).

    ``sqrt_w`` is the per-row diagonal of W^{1/2} (length n).  The Hessian of
    the regulariser is ``2 * reg * A^T W A`` (A^T B A structure, T5); the added
    term is convex so fhat stays a smoothed surrogate with the same A^T B A
    Hessian shape.
    """

    def __init__(
        self,
        base: Smoothed,
        problem: GroupProblem,
        sqrt_w: np.ndarray,
        x0: np.ndarray,
        reg: float,
    ) -> None:
        self.base = base
        self.A = problem.A
        sw = np.asarray(sqrt_w, dtype=np.float64).ravel()
        if sw.shape[0] != problem.A.shape[0]:
            raise ValueError(f"sqrt_w has length {sw.shape[0]}, expected {problem.A.shape[0]}")
        self.sqrt_w = sw
        self.x0 = np.asarray(x0, dtype=np.float64).ravel()
        self.reg = float(reg)
        self.d = base.f.d
        # precompute A^T W A and A^T W b contribution lazily; store scaled A
        self._AW = self.A * sw[:, None]                      # W^{1/2} A   [n, d]
        self._AtWA = self._AW.T @ self.A                     # A^T W A     [d, d]

    # -- regulariser --
    def _reg_value(self, x: np.ndarray) -> float:
        d = self._AW @ (x - self.x0)                          # W^{1/2} A (x - x0)
        return self.reg * float(d @ d)

    def _reg_grad(self, x: np.ndarray) -> np.ndarray:
        # 2 reg A^T W A (x - x0)
        return 2.0 * self.reg * (self._AtWA @ (x - self.x0))

    def _reg_hess(self) -> np.ndarray:
        return 2.0 * self.reg * self._AtWA

    def value(self, x: np.ndarray) -> float:
        return self.base.value(x) + self._reg_value(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.base.grad(x) + self._reg_grad(x)

    def hess(self, x: np.ndarray) -> np.ndarray:
        return self.base.hess(x) + self._reg_hess()


# ------------------------------------------------------------------------------------------------
# Factories (SPEC section 7 signatures)
# ------------------------------------------------------------------------------------------------
def make_smoothed(problem: GroupProblem, beta: float, delta: float) -> Smoothed:
    """E7/E10 smoothed surrogate f~_{beta,delta} for the worst-group MSE objective."""
    return Smoothed(_Folded(problem), beta, delta)


def make_regularized(
    problem: GroupProblem,
    beta: float,
    delta: float,
    sqrt_w: np.ndarray,
    x0: np.ndarray,
    coef: float,
) -> Regularized:
    """T2 regularised surrogate fhat = ftilde + coef * || W^{1/2} A (x - x0) ||^2.

    ``coef`` is the *regulariser coefficient* (already the final scalar); the
    caller resolves the Algorithm-1 vs analysis coefficient (SPEC section 6
    item 8) and passes the chosen value here.
    """
    base = make_smoothed(problem, beta, delta)
    return Regularized(base, problem, sqrt_w, x0, coef)


# ------------------------------------------------------------------------------------------------
# Standalone helpers used by tests / invariants
# ------------------------------------------------------------------------------------------------
def smoothed_value(problem: GroupProblem, x: np.ndarray, beta: float, delta: float) -> float:
    """f~_{beta,delta}(x) without building a full object (E7)."""
    return make_smoothed(problem, beta, delta).value(x)


def smoothed_grad(problem: GroupProblem, x: np.ndarray, beta: float, delta: float) -> np.ndarray:
    return make_smoothed(problem, beta, delta).grad(x)
