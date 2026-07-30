"""Block Lewis weights and the trust-region geometry (E12, E13, E14, E15).

References (all grep-able in ``paper/``):
  * E12 block-Lewis iteration: ``paper/mo25_main.tex:1814-1821`` (MO25 Algorithm 2,
    specialised to all-inner-norms-ell2, p=inf so the exponent 1/2 - 1/p = 1/2);
    consumption in Algorithm 1: ``paper/body.tex:169-174``.
  * E13 ellipsoid guarantee
        ||A x - c b||_{G,inf} <= || W^{1/2}(A x - c b) ||_2 <= sqrt(2(rank(A)+1)) ||A x - c b||_{G,inf}
    ``paper/body.tex:170-172``; proof ``paper/other_proofs.tex:87-104``.
  * E14 weighted-LS initializer  x0 = (A^T W A)^{-1} A^T W b  ``paper/body.tex:176``.
  * E15 naive-geometry distortion  ||Ax-b||_2/sqrt(m) <= ||Ax-b||_{G,inf} <= ||Ax-b||_2
    ``paper/body.tex:599-602``.

Conventions: a block-Lewis *overestimate* (``paper/other_proofs.tex:20``) is a
vector ``w in R^m_{>=0}`` with ``W`` the diagonal [n,n] matrix whose entries are
``w_i`` on the rows of group ``i``, satisfying  sum_{j in S_i} tau_j(W^{1/2-1/p} A) <= w_i.
The geometry actually used downstream is ``M = A^T W^{1-2/p} A`` (note **A**, not
the augmented Ahat) -- ``paper/body.tex:511``; for p=inf the exponent 1-2/p = 1 so
``M = A^T W A``.

The W = I_n reset (Algorithm 1 line 3, ``paper/body.tex:173-174``): if
``sum_i w_i >= m`` we fall back to the naive Euclidean geometry, which is why the
Lewis arm at the identity setting is *exactly* the Euclidean arm (degeneracy test).
"""

from __future__ import annotations

import numpy as np

from .types import GroupProblem

__all__ = [
    "block_lewis_weights",
    "geometry_matrix",
    "weighted_ls_init",
    "leverage_scores",
    "reset_to_identity",
    "expand_weights",
]


# ------------------------------------------------------------------------------------------------
# Leverage scores
# ------------------------------------------------------------------------------------------------
def leverage_scores(A: np.ndarray, Wdiag: np.ndarray | None = None) -> np.ndarray:
    """tau_j(A) = a_j^T (A^T A)^{-1} a_j  (``paper/other_proofs.tex:11-13``).

    If ``Wdiag`` (per-row diagonal of a nonneg weight matrix) is given, returns
    the leverage scores of ``W^{1/2} A``: tau_j = a_j^T (A^T W A)^{-1} a_j * W_jj,
    i.e. the per-row diagonal of ``A (A^T W A)^{-1} A^T W``.

    A pseudo-inverse is used so rank-deficient matrices do not raise: the
    *overestimate* definition only needs the scores to be defined where the
    pseudoinverse puts its mass, and ``numpy``'s ``pinv`` returns 0 on the
    nullspace which keeps the inequality ``sum tau_j <= rank`` intact.
    """
    A = np.asarray(A, dtype=np.float64)
    n, d = A.shape
    if Wdiag is None:
        G = A.T @ A
    else:
        w = np.asarray(Wdiag, dtype=np.float64)
        G = (A * w[:, None]).T @ A          # A^T W A
    Ginv = np.linalg.pinv(G)
    # tau_j = a_j^T Ginv a_j  (diagonal of A Ginv A^T)
    tau = np.einsum("ij,jk,ik->i", A, Ginv, A)
    if Wdiag is not None:
        tau = tau * np.asarray(Wdiag, dtype=np.float64)
    # clip tiny negatives from float roundoff
    tau = np.clip(tau, 0.0, None)
    return tau


# ------------------------------------------------------------------------------------------------
# Block Lewis weights (MO25 Algorithm 2, specialised; E12)
# ------------------------------------------------------------------------------------------------
def block_lewis_weights(
    problem: GroupProblem,
    p: float | None = None,
    n_iters: int | None = None,
) -> np.ndarray:
    """Block-Lewis overestimates ``w [m]`` (E12).

    Parameters
    ----------
    problem : GroupProblem
    p : float | None
        ``None`` means ``p = inf`` (the max objective, Algorithm 1 line 1).  For
        finite p the algorithm is the same; only the downstream geometry exponent
        1-2/p changes (handled in :func:`geometry_matrix`).
    n_iters : int | None
        ``T = O(log m)`` (``paper/mo25_main.tex:1890-1891``).  ``None`` ->
        ``ceil(2 ln m)`` (SPEC section 6 item 9).  The averaging in MO25 line
        1820-1821 (``w = (3/2)(1/T) sum_t w^{(t)}``) is applied over ``T`` sweeps.

    Returns
    -------
    w : np.ndarray, shape [m], >= 0.  By construction an overestimate whose sum is
        <= 2 rank(Ahat) (``paper/other_proofs.tex`` Lemma 5.6 / Theorem 3.4).
    """
    # augmented matrix Ahat = [A | b]  (``paper/other_proofs.tex:53``)
    Ahat = problem.augmented()                       # [n, d+1]
    m = problem.m
    rank = int(np.linalg.matrix_rank(Ahat))
    if n_iters is None:
        n_iters = max(1, int(np.ceil(2.0 * np.log(max(m, np.e)))))
    T = max(1, int(n_iters))

    # w^(0) = ((rank(Ahat))/m) * 1_m   (their n = our rank(Ahat) <= d+1; mo25_main.tex:1814)
    w = np.full(m, rank / float(m), dtype=np.float64)
    acc = np.zeros(m, dtype=np.float64)
    slices = problem.slices
    for t in range(T):
        # W_jj = w_i for j in S_i
        Wdiag = np.repeat(w, problem.sizes.astype(np.int64))
        # leverage scores of W^{1/2} Ahat  (mo25_main.tex:1817, exact-solve OverLev)
        tau = leverage_scores(Ahat, Wdiag)
        # w_i^{(t+1)} = sum_{j in S_i} tau_j   (mo25_main.tex:1818)
        w_new = np.empty(m, dtype=np.float64)
        for i, (s, e) in enumerate(slices):
            w_new[i] = float(np.sum(tau[s:e]))
        acc += w_new
        w = w_new
    # w = (3/2) * (1/T) * sum_t w^{(t)}   (mo25_main.tex:1820-1821)
    w = 1.5 * (acc / float(T))
    w = np.clip(w, 0.0, None)
    return w


def reset_to_identity(w: np.ndarray, m: int) -> np.ndarray:
    """Algorithm 1 line 3 reset: if ``sum_i w_i >= m`` return uniform ``1``.

    The reset collapses the Lewis geometry to ``M = A^T I A = A^T A``, i.e. the
    *naive Euclidean* geometry (``paper/body.tex:173-174``).  This is the
    degeneracy point: the Lewis arm at the identity setting equals the Euclidean
    arm exactly.
    """
    if float(np.sum(w)) >= float(m):
        return np.ones(m, dtype=np.float64)
    return w


def expand_weights(w: np.ndarray, problem: GroupProblem) -> np.ndarray:
    """Per-row diagonal of W: ``W_jj = w_i`` for ``j in S_i`` (length n)."""
    return np.repeat(np.asarray(w, dtype=np.float64), problem.sizes.astype(np.int64))


# ------------------------------------------------------------------------------------------------
# Geometry matrix M (E13, E15)
# ------------------------------------------------------------------------------------------------
def geometry_matrix(
    problem: GroupProblem,
    w: np.ndarray | None = None,
    p: float | None = None,
) -> np.ndarray:
    """Trust-region metric ``M`` ([d,d] PSD).

    * ``w is None``  -> naive Euclidean  ``M = A^T A``  (E15, ``paper/body.tex:599-602``).
    * else           -> Lewis geometry  ``M = A^T W^{1-2/p} A``
      (``paper/body.tex:511``); for ``p = None`` (inf) the exponent is 1 so
      ``M = A^T W A``; for finite p it is ``A^T W^{1-2/p} A``.

    ``w`` is the per-group vector; it is expanded to the per-row diagonal here.
    """
    A = problem.A
    if w is None:
        return A.T @ A
    w = np.asarray(w, dtype=np.float64)
    if p is None:
        exponent = 1.0                                  # 1 - 2/inf = 1
    else:
        if p <= 0:
            raise ValueError(f"p must be > 0 (or None=inf), got {p}")
        exponent = 1.0 - 2.0 / float(p)
    Wdiag = np.repeat(w, problem.sizes.astype(np.int64))     # per-row W_jj
    Wpow = Wdiag ** exponent
    return (A * Wpow[:, None]).T @ A                     # A^T W^{1-2/p} A


def weighted_ls_init(problem: GroupProblem, w: np.ndarray | None, p: float | None = None) -> np.ndarray:
    """E14 weighted-LS initializer  x0 = (A^T W A)^{-1} A^T W b  (``paper/body.tex:176``).

    For the naive geometry (``w is None``) this is ordinary pooled OLS.  Uses a
    pseudo-inverse for rank-deficient designs (synthetic instance).  The exponent
    matches :func:`geometry_matrix` so this is the minimiser of
    ``|| W^{1/2-1/p} (A x - b) ||_2`` (the norm the geometry approximates).
    """
    A = problem.A
    b = problem.b
    if w is None:
        exponent = 1.0
        Wdiag = np.ones(problem.n)
    else:
        if p is None:
            exponent = 1.0
        else:
            exponent = 1.0 - 2.0 / float(p)
        Wdiag = np.repeat(np.asarray(w, dtype=np.float64), problem.sizes.astype(np.int64))
    Wpow = Wdiag ** exponent
    AW = A * Wpow[:, None]
    AtWA = AW.T @ A
    AtWb = AW.T @ b
    if np.linalg.matrix_rank(AtWA) < problem.d:
        return np.asarray(np.linalg.pinv(AtWA) @ AtWb, dtype=np.float64)
    return np.asarray(np.linalg.solve(AtWA, AtWb), dtype=np.float64)
