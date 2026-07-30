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
    """Block-Lewis overestimates ``w [m]`` (E12), MO25 Algorithm 2 verbatim.

    Follows ``paper/mo25_main.tex:1813-1822`` (their input matrix is the augmented
    ``Â=[A|b]`` whose column count ``n_cols = d+1`` plays the role of MO25's ``n``;
    SPEC section 3.5):

      1. ``b^{(1)} = (n_cols / m) * 1_m``                       (line 1814, init)
      2. for ``t = 1, ..., T-1``:                               (line 1815)
           ``B^{(t)}_jj = b^{(t)}_i`` for ``j in S_i``
           ``tilde_tau^{(t)} = OverLev((B^{(t)})^{1/2 - 1/p} Â)``  (line 1817)
           ``b^{(t+1)}_i = sum_{j in S_i} tilde_tau_j``           (line 1818)
      3. ``b_bar = (1/T) sum_{t=1}^{T} b^{(t)}``               (line 1820, **includes
         the init** ``b^{(1)}`` and the ``T-1`` computed iterates ``b^{(2)}..b^{(T)}``)
      4. return ``w = (3/2) * b_bar``                            (line 1821)

    Parameters
    ----------
    problem : GroupProblem
    p : float | None
        ``None`` means ``p = inf`` (the max objective, Algorithm 1 line 1).  For
        finite ``p`` the leverage exponent ``1/2 - 1/p`` (line 1817) is applied --
        the per-row weighting inside ``OverLev`` is ``w_j^{1-2/p}``, *not* the
        ``w_j`` of the ``p = inf`` specialisation.  Downstream geometry uses the
        same exponent (handled in :func:`geometry_matrix`).
    n_iters : int | None
        ``T = O(log m)`` (``paper/mo25_main.tex:1890-1891``).  ``None`` ->
        ``ceil(2 ln m)`` (SPEC section 6 item 9).  Exactly ``T-1`` sweeps are run
        (line 1815) and the average spans ``T`` iterates (line 1820).

    Returns
    -------
    w : np.ndarray, shape [m], >= 0.  By construction an overestimate whose sum is
        <= 2 rank(Ahat) (``paper/other_proofs.tex`` Lemma 5.6 / Theorem 3.4).
    """
    Ahat = problem.augmented()                       # [n, d+1]
    m = problem.m
    if n_iters is None:
        n_iters = max(2, int(np.ceil(2.0 * np.log(max(m, np.e)))))
    T = max(2, int(n_iters))     # need T-1 >= 1 sweep and T >= 2 terms to average

    # exponent for the per-row weighting inside OverLev (MO25 line 1817):
    # OverLev((B)^{1/2-1/p} A) returns leverage scores of (B^{1/2-1/p} A), i.e.
    #   tau_j = w_j^{1-2/p} * a_j^T (A^T W^{1-2/p} A)^{-1} a_j .
    # p = inf  -> 1-2/p = 1  -> plain W^{1/2} A leverage scores.
    if p is None:
        q = 1.0
    else:
        if p <= 0:
            raise ValueError(f"p must be > 0 (or None=inf), got {p}")
        q = 1.0 - 2.0 / float(p)

    # b^{(1)} = (n_cols / m) * 1_m   (MO25 line 1814; n_cols = column count of Â)
    n_cols = Ahat.shape[1]                  # = problem.d + 1
    w = np.full(m, n_cols / float(m), dtype=np.float64)
    # the MO25 average (line 1820) INCLUDES the init b^{(1)}
    acc = w.copy()
    slices = problem.slices
    for _ in range(T - 1):                   # line 1815: t = 1, ..., T-1
        # W_jj = w_i for j in S_i  (line 1816)
        Wdiag = np.repeat(w, problem.sizes.astype(np.int64))
        # OverLev((B)^{1/2-1/p} Â)  (line 1817, exact solve): leverage scores of
        # W^{1/2-1/p} Â  ==  leverage_scores(Â, Wdiag**q)  (see leverage_scores)
        tau = leverage_scores(Ahat, Wdiag ** q)
        # b^{(t+1)}_i = sum_{j in S_i} tilde_tau_j   (line 1818)
        w_new = np.empty(m, dtype=np.float64)
        for i, (s, e) in enumerate(slices):
            w_new[i] = float(np.sum(tau[s:e]))
        acc += w_new                          # average spans b^{(2)} .. b^{(T)}
        w = w_new
    # w = (3/2) * (1/T) * sum_{t=1}^{T} b^{(t)}   (MO25 line 1820-1821)
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
