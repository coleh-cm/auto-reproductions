"""Block Lewis weights and the data-dependent geometry (SPEC E6, E7, E8, E9, E11).

Re-derived concretely from [MO25, Algorithm 2] (arXiv:2311.10013) for *exact*
leverage scores (d is tiny, so a Cholesky on the (d+1)x(d+1) Gram is exact and
cheap).  See SPEC E8 for the pseudocode we implement.

Notation:  w in R^m  are per-block weights;  W in R^{n x n} diagonal with
W_{jj}=w_i for j in S_i.  The geometry is M = A^T W^{1-2/p} A  (p=inf -> A^T W A),
which is what Algorithm 1 line 4 and the proof of Lemma 3.5 actually use
(`other_proofs.tex:74`; U14 corrects the statement typo).
"""
from __future__ import annotations

import numpy as np

from .problem import Problem


# ---------------------------------------------------------------------------
# leverage scores
# ---------------------------------------------------------------------------

def leverage_scores(M: np.ndarray) -> np.ndarray:
    """tau_j(M) = a_j^T (M^T M)^{-1} a_j for rows a_j of M.  M is [n, k].

    Computed as the diagonal of  M (M^T M)^{-1} M^T  = row-wise quadratic form.
    Uses a pseudoinverse if M^T M is rank-deficient (`body.tex:27` convention).
    """
    G = M.T @ M
    # symmetric positive (semi)definite solve; add tiny jitter for Cholesky stability
    k = G.shape[0]
    try:
        L = np.linalg.cholesky(G + 1e-12 * np.eye(k))
        # solve G^{-1} M^T  via Cholesky
        # MinvMT = G^{-1} M^T
        Mt = M.T
        # forward/back solve
        y = np.linalg.solve(L, Mt)              # L y = M^T
        GinvMt = np.linalg.solve(L.T, y)        # L^T z = y  => z = G^{-1} M^T
        # tau_j = row_j(M) . (G^{-1} row_j(M)^T)  = diag(M @ GinvMt)
        tau = np.einsum("ij,ji->i", M, GinvMt)
    except np.linalg.LinAlgError:
        Ginv = np.linalg.pinv(G)
        tau = np.einsum("ij,jk,ik->i", M, Ginv, M)
    return np.maximum(tau, 0.0)


# ---------------------------------------------------------------------------
# block Lewis weights (E8)
# ---------------------------------------------------------------------------

def block_lewis_weights(problem: Problem, p: float, rounds: int | None = None,
                        seed=None) -> np.ndarray:
    """Block Lewis overestimate w in R^m  (E8 / SPEC).  Returns per-block weights.

    Augmented matrix  A_hat = [A | b] in R^{n x (d+1)}  (Alg.1 line 1, Lemma 3.5).
    Exponent  e_p = 1/2 - 1/p   (p=inf -> 1/2).

    Iteration (MO25 alg:blw, exact-lev regime):
        v <- (n_cols/m) * 1_m                       # n_cols = d+1
        for t in 1..T-1,  T = ceil(2 ln m / ln(1.5)):
            tau_j = leverage scores of  V^{e_p} A_hat   (V block-diag, V_{jj}=v_i)
            v_i <- sum_{j in S_i} tau_j
        w <- (3/2) * mean_t v^{(t)}

    Guarantee: w is a block Lewis overestimate (E6) with ||w||_1 <= (3/2)(d+1)
    <= 2(rank(A)+1)  (SPEC E8; MO25 Lemma 5.6).
    """
    A = problem["A"]
    b = problem["b"]
    off = problem["offsets"]
    m = problem["m"]
    n = A.shape[0]
    d = problem["d"]
    n_cols = d + 1
    A_hat = np.column_stack([A, b])              # [n, d+1]

    if np.isinf(p):
        e_p = 0.5
    else:
        e_p = 0.5 - 1.0 / p

    if rounds is None:
        # SPEC E8: T = ceil(2 ln m / ln(3/2)); iterate t=1..T-1  => T-1 update steps
        rounds = int(np.ceil(2.0 * np.log(max(m, 2)) / np.log(1.5)))
    T = max(rounds, 1)
    # SPEC E8 / MO25 alg:blw run t=1..T-1 (T-1 update steps), averaging v^(1)..v^(T-1).
    n_steps = max(T - 1, 1)

    v = np.full(m, n_cols / m, dtype=np.float64)  # init
    v_history = []
    # block index per row
    row_block = np.empty(n, dtype=np.int64)
    for i in range(m):
        row_block[off[i]:off[i + 1]] = i

    for t in range(n_steps):
        # V^{e_p} A_hat: rows scaled by v_{block}^{e_p}
        scale = np.power(v[row_block], e_p)
        M = scale[:, None] * A_hat
        tau = leverage_scores(M)                  # [n]
        # aggregate per block
        v_new = np.zeros(m, dtype=np.float64)
        np.add.at(v_new, row_block, tau)
        v_history.append(v_new.copy())
        v = v_new

    mean_v = np.mean(np.array(v_history), axis=0)
    w = 1.5 * mean_v
    # guard against zeros / negatives
    w = np.maximum(w, 1e-12)
    return w


# ---------------------------------------------------------------------------
# geometry matrix M (E11/E13)
# ---------------------------------------------------------------------------

def geometry_M(problem: Problem, w: np.ndarray | None, p: float) -> np.ndarray:
    """M = A^T W^{1-2/p} A  (p=inf -> A^T W A); naive fallback A^T A when w is None
    (Euclidean geometry, the "W=I" branch of E11).  W is block-diagonal with
    W_{jj} = w_i for j in S_i.

    E11 switch: if sum_i w_i >= m, the paper resets W <- I  (Alg.1 lines 2-3).
    The caller passes the *already-switched* choice: pass w=None for the naive
    Euclidean geometry, or the computed w for Lewis geometry.
    """
    A = problem["A"]
    off = problem["offsets"]
    m = problem["m"]
    if w is None:
        return A.T @ A
    if np.isinf(p):
        exponent = 1.0  # 1 - 2/p = 1 for p=inf
    else:
        exponent = 1.0 - 2.0 / p
    d = A.shape[1]
    M = np.zeros((d, d), dtype=np.float64)
    for i in range(m):
        Ai = A[off[i]:off[i + 1], :]
        M += (w[i] ** exponent) * (Ai.T @ Ai)
    return M


def should_reset_W(w: np.ndarray, m: int) -> bool:
    """E11: reset W <- I when sum_i w_i >= m  (Alg.1 line 2, body.tex:173)."""
    return float(np.sum(w)) >= m


# ---------------------------------------------------------------------------
# weighted least-squares initialization (E9)
# ---------------------------------------------------------------------------

def wls_init(problem: Problem, D_diag_rows: np.ndarray) -> np.ndarray:
    """x_0 = (A^T D A)^{-1} A^T D b  (E9; Alg.1 line 4 / Alg.5 line 2).

    D is a diagonal matrix given as a *per-row* vector of length n.  For p=inf
    the Lewis geometry passes D = w_rows (W block-diagonal); the Euclidean
    geometry passes D = ones.  Pseudoinverse if rank-deficient (body.tex:27).
    """
    A = problem["A"]
    b = problem["b"]
    d = problem["d"]
    sqrtD = np.sqrt(np.maximum(D_diag_rows, 0.0))
    AD = sqrtD[:, None] * A
    bD = sqrtD * b
    G = AD.T @ AD
    try:
        x0 = np.linalg.solve(G + 1e-12 * np.eye(d), AD.T @ bD)
    except np.linalg.LinAlgError:
        x0 = np.linalg.pinv(G) @ (AD.T @ bD)
    return x0


def lewis_warm_start(problem: Problem, p: float = np.inf) -> tuple[np.ndarray, np.ndarray, bool]:
    """Full Alg.1 lines 1-4 pipeline: compute block Lewis weights on [A|b],
    apply the E11 switch, and return the warm-start x_0 plus the per-row weight
    vector w_rows actually used (and whether the reset branch fired).

    Returns (x0 [d], w_rows [n], reset_fired: bool), where w_rows is the per-row
    D used by wls_init (D = W for p=inf, D = W^{1-2/p} for finite p, or ones if
    the E11 reset fired).
    """
    w = block_lewis_weights(problem, p=p)
    reset = should_reset_W(w, problem["m"])
    if reset:
        w_rows = np.ones(problem["A"].shape[0], dtype=np.float64)
    else:
        # E9 requires D = W (p=inf) or D = W^{1-2/p} (finite p) per row.  For
        # p=inf the exponent 1-2/p = 1 so D = w; for finite p each block's
        # per-row weight must be w_i^{1-2/p} (SPEC.md:114; other_proofs.tex:74,
        # corrected by U14).  block_lewis_weights returns p-correct weights w,
        # so we apply the p-dependent exponent here before forming the per-row
        # vector.  (Latent until a finite-p interpolation solver calls this;
        # ball_oracle hard-codes p=inf where this branch is a no-op.)
        exp = 1.0 if np.isinf(p) else (1.0 - 2.0 / float(p))
        w_D = w ** exp if exp != 1.0 else w
        w_rows = np.repeat(w_D, np.diff(problem["offsets"]))
    x0 = wls_init(problem, w_rows)
    return x0, w_rows, reset
