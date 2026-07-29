"""Problem data structure and helpers for GDR least-squares (SPEC §1, §4).

Paper convention (`body.tex:27`): the per-group factors 1/sqrt(n_i) are *folded*
into the data at load time, so every formula below uses the folded (A, b) and
never carries n_i. After folding the group loss is exactly

    ell_i(x) = (1/n_i) * ||(orig A_{S_i}) x - (orig b_{S_i})||^2
             = ||(folded A_{S_i}) x - (folded b_{S_i})||^2     (folded = orig/sqrt(n_i))

and the worst-group objective is F(x) = max_i ell_i(x) = ||Ax - b||_{G_inf}^2
(its unsquared square root f(x) = ||Ax - b||_{G_inf} = max_i ||r_i||_2 is what the
theory smooths, `body.tex:27`).
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class Problem(TypedDict):
    # stacked, *folded* design and responses (SPEC §1)
    A: np.ndarray            # [n, d] float64
    b: np.ndarray            # [n]   float64
    offsets: np.ndarray      # [m+1] int64, contiguous group row ranges; group i -> rows [offsets[i], offsets[i+1])
    name: str
    m: int
    d: int
    n_i: np.ndarray          # [m] int, *pre-fold* per-group sizes (for reporting / unfolding)
    meta: dict


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def make_problem(A_blocks, b_blocks, name: str, n_i_pre: np.ndarray | None = None,
                 meta: dict | None = None) -> Problem:
    """Build a folded Problem from a list of per-group (orig) design/response blocks.

    Folding: A_{S_i} <- A_{S_i}/sqrt(n_i), b_{S_i} <- b_{S_i}/sqrt(n_i)  (body.tex:27).
    Groups are stored contiguously in the given order; offsets record the ranges.
    """
    m = len(A_blocks)
    if m == 0:
        raise ValueError("need at least one group")
    d = int(A_blocks[0].shape[1])
    n_i = np.array([int(bl.shape[0]) for bl in A_blocks], dtype=np.int64)
    if n_i_pre is None:
        n_i_pre = n_i.copy()
    if len(n_i_pre) != m:
        raise ValueError("n_i_pre length mismatch")
    # fold
    folded_A = []
    folded_b = []
    for i in range(m):
        ni = int(n_i[i])
        if ni == 0:
            raise ValueError(f"group {i} has zero rows")
        s = np.sqrt(ni)
        folded_A.append(np.asarray(A_blocks[i], dtype=np.float64) / s)
        folded_b.append(np.asarray(b_blocks[i], dtype=np.float64) / s)
    A = np.vstack(folded_A)
    b = np.concatenate(folded_b)
    offsets = np.zeros(m + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_i)
    return Problem(A=A, b=b, offsets=offsets, name=name, m=m, d=d,
                   n_i=np.asarray(n_i_pre, dtype=np.int64), meta=meta or {})


# ---------------------------------------------------------------------------
# residual / group helpers
# ---------------------------------------------------------------------------

def residuals(problem: Problem, x: np.ndarray) -> np.ndarray:
    """r = A x - b, shape [n].   (folded data ⇒ r_i is the folded group residual)."""
    return problem["A"] @ x - problem["b"]


def group_residuals(problem: Problem, x: np.ndarray) -> list[np.ndarray]:
    """List of per-group folded residual vectors r_i = A_{S_i} x - b_{S_i}."""
    r = residuals(problem, x)
    off = problem["offsets"]
    return [r[off[i]:off[i + 1]] for i in range(problem["m"])]


def group_losses(problem: Problem, x: np.ndarray) -> np.ndarray:
    """ell_i(x) = ||r_i||^2  (folded ⇒ == (1/n_i)||orig residual||^2), shape [m]."""
    out = np.empty(problem["m"], dtype=np.float64)
    off = problem["offsets"]
    r = residuals(problem, x)
    for i in range(problem["m"]):
        ri = r[off[i]:off[i + 1]]
        out[i] = float(ri @ ri)
    return out


def group_norms(problem: Problem, x: np.ndarray) -> np.ndarray:
    """||r_i||_2 per group, shape [m]  (the 'inner' norm the theory uses)."""
    out = np.empty(problem["m"], dtype=np.float64)
    off = problem["offsets"]
    r = residuals(problem, x)
    for i in range(problem["m"]):
        ri = r[off[i]:off[i + 1]]
        out[i] = float(np.sqrt(ri @ ri))
    return out


def max_loss(problem: Problem, x: np.ndarray) -> float:
    """F(x) = max_i ||r_i||^2 = ||Ax-b||_{G_inf}^2  (the benchmarked objective)."""
    return float(group_losses(problem, x).max())


def max_loss_unsquared(problem: Problem, x: np.ndarray) -> float:
    """f(x) = max_i ||r_i||_2 = ||Ax-b||_{G_inf}  (theory's smoothed-scale objective)."""
    return float(group_norms(problem, x).max())


def group_norm_p(problem: Problem, x: np.ndarray, p: float) -> float:
    """||Ax-b||_{G_p} = (sum_i ||r_i||_2^p)^{1/p}; p=inf -> max_i ||r_i||_2."""
    gn = group_norms(problem, x)
    if np.isinf(p):
        return float(gn.max())
    return float(np.sum(gn ** p) ** (1.0 / p))
