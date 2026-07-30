"""Frozen data contract for the GDR linear-regression reproduction.

Implements ``GroupProblem`` exactly as specified in SPEC.md section 7
("Frozen component interfaces").  All other modules build against this
contract; do not rename public attributes.

Conventions (SPEC section 4):
  * 0-indexed groups ``i in {0..m-1}`` map to contiguous row slices ``S_i`` of
    ``A`` (row-major, Fortran-free).  ``group_id`` is nondecreasing.
  * Group loss (E1): ``ell_i(x) = (1/n_i) ||A_{S_i} x - b_{S_i}||_2^2``
    (`paper/experiments.tex:9`).  This is the *unsquared-MSE* convention used
    by every reported number; the theory text folds ``1/sqrt(n_i)`` into the
    data (`paper/body.tex:27`) so its ``f`` is the square root of ``F``.
  * Worst-group objective (E2): ``F(x) = max_i ell_i(x)``.
  * ERM (E3): minimizer of the *group-averaged* loss
    ``(1/m) sum_i (1/n_i) ||A_{S_i} x - b_{S_i}||^2``
    (`paper/intro.tex:5`); the ``1/m`` is a constant so this is weighted
    least squares with per-row weight ``1/n_i``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GroupProblem", "group_slices_from_id"]


def group_slices_from_id(group_id: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous ``[start, end)`` row slices for each group id.

    Assumes ``group_id`` is nondecreasing (the contract).  Raises if it is not,
    because a non-contiguous layout would silently break every downstream
    slice.
    """
    gid = np.asarray(group_id)
    if gid.ndim != 1:
        raise ValueError(f"group_id must be 1-D, got shape {gid.shape}")
    if gid.size == 0:
        raise ValueError("group_id is empty — a GroupProblem needs >=1 group")
    diffs = np.diff(gid)
    if np.any(diffs < 0):
        raise ValueError("group_id must be nondecreasing (contiguous groups)")
    # boundaries where the id changes (or at the very start)
    breaks = np.concatenate(([0], np.where(diffs > 0)[0] + 1, [gid.size]))
    return [(int(breaks[k]), int(breaks[k + 1])) for k in range(len(breaks) - 1)]


@dataclass(frozen=True)
class GroupProblem:
    """A grouped least-squares regression problem.

    Attributes
    ----------
    A : np.ndarray, float64, shape [n, d]
        Stacked design matrix, rows grouped contiguously.
    b : np.ndarray, float64, shape [n]
        Stacked response.
    group_id : np.ndarray, int32, shape [n]
        Group membership of each row, values ``0..m-1``, nondecreasing.
    """

    A: np.ndarray
    b: np.ndarray
    group_id: np.ndarray

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)
        gid = np.asarray(self.group_id, dtype=np.int32)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D [n,d], got shape {A.shape}")
        if b.ndim != 1:
            raise ValueError(f"b must be 1-D [n], got shape {b.shape}")
        if gid.ndim != 1:
            raise ValueError(f"group_id must be 1-D [n], got shape {gid.shape}")
        if A.shape[0] != b.shape[0] or A.shape[0] != gid.shape[0]:
            raise ValueError(
                f"row mismatch: A={A.shape[0]} b={b.shape[0]} gid={gid.shape[0]}"
            )
        if gid.size == 0:
            raise ValueError("empty problem")
        if gid.min() < 0:
            raise ValueError("group_id contains negative values")
        # store as the canonical dtypes (dataclass(frozen=True) -> use object.__setattr__)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "group_id", gid)
        # cache slices lazily via a private attribute
        object.__setattr__(self, "_slices", group_slices_from_id(gid))

    # -- structural properties ------------------------------------------------
    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def d(self) -> int:
        return self.A.shape[1]

    @property
    def m(self) -> int:
        return len(self._slices)

    @property
    def sizes(self) -> np.ndarray:
        """int64 [m]: number of rows per group."""
        return np.array([e - s for s, e in self._slices], dtype=np.int64)

    @property
    def slices(self) -> list[tuple[int, int]]:
        """Contiguous ``[start, end)`` row slice per group."""
        return list(self._slices)

    # -- per-group helpers -----------------------------------------------------
    def group_block(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self._slices[i]
        return self.A[s:e], self.b[s:e]

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """[d] -> [n]: ``A x - b`` (`paper/intro.tex:7`)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.d:
            raise ValueError(f"x has dim {x.shape[0]}, problem has d={self.d}")
        return self.A @ x - self.b

    def group_residual_norms_sq(self, x: np.ndarray) -> np.ndarray:
        """[d] -> [m]: ``||A_{S_i} x - b_{S_i}||_2^2`` per group (no 1/n_i)."""
        r = self.residuals(x)
        out = np.empty(self.m, dtype=np.float64)
        for i, (s, e) in enumerate(self._slices):
            out[i] = float(r[s:e] @ r[s:e])
        return out

    def group_losses(self, x: np.ndarray) -> np.ndarray:
        """[d] -> [m]: E1, ``(1/n_i)||A_{S_i} x - b_{S_i}||_2^2``.

        (`paper/experiments.tex:9`)
        """
        sq = self.group_residual_norms_sq(x)
        sizes = self.sizes.astype(np.float64)
        return sq / sizes

    def worst_loss(self, x: np.ndarray) -> float:
        """E2, ``max_i ell_i(x)`` (`paper/experiments.tex:13`)."""
        return float(np.max(self.group_losses(x)))

    def erm(self) -> np.ndarray:
        """E3 closed form: argmin (1/m) sum_i (1/n_i) ||A_{S_i} x - b_{S_i}||^2.

        The ``1/m`` is a constant, so this is weighted least squares with
        per-row weight ``1/n_i``.  Solved via the normal equations with a
        pseudo-inverse fallback for rank-deficient designs.
        """
        w = 1.0 / self.sizes.astype(np.float64)  # per-group weight
        # expand to per-row weight
        row_w = np.repeat(w, self.sizes.astype(np.int64))
        W = row_w[:, None]  # broadcast over columns of A
        AtWA = (self.A * W).T @ self.A
        AtWb = (self.A * W).T @ self.b
        # solve; pseudo-inverse handles rank-deficient (e.g. synthetic) cases
        sol = np.linalg.solve(AtWA, AtWb) if np.linalg.matrix_rank(AtWA) == self.d else (
            np.linalg.pinv(AtWA) @ AtWb
        )
        return np.asarray(sol, dtype=np.float64)

    def augmented(self) -> np.ndarray:
        """Return ``A_hat = [A | b]`` in R^{n x (d+1)}`` (`paper/other_proofs.tex:53`).

        Used by ``gdr.lewis`` to compute block Lewis weights on the augmented
        matrix, per Algorithm 1 line 1 (`paper/body.tex:169`).
        """
        return np.concatenate([self.A, self.b[:, None]], axis=1)
