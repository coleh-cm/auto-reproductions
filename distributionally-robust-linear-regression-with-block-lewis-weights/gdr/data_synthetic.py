"""Synthetic heterogeneous regression (SPEC D1, experiments.tex:4-38).

The paper states only: d=10, m=100 groups, 5 adversarial, stacked-Gram
condition number ~1e5 (experiments.tex:38); shared orthonormal eigenbasis;
normal groups = moderate condition number, aligned geometry, optima near a
common center, independent noise; adversarial groups = one extremely large-
curvature direction (distinct per group), optima far from the population
center along that direction, very small noise (experiments.tex:19-25).  All
concrete numbers (eigenvalue ranges, noise scales, distances, n_i, seed) are
*not stated* (U1) -- the referenced Jupyter notebook is absent from the arXiv
tarball.  Our concrete instantiation is declared below and in arms.json and is
verified to reproduce: cond(A^T A) ~ 1e5 and a clear ERM-vs-robust worst-group
gap (experiments.tex:38).

Construction (shared orthonormal basis U in R^{d x d}):
  * normal groups  (m - n_adv of them): rows of A_{S_i} ~ N(0, Sigma_i) with
    Sigma_i = U diag(e_i) U^T, e_i drawn log-uniform in [E_LO, E_HI] (moderate
    condition per group); optimum x*_i = center + N(0, sig_x^2 I); responses
    b_{S_i} = A_{S_i} x*_i + N(0, sig^2)  (independent noise, moderately curved).
  * adversarial groups (n_adv of them): each owns a distinct basis direction
    u_{k_i}; Sigma_i has eigenvalue E_ADV along u_{k_i} and moderate values
    (log-uniform [E_LO, E_HI]) on the rest; optimum x*_i = center + DIST*u_{k_i}
    (far along the sharp direction); responses with very small noise SIG_ADV.

All randomness is seeded (deterministic).  Folding (1/sqrt(n_i)) is applied in
make_problem, so the returned problem is on the folded scale.
"""
from __future__ import annotations

import numpy as np

from .problem import Problem, make_problem


def make_synthetic(
    d: int = 10,
    m: int = 100,
    n_adv: int = 5,
    n_per_group: int = 50,
    seed: int = 0,
    E_LO: float = 0.01,
    E_HI: float = 1.0,
    E_ADV: float = 1.0e6,       # curvature of the adversarial sharp direction
    DIST: float = 5.0,          # how far adversarial optima sit along their sharp direction
                               # (SPEC §8A / arms.json _dataset_overrides record DIST=5.0 as
                               #  the disclosed choice; this default must match that record.
                               #  DIST only scales b via x_star=center+DIST*v, so after the
                               #  OPT=1 normalization in run_arm the relative gaps — and thus
                               #  the gate FINAL lines — are invariant under DIST scaling;
                               #  only the absolute OPT/loss scale changes, by (DIST'/DIST)^2.)
    sig_x: float = 0.3,         # spread of normal-group optima around the center
    sig: float = 0.5,           # noise scale for normal groups
    sig_adv: float = 0.02,      # noise scale for adversarial groups (very small)
    center: np.ndarray | None = None,
) -> Problem:
    """Construct a folded synthetic GDR problem (D1).

    The n_adv adversarial groups have their sharp (huge-curvature) directions
    drawn in a *2D subspace* at spread angles so they CONFLICT -- no single x can
    fit them all, so ERM (which averages, dominated by the 95 aligned normal
    groups) sits near the population center and incurs a large loss on the
    adversarial groups, while the robust optimum redistributes (experiments.tex:
    19-25, 33).  Normal groups share the orthonormal basis U and have moderate
    log-uniform eigenvalues; their optima concentrate near the center.

    NOTE: the rotated 2D-subspace spike is a literal departure from the
    "Hessian shares eigenvectors" phrasing of experiments.tex:19, recorded as a
    paper-internal inconsistency (SPEC U19) and empirically verified necessary:
    the paper-literal shared-eigenbasis construction (spike on a distinct U
    column per adversarial group) yields only a ~6% ERM-vs-robust gap with the
    ERM worst group = a NORMAL group, contradicting the paper's own phenomenon 3
    and the "clear gap" of experiments.tex:38; this rotated construction yields
    the measured ~47% gap (ERM worst group = adversarial).  See gdr/data.py
    deviation_note for the full numbers.
    """
    rng = np.random.default_rng(seed)
    if center is None:
        center = np.zeros(d)
    # shared orthonormal basis U for the normal groups' covariances.
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    # adversarial sharp directions live in span(U[:,0], U[:,1]) at spread angles
    # (conflict): the 5 directions cannot be simultaneously fit by one x.
    ang = np.linspace(0.0, np.pi, n_adv, endpoint=False) + rng.uniform(0.0, 0.3)
    adv_dirs = [U[:, 0] * np.cos(a) + U[:, 1] * np.sin(a) for a in ang]

    A_blocks = []
    b_blocks = []
    n_adv_used = 0
    for i in range(m):
        is_adv = i < n_adv  # first n_adv groups are adversarial
        if is_adv:
            v = adv_dirs[n_adv_used]            # unit sharp direction
            n_adv_used += 1
            # moderate base eigenvalues on U, plus a rank-1 spike E_ADV along v
            eigs = np.exp(rng.uniform(np.log(E_LO), np.log(E_HI), size=d))
            Sigma = (U * eigs) @ U.T + E_ADV * np.outer(v, v)
            x_star_i = center + DIST * v
            noise_sd = sig_adv
        else:
            eigs = np.exp(rng.uniform(np.log(E_LO), np.log(E_HI), size=d))
            Sigma = (U * eigs) @ U.T
            x_star_i = center + sig_x * rng.standard_normal(d)
            noise_sd = sig
        # rows ~ N(0, Sigma) via the symmetric square root
        w, Ve = np.linalg.eigh(Sigma)
        w = np.clip(w, 0.0, None)
        Gsqrt = (Ve * np.sqrt(w)) @ Ve.T          # [d, d], Sigma^{1/2}
        Z = rng.standard_normal((n_per_group, d))
        Ai = Z @ Gsqrt.T
        bi = Ai @ x_star_i + noise_sd * rng.standard_normal(n_per_group)
        A_blocks.append(Ai)
        b_blocks.append(bi)

    prob = make_problem(A_blocks, b_blocks, name="synthetic",
                        n_i_pre=np.full(m, n_per_group),
                        meta={"seed": seed, "n_adv": n_adv, "E_ADV": E_ADV,
                              "DIST": DIST, "E_LO": E_LO, "E_HI": E_HI,
                              "n_per_group": n_per_group})
    return prob


def stacked_gram_cond(problem: Problem) -> float:
    G = problem["A"].T @ problem["A"]
    ev = np.linalg.eigvalsh(G)
    return float(ev.max() / max(ev.min(), 1e-300))
