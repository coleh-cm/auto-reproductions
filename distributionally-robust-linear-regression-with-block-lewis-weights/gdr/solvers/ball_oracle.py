"""Ball-oracle method ("ours") — trust-region Newton on the smoothed surrogate.

Paper: the paper's ball-oracle arm (E19, experiments.tex:71-78).  Geometry is
selected by ``cfg['geometry']`` in {"euclidean", "lewis"}:

    euclidean => M = A^T A              (naive ball; W = I, SPEC E11/Table 1)
    lewis     => w = block_lewis_weights(problem, p=inf),  M = geometry_M(problem, w, p=inf)  (E8/E11)

Outer loop (max_outer): minimize the smoothed surrogate f_tilde_{beta,delta}
over the trust region {x : ||x - q||_M <= r} by damped trust-region Newton
(More-Sorensen), iterating from x = q:

    g, H = smoothed_grad_hess(problem, x, beta, delta);  H += 1e-12 I       # E4
    s = -(H + nu M)^{-1} g ;  nu = 0 if ||s||_M <= r else bisect nu to ||s||_M ~ r
    x <- x + s ;  stop inner when ||g||_{M^{-1}} <= tol_inner or inner cap (50)

One outer iteration = one such trust-region solve (start from q,
experiments.tex:101).  After the solve the center is updated q <- x and the
radius optionally shrunk r <- shrink * r (experiments.tex:78).  Best-of-grid
over (r0, shrink, beta, delta, tol_inner); keep the lowest final F(x)
(experiments.tex:92).

Degeneracy note (SPEC, ball_oracle task): with geometry='euclidean', r=+inf
(no active trust region) and beta, delta -> 0 the inner iterate reduces to
plain damped Newton on f_tilde; the More-Sorensen step below returns nu=0
(i.e. s = -H^{-1} g) whenever ||s||_M <= r, so at r=+inf the trust region is
inactive and the inner Newton is the standard Newton step.
"""

from __future__ import annotations

import itertools
import time
import numpy as np

from gdr.objectives import smoothed_grad_hess  # frozen interface (E4)
from gdr.solvers import (
    DEFAULTS, as_list, get_opt, gap_now, seed_history,
    beta_delta_pairs, solve_pd, norm_M, norm_Minv, max_loss,
)

_INNER_CAP = 50          # inner Newton cap (SPEC E19)
_NU_GROW = 60            # doublings to find nu_hi
_NU_BISECT = 20          # bisection refinements of nu


def _geometry(problem, cfg):
    """Build the geometry matrix M [d,d] for the chosen geometry (E11).

    euclidean: M = A^T A.  lewis: w = block_lewis_weights(problem, p=inf)
    (E8), M = geometry_M(problem, w, p=inf) (E11).
    """
    A = problem["A"]                       # [n, d]
    geom = cfg.get("geometry", "euclidean")
    if geom == "lewis":
        from gdr.lewis import block_lewis_weights, geometry_M
        w = block_lewis_weights(problem, p=np.inf)   # [m]  (E8)
        M = geometry_M(problem, w, p=np.inf)         # [d, d]  (E11)
    else:
        M = A.T @ A                         # [d, d]  naive euclidean geometry
    M = np.asarray(M, dtype=np.float64)
    # Symmetrize defensively and add a tiny floor for the M^{-1}-norm solves.
    M = 0.5 * (M + M.T) + 1e-14 * np.eye(problem["d"], dtype=np.float64)
    return M


def _trust_step(g, H, M, r):
    """More-Sorensen trust-region step s = -(H + nu M)^{-1} g with ||s||_M <= r.

    g [d], H [d,d] (already regularized), M [d,d], r >= 0 (or +inf).
    Returns s [d].  nu = 0 if the unconstrained Newton step already satisfies
    the trust region; otherwise bisect nu in [nu_lo, nu_hi] until ||s(nu)||_M
    is at or just below r (SPEC E19; Moré-Sorensen).
    """
    d = g.shape[0]
    if not np.isfinite(r):                  # no active trust region
        return -solve_pd(H, g)             # [d]  plain Newton step
    s0 = -solve_pd(H, g)                   # [d]  nu = 0 step
    n0 = norm_M(s0, M)                     # scalar  ||s(0)||_M
    if n0 <= r:
        return s0                          # trust region inactive

    # Increase nu until ||s(nu)||_M <= r (||s(nu)||_M is monotone decreasing).
    nu_hi = 1.0
    for _ in range(_NU_GROW):
        s_hi = -solve_pd(H + nu_hi * M, g)  # [d]
        if norm_M(s_hi, M) <= r:
            break
        nu_hi *= 2.0
    nu_lo = 0.0
    # Bisect to bring ||s(nu)||_M down to ~ r.
    for _ in range(_NU_BISECT):
        nu_mid = 0.5 * (nu_lo + nu_hi)
        s_mid = -solve_pd(H + nu_mid * M, g)  # [d]
        if norm_M(s_mid, M) <= r:
            nu_hi = nu_mid
        else:
            nu_lo = nu_mid
    return -solve_pd(H + nu_hi * M, g)      # [d]  boundary-respecting step


def _solve_region(problem, M, q, r, beta, delta, tol_inner):
    """Inner trust-region Newton solve: minimize f_tilde over {||x-q||_M <= r}.

    Starts at x = q and runs up to _INNER_CAP Newton steps.  Returns the final
    x [d].  (E19 inner loop; experiments.tex:72-77.)
    """
    d = problem["d"]
    I = np.eye(d, dtype=np.float64)
    x = np.asarray(q, dtype=np.float64).copy()  # [d]
    for _ in range(_INNER_CAP):
        _val, g, H = smoothed_grad_hess(problem, x, beta, delta)  # E4; g [d], H [d,d]
        H = H + 1e-12 * I                 # regularize for definiteness (SPEC E19)
        if norm_Minv(g, M) <= tol_inner:  # inner stopping (E19)
            break
        s = _trust_step(g, H, M, r)        # [d]  More-Sorensen step
        x = x + s                          # [d]
    return x


def _run_single(problem, M, x0, max_outer, deadline,
                r0, shrink, beta, delta, tol_inner, opt):
    """One (r0, shrink, beta, delta, tol_inner) run.  Returns (history, final F)."""
    h = seed_history(problem, x0, opt)     # iter 0 recorded
    q = h["x"].copy()                      # [d]  center
    r = float(r0)                          # scalar  trust-region radius
    t0 = time.perf_counter()
    for k in range(1, max_outer + 1):
        if h["gap"][-1] <= 0.0:
            break
        if time.perf_counter() > deadline:
            break
        x = _solve_region(problem, M, q, r, beta, delta, tol_inner)  # [d]  E19
        q = x                              # update center (experiments.tex:78)
        r = shrink * r                     # shrink radius (experiments.tex:78)
        h["iter"].append(k)
        h["gap"].append(gap_now(problem, x, opt))
        t_elapsed = time.perf_counter() - t0
        h["time"].append(t_elapsed)
        if h["gap"][-1] <= 0.0:
            break
        if t_elapsed > deadline - t0:
            break
    h["x"] = q
    return h, float(max_loss(problem, q))


def _tol_inner_list(cfg):
    """Parse the tol_inner grid: tol_inner_grid (list) or single tol_inner."""
    if cfg.get("tol_inner_grid") is not None:
        return [float(v) for v in as_list(cfg["tol_inner_grid"], [])]
    return [float(cfg.get("tol_inner", DEFAULTS["tol_inner"]))]


def run(problem, cfg, x0, max_outer, time_budget):
    """Best-of-grid ball-oracle.  See module docstring."""
    cfg = cfg or {}
    opt = get_opt(problem, cfg)
    x0 = np.asarray(x0, dtype=np.float64)          # [d]
    M = _geometry(problem, cfg)                    # [d, d]  geometry (E11)
    r0_grid = as_list(cfg.get("r0_grid"), DEFAULTS["r0_grid"])
    shrink_grid = as_list(cfg.get("shrink_grid"), DEFAULTS["shrink_grid"])
    tol_grid = _tol_inner_list(cfg)

    t_start = time.perf_counter()
    deadline = t_start + float(time_budget)
    best = None
    best_final = np.inf
    for r0, shrink, tol_inner in itertools.product(r0_grid, shrink_grid, tol_grid):
        for beta, delta in beta_delta_pairs(cfg):
            if time.perf_counter() > deadline:
                break
            h, final = _run_single(problem, M, x0, max_outer, deadline,
                                   float(r0), float(shrink),
                                   beta, delta, float(tol_inner), opt)
            if final < best_final:
                best_final = final
                best = h
        if time.perf_counter() > deadline:
            break
    if best is None:
        best = seed_history(problem, x0, opt)
    return best
