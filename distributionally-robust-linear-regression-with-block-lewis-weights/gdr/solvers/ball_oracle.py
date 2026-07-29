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

from gdr.objectives import smoothed, smoothed_grad_hess  # frozen interface (E3, E4)
from gdr.solvers import (
    DEFAULTS, as_list, get_opt, gap_now, seed_history,
    beta_delta_pairs, solve_pd, norm_M, norm_Minv, max_loss,
)

_INNER_CAP = 50          # inner Newton cap (SPEC E19)
_NU_GROW = 60            # doublings to find nu_hi
_NU_BISECT = 20          # bisection refinements of nu
_LS_STEPS = 40           # backtracking line-search halvings (damped Newton)
_LS_C1 = 1e-4            # Armijo sufficient-decrease constant


def _geometry(problem, cfg):
    """Build the geometry matrix M [d,d] for the chosen geometry (E11).

    euclidean: M = A^T A.  lewis: w = block_lewis_weights(problem, p=inf)
    (E8), M = geometry_M(problem, w, p=inf) (E11).
    """
    A = problem["A"]                       # [n, d]
    geom = cfg.get("geometry", "euclidean")
    if geom == "lewis":
        from gdr.lewis import block_lewis_weights, geometry_M, should_reset_W
        w = block_lewis_weights(problem, p=np.inf)   # [m]  (E8)
        # E11 switch (Alg.1 lines 2-3): if sum_i w_i >= m, reset W <- I
        # (naive euclidean geometry).  This fires only when m is small
        # (||w||_1 <= 2(d+1)), so on the paper's instances (m=51,100) the Lewis
        # geometry is used; it also gives a clean degeneracy: at the reset the
        # Lewis arm is bit-identical to the Euclidean arm.
        if should_reset_W(w, problem["m"]):
            M = A.T @ A
        else:
            M = geometry_M(problem, w, p=np.inf)      # [d, d]  (E11)
    else:
        M = A.T @ A                         # [d, d]  naive euclidean geometry
    M = np.asarray(M, dtype=np.float64)
    # Symmetrize defensively and add a tiny floor for the M^{-1}-norm solves.
    M = 0.5 * (M + M.T) + 1e-14 * np.eye(problem["d"], dtype=np.float64)
    return M


def _trust_step(g, H, M, r, q, x):
    """More-Sorensen trust-region step over the FIXED ball {y : ||y - q||_M <= r}.

    Minimizes the quadratic model g^T s + (1/2) s^T H s at the current iterate x
    subject to ||(x + s) - q||_M <= r (E19 subproblem, body.tex:31
    O(q) := min_{||x-q||_M <= r_q} f(x)).  The KKT system is

        (H + nu M) s = nu M (q - x) - g

    with nu = 0 when the unconstrained Newton step already stays inside the
    ball; otherwise bisect nu until ||(x + s(nu)) - q||_M <= r (Moré-Sorensen).

    g [d], H [d,d] (regularized), M [d,d], r >= 0 (or +inf), q [d] ball center,
    x [d] current iterate.  Returns s [d].
    """
    offset = q - x                                 # [d]  ball-center offset
    if not np.isfinite(r):                         # no active trust region
        return -solve_pd(H, g)                     # [d]  plain Newton step
    # nu = 0 step: solve H s = -g (offset term vanishes at nu=0).
    s0 = -solve_pd(H, g)                           # [d]
    if norm_M((x + s0) - q, M) <= r:               # inside the fixed ball
        return s0
    # nu > 0: (H + nu M) s = nu M offset - g.  ||x+s-q||_M is monotone decreasing
    # in nu (the M-projection pulls s toward offset).  Grow nu then bisect.
    nu_hi = 1.0
    for _ in range(_NU_GROW):
        s_hi = solve_pd(H + nu_hi * M, nu_hi * (M @ offset) - g)  # [d]
        if norm_M((x + s_hi) - q, M) <= r:
            break
        nu_hi *= 2.0
    nu_lo = 0.0
    for _ in range(_NU_BISECT):
        nu_mid = 0.5 * (nu_lo + nu_hi)
        s_mid = solve_pd(H + nu_mid * M, nu_mid * (M @ offset) - g)  # [d]
        if norm_M((x + s_mid) - q, M) <= r:
            nu_hi = nu_mid
        else:
            nu_lo = nu_mid
    return solve_pd(H + nu_hi * M, nu_hi * (M @ offset) - g)  # [d]  boundary step


def _solve_region(problem, M, q, r, beta, delta, tol_inner):
    """Inner trust-region Newton solve: minimize f_tilde over {||x-q||_M <= r}.

    Starts at x = q and runs up to _INNER_CAP damped Newton steps, each
    constrained to the FIXED ball centered at q (E19 subproblem, body.tex:31).
    "Damped" Newton (experiments.tex:71): every accepted step must give an
    Armijo sufficient decrease of the smoothed objective f_tilde.  The
    More-Sorensen step ``s`` is a descent direction of f_tilde (``g^T s < 0``
    because it decreases the model whose Hessian is PSD-regularized), so a
    backtracking line search along ``s`` always finds a decrease in finitely
    many halvings.  This makes f_tilde monotone non-increasing across inner
    *and* outer iterations -- the genuine invariant the paper's "damped
    Newton" implies (and the one tests/test_invariants.py checks).  Taking
    the step unconditionally (no damping) lets a large trust region overshoot
    and *increase* f_tilde on ill-scaled / high-curvature instances.  Returns
    the final x [d].  (E19 inner loop; experiments.tex:72-77.)
    """
    d = problem["d"]
    I = np.eye(d, dtype=np.float64)
    x = np.asarray(q, dtype=np.float64).copy()  # [d]  == q, so ||x-q||_M = 0
    f_cur, _g0, _H0 = smoothed_grad_hess(problem, x, beta, delta)  # f_tilde(q)
    for _ in range(_INNER_CAP):
        _val, g, H = smoothed_grad_hess(problem, x, beta, delta)  # E4; g [d], H [d,d]
        H = H + 1e-12 * I                 # regularize for definiteness (SPEC E19)
        if norm_Minv(g, M) <= tol_inner:  # inner stopping (E19)
            break
        s = _trust_step(g, H, M, r, q, x)  # [d]  More-Sorensen step (fixed ball at q)
        gs = float(g @ s)                  # directional derivative (descent: gs < 0)
        # Backtracking line search: accept the first alpha in {1, .5, .25, ...}
        # that gives Armijo sufficient decrease of f_tilde.  A smaller alpha
        # keeps x+alpha*s inside the fixed ball (||.||_M is convex), so the
        # trust-region constraint stays satisfied.
        alpha = 1.0
        accepted = False
        for _bt in range(_LS_STEPS):
            x_try = x + alpha * s
            f_try = smoothed(problem, x_try, beta, delta)
            if f_try <= f_cur + _LS_C1 * alpha * gs:   # Armijo (gs < 0)
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break                        # cannot decrease further -> at a min
        x = x + alpha * s
        f_cur = f_try
    return x


def _run_single(problem, M, x0, max_outer, deadline,
                r0, shrink, beta, delta, tol_inner, opt):
    """One (r0, shrink, beta, delta, tol_inner) run.  Returns (history, final F).

    Records the center ``q`` after each outer step in ``history['x_traj']``
    (a list of python lists, JSON-serializable) so callers can verify the
    guaranteed per-iteration invariant of the inner trust-region solve — that
    the smoothed surrogate f_tilde is non-increasing across outer iterations
    (body.tex:31 / E19) — rather than only start-vs-final.
    """
    h = seed_history(problem, x0, opt)     # iter 0 recorded
    h["x_traj"] = [np.asarray(x0, dtype=np.float64).tolist()]  # iter 0 center
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
        h["x_traj"].append(np.asarray(q, dtype=np.float64).tolist())
        t_elapsed = time.perf_counter() - t0
        h["time"].append(t_elapsed)
        if h["gap"][-1] <= 0.0:
            break
        if t_elapsed > deadline - t0:
            break
    h["x"] = q
    return h, float(max_loss(problem, q))


def _tol_inner_list(cfg):
    """Parse the tol_inner grid: accept a list under tol_inner_grid or tol_inner,
    or a single scalar under tol_inner (SPEC sec5 grid convention)."""
    if cfg.get("tol_inner_grid") is not None:
        return [float(v) for v in as_list(cfg["tol_inner_grid"], [])]
    tol = cfg.get("tol_inner", DEFAULTS["tol_inner"])
    if isinstance(tol, (list, tuple, np.ndarray)):
        return [float(v) for v in tol]
    return [float(tol)]


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
    # Deterministic best-of-grid: evaluate EVERY config (grids are small and on
    # rescaled data each run is fast), so the result does not depend on wall-clock
    # timing/machine load.  The per-run deadline inside _run_single is the only
    # safety net (a diverging config cannot hang forever).
    for r0, shrink, tol_inner in itertools.product(r0_grid, shrink_grid, tol_grid):
        for beta, delta in beta_delta_pairs(cfg):
            if time.perf_counter() > deadline and best is not None:
                break
            h, final = _run_single(problem, M, x0, max_outer, deadline,
                                   float(r0), float(shrink),
                                   beta, delta, float(tol_inner), opt)
            if final < best_final:
                best_final = final
                best = h
        if time.perf_counter() > deadline and best is not None:
            break
    if best is None:
        best = seed_history(problem, x0, opt)
    return best
