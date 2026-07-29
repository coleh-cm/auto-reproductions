"""Solver arms for distributionally-robust linear regression
(Manoj & Patel, arXiv:2607.00252, ICLR 2026).

Each arm module exposes a single function

    run(problem, cfg, x0, max_outer, time_budget) -> history

with the frozen signature/return type of SPEC sec4:

    history = {'iter': list[int], 'gap': list[float],
              'time': list[float], 'x': np.ndarray}

where ``gap[k] = max_loss(problem, x_k) - opt`` (squared scale, the scale the
paper plots), ``opt = float(cfg['opt'])`` is the CVXPY reference (E20),
``iter[k]`` is the 1-based OUTER iteration count for that method
(experiments.tex:96-104), and ``time[k]`` is wall time since ``run`` started
(``time.perf_counter``).  Iteration 0 is always the initial point ``x0`` with
its gap and time ``0.0``.  An arm stops early once ``gap <= 0`` or once the
wall clock exceeds ``time_budget``.

These modules import the sibling units ``gdr.objectives`` and ``gdr.lewis``
and the reference ``gdr.reference``; they code to those frozen interfaces and
do not duplicate the maths.

Shared helpers (this file)
--------------------------
* ``get_opt(problem, cfg)``        -- resolve OPT from cfg['opt'] or solve_opt.
* ``gap_now(problem, x, opt)``     -- F(x) - opt (float, squared scale).
* ``as_list(v, default)``          -- accept a scalar or a list grid value.
* ``solve_pd(H, rhs)``             -- linear solve with pinv fallback (U16).
* ``norm_M(v, M)`` / ``norm_Minv(v, M)`` -- M-/M^{-1}-norms.
* ``beta_delta_pairs(cfg)``       -- parse the (beta, delta) grid (SPEC sec5).
* ``DEFAULTS``                     -- documented default grids (U2).
"""

from __future__ import annotations

import itertools
import time
import numpy as np

# ``max_loss`` / ``group_losses`` belong to the frozen objective interface
# (SPEC sec4 lists them under gdr/objectives.py).  We prefer that source; if
# the sibling layout instead exposes them from gdr/problem.py (an alternate
# organization some units adopt), fall back transparently so the arms work
# under either layout.  Re-exported below for the arm modules.
try:
    from gdr.objectives import max_loss, group_losses  # frozen interface
except ImportError:  # pragma: no cover - depends on sibling layout
    from gdr.problem import max_loss, group_losses      # sibling fallback

# Numpy float64 everywhere (SPEC discipline).  The DEFAULTS below are our
# disclosed tuning-grid choices; the paper states only that "every method is
# tuned via grid search" with no grid values (experiments.tex:83-92; U2).  We
# pick modest log-spaced grids sized so a missing cfg key never crashes and a
# best-of-grid search has a reasonable chance of finding a good config.
DEFAULTS = {
    # First-order / subgradient step sizes (squared-loss scale).
    "lr_grid": [1e-3, 1e-2, 1e-1],
    # Subgradient step-size schedules (experiments.tex:57-58).
    "schedule_grid": ["const", "1/sqrt_t"],
    # Smoothing parameters (E3).  beta = eps/(4 log m), delta = eps/4
    # (body.tex:181); for the ACS instance (OPT ~ 100-138) and a 1% target
    # (eps ~ 1) this puts beta ~ 0.05, delta ~ 0.25, so the grid straddles
    # that region.
    "beta_grid": [1e-2, 1e-1, 1.0],
    "delta_grid": [1e-2, 1e-1, 1.0],
    # Heavy-Ball momentum coefficients (Polyak).
    "momentum_grid": [0.5, 0.9],
    # Log-barrier IPM (E21, experiments.tex:68-69): initial barrier weight and
    # per-outer-step reduction factor mu <- theta * mu.
    "mu0_grid": [1.0, 10.0],
    "theta_grid": [0.1, 0.5],
    "inner_tol": 1e-8,
    # Ball-oracle (E19, experiments.tex:71-78): initial trust-region radius,
    # radius shrink factor, and inner Newton tolerance.  We keep the default
    # radii finite: at r = +inf the trust region is inactive and the inner
    # step is plain Newton on the *indefinite* smoothed Hessian (U13 negative
    # rank-1 term), which can diverge — the trust region is precisely what
    # makes the method stable, so the default grid uses finite radii.  r = +inf
    # is still supported (passed explicitly for the degeneracy probe) and then
    # reduces to s = -H^{-1} g per the More-Sorensen rule (nu = 0).
    "r0_grid": [1.0, 10.0, 100.0],
    "shrink_grid": [0.5, 1.0],
    "tol_inner": 1e-8,
}


def as_list(v, default):
    """Coerce a cfg value to a list grid.

    ``None`` -> ``list(default)``; a scalar -> ``[v]``; a list/tuple/ndarray ->
    ``list(v)``.  Lets every arm accept either a single value or a grid under
    the same key.
    """
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) if np.isscalar(x) and not isinstance(x, str) else x
                for x in v]
    return [v]


def get_opt(problem, cfg):
    """Return OPT (float, squared scale).

    If ``cfg['opt']`` is a finite number, use it; otherwise solve the E20
    epigraph QP via ``gdr.reference.solve_opt`` (the opt_reference arm relies
    on this fallback).  SPEC sec4.
    """
    if cfg is not None and "opt" in cfg and cfg["opt"] is not None:
        try:
            opt = float(cfg["opt"])
            if np.isfinite(opt):
                return opt
        except (TypeError, ValueError):
            pass
    from gdr.reference import solve_opt  # sibling unit (frozen interface)

    _x_star, opt = solve_opt(problem, solver=cfg.get("solver", "clarabel")
                             if cfg is not None else "clarabel")
    return float(opt)


def gap_now(problem, x, opt):
    """``F(x) - opt`` (float, squared scale; SPEC sec1; experiments.tex:52)."""
    return float(max_loss(problem, x)) - float(opt)


def solve_pd(H, rhs):
    """Solve ``H z = rhs`` for a small SPD-ish matrix, pinv fallback.

    d is tiny (=10), so we use a dense ``np.linalg.solve`` and fall back to
    ``np.linalg.pinv`` on singularity (rank-deficiency / pseudoinverse
    convention, body.tex:27; U16).  ``H`` is [k,k] float64, ``rhs`` is [k] or
    [k,j] float64; returns the same shape as ``rhs``.
    """
    H = np.asarray(H, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    try:
        return np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(H) @ rhs


def norm_M(v, M):
    """``||v||_M = sqrt(v^T M v)`` (float).  v [d], M [d,d]."""
    v = np.asarray(v, dtype=np.float64)
    return float(np.sqrt(max(0.0, v @ (M @ v))))


def norm_Minv(v, M):
    """``||v||_{M^{-1}} = sqrt(v^T M^{-1} v)`` (float).  v [d], M [d,d]."""
    v = np.asarray(v, dtype=np.float64)
    return float(np.sqrt(max(0.0, v @ solve_pd(M, v))))


def beta_delta_pairs(cfg):
    """Parse the (beta, delta) smoothing grid.

    Accepts either an explicit list of ``(beta, delta)`` tuples under
    ``cfg['beta_delta_grid']`` (SPEC sec5 notation) or the cartesian product
    of ``cfg['beta_grid']`` and ``cfg['delta_grid']`` (falling back to
    DEFAULTS).  Yields ``(beta, delta)`` float pairs.
    """
    if "beta_delta_grid" in cfg and cfg["beta_delta_grid"] is not None:
        pairs = cfg["beta_delta_grid"]
        for bd in pairs:
            yield float(bd[0]), float(bd[1])
        return
    betas = as_list(cfg.get("beta_grid"), DEFAULTS["beta_grid"])
    deltas = as_list(cfg.get("delta_grid"), DEFAULTS["delta_grid"])
    for beta, delta in itertools.product(betas, deltas):
        yield float(beta), float(delta)


def new_history(x0):
    """Empty history pre-seeded with the initial point at iter 0."""
    x0 = np.asarray(x0, dtype=np.float64).copy()  # [d]
    return {"iter": [], "gap": [], "time": [], "x": x0}


def seed_history(problem, x0, opt):
    """History with iter 0 recorded (gap = F(x0) - opt, time = 0.0)."""
    h = new_history(x0)
    h["iter"].append(0)
    h["gap"].append(gap_now(problem, h["x"], opt))
    h["time"].append(0.0)
    return h


__all__ = [
    "DEFAULTS",
    "as_list",
    "get_opt",
    "gap_now",
    "solve_pd",
    "norm_M",
    "norm_Minv",
    "beta_delta_pairs",
    "new_history",
    "seed_history",
    "max_loss",
    "group_losses",
]
