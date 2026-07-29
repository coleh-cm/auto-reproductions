"""Runner: dispatch an arm, track its history, and compute the time/iteration gate (SPEC §4).

Every arm in gdr/solvers/*.py implements

    run(problem, cfg, x0, max_outer, time_budget) -> History
    History = {'iter': list[int], 'gap': list[float], 'time': list[float], 'x': np.ndarray[d]}

where `gap[k] = F(x_k) - OPT`  (the worst-group suboptimality on the squared scale,
matching experiments.tex:52).  We centralize OPT computation and the gap/timing
bookkeeping here so every arm reports the same quantity.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .problem import Problem, max_loss


def run_arm(arm_name: str, cfg: dict, problem: Problem, warm_start: np.ndarray,
           opt: float, max_outer: int, time_budget: float) -> dict:
    """Import the solver module `gdr.solvers.<arm_name>` and run it.

    `opt` is the reference robust optimum (E20) so the arm does not re-solve it.
    """
    import importlib
    mod = importlib.import_module(f"gdr.solvers.{arm_name}")
    t0 = time.perf_counter()
    hist = mod.run(problem, cfg, warm_start, max_outer, time_budget)
    # normalize: ensure arrays are python lists of floats; recompute gaps if missing
    hist = dict(hist)
    if "gap" not in hist or hist["gap"] is None:
        xs = hist.get("x_traj", None)
        gaps = []
        for k in range(len(hist["iter"])):
            if xs is not None and k < len(xs):
                gaps.append(float(max_loss(problem, xs[k]) - opt))
            else:
                gaps.append(float("nan"))
        hist["gap"] = gaps
    else:
        hist["gap"] = [float(g) for g in hist["gap"]]
    hist["iter"] = [int(i) for i in hist["iter"]]
    hist["time"] = [float(t) for t in hist["time"]]
    hist["arm"] = arm_name
    hist["opt"] = float(opt)
    return hist


def time_to_gap(history: dict, rel_gap: float = 0.01):
    """First (iteration, seconds) at which (F(x_k)-OPT)/OPT <= rel_gap, else (None, None).

    If OPT <= 0 the relative gap is undefined; fall back to absolute gap <= rel_gap.
    """
    gaps = history["gap"]
    iters = history["iter"]
    times = history["time"]
    opt = history["opt"]
    for k in range(len(gaps)):
        g = gaps[k]
        if not np.isfinite(g):
            continue
        if opt > 0:
            if g / opt <= rel_gap:
                return iters[k], times[k]
        else:
            if g <= rel_gap:
                return iters[k], times[k]
    return None, None


def best_gap(history: dict) -> float:
    """Lowest finite worst-group suboptimality reached."""
    gaps = [g for g in history["gap"] if np.isfinite(g)]
    return float(min(gaps)) if gaps else float("inf")


# ---------------------------------------------------------------------------
# CLI: `python -m gdr.runner --arm <name> --dataset <name>` prints one
# `FINAL <arm>=<value>` line where <value> is the number of OUTER iterations
# to reach a 1% relative worst-group suboptimality (the metric of the paper's
# tab:acs_runtime, experiments.tex:176-186), or -1 if not reached within the
# budget.  The same names as arms.json are used; the gate checks this line.
# ---------------------------------------------------------------------------

# Per-arm default configs (disclosed tuning grids; SPEC sec5; U2: grids unstated).
# The CLI rescales every problem to O(1) losses (U15: WLOG OPT=1) before running,
# so these grids are on the O(1) scale and work uniformly across datasets.
ARM_CFGS = {
    "subgradient": {"lr_grid": [1e-2, 1e-3, 1e-4], "schedule": ["const", "1/sqrt_t"]},
    "smoothed_gd": {"lr_grid": [1e-2, 1e-3, 1e-4],
                    "beta_grid": [1e-2, 1e-1, 1.0], "delta_grid": [1e-2, 1e-1, 1.0]},
    "smoothed_hb": {"lr_grid": [1e-2, 1e-3], "momentum_grid": [0.5, 0.9],
                    "beta_grid": [1e-2, 1e-1, 1.0], "delta_grid": [1e-2, 1e-1, 1.0]},
    "smoothed_nesterov": {"lr_grid": [1e-2, 1e-3],
                          "beta_grid": [1e-2, 1e-1, 1.0], "delta_grid": [1e-2, 1e-1, 1.0]},
    "ipm": {"mu0_grid": [1.0, 10.0], "theta_grid": [0.1, 0.25, 0.5], "inner_tol": [1e-8]},
    "ball_oracle_euclidean": {"geometry": "euclidean", "r0_grid": [0.5, 1.0, 5.0],
                              "shrink_grid": [0.5, 1.0], "beta_grid": [1e-2, 1e-1, 1.0],
                              "delta_grid": [1e-2, 1e-1, 1.0], "tol_inner": [1e-6]},
    "ball_oracle_lewis": {"geometry": "lewis", "r0_grid": [0.5, 1.0, 5.0],
                          "shrink_grid": [0.5, 1.0], "beta_grid": [1e-2, 1e-1, 1.0],
                          "delta_grid": [1e-2, 1e-1, 1.0], "tol_inner": [1e-6]},
    "opt_reference": {},
}


def _arm_module_name(arm: str) -> str:
    """ball_oracle_euclidean / ball_oracle_lewis both map to the ball_oracle module."""
    return "ball_oracle" if arm.startswith("ball_oracle") else arm


def _erm_warmstart(problem) -> np.ndarray:
    """Universal ERM warm start (experiments.tex:92,148): plain least squares."""
    return np.linalg.lstsq(problem["A"], problem["b"], rcond=None)[0]


def main(argv=None) -> int:
    import argparse, json, os, sys
    from gdr.data import load_problem
    from gdr.reference import solve_opt

    ap = argparse.ArgumentParser(description="Run one GDR arm on one dataset.")
    ap.add_argument("--arm", required=True, help="arm name (arms.json key)")
    ap.add_argument("--dataset", default="synthetic",
                    help="synthetic | acs_income | smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-outer", type=int, default=60)
    ap.add_argument("--time-budget", type=float, default=120.0)
    ap.add_argument("--rel-gap", type=float, default=0.01,
                    help="relative-gap target (paper's 1%%)")
    ap.add_argument("--opt-cache", default=None,
                    help="json file caching OPT per dataset (for fast re-run)")
    args = ap.parse_args(argv)

    if args.arm not in ARM_CFGS:
        print(f"unknown arm: {args.arm}", file=sys.stderr)
        return 2
    cfg = dict(ARM_CFGS[args.arm])

    # OPT (cached to avoid re-solving the QCQP each arm)
    cache_key = f"{args.dataset}_seed{args.seed}"
    opt = None
    if args.opt_cache and os.path.isfile(args.opt_cache):
        try:
            with open(args.opt_cache) as f:
                opt = json.load(f).get(cache_key)
        except Exception:
            opt = None
    problem = load_problem(args.dataset, seed=args.seed)
    if problem is None:  # ACS blocked (download failed)
        print(f"FINAL {args.arm}_{args.dataset}=BLOCKED")
        return 0
    if opt is None:
        _xstar, opt = solve_opt(problem)
        opt = float(opt)
        if args.opt_cache:
            try:
                d = {}
                if os.path.isfile(args.opt_cache):
                    with open(args.opt_cache) as f:
                        d = json.load(f)
                d[cache_key] = opt
                with open(args.opt_cache, "w") as f:
                    json.dump(d, f)
            except Exception:
                pass

    # Rescale to O(1) losses (U15: WLOG OPT=1 rescaling; argmin x* unchanged,
    # relative gaps unchanged).  Lets the standard O(1) tuning grids work on every
    # dataset (ACS ~1e2, synthetic ~1e6) without per-arm scale fiddling.
    from gdr.problem import rescale_problem, max_loss as _max_loss
    x0_raw = _erm_warmstart(problem)
    L0 = float(_max_loss(problem, x0_raw))
    problem = rescale_problem(problem, L0)
    opt = opt / L0
    x0 = _erm_warmstart(problem)  # same x* (scale-invariant)
    cfg["opt"] = opt
    mod_name = _arm_module_name(args.arm)
    if mod_name == "opt_reference":
        print(f"FINAL {args.arm}_{args.dataset}=0")
        return 0
    h = run_arm(mod_name, cfg, problem, x0, opt,
                max_outer=args.max_outer, time_budget=args.time_budget)
    it, _sec = time_to_gap(h, rel_gap=args.rel_gap)
    value = int(it) if it is not None else -1
    final_rel = (h["gap"][-1] / opt) if opt > 0 else float("nan")
    print(f"[{args.arm}/{args.dataset}] iters={len(h['iter'])-1} "
          f"final_rel_gap={final_rel:.4g} iters_to_{int(args.rel_gap*100)}%={value}",
          file=sys.stderr)
    print(f"FINAL {args.arm}_{args.dataset}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
