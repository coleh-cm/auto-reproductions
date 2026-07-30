"""Experiment harness: build instance, compute OPT, tune+run each arm, emit FINAL.

Tuning protocol matches the paper (``paper/experiments.tex:81-92``): each arm is
grid-searched over its relevant hyperparameters and the configuration achieving
the lowest worst-group loss within a fixed budget is selected; the reported
metric (iterations to 1% relative gap, ``paper/experiments.tex:174-186``) is then
computed for that best config.  The 1%-reference is unstated (SPEC section 6
item 4), so both ``base="init"`` (gap/gap0) and ``base="opt"`` (gap/OPT) are
recorded; the FINAL line uses ``base="init"`` (scale-invariant, the most natural
reading of "relative suboptimality").

Determinism: every stochastic component takes the seed recorded in the run
config (SPEC section 7 determinism contract); a single seed lineage per row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable

import numpy as np

from .types import GroupProblem
from . import solvers, metrics, lewis

__all__ = ["main"]


# --------------------------------------------------------------------------------------------
# Tuning grids (reconstructed, SPEC section 6 item 3).  Modest, plausible ranges;
# the grid search keeps the arm that achieves the lowest worst-group loss.
# --------------------------------------------------------------------------------------------
GRIDS = {
    "subgradient": [
        {"step": st, "schedule": sch}
        for st in (1e-6, 1e-5, 1e-4, 1e-3) for sch in ("fixed", "inv_sqrt")
    ],
    "smoothed_gd": [
        {"method": "gd", "beta": b, "delta": dl, "step": st}
        for b in (0.5, 1.0, 5.0) for dl in (0.5, 1.0) for st in (1e-4, 1e-3, 1e-2)
    ],
    "smoothed_heavy_ball": [
        {"method": "heavy_ball", "beta": b, "delta": dl, "step": st, "momentum": mu}
        for b in (0.5, 1.0, 5.0) for dl in (0.5, 1.0)
        for st in (1e-4, 1e-3, 1e-2) for mu in (0.3, 0.5, 0.7, 0.9)
    ],
    "smoothed_nesterov": [
        {"method": "nesterov", "beta": b, "delta": dl, "step": st}
        for b in (0.5, 1.0, 5.0) for dl in (0.5, 1.0) for st in (1e-4, 1e-3, 1e-2)
    ],
    "ipm": [
        {"barrier0": b0, "growth": gr, "damping": 1e-6}
        for b0 in (1.0, 10.0) for gr in (5.0, 10.0, 20.0)
    ],
    "ball_oracle_euclidean": [
        {"geometry": "naive", "radius0": r, "decay": 1.0, "beta": b, "delta": dl,
         "reg_on": reg, "inner_iters": 50}
        for r in (1.0, 5.0, 50.0, 1e3) for b in (0.05, 0.2, 1.0)
        for dl in (0.05, 0.2) for reg in (False, True)
    ],
    "ball_oracle_lewis": [
        {"geometry": "lewis", "lewis_p": None, "lewis_iters": None,
         "radius0": r, "decay": 1.0, "beta": b, "delta": dl,
         "reg_on": reg, "inner_iters": 50}
        for r in (1.0, 5.0, 50.0, 1e3) for b in (0.05, 0.2, 1.0)
        for dl in (0.05, 0.2) for reg in (False, True)
    ],
}

ARM_SOLVER = {
    "subgradient": solvers.solve_subgradient,
    "smoothed_gd": solvers.solve_smooth,
    "smoothed_heavy_ball": solvers.solve_smooth,
    "smoothed_nesterov": solvers.solve_smooth,
    "ipm": solvers.solve_ipm,
    "ball_oracle_euclidean": solvers.solve_ball_oracle,
    "ball_oracle_lewis": solvers.solve_ball_oracle,
}

BUDGET = 100          # outer-iteration budget for the reported curve
TUNE_BUDGET = 80      # budget used during the grid search


def _build_instance(name: str) -> GroupProblem:
    if name == "synthetic":
        from .data_synth import make_synthetic
        return make_synthetic()
    if name == "acs":
        from .data_acs import make_acs
        return make_acs()
    raise ValueError(f"unknown instance {name!r}")


def _tune_and_run(arm: str, problem: GroupProblem, x0: np.ndarray, budget: int,
                  tune_budget: int) -> tuple[dict, solvers.History]:
    """Grid-search the arm, return (best_cfg, history at the reported budget)."""
    grids = GRIDS[arm]
    solver = ARM_SOLVER[arm]
    best = None
    for cfg in grids:
        try:
            h = solver(problem, x0, cfg, tune_budget)
        except Exception:
            continue
        final = problem.worst_loss(h.x[-1])
        if not np.isfinite(final):
            continue
        if best is None or final < best[0]:
            best = (final, cfg)
    if best is None:
        raise RuntimeError(f"arm {arm!r}: every grid config failed -- refusing to report OK on no fit")
    cfg = best[1]
    h = solver(problem, x0, cfg, budget)
    return cfg, h


def run_arm(arm: str, problem: GroupProblem, x0: np.ndarray, opt: float,
            budget: int = BUDGET, tune_budget: int = TUNE_BUDGET) -> dict:
    """Run one arm fully; return a result dict (cfg, curve, metrics)."""
    if arm == "reference_cvxpy":
        xstar, opt_v = solvers.reference_optimum(problem)
        return {"arm": arm, "opt": opt_v, "xstar": xstar.tolist()}
    cfg, h = _tune_and_run(arm, problem, x0, budget, tune_budget)
    worst = h.worst_losses(problem)
    curve = metrics.gap_curve(worst, opt)
    gap0 = float(curve[0])
    it_init = metrics.cost_to_rel_gap(curve, gap0, 0.01, base="init")
    try:
        it_opt = metrics.cost_to_rel_gap(curve, gap0, 0.01, base="opt", opt=opt)
    except ValueError:
        it_opt = None
    return {
        "arm": arm,
        "cfg": _jsonable(cfg),
        "budget": budget,
        "worst_losses": worst.tolist(),
        "curve": curve.tolist(),
        "gap0": gap0,
        "final_gap": float(curve[-1]),
        "iterations_to_1pct_init": (None if it_init is None else int(it_init)),
        "iterations_to_1pct_opt": (None if it_opt is None else int(it_opt)),
        "wall_final": float(h.wall[-1]),
    }


def _jsonable(cfg):
    out = {}
    for k, v in cfg.items():
        out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    return out


def _final_value(result: dict):
    if result["arm"] == "reference_cvxpy":
        return result["opt"]
    it = result.get("iterations_to_1pct_init")
    return "not_reached" if it is None else int(it)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GDR reproduction harness")
    ap.add_argument("command", choices=["arm", "all", "smoke", "reference", "context"])
    ap.add_argument("--arm")
    ap.add_argument("--instance", default="acs")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--tune-budget", type=int, default=TUNE_BUDGET)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    if args.command == "smoke":
        # tiny synthetic, one ball-oracle arm, one FINAL line -- proves the path
        # runs; NOT evidence about the paper (never report its number as a result).
        from .data_synth import make_synthetic
        cfg = {**__import__("gdr.data_synth", fromlist=["SYNTH_CONFIG"]).SYNTH_CONFIG,
               "m": 6, "d": 3, "n_adversarial": 1, "n_per_group": 6}
        problem = make_synthetic(cfg)
        x0 = problem.erm()
        xstar, opt = solvers.reference_optimum(problem)
        h = solvers.solve_ball_oracle(
            problem, x0,
            {"geometry": "naive", "radius0": 50.0, "decay": 1.0, "beta": 0.2,
             "delta": 0.1, "reg_on": False, "inner_iters": 20}, 3)
        curve = metrics.gap_curve(h.worst_losses(problem), opt)
        print(f"FINAL smoke={float(curve[-1]):.6g}")
        return 0

    problem = _build_instance(args.instance)
    x0 = problem.erm()
    # reference optimum
    xstar, opt = solvers.reference_optimum(problem)
    gap0 = float(problem.worst_loss(x0) - opt)

    if args.command == "reference":
        print(f"FINAL reference_cvxpy={opt}")
        return 0

    if args.command == "context":
        ctx = metrics.statistical_context(problem, x0, xstar)
        print(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in ctx.items()}, indent=2))
        return 0

    if args.command == "arm":
        if not args.arm:
            print("arm command requires --arm", file=sys.stderr); return 2
        r = run_arm(args.arm, problem, x0, opt, args.budget, args.tune_budget)
        r["opt"] = opt
        with open(os.path.join(args.out, f"{args.instance}_{args.arm}.json"), "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"FINAL {args.arm}={_final_value(r)}")
        return 0

    if args.command == "all":
        arms = ["reference_cvxpy", "subgradient", "smoothed_gd", "smoothed_heavy_ball",
                "smoothed_nesterov", "ipm", "ball_oracle_euclidean", "ball_oracle_lewis"]
        results = {}
        print(f"FINAL reference_cvxpy={opt}")
        results["reference_cvxpy"] = {"opt": opt}
        results["_meta"] = {"instance": args.instance, "m": problem.m, "d": problem.d,
                            "n": problem.n, "opt": opt, "gap0": gap0, "budget": args.budget}
        for arm in arms[1:]:
            try:
                r = run_arm(arm, problem, x0, opt, args.budget, args.tune_budget)
            except Exception as e:
                print(f"# arm {arm} FAILED: {e}", file=sys.stderr)
                print(f"FINAL {arm}=error")
                results[arm] = {"error": str(e)}
                continue
            results[arm] = r
            print(f"FINAL {arm}={_final_value(r)}")
        with open(os.path.join(args.out, f"{args.instance}_all.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
