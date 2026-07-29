"""Entrypoint for a single arm on a single dataset (SPEC §4, §5; experiments §8).

    python run_arm.py --arm <name> [--geometry euclidean|lewis] \
                     [--dataset synthetic|acs_income] \
                     [--max-outer N] [--time-budget S] [--rel-gap g] [--seed s]

Loads the (folded) problem, computes (or loads a cached) OPT via the CVXPY
epigraph QP (E20), warm-starts every arm at the ERM solution (experiments.tex:
92,148), runs the arm's best-of-grid search, and prints exactly one line

    FINAL <dataset>_<arm>=<value>

where <value> is the number of the arm's own outer iterations needed to reach
the relative worst-group suboptimality (F(x)-OPT)/OPT <= rel_gap (T1 gate,
experiments.tex:176-186), or ``NR`` if it does not reach the target within the
budget.  The full history (iter/gap/time/x) is saved to results/<dataset>_<arm>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# make `gdr` importable when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gdr.problem import max_loss
from gdr.reference import solve_opt
from gdr.runner import run_arm, time_to_gap


# ---------------------------------------------------------------------------
# per-arm tuning grids (disclosed choices; paper states only "grid search",
# experiments.tex:83-92; U2).  Kept modest so the full run finishes in minutes.
# ---------------------------------------------------------------------------
ARM_CONFIGS = {
    "subgradient": {
        "lr_grid": [1e-3, 1e-2, 1e-1, 1.0],
        "schedule_grid": ["const", "1/sqrt_t"],
    },
    "smoothed_gd": {
        "lr_grid": [1e-3, 1e-2, 1e-1],
        "beta_grid": [1e-2, 1e-1, 1.0],
        "delta_grid": [1e-2, 1e-1, 1.0],
    },
    "smoothed_hb": {
        "lr_grid": [1e-3, 1e-2, 1e-1],
        "momentum_grid": [0.5, 0.9],
        "beta_grid": [1e-2, 1e-1, 1.0],
        "delta_grid": [1e-2, 1e-1, 1.0],
    },
    "smoothed_nesterov": {
        "lr_grid": [1e-3, 1e-2, 1e-1],
        "momentum_grid": [0.5, 0.9],
        "beta_grid": [1e-2, 1e-1, 1.0],
        "delta_grid": [1e-2, 1e-1, 1.0],
    },
    "ipm": {
        "mu0_grid": [1.0, 10.0, 100.0],
        "theta_grid": [0.1, 0.3, 0.5],
        "inner_tol": 1e-8,
    },
    "ball_oracle": {
        "r0_grid": [0.5, 1.0, 5.0],
        "shrink_grid": [0.5, 1.0],
        "beta_grid": [1e-2, 1e-1],
        "delta_grid": [1e-2, 1e-1],
        "tol_inner": 1e-8,
    },
    "opt_reference": {},
}


def load_problem(dataset: str, seed: int):
    if dataset == "synthetic":
        from gdr.data_synthetic import make_synthetic
        return make_synthetic(seed=seed)
    elif dataset == "acs_income":
        from gdr.data_acs import make_acs_income
        return make_acs_income(seed=seed)
    else:
        raise ValueError(f"unknown dataset {dataset}")


def get_opt_cached(problem, dataset: str, solver: str = "CLARABEL"):
    """OPT via the E20 epigraph QP, cached under results/opt_<dataset>.json."""
    os.makedirs("results", exist_ok=True)
    cache = f"results/opt_{dataset}.json"
    if os.path.isfile(cache):
        try:
            with open(cache) as f:
                d = json.load(f)
            if d.get("m") == problem["m"] and d.get("n") == problem["A"].shape[0]:
                return float(d["opt"])
        except Exception:
            pass
    x_star, opt = solve_opt(problem, solver=solver)
    with open(cache, "w") as f:
        json.dump({"opt": float(opt), "m": problem["m"],
                   "n": int(problem["A"].shape[0])}, f, indent=2)
    return float(opt)


def erm_warm_start(problem) -> np.ndarray:
    """All methods use the same ERM warm start (experiments.tex:92,148)."""
    return np.linalg.lstsq(problem["A"], problem["b"], rcond=None)[0]


def normalize_problem(problem, opt):
    """Rescale (A, b) by 1/sqrt(opt) so the normalized OPT == 1 (theory's WLOG
    convention, body.tex:511-513 / U15).  x* is unchanged; group losses and the
    objective all divide by opt, so the *relative* gap (F-OPT)/OPT == F'-1 is
    unchanged.  This keeps every arm on an O(1) loss scale, which the IPM's
    damped-Newton centering needs (it stalls on O(1e5) losses at cond 1e4+ even
    though it converges at the same cond with O(1) losses).  Leverage scores and
    block Lewis weights are scale-invariant, so the geometry is unaffected up to
    the 1/opt factor folded into M.
    """
    s = float(np.sqrt(opt)) if opt > 0 else 1.0
    if s == 1.0:
        return problem, 1.0
    prob = dict(problem)
    prob["A"] = problem["A"] / s
    prob["b"] = problem["b"] / s
    return prob, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--geometry", default=None, choices=["euclidean", "lewis", None])
    ap.add_argument("--dataset", default="acs_income",
                    choices=["synthetic", "acs_income"])
    ap.add_argument("--max-outer", type=int, default=300)
    ap.add_argument("--time-budget", type=float, default=120.0)
    ap.add_argument("--rel-gap", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", default="CLARABEL")
    args = ap.parse_args()

    problem = load_problem(args.dataset, args.seed)
    opt = get_opt_cached(problem, args.dataset, args.solver)
    # normalize so OPT == 1 (U15); arms run on the normalized problem
    problem, scale = normalize_problem(problem, opt)
    opt_norm = 1.0
    x0 = erm_warm_start(problem)

    arm = args.arm
    cfg = dict(ARM_CONFIGS.get(arm, {}))
    cfg["opt"] = opt_norm
    cfg["solver"] = args.solver
    if arm == "ball_oracle" and args.geometry is not None:
        cfg["geometry"] = args.geometry
        arm_name = f"ball_oracle_{args.geometry}"
    else:
        arm_name = arm

    t_start = time.perf_counter()
    hist = run_arm(arm, cfg, problem, x0, opt_norm,
                  max_outer=args.max_outer, time_budget=args.time_budget)
    elapsed = time.perf_counter() - t_start

    it_to_gap, _t = time_to_gap(hist, rel_gap=args.rel_gap)
    value = "NR" if it_to_gap is None else str(it_to_gap)

    # save full history (committed evidence).  Gaps are on the normalized
    # (OPT=1) scale; the relative gap F'-1 == (F-OPT)/OPT is what the gate reads.
    os.makedirs("results", exist_ok=True)
    F0_norm = float(max_loss(problem, x0))
    out = {
        "arm": arm_name, "dataset": args.dataset, "opt": opt, "opt_norm": opt_norm,
        "scale": scale, "F0_norm": F0_norm, "rel_gap": args.rel_gap,
        "iters_to_rel_gap": it_to_gap, "elapsed": elapsed,
        "max_outer": args.max_outer, "time_budget": args.time_budget,
        "history": {k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in hist.items()},
    }
    with open(f"results/{args.dataset}_{arm_name}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"FINAL {args.dataset}_{arm_name}={value}")


if __name__ == "__main__":
    main()
