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
