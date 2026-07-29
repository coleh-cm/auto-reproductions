"""Gradient descent on the LSE-smoothed surrogate f_tilde_{beta,delta} (E3).

Paper: smoothed gradient-descent baseline (E22, experiments.tex:60-66).
One gradient step per outer iteration (experiments.tex:99):

    x_{k+1} = x_k - lr * grad f_tilde(x_k)        # [d]

with ``grad f_tilde = smoothed_grad_hess(problem, x, beta, delta)[1]`` (E4).
Best-of-grid over (lr, beta, delta); keep the config with the lowest final
F(x) (experiments.tex:92), return that curve.
"""

from __future__ import annotations

import time
import numpy as np

from gdr.objectives import smoothed_grad_hess  # frozen interface (E4)
from gdr.solvers import (
    DEFAULTS, as_list, get_opt, gap_now, seed_history,
    beta_delta_pairs, max_loss,
)


def _run_single(problem, x0, max_outer, deadline, lr, beta, delta, opt):
    """One (lr, beta, delta) run from x0.  Returns (history, final F)."""
    h = seed_history(problem, x0, opt)     # iter 0 recorded
    x = h["x"].copy()                      # [d]
    t0 = time.perf_counter()
    for k in range(1, max_outer + 1):
        if h["gap"][-1] <= 0.0:
            break
        if time.perf_counter() > deadline:
            break
        _val, g, _H = smoothed_grad_hess(problem, x, beta, delta)  # E4; g [d]
        x = x - lr * g                      # [d]  GD step on f_tilde (E3)
        h["iter"].append(k)
        h["gap"].append(gap_now(problem, x, opt))
        t = time.perf_counter() - t0
        h["time"].append(t)
        if h["gap"][-1] <= 0.0:
            break
        if t > deadline - t0:
            break
    h["x"] = x
    return h, float(max_loss(problem, x))


def run(problem, cfg, x0, max_outer, time_budget):
    """Best-of-grid smoothed gradient descent.  See module docstring."""
    cfg = cfg or {}
    opt = get_opt(problem, cfg)
    x0 = np.asarray(x0, dtype=np.float64)          # [d]
    lr_grid = as_list(cfg.get("lr_grid"), DEFAULTS["lr_grid"])

    t_start = time.perf_counter()
    deadline = t_start + float(time_budget)
    best = None
    best_final = np.inf
    for lr in lr_grid:
        for beta, delta in beta_delta_pairs(cfg):
            if time.perf_counter() > deadline:
                break
            h, final = _run_single(problem, x0, max_outer, deadline,
                                   float(lr), beta, delta, opt)
            if final < best_final:
                best_final = final
                best = h
    if best is None:
        best = seed_history(problem, x0, opt)
    return best
