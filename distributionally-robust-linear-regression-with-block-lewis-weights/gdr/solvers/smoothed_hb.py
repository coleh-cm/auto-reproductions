"""Heavy-Ball (Polyak) momentum on the LSE-smoothed surrogate (E3).

Paper: smoothed Heavy-Ball baseline (E22, experiments.tex:60-66).  One momentum
step per outer iteration (experiments.tex:99):

    x_{k+1} = x_k - lr * grad f_tilde(x_k) + momentum * (x_k - x_{k-1})   # [d]

with ``grad f_tilde = smoothed_grad_hess(...)[1]`` (E4).  Best-of-grid over
(lr, momentum, beta, delta); keep the lowest final F(x) (experiments.tex:92).
"""

from __future__ import annotations

import time
import numpy as np

from gdr.objectives import smoothed_grad_hess  # frozen interface (E4)
from gdr.solvers import (
    DEFAULTS, as_list, get_opt, gap_now, seed_history,
    beta_delta_pairs, max_loss,
)


def _run_single(problem, x0, max_outer, deadline, lr, momentum, beta, delta, opt):
    """One (lr, momentum, beta, delta) run from x0.  Returns (history, final F)."""
    h = seed_history(problem, x0, opt)     # iter 0 recorded
    x_prev = h["x"].copy()                 # [d]  x_{k-1}
    x = h["x"].copy()                      # [d]  x_k
    t0 = time.perf_counter()
    for k in range(1, max_outer + 1):
        if h["gap"][-1] <= 0.0:
            break
        if time.perf_counter() > deadline:
            break
        _val, g, _H = smoothed_grad_hess(problem, x, beta, delta)  # E4; g [d]
        x_new = x - lr * g + momentum * (x - x_prev)  # [d]  Heavy-Ball step
        x_prev, x = x, x_new              # shift registers
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
    """Best-of-grid smoothed Heavy-Ball.  See module docstring."""
    cfg = cfg or {}
    opt = get_opt(problem, cfg)
    x0 = np.asarray(x0, dtype=np.float64)          # [d]
    lr_grid = as_list(cfg.get("lr_grid"), DEFAULTS["lr_grid"])
    mom_grid = as_list(cfg.get("momentum_grid"), DEFAULTS["momentum_grid"])

    t_start = time.perf_counter()
    deadline = t_start + float(time_budget)
    best = None
    best_final = np.inf
    for lr in lr_grid:
        for momentum in mom_grid:
            for beta, delta in beta_delta_pairs(cfg):
                if time.perf_counter() > deadline:
                    break
                h, final = _run_single(problem, x0, max_outer, deadline,
                                       float(lr), float(momentum),
                                       beta, delta, opt)
                if final < best_final:
                    best_final = final
                    best = h
            if time.perf_counter() > deadline:
                break
        if time.perf_counter() > deadline:
            break
    if best is None:
        best = seed_history(problem, x0, opt)
    return best
