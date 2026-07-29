"""Subgradient descent on the nonsmooth max-loss objective F(x) = max_i ||r_i||^2.

Paper: subgradient baseline (E22, experiments.tex:57-58).  The subgradient at
the argmax group i* = argmax_i ||r_i||^2 is

    g = 2 * A_{S_{i*}}^T (A_{S_{i*}} x - b_{S_{i*}})        # [d]

(one full subgradient step per outer iteration; experiments.tex:99).  We run
best-of-grid over (lr, schedule) with schedules {const, 1/sqrt_t} and keep the
config with the lowest final F(x) (experiments.tex:92), returning that curve.
"""

from __future__ import annotations

import time
import numpy as np

from gdr.solvers import (
    DEFAULTS, as_list, get_opt, gap_now, seed_history,
    max_loss, group_losses,
)


def _subgradient(problem, x):
    """Subgradient of F at the argmax group (E22, experiments.tex:57-58).

    Returns g [d] float64.
    """
    A = problem["A"]                       # [n, d]
    b = problem["b"]                       # [n]
    offsets = problem["offsets"]           # [m+1] int64
    ell = group_losses(problem, x)         # [m]  squared group norms
    i = int(np.argmax(ell))                # scalar argmax group i*
    s, e = int(offsets[i]), int(offsets[i + 1])  # rows [s:e]
    r = A[s:e, :] @ x - b[s:e]             # [n_i]  r_{i*}
    g = 2.0 * (A[s:e, :].T @ r)            # [d]  2 A_{S_i}^T r_i
    return g


def _run_single(problem, x0, max_outer, deadline, lr, schedule, opt):
    """One (lr, schedule) run from x0.  Returns (history, final F)."""
    h = seed_history(problem, x0, opt)     # iter 0 recorded
    x = h["x"].copy()                      # [d]
    t0 = time.perf_counter()
    for k in range(1, max_outer + 1):
        if h["gap"][-1] <= 0.0:
            break
        if time.perf_counter() > deadline:
            break
        step = lr if schedule == "const" else lr / np.sqrt(float(k))  # scalar
        g = _subgradient(problem, x)        # [d]
        x = x - step * g                    # [d]  subgradient step (E22)
        h["iter"].append(k)
        h["gap"].append(gap_now(problem, x, opt))
        t = time.perf_counter() - t0
        h["time"].append(t)
        if h["gap"][-1] <= 0.0:
            break
        if t > deadline - t0:  # time_budget exhausted
            break
    h["x"] = x
    return h, float(max_loss(problem, x))


def run(problem, cfg, x0, max_outer, time_budget):
    """Best-of-grid subgradient descent.  See module docstring."""
    cfg = cfg or {}
    opt = get_opt(problem, cfg)
    x0 = np.asarray(x0, dtype=np.float64)          # [d]
    lr_grid = as_list(cfg.get("lr_grid"), DEFAULTS["lr_grid"])
    # The runner (gdr/runner.py:92) passes the schedule grid under key 'schedule';
    # the in-solver default and tests use 'schedule_grid'.  Read both so a
    # non-default grid passed via the runner is not silently dropped (the prior
    # code read only 'schedule_grid' and fell back to the default, masking the
    # runner value).
    sched_grid = as_list(cfg.get("schedule") or cfg.get("schedule_grid"),
                         DEFAULTS["schedule_grid"])

    t_start = time.perf_counter()
    deadline = t_start + float(time_budget)
    best = None
    best_final = np.inf
    for lr in lr_grid:
        for schedule in sched_grid:
            if time.perf_counter() > deadline:
                break
            h, final = _run_single(problem, x0, max_outer, deadline,
                                    float(lr), schedule, opt)
            if final < best_final:
                best_final = final
                best = h
    if best is None:  # time_budget already exhausted before any combo ran
        best = seed_history(problem, x0, opt)
    return best
