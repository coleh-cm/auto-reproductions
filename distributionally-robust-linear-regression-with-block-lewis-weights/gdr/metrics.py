"""Evaluation metrics (E5, E6).

E5  gap curve: best-so-far ``F(x_t) - OPT`` per natural outer iteration
    (``paper/experiments.tex:52,167``).
E6  cost to a relative gap: first ``t`` with ``gap_t / gap_ref <= rel``; the
    reference (initial gap vs OPT) is unstated by the paper (SPEC section 6
    item 4), so both ``base="init"`` (``gap_t / gap_0``) and ``base="opt"``
    (``gap_t / OPT``) are computed and the harness records both.

Every function here operates on already-computed scalars so the metric is fully
independent of how an arm produced its iterates -- once it matches the paper's
definition it is not touched again.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .types import GroupProblem

__all__ = ["gap_curve", "cost_to_rel_gap", "statistical_context"]


def gap_curve(
    worst_losses: list[float] | np.ndarray,
    opt: float,
) -> np.ndarray:
    """E5: best-so-far ``F(x_t) - OPT`` for t = 0..T.

    ``worst_losses[t] = F(x_t)`` (the worst-group MSE at iterate t).  "Best-so-far"
    means the running minimum of the gap, so a method that temporarily worsens is
    not credited with the later re-improvement unless it actually beats its prior
    best -- this matches the paper's "report the gap ... best-so-far per curve"
    convention (``paper/experiments.tex:52``).
    """
    w = np.asarray(worst_losses, dtype=np.float64)
    if w.size == 0:
        raise ValueError("gap_curve: empty worst_losses (no iterates) -- refusing to report OK on nothing")
    gaps = w - float(opt)
    # A large negative gap means OPT was computed ABOVE the true minimum (wrong
    # reference).  A tiny negative slack (~1e-6) is solver tolerance and is
    # clipped to 0; a large one is a bug we must surface, not hide (task: "No
    # success path may report OK on an empty result" — a wrong OPT is exactly
    # that, and a silently-clipped curve would let a broken run pass the gate).
    if np.min(gaps) < -1e-3:
        raise ValueError(
            f"gap_curve: gap {np.min(gaps):.3g} << 0 — OPT ({opt}) exceeds a "
            f"measured F(x_t) ({np.min(w):.6g}); the reference optimum is wrong"
        )
    gaps = np.maximum(gaps, 0.0)          # clip solver-tolerance slack to 0
    best = np.minimum.accumulate(gaps)
    return best


def cost_to_rel_gap(
    curve: np.ndarray,
    gap0: float,
    rel: float,
    base: str = "init",
    opt: float | None = None,
) -> int | None:
    """E6: first index ``t`` (0-based) with ``gap_t / gap_ref <= rel``.

    base="init" : gap_ref = gap0 = F(x_0) - OPT   (relative to the initial gap)
    base="opt"  : gap_ref = OPT                   (relative to the optimum value)

    Returns ``None`` if the target is never reached within the budget (the paper
    marks the subgradient arm with "--- (not reached)" exactly this way,
    ``paper/experiments.tex:178``).  ``gap0`` must be > 0 for base="init" --
    a zero initial gap means the warm start is already optimal and the 1%-target
    is ill-defined; we raise rather than silently return 0.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.size == 0:
        raise ValueError("cost_to_rel_gap: empty curve -- refusing to report OK on nothing")
    if base == "init":
        if gap0 <= 0.0:
            raise ValueError(f"base='init' needs gap0>0, got {gap0}; warm start already optimal")
        ref = float(gap0)
    elif base == "opt":
        if opt is None:
            raise ValueError("base='opt' requires opt")
        if float(opt) <= 0.0:
            raise ValueError(f"base='opt' needs opt>0, got {opt}")
        ref = float(opt)
    else:
        raise ValueError(f"unknown base {base!r}")
    target = ref * float(rel)
    hits = np.where(curve <= target)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def statistical_context(problem: GroupProblem, x_erm: np.ndarray, x_robust: np.ndarray) -> dict:
    """The "Statistical context" vector the paper reports
    (``paper/experiments.tex:189``): ERM per-group losses (mean/std/worst/argmax),
    robust-optimum per-group losses (mean/min/max, Max/Mean ratio), and the
    per-group loss change from ERM to robust.  Scale-dependent numbers are
    reported verbatim and flagged in REPRODUCTION.md; the headline gate metric
    (iterations to 1% gap) is scale-invariant.
    """
    erm_losses = problem.group_losses(x_erm)
    rob_losses = problem.group_losses(x_robust)
    return {
        "erm_losses": erm_losses,
        "robust_losses": rob_losses,
        "erm_avg": float(np.mean(erm_losses)),
        "erm_std": float(np.std(erm_losses)),
        "erm_worst": float(np.max(erm_losses)),
        "erm_worst_group": int(np.argmax(erm_losses)),
        "robust_avg": float(np.mean(rob_losses)),
        "robust_min": float(np.min(rob_losses)),
        "robust_max": float(np.max(rob_losses)),
        "robust_max_over_mean": float(np.max(rob_losses) / np.mean(rob_losses)),
        "erm_max_over_mean": float(np.max(erm_losses) / np.mean(erm_losses)),
        "per_group_delta": rob_losses - erm_losses,
        "worst_erm_group_delta": float((rob_losses - erm_losses)[int(np.argmax(erm_losses))]),
    }
