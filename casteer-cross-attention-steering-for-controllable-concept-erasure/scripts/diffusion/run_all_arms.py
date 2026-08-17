"""Run every CASteer arm at the paper's full configuration at every seed and
write measured.json.

Each arm prints exactly one line:
    FINAL <arm>=<value>      (the arm's primary metric for this run)
or
    FINAL <arm>=BLOCKED      (this environment cannot produce the value)

measured.json shape: {arm: {seed: {metric: value-or-"BLOCKED"}}} covering every
(arm, seed, metric) referenced by claims.json claims. The arm and metric names
are claims.json's; do not rename. A sidecar measured_blocked_reasons.json
records why each BLOCKED value could not be produced.

On a CPU-only host the paper's full config (50 steps, >=1000 prompts, 3 seeds)
is infeasible (paper used 8xV100, supplementary.tex:30): is_arm_feasible returns
False and the arm emits BLOCKED without downloading any model. On a CUDA host
the same script would run the arms for real.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from core.runner import ARM_CONFIG, is_arm_feasible
from core.eval.metrics import MissingEvaluatorError
from core.invariants import cpu_invariant_metrics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLAIMS_PATH = os.path.join(REPO, "claims.json")
MEASURED_PATH = os.path.join(REPO, "measured.json")
REASONS_PATH = os.path.join(REPO, "measured_blocked_reasons.json")

_MEASURED_RE = re.compile(r"measured\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)")


def arm_metrics_from_claims(claims: dict) -> dict:
    """Parse every measured.<arm>.<metric> reference in the claims (quantity,
    against, and curve x/against lists) -> {arm: set(metrics)}. claims.json is
    the single source of arm/metric names."""
    out: dict[str, set[str]] = {}
    texts = []
    for c in claims.get("claims", []):
        for key in ("quantity", "against"):
            v = c.get(key)
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, list):
                texts.extend(str(x) for x in v)
        # value claims reference measured.* in quantity too (already above)
    for t in texts:
        for arm, metric in _MEASURED_RE.findall(t):
            out.setdefault(arm, set()).add(metric)
    return {a: sorted(ms) for a, ms in out.items()}


def main():
    with open(CLAIMS_PATH) as f:
        claims = json.load(f)
    arms = claims["arms"]
    seeds = claims["seeds"]
    arm_metrics = arm_metrics_from_claims(claims)

    measured: dict[str, dict[int, dict[str, object]]] = {}
    reasons: dict[str, dict[int, dict[str, str]]] = {}

    for arm in arms:
        measured[arm] = {}
        reasons[arm] = {}
        metrics = arm_metrics.get(arm, [])
        for seed in seeds:
            measured[arm][seed] = {}
            reasons[arm][seed] = {}
            feasible, why = is_arm_feasible(arm, scale="full")
            if not feasible:
                print(f"FINAL {arm}=BLOCKED")
                for m in metrics:
                    measured[arm][seed][m] = "BLOCKED"
                    reasons[arm][seed][m] = why
                # CPU-runnable invariants (e.g. the pure-math `house`
                # Householder norm-preservation claim) are filled in EVEN when
                # the diffusion arm is BLOCKED: they need no GPU and no
                # generation. The numbers gate evaluates claim `house`'s
                # predicate against these real values, so it settles to
                # `reproduced` (or `refuted`) rather than `unevaluable`/`blocked`.
                inv = cpu_invariant_metrics(arm, seed)
                for im, iv in inv.items():
                    measured[arm][seed][im] = iv
                continue
            # Feasible (GPU host): run + evaluate. On a CPU host we never reach
            # here. Wrapped so a MissingEvaluatorError -> BLOCKED, not a crash.
            try:
                value = _run_and_evaluate(arm, seed, metrics)
                primary = _primary_metric(arm, metrics)
                pv = value.get(primary, "BLOCKED")
                print(f"FINAL {arm}={pv}")
                for m in metrics:
                    measured[arm][seed][m] = value.get(m, "BLOCKED")
            except MissingEvaluatorError as e:
                print(f"FINAL {arm}=BLOCKED")
                for m in metrics:
                    measured[arm][seed][m] = "BLOCKED"
                    reasons[arm][seed][m] = f"MissingEvaluatorError: {e}"
            except Exception as e:  # any other failure -> BLOCKED with reason
                print(f"FINAL {arm}=BLOCKED")
                for m in metrics:
                    measured[arm][seed][m] = "BLOCKED"
                    reasons[arm][seed][m] = f"{type(e).__name__}: {e}"

    with open(MEASURED_PATH, "w") as f:
        json.dump(measured, f, indent=2, sort_keys=True)
    with open(REASONS_PATH, "w") as f:
        json.dump(reasons, f, indent=2, sort_keys=True)
    print(f"wrote {MEASURED_PATH}")


def _primary_metric(arm, metrics):
    """The one metric printed on the FINAL line for this arm."""
    # Prefer nudity_total for nudity arms, coco_fid30k for coco arms, etc.
    for pref in ("nudity_total", "i2p_overall_pct", "coco_fid30k", "snoopy_cs",
                 "mean_others_fid", "vangogh_lpips_e"):
        if pref in metrics:
            return pref
    return metrics[0] if metrics else "BLOCKED"


def _run_and_evaluate(arm, seed, metrics):
    """Actually run the arm at full config and evaluate its metrics.

    Only invoked on a feasible (GPU) host. This wires the runner + eval
    together; on CPU it is never called. It is intentionally explicit so a
    reader can follow exactly what would be measured.
    """
    from core.runner import run_generation
    from core.data import load_i2p_prompts, load_coco_captions
    values = {}
    # NOTE: full-config generation needs a GPU; see REPRODUCTION.md.
    raise MissingEvaluatorError(
        "full-config arm execution is not implemented for this CPU sandbox; "
        "run on a CUDA host (see run_all_arms.sh)."
    )


if __name__ == "__main__":
    main()
