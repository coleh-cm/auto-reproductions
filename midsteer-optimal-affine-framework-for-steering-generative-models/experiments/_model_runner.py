"""Shared runner for the BLOCKED model arms (E2-E5). Each model experiment imports this
and calls run_blocked_or_real with its arms/metrics/seeds; in this CPU sandbox every
model arm takes the BLOCKED branch and writes BLOCKED for every metric at every seed.
On a GPU+HF_TOKEN host the real branch would load the model, estimate stats, steer,
generate, and score with midsteer_core.eval."""
from __future__ import annotations
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midsteer_core.data import is_model_arm_blocked


def run_blocked_or_real(exp_name, arms, metric_keys, betas, out_partial, real_fn=None):
    """Run every arm x seed; BLOCKED here, else real_fn(arm, seed, beta)->{metric: value}.

    Writes results/<exp>_partial.json: {arm: {seed_str: {metric: value_or_BLOCKED}}}.
    Per-arm diagnostics go to stderr; the single authoritative 'FINAL <arm>=...' line
    per arm is emitted by assemble_measured.py at the end of run_all_arms.sh, so each
    arm name appears in exactly one FINAL line (no duplicates).
    """
    seeds = [0, 1, 2]
    blocked = is_model_arm_blocked()
    partial = {}
    for arm in arms:
        arm_beta = betas.get(arm) if isinstance(betas, dict) else betas
        partial[arm] = {}
        any_real = False
        for seed in seeds:
            partial[arm][str(seed)] = {}
            if blocked:
                for m in metric_keys:
                    partial[arm][str(seed)][m] = "BLOCKED"
            else:
                vals = real_fn(arm, seed, arm_beta) if real_fn else {}
                if not vals:
                    raise RuntimeError(f"{exp_name}: empty result for arm={arm} seed={seed}")
                partial[arm][str(seed)].update(vals)
                any_real = True
        # Per-arm diagnostic to stderr only (do NOT print FINAL here — assemble_measured
        # emits the single authoritative FINAL <arm> line per arm).
        if blocked or not any_real:
            sys.stderr.write(f"[{exp_name}] {arm}: BLOCKED (no CUDA / no HF_TOKEN in this sandbox)\n")
        else:
            pm = metric_keys[0]
            last = partial[arm][str(seeds[-1])].get(pm, "BLOCKED")
            sys.stderr.write(f"[{exp_name}] {arm}: {pm}={last}\n")
    os.makedirs(os.path.dirname(out_partial), exist_ok=True)
    with open(out_partial, 'w') as f:
        json.dump(partial, f, indent=2)
    if blocked:
        sys.stderr.write(f"[{exp_name}] BLOCKED: no CUDA and/or no HF_TOKEN in this sandbox; "
                          f"model arms not run, no synthetic fallback.\n")
    return partial
