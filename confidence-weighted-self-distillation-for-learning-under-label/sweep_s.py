#!/usr/bin/env python3
"""Sweep the one unstated hyperparameter ``s`` (gate sharpness, Eq. 2) across
both gradient modes and all claims seeds, and record the CWSD test accuracy
plus the per-seed verdicts of the three gated cwsd headline claims.

WHY THIS EXISTS
---------------
The paper's Eq. (2) defines ``s`` ("s controls how sharply the gate opens
around tau", paper/paper.md:164) but §3 never assigns it a value (the
hyperparameter sentence at :361-375 lists only lambda=1, tau=0.9, T=2 and
stops). The cwsd headline claims (+2.5 points, the 0.9620 value, the
cwsd>baseline ordering) therefore depend on a value the paper never states.

Per the SPEC requirement ("For any claim whose verdict depends on a value the
paper never states, sweep that value and record whether the verdict survives"),
this script sweeps ``s`` over a wide range — including the sharp-gate default
``s=0.15`` and the shallow-gate regime ``s>=2`` where the gate-path gradient
term (proportional to 1/s) becomes negligible — under BOTH stop-grad readings:

  * ``literal`` (the gated cwsd arm; stopgrad on p_tilde ONLY, gate weight w
    differentiable — Eqs. 2-4 as written).
  * ``detached`` (the counterfactual; whole target constant, dL/dz=(p-t)/B only
    — the standard self-distillation convention the paper does not mark on w).

The literal and detached gradients CONVERGE as s grows (the gate-path term is
proportional to 1/s and vanishes), so the literal headline reproduces Table 1
for sufficiently shallow gates (s >= ~0.7 for the ordering, s >= ~2.0 for the
value/magnitude), NOT "at any s failing" — the sweep is what establishes the
s-dependence honestly. A prior pass truncated this sweep at s=0.30 and wrongly
concluded the literal gradient never reproduces; this script extends it to
s=5.0 across seeds 0/1/2 in both modes.

Output: s_sweep.json (committed; machine-readable sweep evidence the SPEC and
REPRODUCTION.md cite). This is NOT claims_result.json (which the numbers gate
owns) and writes no verdicts the gate did not produce; it records measured
accuracies and the per-claim per-seed pass/fail under the tolerances declared
in claims.json, for transparency.

Usage:  python sweep_s.py            # writes s_sweep.json (~10 min: 60 runs)
        python sweep_s.py --quick    # the s-grid actually used for the committed file
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

SEEDS = [0, 1, 2]
# Grid spans the sharp-gate default (0.15) through the shallow-gate regime
# (>=2) where literal -> detached. Crossover for the ordering is at s~0.7;
# for the value/magnitude at s~2.0.
S_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00, 2.00, 5.00]
MODES = ["literal", "detached"]

# Claims.json tolerances (the gate's own thresholds), replicated here only to
# annotate per-seed pass/fail for transparency. The gate adjudicates; this
# script does not write claims_result.json.
BASELINE_CLAIMED = 0.937
BASELINE_TOL = 0.01
CWSD_CLAIMED = 0.962
CWSD_VALUE_TOL = 0.015
GAP_CLAIMED = 0.025
GAP_TOL = 0.02


def run_one(lam, seed, mode, s, py):
    out = subprocess.run(
        [py, "run_experiment.py", "--lambda", str(lam), "--seed", str(seed),
         "--grad-mode", mode, "--s", str(s)],
        capture_output=True, text=True,
    )
    line = [l for l in out.stdout.splitlines() if l.startswith("FINAL accuracy=")]
    if not line:
        return None
    return float(line[0].split("=")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s_sweep.json")
    ap.add_argument("--quick", action="store_true",
                    help="use the committed s-grid (default; same as no flag)")
    args = ap.parse_args()
    py = ".venv/bin/python"
    if not Path(py).exists():
        py = "python3"

    grid = S_GRID  # --quick is the default grid; kept for CLI compatibility

    # Baseline (lambda=0) is grad-mode- and s-independent: one run per seed.
    baseline = {}
    for seed in SEEDS:
        baseline[str(seed)] = run_one(0.0, seed, "literal", 0.15, py)
        print(f"baseline seed={seed} -> {baseline[str(seed)]}", file=sys.stderr)

    sweep = {"baseline": baseline, "grid": grid, "seeds": SEEDS, "modes": MODES}
    for mode in MODES:
        sweep[mode] = {}
        for s in grid:
            sk = f"{s:.2f}"
            sweep[mode][sk] = {}
            for seed in SEEDS:
                acc = run_one(1.0, seed, mode, s, py)
                sweep[mode][sk][str(seed)] = acc
                # per-seed pass/fail under the declared tolerances (transparency)
                b = baseline[str(seed)]
                gap = (acc - b) if acc is not None else None
                verdicts = {
                    "accuracy": acc,
                    "gap_vs_baseline": gap,
                    "ordering_gap_positive": (gap is not None and gap > 0),
                    "cwsd_value_pass": (acc is not None
                                        and abs(acc - CWSD_CLAIMED) <= CWSD_VALUE_TOL),
                    "magnitude_pass": (gap is not None
                                       and abs(gap - GAP_CLAIMED) <= GAP_TOL),
                }
                sweep[mode][sk][str(seed)] = verdicts
                print(f"{mode} s={sk} seed={seed} acc={acc} gap={gap}",
                      file=sys.stderr)

    # aggregate (across-seed) survival of each gated cwsd claim at this s/mode
    for mode in MODES:
        for s in grid:
            sk = f"{s:.2f}"
            rows = [sweep[mode][sk][str(sd)] for sd in SEEDS]
            sweep[mode][sk]["_aggregate"] = {
                "ordering_positive_every_seed":
                    all(r["ordering_gap_positive"] for r in rows),
                "cwsd_value_pass_every_seed":
                    all(r["cwsd_value_pass"] for r in rows),
                "magnitude_pass_every_seed":
                    all(r["magnitude_pass"] for r in rows),
                "mean_gap": sum(r["gap_vs_baseline"] for r in rows) / len(rows),
                "spread_gap": (max(r["gap_vs_baseline"] for r in rows)
                               - min(r["gap_vs_baseline"] for r in rows)),
            }

    Path(args.out).write_text(json.dumps(sweep, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
