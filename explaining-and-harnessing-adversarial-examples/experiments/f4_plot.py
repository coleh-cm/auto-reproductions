#!/usr/bin/env python
"""experiments/f4_plot.py — regenerate the Figure-4 (eps-sweep) curve panel
from the COMMITTED curve data, without retraining.

The curve claims fc1..fc3 (claims.json) are settled against the per-eps
sequences in ``results/f4_eps_curve.json`` (and the per-seed files under
``results/_per_seed``). This script reads that committed curve data and
regenerates the Figure 4 LEFT panel (softmax logits vs eps along the FGSM
direction for a class-4 example) as a PNG beside the paper's
``paper/source/eps_curve.pdf`` for a reader to compare.

Per the reproduction contract: this regenerated figure is for a reader to
COMPARE against the paper's Figure 4 and is NOT evidence — the gate's
verdicts on fc1..fc3 (from the curve DATA) are the evidence.

Axis ranges / units are asserted against the paper's figure (read with the
vision tool, transcript ``paper/figure_transcripts.md``):
  * x-axis: eps in [-15, 15], ticks -15,-10,-5,0,5,10,15, label "epsilon"
  * y-axis: "argument to softmax" (raw pre-softmax logits — same units as the
    paper; the paper's numeric y-range [-2000, 1000] reflects a fully-trained
    maxout, this sub-scale run is smaller-magnitude — the curve claims are
    SHAPE claims and scale-invariant, see REPRODUCTION.md).

Runnable::

    python experiments/f4_plot.py                      # seed 0 (results/f4_eps_curve.json)
    python experiments/f4_plot.py --per-seed 1          # results/_per_seed/f4_eps_curve__seed1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT / "experiments"))

from f4_eps_curve import _plot_figure4  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regenerate the Figure-4 curve panel from committed curve data.")
    p.add_argument("--data", type=str,
                   default=str(REPRO_ROOT / "results" / "f4_eps_curve.json"),
                   help="path to the f4_eps_curve.json curve-data file")
    p.add_argument("--per-seed", type=int, default=None,
                   help="instead read results/_per_seed/f4_eps_curve__seed<SEED>.json")
    p.add_argument("--fig-dir", dest="fig_dir", type=str,
                   default=str(REPRO_ROOT / "results" / "figures"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.per_seed is not None:
        data_path = (REPRO_ROOT / "results" / "_per_seed"
                     / f"f4_eps_curve__seed{args.per_seed}.json")
    else:
        data_path = Path(args.data)
    if not data_path.exists():
        print(f"[f4_plot] missing curve data: {data_path}", file=sys.stderr)
        return 1
    record = json.loads(data_path.read_text())
    if "logits" not in record or "eps_values" not in record:
        print(f"[f4_plot] {data_path} has no curve sequences (logits/eps_values); "
              "run experiments/f4_eps_curve.py first.", file=sys.stderr)
        return 1
    fname = None
    if args.per_seed is not None:
        fname = f"f4_eps_curve_repro_seed{args.per_seed}.png"
    out = _plot_figure4(record, Path(args.fig_dir), fname=fname)
    print(f"[f4_plot] regenerated Figure 4 panel from {data_path} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
