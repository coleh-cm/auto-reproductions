"""Tests for the regenerated Figure-4 curve panel (``experiments/f4_plot.py`` /
``experiments/f4_eps_curve._plot_figure4``).

The reproduction contract requires: for any curve claim, regenerate the figure
it came from, commit it beside the paper's, and assert the axis ranges and units
match the paper's figure. Figure 4 is the only figure a curve claim (fc1..fc3)
comes from here. The PNG is for a reader to compare and is NOT evidence (the
gate's verdicts on the curve DATA are); these tests guard the axis-assertion
contract and the non-empty-output contract, not the picture's appearance.

Positive: a record with the paper's exact axis semantics (x [-15,15], y label
"argument to softmax", 10 logits per eps) regenerates a non-empty PNG and the
axis assertion passes.
Negative: a record with a tampered x-range or y-label makes the plotter RAISE
(an axis-range mismatch must fail loudly, never silently produce a
right-shaped curve on the wrong scale).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT / "experiments"))

from f4_eps_curve import _plot_figure4  # noqa: E402


def _good_record():
    """A minimal record carrying the paper's Figure-4 axis semantics."""
    n = 5  # coarse grid is enough to exercise the plotter
    eps = [-15.0 + 7.5 * i for i in range(n)]  # [-15, -7.5, 0, 7.5, 15]
    logits = [[float(c) for c in range(10)] for _ in range(n)]
    return {
        "eps_values": eps,
        "logits": logits,
        "correct_class": 4,
        "hyperparams": {"units": 240},
        "figure": {
            "x_axis_range_read": [-15.0, 15.0],
            "y_axis_label": "argument to softmax",
        },
        "seed": 0,
    }


def test_plot_figure4_positive_axis_match_writes_png(tmp_path):
    """Known-correct axis semantics -> non-empty PNG written, no raise."""
    rec = _good_record()
    out = _plot_figure4(rec, tmp_path)
    assert out.exists()
    assert out.stat().st_size > 0
    # PNG magic bytes
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_figure4_negative_wrong_x_range_raises(tmp_path):
    """A tampered x-range must fail loudly (axis-range assertion)."""
    rec = _good_record()
    rec["figure"]["x_axis_range_read"] = [-10.0, 10.0]  # NOT the paper's range
    with __import__("pytest").raises(AssertionError):
        _plot_figure4(rec, tmp_path)


def test_plot_figure4_negative_wrong_y_label_raises(tmp_path):
    """A tampered y-axis label (different units) must fail loudly."""
    rec = _good_record()
    rec["figure"]["y_axis_label"] = "probability"  # NOT the paper's units
    with __import__("pytest").raises(AssertionError):
        _plot_figure4(rec, tmp_path)


def test_plot_figure4_negative_eps_not_spanning_range_raises(tmp_path):
    """eps_values must actually span [-15, 15] (not just claim to)."""
    rec = _good_record()
    rec["eps_values"] = [-7.5, 0.0, 7.5]  # does not reach the paper's bounds
    with __import__("pytest").raises(AssertionError):
        _plot_figure4(rec, tmp_path)


def test_plot_figure4_custom_fname(tmp_path):
    """A custom filename is honored (per-seed panels get distinct names)."""
    rec = _good_record()
    out = _plot_figure4(rec, tmp_path, fname="custom.png")
    assert out.name == "custom.png"
    assert out.exists()


def test_committed_figure_png_exists_and_nonempty():
    """The committed seed-0 regenerated panel exists and is a real PNG.

    This is the figure a reader compares against paper/source/eps_curve.pdf.
    It is NOT evidence (REPRODUCTION.md says so); this test only guards that
    the committed artifact is present and well-formed.
    """
    p = REPRO_ROOT / "results" / "figures" / "f4_eps_curve_repro.png"
    assert p.exists(), "results/figures/f4_eps_curve_repro.png must be committed"
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_committed_curve_data_matches_axis_contract():
    """The committed curve data carries the paper's Figure-4 axis semantics."""
    rec = json.loads(
        (REPRO_ROOT / "results" / "f4_eps_curve.json").read_text())
    assert rec["figure"]["x_axis_range_read"] == [-15.0, 15.0]
    assert rec["figure"]["y_axis_label"] == "argument to softmax"
    assert rec["eps_values"][0] == -15.0
    assert rec["eps_values"][-1] == 15.0
    # one logit vector of length 10 per eps (10 MNIST classes)
    assert all(len(row) == 10 for row in rec["logits"])
    assert len(rec["logits"]) == len(rec["eps_values"])
