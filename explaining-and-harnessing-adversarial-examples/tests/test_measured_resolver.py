"""Regression test for the make_measured.py metric resolver.

The numbers-gate evaluator formats every metric value in ``measured.json`` as a
scalar (``f"{v:.4f}"``-style). A list value crashes it with::

    TypeError: unsupported format string passed to list.__format__

The one pointer that resolves to a list is
``arms.adversarial.per_seed[*].test_error`` (metric
``per_seed_test_errors`` on arm ``m5_maxout1600_advtrain``). Each per-seed
result file's ``per_seed`` array contains exactly ONE element (the seed that run
trained under), so ``[*]`` yields a one-element list ``[x]``. The resolver must
collapse that to the scalar ``x`` (THIS seed's test error), so the gate's
cross-seed gather produces a flat list of scalars ``[x0, x1, x2]`` that
``mean(...)`` / ``max(...)`` / ``min(...)`` in claims c12/c13 operate on as the
paper intends. A multi-element list is a genuine ambiguity and must be BLOCKED
rather than silently flattened.

This test also asserts the shipped ``measured.json`` contains NO list values
(every metric is a scalar or the ``"BLOCKED"`` sentinel), which is the property
the gate depends on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT))  # make_measured is at repo root

from make_measured import _metrics_for_arm, BLOCKED  # noqa: E402


def test_single_element_list_pointer_collapses_to_scalar():
    """A ``[*]`` pointer over a one-element array returns the scalar, not [x]."""
    blob = {"arms": {"adversarial": {"per_seed": [
        {"test_error": 0.0144, "seed": 0,
         "best_epoch_index": 11, "epochs_chosen": 12}]}}}
    metrics = {"per_seed_test_errors":
               "results/m5_large_advtrain.json:arms.adversarial.per_seed[*].test_error"}
    out = _metrics_for_arm(blob, metrics)
    assert out["per_seed_test_errors"] == 0.0144, (
        f"one-element [*] should collapse to scalar, got {out!r}")
    assert not isinstance(out["per_seed_test_errors"], list)


def test_multi_element_list_pointer_is_blocked():
    """A ``[*]`` pointer over a multi-element array is ambiguous -> BLOCKED.

    A per-seed result file must contain exactly one seed; a multi-element
    ``per_seed`` would mean the resolver cannot pick one number, so we BLOCK it
    loudly rather than silently average/min/max and lie about which seed ran."""
    blob = {"arms": {"adversarial": {"per_seed": [
        {"test_error": 0.0144, "seed": 0},
        {"test_error": 0.0154, "seed": 1}]}}}
    metrics = {"per_seed_test_errors":
               "results/m5_large_advtrain.json:arms.adversarial.per_seed[*].test_error"}
    out = _metrics_for_arm(blob, metrics)
    assert out["per_seed_test_errors"] == BLOCKED, (
        f"multi-element [*] should BLOCK, got {out!r}")


def test_scalar_pointer_unchanged():
    """A plain scalar pointer is returned verbatim (no spurious unwrapping)."""
    blob = {"arms": {"baseline": {"mean_test_error": 0.0195}}}
    metrics = {"mean_test_error":
               "results/m5_large_advtrain.json:arms.baseline.mean_test_error"}
    out = _metrics_for_arm(blob, metrics)
    assert out["mean_test_error"] == 0.0195


def test_measured_json_has_no_list_values():
    """The shipped measured.json must contain only scalars or BLOCKED.

    The numbers gate formats each value as a scalar; a list value is the crash
    this test guards against. Every arm x seed x metric must be a number or the
    string ``"BLOCKED"`` (never a list/dict).
    """
    measured = json.loads((REPRO_ROOT / "measured.json").read_text())
    bad = []
    for arm, per_seed in measured.items():
        if arm == "_meta":
            continue
        for seed, md in per_seed.items():
            for metric, v in md.items():
                if v == BLOCKED:
                    continue
                if not isinstance(v, (int, float)):
                    bad.append((arm, seed, metric, type(v).__name__, v))
    assert not bad, f"measured.json has non-scalar/non-BLOCKED values: {bad}"
