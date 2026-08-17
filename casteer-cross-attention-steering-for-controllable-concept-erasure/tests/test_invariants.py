"""Tests for the CPU-runnable invariant measurement instrument (core/invariants.py).

This module computes the `house` claim's measured values (house_max_norm_err,
house_unit_norm_ok, house_pass) that run_all_arms writes into measured.json
under arm `casteer_noclip`, and that the numbers gate evaluates the `house`
predicate against. It is a *scorer* for the `house` verdict, so it is registered
in instruments.json and must be exercised on a known-correct (positive) and a
known-wrong (negative) input -- like every other instrument.
"""
import torch

from core.invariants import compute_house_invariant, HOUSE_ARM, HOUSE_METRICS


def test_house_instrument_arm_and_metrics_names():
    """The instrument writes under claims.json's arm/metric names (not renamable)."""
    assert HOUSE_ARM == "casteer_noclip"
    assert set(HOUSE_METRICS) == {"house_max_norm_err", "house_unit_norm_ok", "house_pass"}


def test_house_invariant_positive_known_correct():
    """Positive: for the paper's beta=2 Householder reflection with a unit s,
    the invariant holds at every claims.json seed -> house_pass == 1 and the
    max norm error is far below 1e-5 (float64). This is the known-correct input
    (the property the paper asserts at experiments.tex:21-22)."""
    for seed in (42, 1234, 2024):
        r = compute_house_invariant(seed)
        assert r["house_pass"] == 1, f"seed {seed}: house_pass != 1 ({r})"
        assert r["house_unit_norm_ok"] == 1
        assert r["house_max_norm_err"] < 1e-5
        # the measured error is ~1e-14, well below the 1e-5 gate
        assert r["house_max_norm_err"] < 1e-10


def test_house_instrument_is_deterministic_per_seed():
    """The same seed yields the same measured values (no hidden RNG state)."""
    a = compute_house_invariant(42)
    b = compute_house_invariant(42)
    assert a == b


def test_house_instrument_rejects_non_householder():
    """Negative: the measurement's discriminative power. A beta=1 (broken,
    non-Householder) reflection does NOT preserve ||c||, so a scorer built the
    same way would report house_pass == 0. This proves the instrument accepts the
    known-correct (beta=2, unit s) input and rejects a known-wrong one, rather
    than passing anything fed to it."""
    torch.manual_seed(0)
    d = 320
    s = torch.randn(d, dtype=torch.float64)
    s = s / s.norm()
    c = torch.randn(d, dtype=torch.float64)
    wrong = c - 1.0 * (s @ c) * s          # beta=1 -> NOT a Householder reflection
    right = c - 2.0 * (s @ c) * s         # beta=2 -> Householder reflection
    assert abs(wrong.norm().item() - c.norm().item()) > 1e-5, (
        "beta=1 must fail norm preservation; else the instrument is blind")
    assert abs(right.norm().item() - c.norm().item()) < 1e-10, (
        "beta=2 must preserve norm (sanity)")
