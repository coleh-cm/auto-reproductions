"""Instrument tests for the numbers-gate CURVE evaluator.

``numbers_gate.evaluate_curve_claim`` decides whether a figure claim (a curve
rising, crossing, staying above/below) holds. Per the reproduction contract it
is exercised on known-correct AND known-wrong synthetic per-seed result files,
and it must BLOCK (never fabricate a verdict) when the data is absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT))

import numbers_gate  # noqa: E402

EPS = [float(e) for e in range(0, 16)]


def _claims_doc(q_seq, r_seq=None) -> dict:
    arm = {
        "script": "experiments/f4_eps_curve.py",
        "command_per_seed": "python experiments/f4_eps_curve.py --seed {seed}",
        "results": "results/f4_eps_curve.json",
        "metrics": {},
        "curve_metrics": {
            "eps_values": "results/f4_eps_curve.json:eps_values",
            "q": "results/f4_eps_curve.json:q",
            "r": "results/f4_eps_curve.json:r",
        },
    }
    blob = {"eps_values": EPS, "q": q_seq, "r": r_seq or q_seq}
    return arm, blob


def _write_seed_files(tmp_path: Path, monkeypatch, blob, seeds=(0, 1, 2)) -> dict:
    """Point the gate's REPRO_ROOT at a tmp dir holding synthetic per-seed files."""
    d = tmp_path / "results" / "_per_seed"
    d.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        (d / f"f4_eps_curve__seed{s}.json").write_text(json.dumps(blob))
    monkeypatch.setattr(numbers_gate, "REPRO_ROOT", tmp_path)
    claims_doc = {"seeds": [0, 1, 2], "arms": {"f4_eps_curve": _claims_doc(blob["q"])[0]}}
    canonical = numbers_gate.build_canonical_tokens(claims_doc)
    return claims_doc, canonical


def _base_claim(comparison, **kw):
    c = {
        "id": "t", "kind": "curve", "compute_invariance": "high",
        "quote": "x", "citation": "paper/source/iclr2015.tex:1",
        "quantity": "measured.f4_eps_curve.q", "x": "measured.f4_eps_curve.eps_values",
        "comparison": comparison,
    }
    c.update(kw)
    return c


def test_crosses_positive(tmp_path, monkeypatch):
    # correct on top at eps=0, wrong on top at eps=15 -> strict sign change
    q = [10.0 - e for e in EPS]
    r = [e * 1.0 for e in EPS]
    blob = {"eps_values": EPS, "q": q, "r": r}
    cd, canonical = _write_seed_files(tmp_path, monkeypatch, blob)
    claim = _base_claim("crosses", reference="measured.f4_eps_curve.r")
    res = numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)
    assert res["verdict"] == "pass" and res["seeds_evaluated"] == [0, 1, 2]


def test_crosses_negative_no_crossing(tmp_path, monkeypatch):
    q = [10.0 + e for e in EPS]  # q always above r
    r = [e * 1.0 for e in EPS]
    cd, canonical = _write_seed_files(tmp_path, monkeypatch,
                                      {"eps_values": EPS, "q": q, "r": r})
    claim = _base_claim("crosses", reference="measured.f4_eps_curve.r")
    res = numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)
    assert res["verdict"] == "fail"


def test_below_positive_and_negative(tmp_path, monkeypatch):
    wrong_all = {"eps_values": EPS, "q": [0.0] * len(EPS), "r": [0.0] * len(EPS)}
    cd, canonical = _write_seed_files(tmp_path, monkeypatch, wrong_all)
    claim = _base_claim("below", claimed=0.5, tolerance=0.0,
                        x_range=[4.0, 15.0])
    assert numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)["verdict"] == "pass"
    # one correct sample inside the range -> fail
    one_correct = {"eps_values": EPS, "q": [0.0] * 10 + [1.0] + [0.0] * 5,
                   "r": [0.0] * 16}
    cd, canonical = _write_seed_files(tmp_path, monkeypatch, one_correct)
    assert numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)["verdict"] == "fail"


def test_below_x_range_is_respected(tmp_path, monkeypatch):
    # the '1' sits OUTSIDE the claimed range [4,15]; inside the range all below
    seq = [1.0, 1.0, 1.0, 1.0] + [0.0] * 12
    cd, canonical = _write_seed_files(tmp_path, monkeypatch,
                                      {"eps_values": EPS, "q": seq, "r": seq})
    claim = _base_claim("below", claimed=0.5, tolerance=0.0, x_range=[4.0, 15.0])
    res = numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)
    assert res["verdict"] == "pass" and res["per_seed"][0]["n_samples"] == 12


def test_increasing_positive_and_negative(tmp_path, monkeypatch):
    monotone = {"eps_values": EPS, "q": [e * 100.0 for e in EPS], "r": EPS}
    cd, canonical = _write_seed_files(tmp_path, monkeypatch, monotone)
    claim = _base_claim("increasing", tolerance=0.5, x_range=[0.0, 15.0])
    assert numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)["verdict"] == "pass"
    # a big mid-range drop beyond tolerance -> fail
    dip = [e * 100.0 for e in EPS]
    dip[8] = dip[7] - 10.0
    cd, canonical = _write_seed_files(tmp_path, monkeypatch,
                                      {"eps_values": EPS, "q": dip, "r": dip})
    assert numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)["verdict"] == "fail"


def test_every_seed_must_pass(tmp_path, monkeypatch):
    good = {"eps_values": EPS, "q": [0.0] * 16, "r": [0.0] * 16}
    cd, canonical = _write_seed_files(tmp_path, monkeypatch, good)
    # corrupt seed 2 only
    bad = dict(good); bad["q"] = [0.0] * 10 + [1.0] + [0.0] * 5
    (tmp_path / "results" / "_per_seed" / "f4_eps_curve__seed2.json").write_text(json.dumps(bad))
    claim = _base_claim("below", claimed=0.5, tolerance=0.0, x_range=[4.0, 15.0])
    assert numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)["verdict"] == "fail"


def test_missing_per_seed_file_blocks(tmp_path, monkeypatch):
    cd, canonical = _write_seed_files(tmp_path, monkeypatch,
                                      {"eps_values": EPS, "q": [0.0] * 16, "r": [0.0] * 16},
                                      seeds=(0,))
    claim = _base_claim("below", claimed=0.5)
    res = numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)
    assert res["verdict"] == "blocked"


def test_matches_with_tolerance(tmp_path, monkeypatch):
    q = [float(e) + 0.2 for e in EPS]
    cd, canonical = _write_seed_files(tmp_path, monkeypatch,
                                      {"eps_values": EPS, "q": q, "r": q})
    claim = _base_claim("matches", claimed=[float(e) for e in EPS], tolerance=0.5)
    assert numbers_gate.evaluate_curve_claim(claim, cd, [0, 1, 2], canonical)["verdict"] == "pass"
    claim2 = _base_claim("matches", claimed=[float(e) for e in EPS], tolerance=0.1)
    assert numbers_gate.evaluate_curve_claim(claim2, cd, [0, 1, 2], canonical)["verdict"] == "fail"
