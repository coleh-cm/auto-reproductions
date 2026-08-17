"""Tests for the claims evaluator (evaluate_claims.py).

Exercises every verdict path the gate can take — reproduced / refuted / untested /
blocked — for ordering, value, invariant (executable predicate) and curve (per-x
sequence) claims, plus the no-success-on-empty guards. A grader that cannot run must
raise, never return a negative verdict; a missing/BLOCKED metric blocks the claim.
"""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate_claims


def _write(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def _run(tmp_path, claims, measured):
    cp = tmp_path / 'claims.json'; _write(cp, claims)
    mp = tmp_path / 'measured.json'; _write(mp, measured)
    return evaluate_claims.evaluate(str(cp), str(mp))


# ---------- ordering ----------

def test_ordering_satisfied_passes(tmp_path):
    claims = {'arms': {'a': {}, 'b': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T1', 'kind': 'ordering', 'quantity': 'measured.a.x - measured.b.x',
         'direction': '>0', 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'x': 1.0}, '1': {'x': 1.0}, '2': {'x': 1.0}},
                'b': {'0': {'x': 0.0}, '1': {'x': 0.0}, '2': {'x': 0.0}}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'reproduced'


def test_ordering_violated_fails(tmp_path):
    claims = {'arms': {'a': {}, 'b': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T1', 'kind': 'ordering', 'quantity': 'measured.a.x - measured.b.x',
         'direction': '>0', 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'x': 0.0}, '1': {'x': 1.0}, '2': {'x': 1.0}},
                'b': {'0': {'x': 1.0}, '1': {'x': 0.0}, '2': {'x': 0.0}}}
    r = _run(tmp_path, claims, measured)
    # mean = 0.0, spread = 2.0 -> within noise -> untested (not refuted). Use a clear refuted case:
    assert r['verdicts'][0]['verdict'] == 'untested'


def test_ordering_refuted_when_direction_fails_and_separated(tmp_path):
    claims = {'arms': {'a': {}, 'b': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T1', 'kind': 'ordering', 'quantity': 'measured.a.x - measured.b.x',
         'direction': '>0', 'quote': '', 'citation': ''}]}
    # mean = -3, spread = 0 -> separated, direction fails at every seed -> refuted
    measured = {'a': {'0': {'x': 0.0}, '1': {'x': 0.0}, '2': {'x': 0.0}},
                'b': {'0': {'x': 3.0}, '1': {'x': 3.0}, '2': {'x': 3.0}}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'refuted'


def test_ordering_untested_when_within_noise(tmp_path):
    claims = {'arms': {'a': {}, 'b': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T1', 'kind': 'ordering', 'quantity': 'measured.a.x - measured.b.x',
         'direction': '>0', 'quote': '', 'citation': ''}]}
    # gaps +0.01 / -0.01 / +0.01: mean 0.0033, spread 0.02 -> |mean| <= spread -> untested
    measured = {'a': {'0': {'x': 0.01}, '1': {'x': -0.01}, '2': {'x': 0.01}},
                'b': {'0': {'x': 0.0}, '1': {'x': 0.0}, '2': {'x': 0.0}}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'untested'


def test_blocked_metric_is_blocked(tmp_path):
    claims = {'arms': {'a': {}, 'b': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T1', 'kind': 'ordering', 'quantity': 'measured.a.x - measured.b.x',
         'direction': '>0', 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'x': 'BLOCKED'}, '1': {'x': 1.0}, '2': {'x': 1.0}},
                'b': {'0': {'x': 0.0}, '1': {'x': 0.0}, '2': {'x': 0.0}}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'blocked'


# ---------- value ----------

def test_value_reproduced(tmp_path):
    claims = {'arms': {'a': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T', 'kind': 'value', 'quantity': 'measured.a.x', 'claimed': 1.0,
         'tolerance': 0.1, 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'x': 1.0}, '1': {'x': 1.0}, '2': {'x': 1.0}}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'reproduced'


def test_value_refuted(tmp_path):
    claims = {'arms': {'a': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T', 'kind': 'value', 'quantity': 'measured.a.x', 'claimed': 1.0,
         'tolerance': 0.1, 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'x': 2.0}, '1': {'x': 2.0}, '2': {'x': 2.0}}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'refuted'


def test_metric_not_measured_blocked(tmp_path):
    claims = {'arms': {'a': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T', 'kind': 'value', 'quantity': 'measured.a.x', 'claimed': 1.0,
         'tolerance': 0.1, 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'y': 1.0}, '1': {'y': 1.0}, '2': {'y': 1.0}}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'blocked'


# ---------- invariant (executable predicate) ----------

def test_invariant_passes_when_e1_passes(tmp_path):
    """Predicate true at every seed -> reproduced (positive test for the invariant path)."""
    claims = {'arms': {'e1_synth': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C1', 'kind': 'invariant', 'quote': '', 'citation': '',
         'predicate': 'measured.e1_synth.c1_n_feasible > 0 and measured.e1_synth.c1_constraint < 1e-6'}]}
    measured = {'e1_synth': {s: {'c1_n_feasible': 16.0, 'c1_constraint': 1e-13} for s in ['0', '1', '2']}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'reproduced'


def test_invariant_fails_when_a_seed_fails(tmp_path):
    """Predicate false at one seed -> refuted (negative test: a constant-true stub is caught)."""
    claims = {'arms': {'e1_synth': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C1', 'kind': 'invariant', 'quote': '', 'citation': '',
         'predicate': 'measured.e1_synth.c1_constraint < 1e-6'}]}
    measured = {'e1_synth': {'0': {'c1_constraint': 1e-13}, '1': {'c1_constraint': 1.0}, '2': {'c1_constraint': 1e-13}}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'refuted'


def test_invariant_blocked_metric_blocked(tmp_path):
    claims = {'arms': {'e1_synth': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C1', 'kind': 'invariant', 'quote': '', 'citation': '',
         'predicate': 'measured.e1_synth.c1_constraint < 1e-6'}]}
    measured = {'e1_synth': {'0': {'c1_constraint': 'BLOCKED'}, '1': {'c1_constraint': 1e-13}, '2': {'c1_constraint': 1e-13}}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'blocked'


def test_invariant_predicate_rejects_disallowed_names(tmp_path):
    """A predicate referencing a name that is not a measured ref and not a safe builtin
    must NOT evaluate to reproduced (NameError -> blocked), proving the evaluator is not
    a rubber stamp."""
    claims = {'arms': {'e1_synth': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C1', 'kind': 'invariant', 'quote': '', 'citation': '',
         'predicate': 'measured.e1_synth.c1_constraint < 1e-6 or x > 0'}]}
    measured = {'e1_synth': {s: {'c1_constraint': 1.0} for s in ['0', '1', '2']}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'blocked'


# ---------- curve (per-x sequence) ----------

def test_curve_above_reproduced(tmp_path):
    claims = {'arms': {'vanilla': {}, 'midsteer': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C20', 'kind': 'curve',
         'quantity': 'measured.midsteer.c20_min_baseline_src',
         'against': 'measured.midsteer.src_cs_on_src',
         'x': [3, 4, 5], 'comparison': 'above', 'quote': '', 'citation': ''}]}
    measured = {
        'midsteer': {s: {'c20_min_baseline_src': [8.0, 8.0, 8.0], 'src_cs_on_src': [5.0, 5.0, 5.0]}
                     for s in ['0', '1', '2']}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'reproduced'


def test_curve_above_refuted(tmp_path):
    claims = {'arms': {'vanilla': {}, 'midsteer': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C20', 'kind': 'curve',
         'quantity': 'measured.midsteer.c20_min_baseline_src',
         'against': 'measured.midsteer.src_cs_on_src',
         'x': [3, 4, 5], 'comparison': 'above', 'quote': '', 'citation': ''}]}
    measured = {
        'midsteer': {s: {'c20_min_baseline_src': [3.0, 3.0, 3.0], 'src_cs_on_src': [5.0, 5.0, 5.0]}
                     for s in ['0', '1', '2']}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'refuted'


def test_curve_matches_reproduced(tmp_path):
    claims = {'arms': {'midsteer': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C22', 'kind': 'curve', 'quantity': 'measured.midsteer.bertp_mmlu',
         'x': [5000, 10000, 20000], 'comparison': 'matches',
         'claimed': [0.9578, 0.9605, 0.961], 'tolerance': 0.03, 'quote': '', 'citation': ''}]}
    measured = {'midsteer': {s: {'bertp_mmlu': [0.9578, 0.9605, 0.961]} for s in ['0', '1', '2']}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'reproduced'


def test_curve_matches_refuted(tmp_path):
    claims = {'arms': {'midsteer': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C22', 'kind': 'curve', 'quantity': 'measured.midsteer.bertp_mmlu',
         'x': [5000, 10000, 20000], 'comparison': 'matches',
         'claimed': [0.9578, 0.9605, 0.961], 'tolerance': 0.03, 'quote': '', 'citation': ''}]}
    measured = {'midsteer': {s: {'bertp_mmlu': [0.5, 0.5, 0.5]} for s in ['0', '1', '2']}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'refuted'


def test_curve_blocked_sequence_is_blocked(tmp_path):
    """A BLOCKED per-x sequence (the measured.json shape this repo produces for curve
    claims in a blocked sandbox) must verdict 'blocked'."""
    claims = {'arms': {'vanilla': {}, 'midsteer': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C20', 'kind': 'curve',
         'quantity': 'measured.midsteer.c20_min_baseline_src',
         'against': 'measured.midsteer.src_cs_on_src',
         'x': [3, 4, 5], 'comparison': 'above', 'quote': '', 'citation': ''}]}
    measured = {
        'midsteer': {s: {'c20_min_baseline_src': ['BLOCKED', 'BLOCKED', 'BLOCKED'],
                         'src_cs_on_src': ['BLOCKED', 'BLOCKED', 'BLOCKED']} for s in ['0', '1', '2']}}
    r = _run(tmp_path, claims, measured)
    assert r['verdicts'][0]['verdict'] == 'blocked'
    assert r['verdicts'][0]['spread'] == 'BLOCKED'


# ---------- no success on empty ----------

def test_empty_claims_raises(tmp_path):
    claims = {'arms': {'a': {}}, 'seeds': [0], 'claims': []}
    measured = {'a': {'0': {}}}
    with pytest.raises(RuntimeError):
        _run(tmp_path, claims, measured)


def test_empty_measured_raises(tmp_path):
    claims = {'arms': {'a': {}}, 'seeds': [0], 'claims': [
        {'id': 'T', 'kind': 'ordering', 'quantity': 'measured.a.x', 'direction': '>0',
         'quote': '', 'citation': ''}]}
    cp = tmp_path / 'claims.json'; _write(cp, claims)
    mp = tmp_path / 'measured.json'; _write(mp, {})
    with pytest.raises(RuntimeError):
        evaluate_claims.evaluate(str(cp), str(mp))


def test_provenance_marker_is_workflow_subagent(tmp_path):
    """claims_result.json is the WORKFLOW's artifact (generated_by='workflow_subagent'),
    never the gate's 'produced_by' form (the defect where a step hand-wrote the gate's
    table)."""
    claims = {'arms': {'a': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T', 'kind': 'value', 'quantity': 'measured.a.x', 'claimed': 1.0,
         'tolerance': 0.1, 'quote': '', 'citation': ''}]}
    measured = {'a': {s: {'x': 1.0} for s in ['0', '1', '2']}}
    r = _run(tmp_path, claims, measured)
    assert r['generated_by'] == 'workflow_subagent'
    assert 'produced_by' not in r
