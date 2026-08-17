"""Tests for the claims evaluator (evaluate_claims.py)."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate_claims


def _write(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def _tmp_claims_measured(tmp_path, claims, measured):
    cp = tmp_path / 'claims.json'; _write(cp, claims)
    mp = tmp_path / 'measured.json'; _write(mp, measured)
    ep = tmp_path / 'e1_synth.json'; _write(ep, {})
    return str(cp), str(mp), str(ep)


def test_ordering_satisfied_passes(tmp_path):
    claims = {'arms': {'a': {}, 'b': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T1', 'kind': 'ordering', 'quantity': 'measured.a.x - measured.b.x',
         'direction': '>0', 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'x': 1.0}, '1': {'x': 1.0}, '2': {'x': 1.0}},
                'b': {'0': {'x': 0.0}, '1': {'x': 0.0}, '2': {'x': 0.0}}}
    cp, mp, ep = _tmp_claims_measured(tmp_path, claims, measured)
    r = evaluate_claims.evaluate(cp, mp, ep)
    assert r['verdicts'][0]['verdict'] == 'pass'


def test_ordering_violated_fails(tmp_path):
    claims = {'arms': {'a': {}, 'b': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T1', 'kind': 'ordering', 'quantity': 'measured.a.x - measured.b.x',
         'direction': '>0', 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'x': 0.0}, '1': {'x': 1.0}, '2': {'x': 1.0}},
                'b': {'0': {'x': 1.0}, '1': {'x': 0.0}, '2': {'x': 0.0}}}
    cp, mp, ep = _tmp_claims_measured(tmp_path, claims, measured)
    r = evaluate_claims.evaluate(cp, mp, ep)
    assert r['verdicts'][0]['verdict'] == 'fail'  # seed 0 violates


def test_blocked_metric_is_blocked(tmp_path):
    claims = {'arms': {'a': {}, 'b': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T1', 'kind': 'ordering', 'quantity': 'measured.a.x - measured.b.x',
         'direction': '>0', 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'x': 'BLOCKED'}, '1': {'x': 1.0}, '2': {'x': 1.0}},
                'b': {'0': {'x': 0.0}, '1': {'x': 0.0}, '2': {'x': 0.0}}}
    cp, mp, ep = _tmp_claims_measured(tmp_path, claims, measured)
    r = evaluate_claims.evaluate(cp, mp, ep)
    assert r['verdicts'][0]['verdict'] == 'blocked'


def test_empty_claims_raises(tmp_path):
    claims = {'arms': {'a': {}}, 'seeds': [0], 'claims': []}
    measured = {'a': {'0': {}}}
    cp, mp, ep = _tmp_claims_measured(tmp_path, claims, measured)
    with pytest.raises(RuntimeError):
        evaluate_claims.evaluate(cp, mp, ep)


def test_empty_measured_raises(tmp_path):
    claims = {'arms': {'a': {}}, 'seeds': [0], 'claims': [
        {'id': 'T', 'kind': 'ordering', 'quantity': 'measured.a.x', 'direction': '>0',
         'quote': '', 'citation': ''}]}
    mp = tmp_path / 'measured.json'; _write(mp, {})
    cp = tmp_path / 'claims.json'; _write(cp, claims)
    ep = tmp_path / 'e1_synth.json'; _write(ep, {})
    with pytest.raises(RuntimeError):
        evaluate_claims.evaluate(str(cp), str(mp), str(ep))


def test_metric_not_measured_blocked(tmp_path):
    # claim references measured.a.x but measured has no 'x' for the seed
    claims = {'arms': {'a': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'T', 'kind': 'value', 'quantity': 'measured.a.x', 'claimed': 1.0,
         'tolerance': 0.1, 'quote': '', 'citation': ''}]}
    measured = {'a': {'0': {'y': 1.0}, '1': {'y': 1.0}, '2': {'y': 1.0}}}
    cp, mp, ep = _tmp_claims_measured(tmp_path, claims, measured)
    r = evaluate_claims.evaluate(cp, mp, ep)
    assert r['verdicts'][0]['verdict'] == 'blocked'


def test_invariant_passes_when_e1_passes(tmp_path):
    claims = {'arms': {'base': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C1', 'kind': 'invariant', 'quote': '', 'citation': ''}]}
    measured = {'base': {'0': {}, '1': {}, '2': {}}}
    e1 = {'0': {'C1': {'pass': True, 'detail': 'ok', 'metrics': {}}},
          '1': {'C1': {'pass': True, 'detail': 'ok', 'metrics': {}}},
          '2': {'C1': {'pass': True, 'detail': 'ok', 'metrics': {}}}}
    cp = tmp_path / 'claims.json'; _write(cp, claims)
    mp = tmp_path / 'measured.json'; _write(mp, measured)
    ep = tmp_path / 'e1_synth.json'; _write(ep, e1)
    r = evaluate_claims.evaluate(str(cp), str(mp), str(ep))
    assert r['verdicts'][0]['verdict'] == 'pass'


def test_invariant_fails_when_a_seed_fails(tmp_path):
    claims = {'arms': {'base': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C1', 'kind': 'invariant', 'quote': '', 'citation': ''}]}
    measured = {'base': {'0': {}, '1': {}, '2': {}}}
    e1 = {'0': {'C1': {'pass': True, 'detail': 'ok', 'metrics': {}}},
          '1': {'C1': {'pass': False, 'detail': 'bad', 'metrics': {}}},
          '2': {'C1': {'pass': True, 'detail': 'ok', 'metrics': {}}}}
    cp = tmp_path / 'claims.json'; _write(cp, claims)
    mp = tmp_path / 'measured.json'; _write(mp, measured)
    ep = tmp_path / 'e1_synth.json'; _write(ep, e1)
    r = evaluate_claims.evaluate(str(cp), str(mp), str(ep))
    assert r['verdicts'][0]['verdict'] == 'fail'


def test_curve_blocked_sequence_is_blocked(tmp_path):
    # curve metric stored as a per-x BLOCKED sequence (the measured.json shape this repo
    # produces for curve claims in a blocked sandbox) must verdict 'blocked'.
    claims = {'arms': {'vanilla': {}, 'midsteer': {}}, 'seeds': [0, 1, 2], 'claims': [
        {'id': 'C20', 'kind': 'curve',
         'quantity': '[min(measured.vanilla.src_cs_on_src, measured.midsteer.src_cs_on_src) for beta in x]',
         'against': '[measured.midsteer.src_cs_on_src for beta in x]',
         'x': [3, 4, 5], 'comparison': 'above', 'quote': '', 'citation': ''}]}
    measured = {
        'vanilla':  {s: {'src_cs_on_src': ['BLOCKED', 'BLOCKED', 'BLOCKED']} for s in ['0', '1', '2']},
        'midsteer': {s: {'src_cs_on_src': ['BLOCKED', 'BLOCKED', 'BLOCKED']} for s in ['0', '1', '2']},
    }
    cp = tmp_path / 'claims.json'; _write(cp, claims)
    mp = tmp_path / 'measured.json'; _write(mp, measured)
    ep = tmp_path / 'e1_synth.json'; _write(ep, {})
    r = evaluate_claims.evaluate(str(cp), str(mp), str(ep))
    assert r['verdicts'][0]['verdict'] == 'blocked'
    assert r['verdicts'][0]['spread'] == 'BLOCKED'
