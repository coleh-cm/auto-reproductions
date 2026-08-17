"""Regression test for the smoke-must-not-clobber-evidence defect.

Before the fix, smoke.sh ran `experiments/run_e1_synth.py` with MIDSTEER_SMOKE=1 and
that script wrote to the CANONICAL evidence file `results/e1_synth.json`, overwriting
the real 3-seed x 16-perturbation results with smoke-sized (1-seed x 4-perturbation)
data. A reviewer who ran smoke.sh to verify the path runs had silently destroyed the
evidence. This test exercises the real smoke entrypoint via `sys.executable` and
asserts the canonical file is byte-for-byte unchanged and the smoke output lands in
the separate `results/e1_synth_smoke.json` (which is gitignored — smoke is not
evidence).
"""
import hashlib
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E1 = os.path.join(REPO, 'results', 'e1_synth.json')
E1_SMOKE = os.path.join(REPO, 'results', 'e1_synth_smoke.json')
ENTRY = os.path.join(REPO, 'experiments', 'run_e1_synth.py')


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_smoke_does_not_clobber_canonical_e1_synth():
    """Smoke (MIDSTEER_SMOKE=1) must not modify results/e1_synth.json."""
    assert os.path.exists(E1), "canonical results/e1_synth.json must exist in the repo"
    before = _sha(E1)
    if os.path.exists(E1_SMOKE):
        os.remove(E1_SMOKE)
    env = dict(os.environ, MIDSTEER_SMOKE='1', PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''))
    proc = subprocess.run([sys.executable, ENTRY], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"smoke run failed: {proc.stderr}"
    assert 'FINAL e1_synth_smoke=PASS' in proc.stdout, f"no FINAL smoke line: {proc.stdout!r}"
    # canonical evidence untouched
    assert _sha(E1) == before, "smoke CLOBBERED the canonical results/e1_synth.json"
    # smoke landed in its own (gitignored) file
    assert os.path.exists(E1_SMOKE), "smoke did not write results/e1_synth_smoke.json"
    import json
    with open(E1_SMOKE) as f:
        sm = json.load(f)
    assert list(sm.keys()) == ['0'], f"smoke file should have 1 seed, got {list(sm.keys())}"
    # cleanup the gitignored smoke artefact so the working tree stays clean
    os.remove(E1_SMOKE)


def test_full_run_path_targets_canonical_file():
    """Without MIDSTEER_SMOKE, the OUT path must be the canonical e1_synth.json
    (regression guard on the path-selection logic, without running the full sweep)."""
    import importlib
    import experiments.run_e1_synth as mod
    # module imported without MIDSTEER_SMOKE in this process -> canonical target
    assert os.environ.get('MIDSTEER_SMOKE') != '1'
    importlib.reload(mod)
    assert os.path.basename(mod.OUT) == 'e1_synth.json', (
        f"full run must target e1_synth.json, got {mod.OUT}")
