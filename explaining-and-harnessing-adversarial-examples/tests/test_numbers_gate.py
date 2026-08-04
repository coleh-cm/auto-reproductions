"""tests/test_numbers_gate.py — exercise the numbers gate (the grader) on a
known-correct and a known-wrong input.

The task contract: anything that decides whether an output is correct — a
grader, scorer, equivalence check — must be exercised on one known-correct and
one known-wrong input, and must invoke ``sys.executable`` rather than bare
``python``. ``numbers_gate.py`` is exactly such a grader: it adjudicates every
claim in claims.json against measured.json and writes claims_result.json. This
test runs it in an isolated temp directory (so the real claims_result.json is
never clobbered) via ``sys.executable`` and asserts:

  - positive: a measured.json where HIGH claim c03 holds  -> verdict "pass"
  - negative: a measured.json where c03 fails            -> verdict "fail"
  - blocked : a measured.json missing the referenced arm  -> verdict "blocked"
    (never a silent "pass")
  - raise   : a malformed measured.json                   -> the gate CRASHES
    with a non-zero exit and a traceback on stderr, never a silent "fail"
    verdict line ("a grader that cannot run must raise, never return a
    negative verdict").

The positive/negative fixtures only populate softmax_reg (enough to adjudicate
c03); every other claim is blocked, which is the honest verdict for an absent
metric and does not affect the per-claim assertions we make here.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(REPO, "numbers_gate.py")
CLAIMS = os.path.join(REPO, "claims.json")
PY = sys.executable  # never bare `python`: a host with only python3 would mis-grade


def _seed_measured(adv_err, clean_err):
    """A measured.json with only softmax_reg populated across seeds 0,1,2,
    carrying exactly the metrics c03 reads (adv_err, clean_err)."""
    return {
        "softmax_reg": {
            str(s): {"clean_err": clean_err, "adv_err": adv_err}
            for s in (0, 1, 2)
        }
    }


def _run_gate(tmpdir):
    """Run the COPIED gate in tmpdir so REPO resolves there (isolating the real
    claims_result.json). Returns (returncode, stdout, stderr)."""
    shutil.copy(GATE, os.path.join(tmpdir, "numbers_gate.py"))
    shutil.copy(CLAIMS, os.path.join(tmpdir, "claims.json"))
    return subprocess.run(
        [PY, "numbers_gate.py"], cwd=tmpdir,
        capture_output=True, text=True,
    )


def _verdict(result_path, claim_id):
    with open(result_path) as fh:
        doc = json.load(fh)
    for c in doc["claims"]:
        if c["id"] == claim_id:
            return c["verdict"]
    raise AssertionError(f"{claim_id} not in claims_result.json")


def test_gate_positive_known_correct(tmp_path):
    """c03 (adv_err - clean_err > 0) holds -> verdict pass; claims_result.json is
    stamped produced_by the gate (never hand-authored).

    The gate's overall exit code is non-zero here because the *other* 18 HIGH
    claims are blocked (only softmax_reg is populated) -- that is the honest
    gate behavior, not a bug. The positive signal is c03's per-claim verdict
    plus the produced_by stamp."""
    (tmp_path / "measured.json").write_text(json.dumps(_seed_measured(99.0, 10.0)))
    r = _run_gate(str(tmp_path))
    assert "FINAL gate=FAIL" in r.stdout, r.stdout + r.stderr  # 18 HIGH blocked
    res = tmp_path / "claims_result.json"
    assert res.exists()
    doc = json.loads(res.read_text())
    assert doc["produced_by"] == "numbers_gate.py"
    assert _verdict(res, "c03") == "pass"


def test_gate_negative_known_wrong(tmp_path):
    """c03 fails (adv_err < clean_err at every seed) -> verdict fail, not pass,
    not blocked."""
    (tmp_path / "measured.json").write_text(json.dumps(_seed_measured(5.0, 10.0)))
    r = _run_gate(str(tmp_path))
    # gate exits 1 because HIGH claim c03 failed (gate_pass False)
    assert r.returncode != 0, "a failed HIGH claim must make the gate exit non-zero"
    res = tmp_path / "claims_result.json"
    assert _verdict(res, "c03") == "fail"


def test_gate_blocked_metric_not_silent_pass(tmp_path):
    """A metric the environment could not produce (arm absent) is verdicted
    blocked, never silently passed."""
    (tmp_path / "measured.json").write_text(json.dumps({}))  # no arms at all
    r = _run_gate(str(tmp_path))
    assert r.returncode != 0  # c03 is HIGH and blocked -> gate_pass False
    res = tmp_path / "claims_result.json"
    assert _verdict(res, "c03") == "blocked"


def test_gate_raises_on_unusable_input(tmp_path):
    """A grader that cannot run must raise, never return a negative verdict.
    A malformed measured.json (not valid JSON) crashes the gate with a
    traceback and a non-zero exit, and does NOT print a clean FINAL gate= line
    (which would be the silent-negative-verdict failure mode)."""
    (tmp_path / "measured.json").write_text("{ this is not json ,,")
    r = _run_gate(str(tmp_path))
    assert r.returncode != 0, "malformed measured.json must crash the gate, not pass"
    assert "Traceback" in r.stderr, "the gate must raise a traceback, not return quietly"
    assert "FINAL gate=" not in r.stdout, (
        "a crashed grader must not emit a clean verdict line")
    assert not (tmp_path / "claims_result.json").exists(), (
        "a crashed grader must not write a claims_result.json")
