"""Instrument tests (research-code skill: graders/scorers/loaders exercised on
one known-correct and one known-wrong input).

Every thing that decides whether an output is correct is an instrument and is
exercised here on a positive (known-correct) and a negative (known-wrong)
input. A grader that cannot run must raise, never return a negative verdict;
these tests assert the instruments DO run and discriminate.

Subprocess invocations use ``sys.executable`` (never bare ``python``) so the
suite works on hosts with only ``python3``.

Instruments (see instruments.json):
  - data-loader            : the paper's own load_digits dataset, by fingerprint
  - accuracy-scorer        : the test-set accuracy metric (evaluate)
  - final-line-parser      : the 'FINAL accuracy=<float>' output contract
  - degeneracy-equivalence : lambda=0 == independent cross-entropy, bitwise
"""

import hashlib
import re
import subprocess
import sys

import numpy as np
import pytest

import run_experiment as r

# --------------------------------------------------------------------------- #
# Shared reference data (fingerprints of the paper's dataset at seed 0).
# These are SHA-256 hashes of the raw bytes of the arrays returned by
# load_data(0); computed once and pinned here so a synthetic or substitute
# corpus is caught. (Vocabulary = 10 classes; 1797 total; 1257/540 split.)
# --------------------------------------------------------------------------- #
EXPECTED = {
    "n_total": 1797,
    "n_features": 64,
    "K": 10,
    "Xtr_shape": (1257, 64),
    "Xte_shape": (540, 64),
    "ytr_shape": (1257,),
    "yte_shape": (540,),
    "Xtr_sha256": "4b186f7c4d31f28fdb60b595be73aa14fdc8e940987f3c6f8ba540bdf50b2cdb",
    "Xte_sha256": "8a53b25f6dc18f3306918f5b81d1e4c24fa094c4a30baea173347fe0782eb070",
    "yte_sha256": "12f9e0d92f950d5f4d65712f513ff9419f212566a442b8893c98af854b0d1e01",
}


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 1. Data loader
# --------------------------------------------------------------------------- #
def _data_fingerprint_ok(Xtr, ytr, Xte, yte) -> bool:
    """Return True iff the four arrays match every pinned fingerprint.

    This is the data-loader instrument: it decides whether the data is the
    paper's own load_digits corpus (by size, vocabulary, shape, dtype, value
    range, and content checksum) rather than a synthetic or substitute corpus.
    """
    if Xtr.shape != EXPECTED["Xtr_shape"]:
        return False
    if Xte.shape != EXPECTED["Xte_shape"]:
        return False
    if ytr.shape != EXPECTED["ytr_shape"]:
        return False
    if yte.shape != EXPECTED["yte_shape"]:
        return False
    if Xtr.dtype != np.float32 or Xte.dtype != np.float32:
        return False
    if ytr.dtype != np.int64 or yte.dtype != np.int64:
        return False
    if not (Xtr.min() >= 0.0 and Xtr.max() <= 1.0):
        return False
    if not (Xte.min() >= 0.0 and Xte.max() <= 1.0):
        return False
    if set(np.concatenate([ytr, yte]).tolist()) != set(range(EXPECTED["K"])):
        return False
    if _sha(Xtr) != EXPECTED["Xtr_sha256"]:
        return False
    if _sha(Xte) != EXPECTED["Xte_sha256"]:
        return False
    if _sha(yte) != EXPECTED["yte_sha256"]:
        return False
    return True


def test_data_loader_positive():
    """Positive: load_data(0) matches every fingerprint of the paper's dataset."""
    Xtr, ytr, Xte, yte = r.load_data(0)
    assert _data_fingerprint_ok(Xtr, ytr, Xte, yte)


def test_data_loader_negative():
    """Negative: a synthetic/fabricated corpus is REJECTED by the fingerprint
    check (a closed-book run that fell back to a synthetic corpus would be
    caught here, not silently passed)."""
    rng = np.random.default_rng(123)
    # fabricate arrays of the RIGHT shape but WRONG content (synthetic fallback)
    Xtr = rng.standard_normal((1257, 64)).astype(np.float32)
    Xte = rng.standard_normal((540, 64)).astype(np.float32)
    ytr = rng.integers(0, 10, size=1257).astype(np.int64)
    yte = rng.integers(0, 10, size=540).astype(np.int64)
    assert not _data_fingerprint_ok(Xtr, ytr, Xte, yte)
    # also: a wrong-size corpus (the 65-token-vocabulary failure mode) is rejected
    Xtr2 = rng.standard_normal((650, 64)).astype(np.float32)
    Xte2 = rng.standard_normal((200, 64)).astype(np.float32)
    ytr2 = rng.integers(0, 10, size=650).astype(np.int64)
    yte2 = rng.integers(0, 10, size=200).astype(np.int64)
    assert not _data_fingerprint_ok(Xtr2, ytr2, Xte2, yte2)


# --------------------------------------------------------------------------- #
# 2. Accuracy scorer (evaluate)
# --------------------------------------------------------------------------- #
def _toy_params(D=4, H=4, K=3):
    """Params that make forward act as a known classifier: identity hidden,
    so z[i] = Xte[i] @ W2 and argmax(z[i]) = the column W2 selects."""
    W1 = np.eye(D, H, dtype=np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # W2 columns are one-hot labels 0,1,2 (padded to H rows)
    W2 = np.zeros((H, K), dtype=np.float32)
    for k in range(K):
        W2[k, k] = 1.0
    b2 = np.zeros(K, dtype=np.float32)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def test_accuracy_scorer_positive():
    """Positive: a model whose argmax predictions match yte on all examples
    scores accuracy 1.0 exactly."""
    P = _toy_params()
    Xte = np.eye(3, 4, dtype=np.float32)  # rows = e_0, e_1, e_2
    yte = np.array([0, 1, 2], dtype=np.int64)
    acc = r.evaluate(P, Xte, yte)
    assert acc == 1.0


def test_accuracy_scorer_negative():
    """Negative: a model whose predictions are all wrong scores 0.0 (not a
    constant/silent-pass), and a partially-wrong case scores the exact wrong
    fraction — proving the scorer discriminates and is not a stub."""
    P = _toy_params()
    Xte = np.eye(3, 4, dtype=np.float32)
    # predictions are [0,1,2]; all-wrong labels:
    yte_wrong = np.array([1, 2, 0], dtype=np.int64)
    assert r.evaluate(P, Xte, yte_wrong) == 0.0
    # partially wrong (1/3 correct) — must NOT equal the positive's 1.0
    # predictions are [0,1,2]; only the middle matches label 1
    yte_partial = np.array([1, 1, 1], dtype=np.int64)
    assert r.evaluate(P, Xte, yte_partial) == pytest.approx(1 / 3)
    assert r.evaluate(P, Xte, yte_partial) != 1.0


def test_accuracy_scorer_must_raise_on_empty():
    """A grader that cannot run must raise, never return a negative verdict.
    evaluate on an empty test set must not silently report 0.0 (which would
    read as 'all wrong'); it raises (or returns nan, which is also not a
    silent negative verdict) — assert it does not return a plain 0.0 on empty.
    """
    P = _toy_params()
    Xte = np.zeros((0, 4), dtype=np.float32)
    yte = np.array([], dtype=np.int64)
    with np.errstate(invalid="ignore"):
        acc = r.evaluate(P, Xte, yte)
    # empty evaluation must NOT be reportable as a clean 0.0 success
    assert not (acc == 0.0 and np.isfinite(acc))


# --------------------------------------------------------------------------- #
# 3. Final-line parser (output contract)
# --------------------------------------------------------------------------- #
def parse_final_line(stdout: str):
    """Reference implementation of the run_all_arms.sh parser. Returns
    (ok, value): ok=True with the float iff stdout contains EXACTLY one line
    matching 'FINAL accuracy=<float>'; otherwise (False, None) -> BLOCKED."""
    lines = stdout.splitlines()
    final_lines = [ln for ln in lines if ln.startswith("FINAL accuracy=")]
    if len(final_lines) != 1:
        return (False, None)
    m = re.match(r"^FINAL accuracy=(-?\d+\.?\d*)$", final_lines[0])
    if not m:
        return (False, None)
    return (True, float(m.group(1)))


def _subprocess_run(args):
    cmd = [sys.executable, "run_experiment.py"] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def test_final_line_parser_positive():
    """Positive: a real run produces exactly one parseable FINAL line."""
    out = _subprocess_run(["--lambda", "0.0", "--steps", "5"])
    assert out.returncode == 0, out.stderr
    ok, val = parse_final_line(out.stdout)
    assert ok is True
    assert val is not None and 0.0 <= val <= 1.0


def test_final_line_parser_negative():
    """Negative: a rejected run (lambda out of range) emits NO FINAL line, so
    the parser returns (False, None) -> BLOCKED (not a fabricated value)."""
    out = _subprocess_run(["--lambda", "1.5", "--steps", "1"])
    assert out.returncode != 0
    ok, val = parse_final_line(out.stdout)
    assert ok is False
    assert val is None
    # a multi-FINAL-line stdout is also rejected
    ok2, val2 = parse_final_line("FINAL accuracy=0.9\nFINAL accuracy=0.8\n")
    assert ok2 is False and val2 is None
    # a malformed line is rejected
    ok3, val3 = parse_final_line("FINAL accuracy=NaN")
    assert ok3 is False and val3 is None


# --------------------------------------------------------------------------- #
# 4. Degeneracy equivalence (lambda=0 == independent cross-entropy)
# --------------------------------------------------------------------------- #
def _independent_ce_loss_and_grads(params, X, Y_onehot):
    """An INDEPENDENT plain cross-entropy routine (does not call make_target
    or loss_and_grads), used as the equivalence-check oracle."""
    h = np.maximum(0.0, X @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    zc = z - z.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(zc).sum(axis=-1, keepdims=True))
    log_p = zc - lse
    p = np.exp(log_p)
    B = X.shape[0]
    loss = float(-np.sum(Y_onehot * log_p) / B)
    dz = (p - Y_onehot) / B
    dW2 = h.T @ dz
    db2 = dz.sum(axis=0)
    dh = dz @ params["W2"].T
    dh = dh * (h > 0).astype(np.float32)
    dW1 = X.T @ dh
    db1 = dh.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def _rand_case(seed=0, B=8):
    rng = np.random.default_rng(seed)
    params = r.init_params(rng)
    X = rng.standard_normal((B, r.D)).astype(np.float32)
    Y = np.eye(r.K, dtype=np.float32)[rng.integers(0, r.K, size=B)]
    return params, X, Y


def test_degeneracy_equivalence_positive():
    """Positive: at lambda=0 the loss and ALL grads are bitwise identical to
    the independent CE routine (the no-op reproduces the baseline exactly)."""
    params, X, Y = _rand_case(seed=7)
    loss_ce, grads_ce = _independent_ce_loss_and_grads(params, X, Y)
    loss_cw, grads_cw = r.loss_and_grads(params, X, Y, 0.0, 0.9, 0.15, 2.0)
    assert loss_cw == loss_ce
    for k in ("W1", "b1", "W2", "b2"):
        assert np.array_equal(grads_cw[k], grads_ce[k]), k


def test_degeneracy_equivalence_negative():
    """Negative: at lambda=1 (gate active, w>0) the loss/grads DIFFER from the
    independent CE routine — the equivalence check rejects a non-degenerate
    input, proving it is not a vacuous always-equal check."""
    params, X, Y = _rand_case(seed=7)
    loss_ce, grads_ce = _independent_ce_loss_and_grads(params, X, Y)
    loss_cw, grads_cw = r.loss_and_grads(params, X, Y, 1.0, 0.9, 0.15, 2.0)
    # the gate is active (w not all zero) for this random case
    out = r.forward(params, X)
    w = 1.0 / (1.0 + np.exp(-(out["p"].max(axis=-1) - 0.9) / 0.15))
    assert w.max() > 0.0
    # so the loss must differ (equivalence is broken) — check rejects
    assert loss_cw != loss_ce
    assert any(not np.array_equal(grads_cw[k], grads_ce[k]) for k in ("W1", "b1", "W2", "b2"))
