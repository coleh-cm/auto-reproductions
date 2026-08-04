"""Mutation tests: break run_experiment.py on purpose and assert the suite
catches each defect (research-code skill: a suite nobody has broken on purpose
is not evidence).

Each defect in mutations.json is injected into a fresh exec'd copy of
run_experiment.py (the original file on disk is NEVER modified). For each
defect we assert:
  - the `find` string is present exactly once in the original source (so the
    defect is well-aimed and the find anchor has not drifted);
  - the invariant the `must_fail` test encodes HOLDS for the original module;
  - the invariant is VIOLATED for the mutated module (i.e. the must_fail test
    would fail -> the suite catches the defect).

No bare `python`: nothing here shells out. The mutated module is built by
exec'ing the source string in a fresh namespace, so the on-disk file is
untouched and the test environment is unaffected.
"""

import json
import types
from pathlib import Path

import numpy as np
import pytest

import run_experiment as r

REPO = Path(__file__).resolve().parent.parent
MUTATIONS = REPO / "mutations.json"
SRC = (REPO / "run_experiment.py").read_text()


def _load_mutated(find: str, replace: str) -> types.ModuleType:
    """Return a fresh module built from run_experiment.py with find->replace
    applied. Asserts find is present exactly once in the original source."""
    assert SRC.count(find) == 1, f"find not unique (count={SRC.count(find)}): {find!r}"
    mutated = SRC.replace(find, replace, 1)
    mod = types.ModuleType("run_experiment_mutated")
    mod.__file__ = str(REPO / "run_experiment.py")
    # exec the mutated source; the `if __name__ == "__main__"` guard does not
    # fire because __name__ is "run_experiment_mutated".
    exec(compile(mutated, mod.__file__, "exec"), mod.__dict__)
    return mod


# --------------------------------------------------------------------------- #
# Reference data + independent cross-entropy (mirrors test_degeneracy.py)
# --------------------------------------------------------------------------- #
def _rand_case(seed=7, B=8):
    rng = np.random.default_rng(seed)
    params = r.init_params(rng)
    X = rng.standard_normal((B, r.D)).astype(np.float32)
    Y = np.eye(r.K, dtype=np.float32)[rng.integers(0, r.K, size=B)]
    return params, X, Y


def _ce_loss_and_grads(params, X, Y_onehot):
    """Independent plain cross-entropy (the must_fail oracle). Uses h>0."""
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


# --------------------------------------------------------------------------- #
# Invariant checkers. Each returns True if the invariant HOLDS for `mod`.
# --------------------------------------------------------------------------- #
def _inv_target_equals_onehot(mod):
    """M1: at lambda=0, make_target returns t bitwise == Y."""
    params, X, Y = _rand_case()
    out = mod.forward(params, X) if hasattr(mod, "forward") else None
    # use the same z path the module uses internally
    h = np.maximum(0.0, X @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    t = mod.make_target(z, Y, 0.0, 0.9, 0.15, 2.0)
    return np.array_equal(t, Y)


def _inv_loss_grads_equal_ce(mod):
    """M2/M3/M4: at lambda=0, loss + all four grads bitwise == independent CE."""
    params, X, Y = _rand_case()
    loss_ce, grads_ce = _ce_loss_and_grads(params, X, Y)
    loss_cw, grads_cw = mod.loss_and_grads(params, X, Y, 0.0, 0.9, 0.15, 2.0)
    if loss_cw != loss_ce:
        return False
    for k in ("W1", "b1", "W2", "b2"):
        if not np.array_equal(grads_cw[k], grads_ce[k]):
            return False
    return True


_CHECKERS = {
    "M1-gate-weight-nonzero-at-lambda-zero": _inv_target_equals_onehot,
    "M2-relu-mask-off-by-one": _inv_loss_grads_equal_ce,
    "M3-loss-reduction-mean-over-elements": _inv_loss_grads_equal_ce,
    "M4-temperature-leaks-into-loss-prediction": _inv_loss_grads_equal_ce,
}


def _load_defects():
    return json.loads(MUTATIONS.read_text())["defects"]


@pytest.mark.parametrize("defect", _load_defects(), ids=lambda d: d["id"])
def test_mutation_is_caught(defect):
    """The must_fail test catches the defect: invariant holds for the original
    module and is VIOLATED for the mutated module."""
    did = defect["id"]
    assert did in _CHECKERS, f"no checker wired for {did}"
    check = _CHECKERS[did]
    # original: invariant holds
    assert check(r) is True, f"original module violates invariant for {did}"
    # mutated: invariant is violated (the suite catches the defect)
    mod = _load_mutated(defect["find"], defect["replace"])
    assert check(mod) is False, (
        f"defect {did} was NOT caught: the invariant still holds after "
        f"applying the mutation; the must_fail test "
        f"{defect['must_fail']} would NOT fail -> suite gap."
    )


@pytest.mark.parametrize("defect", _load_defects(), ids=lambda d: d["id"])
def test_find_anchor_is_unique(defect):
    """The find string is present exactly once in run_experiment.py, so the
    defect is well-aimed and the anchor has not drifted."""
    assert SRC.count(defect["find"]) == 1, defect["find"]
