"""Tests for the structural-invariant metrics run_experiment.py emits via
--metrics_out (the measured evidence the numbers gate adjudicates the
paper's equation-invariants against).

These metrics are NOT the 4000-step accuracy number; they are cheap,
deterministic invariants of Eqs. 1-4 computed on one batch with the trained
params. A positive test proves they report in-bounds values for a correct
implementation; a negative test proves they go out of bounds when an
invariant is broken (so the metric is not a vacuous always-pass). The
gate-weight-bound negative is exercised at the source -- a buggy w formula
-- by tests/test_mutations.py (M5).
"""

import numpy as np

import run_experiment as r


def _case(seed=7, B=16):
    rng = np.random.default_rng(seed)
    params = r.init_params(rng)
    X = rng.standard_normal((B, r.D)).astype(np.float32)
    Y = np.eye(r.K, dtype=np.float32)[rng.integers(0, r.K, size=B)]
    return params, X, Y


def test_structural_metrics_cwsd_in_bounds():
    """Positive (CWSD arm, lambda=1): every structural metric is in the range
    the paper's equations force."""
    params, X, Y = _case()
    m = r.structural_metrics(params, X, Y, lam=1.0, tau=0.9, s=0.15, T=2.0)
    assert m["param_count"] == 4                       # single network (paper §1)
    assert m["gate_w_min"] > 0.0                        # Eq. (2): sigmoid > 0
    assert m["gate_w_max"] < 1.0                        # Eq. (2): w < lam = 1
    assert m["target_min"] >= 0.0                       # Eq. (3): convex combo >= 0
    assert m["target_sum_err"] < 1e-6                   # Eq. (3): convex combo sums to 1
    assert m["stopgrad_grad_err"] < 5e-3                 # Eq. (3): dL/dz=(p-t)/B, t constant


def test_structural_metrics_baseline_degeneracy_is_zero():
    """Positive (baseline arm, lambda=0): the degeneracy metrics are EXACTLY
    0.0 (the lambda=0 path is bitwise identical to independent cross-entropy),
    and the gate weight is exactly zero (the paper's own verification gate)."""
    params, X, Y = _case()
    m = r.structural_metrics(params, X, Y, lam=0.0, tau=0.9, s=0.15, T=2.0)
    assert m["param_count"] == 4
    assert m["gate_w_max"] == 0.0      # w = 0 * sigmoid(...) == 0 exactly
    assert m["gate_w_min"] == 0.0
    assert m["degeneracy_loss_err"] == 0.0
    assert m["degeneracy_grad_err"] == 0.0


def test_structural_metrics_simplex_catches_broken_target(monkeypatch):
    """Negative: if make_target returns a target that is NOT in the simplex
    (the convex-combination invariant of Eq. 3 is broken), target_sum_err
    goes OUT of bounds -- so the metric is not a vacuous always-pass."""
    params, X, Y = _case()
    real = r.make_target

    def broken(z, Y_onehot, lam, tau, s, T):
        return 2.0 * real(z, Y_onehot, lam, tau, s, T)  # sums to 2, not 1

    monkeypatch.setattr(r, "make_target", broken)
    m = r.structural_metrics(params, X, Y, lam=1.0, tau=0.9, s=0.15, T=2.0)
    assert m["target_sum_err"] >= 1e-6, m["target_sum_err"]


def test_structural_metrics_degeneracy_catches_active_gate():
    """Negative: at lambda=1 (gate active) the loss/grads DIFFER from
    independent cross-entropy, so a degeneracy-style check on the CWSD arm
    would be non-zero -- the degeneracy metric is not a vacuous always-zero.
    (structural_metrics only emits degeneracy_* at lambda=0; here we check
    the underlying comparison directly to prove it discriminates.)"""
    params, X, Y = _case()
    loss_cw, grads_cw = r.loss_and_grads(params, X, Y, 1.0, 0.9, 0.15, 2.0)
    loss_ce, grads_ce = r.ce_loss_and_grads_independent(params, X, Y)
    assert abs(loss_cw - loss_ce) > 0.0
    assert any(not np.array_equal(grads_cw[k], grads_ce[k])
               for k in ("W1", "b1", "W2", "b2"))
