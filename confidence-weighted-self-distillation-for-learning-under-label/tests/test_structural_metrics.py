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


def test_stopgrad_grad_err_is_nonvacuous():
    """The stopgrad_grad_err check must DISTINGUISH a correct stop-grad from a
    no-stopgrad implementation. The fixed check (_stopgrad_grad_err) finite-
    differences the loss with t FROZEN at the unperturbed params, so it matches
    the analytic dL/dz=(p-t)/B (with t constant) -- the Eq. (3) stop-grad.
    A no-stopgrad analytic grad would include the dt/dz chain; against the
    frozen-target FD it would DIVERGE. We simulate a no-stopgrad grad by
    finite-differencing loss_and_grads(...)[0] directly (which recomputes t
    from the perturbed z, i.e. the no-stopgrad gradient) on the SAME peaked
    network the fixed check uses, and assert it does NOT match the stopgrad
    analytic grad (it diverges by orders of magnitude). This proves the fixed
    check is not passing by coincidence (near-uniform p): on the peaked net a
    no-stopgrad implementation would be caught."""
    rng = np.random.default_rng(123)
    P = {
        "W1": (rng.standard_normal((4, 5)) * 0.1).astype(np.float32),
        "b1": np.zeros(5, np.float32),
        "W2": (rng.standard_normal((5, 3)) * 8.0).astype(np.float32),
        "b2": np.zeros(3, np.float32),
    }
    X = rng.standard_normal((3, 4)).astype(np.float32)
    Y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=3)]
    _, grads = r.loss_and_grads(P, X, Y, 1.0, 0.9, 0.15, 2.0)
    eps = 1e-4
    # The fixed check: FD of the loss with t FROZEN -- must MATCH the analytic
    # stopgrad grad. (Re-derive here rather than call _stopgrad_grad_err so the
    # test is independent of the production helper's internals.)
    h0 = np.maximum(0.0, X @ P["W1"] + P["b1"])
    z0 = h0 @ P["W2"] + P["b2"]
    t0 = r.make_target(z0, Y, 1.0, 0.9, 0.15, 2.0)
    frozen_errs = []
    for name in ("W1", "b1", "W2", "b2"):
        num = np.zeros_like(P[name])
        for idx in np.ndindex(P[name].shape):
            orig = P[name][idx]
            P[name][idx] = orig + eps
            lp = r._loss_with_frozen_target(P, X, t0)
            P[name][idx] = orig - eps
            lm = r._loss_with_frozen_target(P, X, t0)
            P[name][idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
        frozen_errs.append(float(np.max(np.abs(num - grads[name]))))
    assert max(frozen_errs) < 5e-3, max(frozen_errs)   # correct stop-grad matches
    # The OLD broken check: FD of loss_and_grads(...)[0] directly (recomputes
    # t from perturbed z = the no-stopgrad gradient) -- must DIVERGE on the
    # peaked net, proving the frozen-target check is non-vacuous here.
    recomputed_errs = []
    for name in ("W1", "b1", "W2", "b2"):
        num = np.zeros_like(P[name])
        for idx in np.ndindex(P[name].shape):
            orig = P[name][idx]
            P[name][idx] = orig + eps
            lp = r.loss_and_grads(P, X, Y, 1.0, 0.9, 0.15, 2.0)[0]
            P[name][idx] = orig - eps
            lm = r.loss_and_grads(P, X, Y, 1.0, 0.9, 0.15, 2.0)[0]
            P[name][idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
        recomputed_errs.append(float(np.max(np.abs(num - grads[name]))))
    assert max(recomputed_errs) > 1.0, max(recomputed_errs)   # no-stopgrad diverges
