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
    """The stopgrad_grad_err check must DISTINGUISH the paper-literal gradient
    (stopgrad ONLY on p_tilde; gate weight w differentiable) from BOTH failure
    modes a literal reading can get wrong:

      (1) NO stop-grad on p_tilde (gradient also flows through p_tilde toward
          matching it -- the trivial-solution hazard the paper warns about,
          paper/paper.md:213-214). Its analytic grad = the full no-stopgrad
          gradient; it matches a no-stopgrad FD (everything recomputed) but NOT
          the frozen-p_tilde FD the check uses.
      (2) DETACHED / no gate path (the WHOLE target constant, dL/dz=(p-t)/B
          only -- the standard self-distillation convention, NOT what Eq. (3)
          marks). Its analytic grad omits the L->t->w->c->z term; it does NOT
          match the frozen-p_tilde FD (which includes the gate path).

    The fixed check (``_stopgrad_grad_err``) finite-differences the loss with
    p_tilde FROZEN at the unperturbed params (and w recomputed), which is the
    literal gradient. We re-derive that FD here independently and assert:
      - the LITERAL analytic (``grad_mode="literal"``) MATCHES the frozen-p_tilde
        FD (correct stop-grad on p_tilde AND gate path included);
      - a NO-stopgrad FD (everything recomputed) DIVERGES from the literal
        analytic (proving p_tilde is actually stopped, not coincidentally
        matching);
      - a DETACHED analytic (``grad_mode="detached"``) DIVERGES from the
        frozen-p_tilde FD (proving the gate path is actually included).

    On the peaked net (W2 scaled 8x) the literal analytic matches at ~1e-3
    while both failure modes diverge by O(1), so the check is not vacuous.
    """
    rng = np.random.default_rng(123)
    P = {
        "W1": (rng.standard_normal((4, 5)) * 0.1).astype(np.float32),
        "b1": np.zeros(5, np.float32),
        "W2": (rng.standard_normal((5, 3)) * 8.0).astype(np.float32),
        "b2": np.zeros(3, np.float32),
    }
    X = rng.standard_normal((3, 4)).astype(np.float32)
    Y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=3)]
    _, grads_lit = r.loss_and_grads(P, X, Y, 1.0, 0.9, 0.15, 2.0, grad_mode="literal")
    _, grads_det = r.loss_and_grads(P, X, Y, 1.0, 0.9, 0.15, 2.0, grad_mode="detached")
    h0 = np.maximum(0.0, X @ P["W1"] + P["b1"])
    z0 = h0 @ P["W2"] + P["b2"]
    eps = 1e-4

    def _fd(loss_fn):
        out = {}
        for name in ("W1", "b1", "W2", "b2"):
            num = np.zeros_like(P[name])
            for idx in np.ndindex(P[name].shape):
                orig = P[name][idx]
                P[name][idx] = orig + eps
                lp = loss_fn(P)
                P[name][idx] = orig - eps
                lm = loss_fn(P)
                P[name][idx] = orig
                num[idx] = (lp - lm) / (2 * eps)
            out[name] = num
        return out

    # frozen-p_tilde FD (p_tilde from z0, w recomputed) = the literal gradient.
    frozen_pt = _fd(lambda Pp: r._loss_with_frozen_ptilde(Pp, X, Y, z0, 1.0, 0.9, 0.15, 2.0))
    frozen_err = max(float(np.max(np.abs(frozen_pt[k] - grads_lit[k])))
                     for k in ("W1", "b1", "W2", "b2"))
    assert frozen_err < 5e-3, frozen_err   # literal analytic matches frozen-p_tilde FD

    # no-stopgrad FD (p_tilde recomputed from perturbed z) -- must DIVERGE from
    # the literal analytic, proving p_tilde is actually stopped.
    def _loss_nostop(Pp):
        h = np.maximum(0.0, X @ Pp["W1"] + Pp["b1"])
        z = h @ Pp["W2"] + Pp["b2"]
        t = r.make_target(z, Y, 1.0, 0.9, 0.15, 2.0)   # p_tilde recomputed (NOT frozen)
        zc = z - z.max(axis=-1, keepdims=True)
        log_p = zc - np.log(np.exp(zc).sum(axis=-1, keepdims=True))
        return float(-np.sum(t * log_p) / X.shape[0])
    nostop = _fd(_loss_nostop)
    nostop_err = max(float(np.max(np.abs(nostop[k] - grads_lit[k])))
                     for k in ("W1", "b1", "W2", "b2"))
    assert nostop_err > 1.0, nostop_err   # no-stopgrad diverges (p_tilde is stopped)

    # detached analytic (no gate path) -- must DIVERGE from the frozen-p_tilde
    # FD (which includes the gate path), proving the gate path is included.
    detached_err = max(float(np.max(np.abs(frozen_pt[k] - grads_det[k])))
                       for k in ("W1", "b1", "W2", "b2"))
    assert detached_err > 1.0, detached_err   # detached diverges (gate path included)
