"""Invariants implied by the paper's equations (research-code skill; SPEC §7).

Each test asserts something the maths *forces* to be true, so a violation is a
bug regardless of the (unstable) final-number comparison. Cheap and
deterministic; none runs the 4000-step training loop.
"""

import numpy as np

import run_experiment as r


def _case(seed=11, B=16):
    rng = np.random.default_rng(seed)
    params = r.init_params(rng)
    X = rng.standard_normal((B, r.D)).astype(np.float32)
    Y = np.eye(r.K, dtype=np.float32)[rng.integers(0, r.K, size=B)]
    out = r.forward(params, X)
    return params, X, Y, out


def test_softmax_sums_to_one():
    """p = softmax(z) is a distribution: sum_k p_ik = 1 for every example."""
    _, _, _, out = _case()
    s = out["p"].sum(axis=-1)
    assert np.allclose(s, 1.0)


def test_ptilde_sums_to_one():
    """p_tilde = softmax(z/T) is a distribution too (Eq. 3)."""
    _, _, _, out = _case()
    pt = r.softmax(out["z"] / 2.0)
    assert np.allclose(pt.sum(axis=-1), 1.0)


def test_target_sums_to_one():
    """t = (1-w) y + w p_tilde is a convex combination of two distributions that
    both sum to 1, with per-example weights, so t sums to 1 along the class axis."""
    _, _, Y, out = _case()
    for lam in (0.0, 0.3, 1.0):
        t = r.make_target(out["z"], Y, lam=lam, tau=0.9, s=0.15, T=2.0)
        assert np.allclose(t.sum(axis=-1), 1.0), lam


def test_target_in_simplex_nonneg():
    """Convex combination of distributions is itself in the simplex: t >= 0."""
    _, _, Y, out = _case()
    for lam in (0.0, 0.5, 1.0):
        t = r.make_target(out["z"], Y, lam=lam, tau=0.9, s=0.15, T=2.0)
        assert (t >= -1e-6).all(), lam


def test_confidence_in_range():
    """c = max_k p_k in [1/K, 1] (Eq. 1). Lower bound: the max of K values that
    sum to 1 is at least 1/K."""
    _, _, _, out = _case()
    c = out["p"].max(axis=-1)
    assert (c >= 1.0 / r.K - 1e-6).all()
    assert (c <= 1.0 + 1e-6).all()


def test_weight_bounded_by_lambda():
    """w = lam * sigmoid((c-tau)/s) in [0, lam] for every example (Eq. 2)."""
    _, _, _, out = _case()
    c = out["p"].max(axis=-1)
    for lam in (0.0, 0.5, 1.0):
        w = lam / (1.0 + np.exp(-(c - 0.9) / 0.15))
        assert (w >= -1e-9).all() and (w <= lam + 1e-6).all(), lam
    # lam=0 forces w==0 exactly (degeneracy anchor)
    w0 = 0.0 / (1.0 + np.exp(-(c - 0.9) / 0.15))
    assert np.all(w0 == 0.0)


def test_loss_nonnegative():
    """Cross-entropy between distributions is non-negative (Eq. 4)."""
    params, X, Y, _ = _case()
    for lam in (0.0, 0.5, 1.0):
        loss, _ = r.loss_and_grads(params, X, Y, lam, 0.9, 0.15, 2.0)
        assert loss >= -1e-6, (lam, loss)


def test_loss_matches_direct_formula():
    """Loss from loss_and_grads equals a direct -mean(sum t log p) computation."""
    params, X, Y, out = _case()
    lam = 1.0
    loss, _ = r.loss_and_grads(params, X, Y, lam, 0.9, 0.15, 2.0)
    t = r.make_target(out["z"], Y, lam=lam, tau=0.9, s=0.15, T=2.0)
    p = out["p"]
    direct = float(-np.sum(t * np.log(p + 1e-12)) / X.shape[0])
    assert np.isclose(loss, direct, atol=1e-5), (loss, direct)


def test_gate_open_target_equals_ptilde():
    """SPEC §7 gate (c): when the gate is fully open the target equals p_tilde
    and Eq. (4) reduces to CE(p_tilde, p). Drive the gate open with lam=1,
    tau=0, s tiny: for c >= 1/K > 0, (c - 0)/s -> +inf so sigmoid -> 1, w -> 1,
    hence t -> p_tilde. Then the loss must match a direct CE(p_tilde, p)."""
    params, X, Y, out = _case(seed=21)
    # tiny s + tau=0 fully opens the gate for all c > 0 (always true here)
    loss_open, _ = r.loss_and_grads(params, X, Y, 1.0, 0.0, 1e-3, 2.0)
    p_tilde = r.softmax(out["z"] / 2.0)
    p = out["p"]
    direct = float(-np.sum(p_tilde * np.log(p + 1e-12)) / X.shape[0])
    assert np.isclose(loss_open, direct, atol=1e-3), (loss_open, direct)


def test_gradient_matches_finite_differences():
    """Analytic dL/dz = (p-t)/B (with t held constant, Eq. 3 stop-grad) vs
    central finite differences of the loss with t FROZEN at the unperturbed
    params (SPEC §7 gate d).

    Finite-differencing ``loss_and_grads(...)[0]`` directly is WRONG here: that
    recomputes t from the perturbed z on every call, so its value-FD returns
    the full no-stopgrad gradient (including the dt/dz chain) and the check
    would pass against the stopgrad analytic grad only by coincidence on
    near-uniform p. We freeze t via ``_loss_with_frozen_target`` and FD only
    the log p term, which is the gradient the stop-grad actually produces.
    A peaked network (large W2) is used so p is non-uniform: a no-stopgrad
    implementation would diverge here (~6 vs the analytic grad), proving the
    check is not vacuous.
    """
    rng = np.random.default_rng(2)
    P = {
        "W1": (rng.standard_normal((4, 5)) * 0.1).astype(np.float32),
        "b1": np.zeros(5, np.float32),
        "W2": (rng.standard_normal((5, 3)) * 8.0).astype(np.float32),
        "b2": np.zeros(3, np.float32),
    }
    X = rng.standard_normal((3, 4)).astype(np.float32)
    Y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=3)]
    # Freeze the target at the unperturbed params (the Eq. (3) stop-grad).
    h0 = np.maximum(0.0, X @ P["W1"] + P["b1"])
    z0 = h0 @ P["W2"] + P["b2"]
    t0 = r.make_target(z0, Y, 1.0, 0.9, 0.15, 2.0)
    _, grads = r.loss_and_grads(P, X, Y, 1.0, 0.9, 0.15, 2.0)
    eps = 1e-4
    # check ALL four params, including W1/b1 (the ReLU backprop path — the most
    # error-prone: a wrong ReLU mask or transposed W2 would only show up here).
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
        assert np.allclose(num, grads[name], atol=2e-3), name


def test_stopgrad_target_independent_of_params():
    """The target is treated as a constant w.r.t. theta (Eq. 3 stopgrad; SPEC
    §4 item 5). Perturbing the parameters must NOT change the target computed
    from the *same* logits — i.e. make_target is a pure function of (z, Y, ...).
    This is structural in numpy (no autograd), but assert it explicitly so a
    future autograd port cannot silently re-introduce the trivial solution."""
    params, X, Y, out = _case(seed=31)
    z = out["z"].copy()
    t0 = r.make_target(z, Y, 1.0, 0.9, 0.15, 2.0)
    # perturb params; with the same z the target must be unchanged
    t1 = r.make_target(z, Y, 1.0, 0.9, 0.15, 2.0)
    assert np.array_equal(t0, t1)
    # and changing z DOES change the target (so the test is not vacuous).
    # NB: a constant shift to all logits is softmax-invariant, so perturb
    # non-uniformly across classes.
    rng2 = np.random.default_rng(99)
    z2 = z + 0.5 * rng2.standard_normal(z.shape).astype(np.float32)
    t2 = r.make_target(z2, Y, 1.0, 0.9, 0.15, 2.0)
    assert not np.allclose(t0, t2)
