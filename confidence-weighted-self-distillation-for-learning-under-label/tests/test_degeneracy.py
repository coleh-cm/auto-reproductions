"""Degeneracy / verification gate (paper §2 last paragraph; SPEC §1, §7 gate a).

The paper's central checkability claim: at ``lambda = 0`` the gate weight ``w``
is identically zero, the target ``t`` equals the one-hot label ``y``, and the
loss of Eq. (4) reduces *exactly* to standard cross-entropy on the observed
labels. These tests enforce that at three levels:

1. structural  : ``t == y`` and ``w == 0`` element-wise (cheap, deterministic).
2. per-step    : the loss and every parameter gradient at ``lambda = 0`` are
                 bitwise identical to an *independently written* cross-entropy
                 routine (not just ``loss_and_grads`` called with ``lam=0``).
3. end-to-end  : a full SGD training loop at ``lambda = 0`` produces bit-identical
                 parameters and accuracy to an independent CE training loop with
                 the same RNG stream. This is the cheapest real evidence that
                 "the method at its no-op setting reproduces the baseline
                 exactly" — a reader can run it without trusting the
                 implementation, and it does not depend on ``s`` (the one
                 unstated hyperparameter), so it cannot be fit to the answer.
"""

import argparse

import numpy as np

import run_experiment as r


# --------------------------------------------------------------------------- #
# Independent reference implementation of plain cross-entropy on the one-hot
# label. Deliberately NOT calling make_target / loss_and_grads, so that a bug
# shared by both would still be caught here.
# --------------------------------------------------------------------------- #
def _ce_forward(params, X):
    h = np.maximum(0.0, X @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    zc = z - z.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(zc).sum(axis=-1, keepdims=True))
    log_p = zc - lse
    p = np.exp(log_p)
    return h, z, p, log_p


def ce_loss_and_grads(params, X, Y_onehot):
    """Plain cross-entropy -mean(sum y log p) and grads (independent code path)."""
    h, _, p, log_p = _ce_forward(params, X)
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


def test_lambda_zero_target_equals_onehot():
    """lam=0 => w=0 => t == y exactly (SPEC gate a).

    Non-circular: we probe ``make_target`` with a ``p_tilde`` (built from z) that
    differs from ``Y`` and confirm the returned target is STILL bitwise equal to
    ``Y``. That can only hold if the mixing weight is exactly zero — any nonzero
    ``w`` would mix in ``p_tilde`` and break equality. We do NOT recompute ``w``
    with the same formula (that would be circular); we let ``array_equal(t, Y)``
    be the witness that ``w == 0``.

    Swept over ``s`` so the gate cannot be fit to the answer via the one
    unstated hyperparameter: the degeneracy must hold for *every* ``s``, not just
    the calibrated default 0.15.
    """
    params, X, Y = _rand_case()
    out = r.forward(params, X)
    # p_tilde differs from Y for this random case (sanity: not all rows equal)
    p_tilde = r.softmax(out["z"] / 2.0)
    assert not np.allclose(p_tilde, Y)
    for s in (0.01, 0.05, 0.15, 0.5, 1.0, 10.0):
        t = r.make_target(out["z"], Y, lam=0.0, tau=0.9, s=s, T=2.0)
        # t == Y bitwise => the only way is w == 0 exactly (p_tilde != Y above)
        assert np.array_equal(t, Y), f"s={s}"


def test_lambda_zero_loss_and_grads_equal_ce_bitwise():
    """Loss and ALL parameter grads at lam=0 are bitwise identical to plain CE.

    Swept over ``s`` so the gate cannot be fit to the answer via the one
    unstated hyperparameter: CE equality must hold for *every* ``s``.

    Swept over ``grad_mode`` so the paper's degeneracy (lambda=0 reproduces
    the baseline EXACTLY, paper/paper.md:253-280) is regression-guarded under
    BOTH grad modes. The structural reason it holds under both: the
    gate-path term is ``lam * ...`` (run_experiment.py:213-214) and vanishes
    at ``lam=0``, so literal and detached are bit-identical there. This test
    pins that claim against a future edit that re-introduces a
    lambda-dependent term at ``lam=0``.
    """
    params, X, Y = _rand_case(seed=7)
    loss_ce, grads_ce = ce_loss_and_grads(params, X, Y)
    for s in (0.01, 0.15, 1.0, 10.0):
        for grad_mode in ("literal", "detached"):
            loss_cw, grads_cw = r.loss_and_grads(
                params, X, Y, 0.0, 0.9, s, 2.0, grad_mode=grad_mode,
            )
            assert loss_cw == loss_ce, (grad_mode, s, loss_cw, loss_ce)
            for k in ("W1", "b1", "W2", "b2"):
                assert np.array_equal(grads_cw[k], grads_ce[k]), (grad_mode, s, k)


def test_lambda_zero_training_matches_ce_training_bitwise():
    """End-to-end: lam=0 SGD loop == independent CE SGD loop, bit-identical."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0,
    )

    def run(use_cw):
        rng = np.random.default_rng(0)
        params = r.init_params(rng, scheme="he")
        yc = r.corrupt_labels(ytr, rng, rate=0.2, mode="uniform-all")
        Yoh = np.eye(r.K, dtype=np.float32)[yc]
        n = Xtr.shape[0]
        step, steps = 0, 300  # short: only proving bitwise equivalence
        while step < steps:
            for idx in r.batches(n, 64, rng, mode="epoch-permutation"):
                if step >= steps:
                    break
                Xb, Yb = Xtr[idx], Yoh[idx]
                if use_cw:
                    _, g = r.loss_and_grads(params, Xb, Yb, 0.0, 0.9, 0.15, 2.0)
                else:
                    _, g = ce_loss_and_grads(params, Xb, Yb)
                for k in ("W1", "b1", "W2", "b2"):
                    params[k] = params[k] - 0.1 * g[k]
                step += 1
        acc = r.evaluate(params, Xte, yte)
        return params, acc

    p_cw, a_cw = run(use_cw=True)
    p_ce, a_ce = run(use_cw=False)
    for k in ("W1", "b1", "W2", "b2"):
        assert np.array_equal(p_cw[k], p_ce[k]), k
    assert a_cw == a_ce


def test_training_step_count_is_exact():
    """The loop runs EXACTLY --steps gradient updates, no more, no fewer.

    Guards against a future edit silently breaking the step-count guard
    (run_experiment.py:556-559). Includes a short-final-batch boundary
    (1257 train / 64 => 20 batches = 1280 step-capacities; probes 0,1,63,64,
    65,100 around the first-epoch boundary).
    """
    for steps in (0, 1, 63, 64, 65, 100):
        saved_fn = r.loss_and_grads
        seen = {"n": 0}

        def counting(params, X, Y_onehot, lam, tau, s, T, **kw):
            loss, g = saved_fn(params, X, Y_onehot, lam, tau, s, T, **kw)
            seen["n"] += 1
            return loss, g
        r.loss_and_grads = counting
        try:
            ns = argparse.Namespace(
                lambda_=0.0, s=0.15, tau=0.9, temperature=2.0, seed=0,
                steps=steps, lr=0.1, batch_size=64, init="he",
                noise_mode="uniform-all", noise_rate=0.2,
                batch_mode="epoch-permutation", rng_layout="init-first",
                grad_mode="literal",
            )
            r.train(ns)
        finally:
            r.loss_and_grads = saved_fn
        assert seen["n"] == steps, (steps, seen["n"])


def test_baseline_seed0_reproduces_paper_table1_value():
    """The lambda=0 baseline at seed 0 reproduces paper Table 1 EXACTLY
    (0.9370 = 506/540; paper/paper.md:400-406, :434-438).

    This pins the paper's headline baseline number as a regression guard: a
    future edit that perturbs the baseline accuracy without breaking the
    bitwise CE-equality degeneracy test (e.g. a change to the RNG stream
    layout or the data split) would otherwise slip through. The degeneracy
    test above proves lambda=0 == independent CE on a *random* case; this
    test proves the *specific* paper configuration lands on the paper's
    specific reported value. Full 4000-step budget (the paper's own); the
    run is deterministic at seed 0 and finishes in ~1s.
    """
    import sklearn  # noqa: F401  (load_digits path exercised via r.load_data)
    ns = argparse.Namespace(
        lambda_=0.0, s=0.15, tau=0.9, temperature=2.0, seed=0,
        steps=4000, lr=0.1, batch_size=64, init="he",
        noise_mode="uniform-all", noise_rate=0.2,
        batch_mode="epoch-permutation", rng_layout="init-first",
        grad_mode="literal",
    )
    acc, _, _, _ = r.train(ns)
    # The paper reports 0.9370 (=506/540); main() formats the FINAL line with
    # %.4f, so the headline value is the 4-decimal rounding of the raw fraction.
    assert f"{acc:.4f}" == "0.9370", (acc, "baseline seed-0 must reproduce paper Table 1: 0.9370")


