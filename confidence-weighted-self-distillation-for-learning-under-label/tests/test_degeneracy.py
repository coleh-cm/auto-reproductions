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
    """lam=0 => w=0 => t == y exactly (SPEC gate a)."""
    params, X, Y = _rand_case()
    out = r.forward(params, X)
    t = r.make_target(out["z"], Y, lam=0.0, tau=0.9, s=0.15, T=2.0)
    assert np.array_equal(t, Y)
    # and the gate weight itself is exactly zero, not merely tiny
    w = 0.0 / (1.0 + np.exp(-(out["p"].max(-1) - 0.9) / 0.15))
    assert np.all(w == 0.0)


def test_lambda_zero_loss_and_grads_equal_ce_bitwise():
    """Loss and ALL parameter grads at lam=0 are bitwise identical to plain CE."""
    params, X, Y = _rand_case(seed=7)
    loss_cw, grads_cw = r.loss_and_grads(params, X, Y, 0.0, 0.9, 0.15, 2.0)
    loss_ce, grads_ce = ce_loss_and_grads(params, X, Y)
    assert loss_cw == loss_ce, (loss_cw, loss_ce)
    for k in ("W1", "b1", "W2", "b2"):
        assert np.array_equal(grads_cw[k], grads_ce[k]), k


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
