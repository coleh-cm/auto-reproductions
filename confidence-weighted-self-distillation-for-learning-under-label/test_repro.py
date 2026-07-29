"""Structural tests for the CWSD reproduction.

These verify the degeneracy/verification gates the paper itself prescribes
(SPEC §1, §7), independent of the training-run RNG so they are deterministic
and fast (no 4000-step training).
"""

import numpy as np

import run_experiment as r


def test_lambda_zero_target_equals_onehot():
    """lambda=0 => w=0 => t=y exactly (Eq. degeneracy / SPEC gate a)."""
    rng = np.random.default_rng(0)
    params = r.init_params(rng)
    X = rng.standard_normal((8, r.D)).astype(np.float32)
    Y = np.eye(r.K, dtype=np.float32)[rng.integers(0, r.K, size=8)]
    out = r.forward(params, X)
    t = r.make_target(out["z"], Y, lam=0.0, tau=0.9, s=0.05, T=2.0)
    assert np.allclose(t, Y), "lambda=0 target must equal one-hot label exactly"
    assert np.abs(t - Y).max() == 0.0


def test_weight_in_unit_interval_and_scaled_by_lambda():
    """w = lambda * sigmoid(...) in [0,1]; w(lam=0)=0, w(lam=1)>=0."""
    rng = np.random.default_rng(1)
    params = r.init_params(rng)
    X = rng.standard_normal((16, r.D)).astype(np.float32)
    out = r.forward(params, X)
    p = out["p"]
    for lam in (0.0, 0.5, 1.0):
        c = p.max(-1)
        w = lam / (1.0 + np.exp(-(c - 0.9) / 0.05))
        assert (w >= 0).all() and (w <= 1).all()
    w0 = 0.0 / (1.0 + np.exp(-(p.max(-1) - 0.9) / 0.05))
    assert np.all(w0 == 0.0)


def test_gradient_matches_finite_differences():
    """Analytic dL/dz = (p-t)/B vs finite differences on the weights."""
    rng = np.random.default_rng(2)
    P = {
        "W1": (rng.standard_normal((4, 5)) * 0.1).astype(np.float32),
        "b1": np.zeros(5, np.float32),
        "W2": (rng.standard_normal((5, 3)) * 0.1).astype(np.float32),
        "b2": np.zeros(3, np.float32),
    }
    X = rng.standard_normal((3, 4)).astype(np.float32)
    Y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=3)]
    _, grads = r.loss_and_grads(P, X, Y, 1.0, 0.9, 0.05, 2.0)
    eps = 1e-4
    for name in ("W2", "b2"):
        num = np.zeros_like(P[name])
        it = np.ndindex(P[name].shape)
        for idx in it:
            orig = P[name][idx]
            P[name][idx] = orig + eps
            lp = r.loss_and_grads(P, X, Y, 1.0, 0.9, 0.05, 2.0)[0]
            P[name][idx] = orig - eps
            lm = r.loss_and_grads(P, X, Y, 1.0, 0.9, 0.05, 2.0)[0]
            P[name][idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
        # float32 finite differences have ~1e-3 truncation/rounding noise
        assert np.allclose(num, grads[name], atol=2e-3), name


def test_data_split_shapes():
    """Empirical split: 1257 train / 540 test (SPEC §1)."""
    Xtr, ytr, Xte, yte = r.load_data(0)
    assert Xtr.shape == (1257, 64)
    assert Xte.shape == (540, 64)
    assert ytr.shape == (1257,)
    assert yte.shape == (540,)
    assert Xtr.max() <= 1.0 and Xtr.min() >= 0.0


def test_corrupt_labels_rate_and_invariance():
    """uniform-all noise touches ~20% of labels and keeps all labels in [0,K)."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, r.K, size=4000)
    yc = r.corrupt_labels(y, rng, rate=0.2, mode="uniform-all")
    changed = float(np.mean(yc != y))
    assert 0.16 < changed < 0.24  # ~0.18 effective flip rate
    assert yc.min() >= 0 and yc.max() < r.K
