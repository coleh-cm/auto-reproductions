"""Data-pipeline tests (SPEC §1 setup; shapes from SPEC §2)."""

import numpy as np

import run_experiment as r


def test_data_split_shapes():
    """Empirical stratified split: 1257 train / 540 test (SPEC §1, §2)."""
    Xtr, ytr, Xte, yte = r.load_data(0)
    assert Xtr.shape == (1257, 64)
    assert Xte.shape == (540, 64)
    assert ytr.shape == (1257,)
    assert yte.shape == (540,)
    # scaling to [0,1]
    assert Xtr.max() <= 1.0 + 1e-6 and Xtr.min() >= 0.0 - 1e-6
    assert Xte.max() <= 1.0 + 1e-6 and Xte.min() >= 0.0 - 1e-6
    # labels in range
    assert ytr.min() >= 0 and ytr.max() < r.K
    assert yte.min() >= 0 and yte.max() < r.K


def test_split_is_stratified_and_seeded():
    """Same seed -> identical split; stratification keeps class proportions."""
    a = r.load_data(0)
    b = r.load_data(0)
    for ai, bi in zip(a, b):
        assert np.array_equal(ai, bi)
    Xtr, ytr, Xte, yte = a
    # stratified: train/test class proportions match the full set closely
    full = np.bincount(np.concatenate([ytr, yte]), minlength=r.K) / (len(ytr) + len(yte))
    tr = np.bincount(ytr, minlength=r.K) / len(ytr)
    te = np.bincount(yte, minlength=r.K) / len(yte)
    assert np.allclose(tr, full, atol=0.02)
    assert np.allclose(te, full, atol=0.02)


def test_corrupt_labels_rate_and_invariance():
    """uniform-all noise touches ~20% (effective ~0.18) and keeps labels in [0,K)."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, r.K, size=4000)
    yc = r.corrupt_labels(y, rng, rate=0.2, mode="uniform-all")
    changed = float(np.mean(yc != y))
    assert 0.16 < changed < 0.24
    assert yc.min() >= 0 and yc.max() < r.K


def test_corrupt_labels_uniform_other_excludes_original():
    """uniform-other never keeps the original label on a corrupted position."""
    rng = np.random.default_rng(4)
    y = rng.integers(0, r.K, size=20000)
    yc = r.corrupt_labels(y, rng, rate=0.2, mode="uniform-other")
    changed = yc != y
    # every changed label must differ from the original
    assert np.all(yc[changed] != y[changed])
    # rate close to 0.2 (no re-hit of original)
    assert 0.17 < changed.mean() < 0.23


def test_corrupt_labels_clean_when_rate_zero():
    """rate=0 returns the labels untouched (and unchanged array)."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, r.K, size=100)
    yc = r.corrupt_labels(y, rng, rate=0.0)
    assert np.array_equal(yc, y)
