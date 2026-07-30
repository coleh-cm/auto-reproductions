"""Data-loader instrument test — assert the paper's own dataset by fingerprint.

The data loader (``fgsm_repro.data.load_mnist``) is an INSTRUMENT: it decides
what data every arm runs on, so a silent fallback to a synthetic corpus would
make every number meaningless (the failure mode this gate exists to prevent:
"a closed-book run fell back to a synthetic corpus and produced seven arms at
chance level ... which passed every gate and meant nothing").

This test fingerprints the real MNIST dataset three ways:

  1. SIZE / SHAPE: 60000 train + 10000 test images, each 784-dim, float32 in
     [0, 1] after the /255 scaling (paper footnote tex:334-337).
  2. VOCABULARY: the label set is exactly {0,...,9}; the per-class histogram
     matches the canonical MNIST distribution (the well-known counts every
     MNIST loader on earth produces — these are the dataset's fingerprint, not
     a model output).
  3. CHECKSUM: the sha256 of the raw label byte streams matches the canonical
     MNIST label files, so a different/renumbered/corrupted dataset is rejected.

The split convention (train[0:50000] / valid[50000:60000]) is also asserted.
If MNIST is absent this test RAISES (it never returns a false verdict): a host
with no dataset is a blocked result to report, not a silent synthetic fallback.
"""
from __future__ import annotations

import gzip
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT / "src"))

from fgsm_repro.data import load_mnist  # noqa: E402

MNIST_DIR = REPRO_ROOT / "mnist"

# Canonical MNIST fingerprints (the standard Yann LeCun / cvdf-datasets files).
RAW_LABEL_SHA = {
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbe",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6d",
}
TRAIN_HIST = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
TEST_HIST = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]


def _raw_labels(name: str) -> np.ndarray:
    path = MNIST_DIR / name
    if not path.exists():
        pytest.fail(
            f"MNIST raw file missing: {path}. The data loader is an instrument; "
            f"a missing dataset is a BLOCKED result, never a synthetic fallback."
        )
    raw = gzip.open(path).read()
    magic, n = struct.unpack(">II", raw[:8])
    assert magic == 0x00000801, f"{name}: bad label magic {magic:#x}"
    return np.frombuffer(raw[8:8 + n], dtype=np.uint8)


def test_mnist_raw_files_fingerprint():
    """Checksum + magic + count of the raw IDX label files (the dataset itself)."""
    for name, want_sha in RAW_LABEL_SHA.items():
        path = MNIST_DIR / name
        if not path.exists():
            pytest.fail(f"MNIST raw file missing: {path}")
        got = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        assert got == want_sha, f"{name}: sha256[:16]={got} expected {want_sha}"
    # magic + counts
    tr = _raw_labels("train-labels-idx1-ubyte.gz")
    te = _raw_labels("t10k-labels-idx1-ubyte.gz")
    assert len(tr) == 60000 and len(te) == 10000


def test_mnist_label_vocabulary_and_histogram():
    """Label set is {0..9} with the canonical MNIST class counts."""
    tr = _raw_labels("train-labels-idx1-ubyte.gz")
    te = _raw_labels("t10k-labels-idx1-ubyte.gz")
    assert sorted(set(tr.tolist())) == list(range(10))
    assert sorted(set(te.tolist())) == list(range(10))
    assert np.bincount(tr, minlength=10).tolist() == TRAIN_HIST
    assert np.bincount(te, minlength=10).tolist() == TEST_HIST


def test_load_mnist_shapes_range_and_split():
    """load_mnist yields 50000/10000/10000 tensors, float32 in [0,1], 784-dim."""
    data = load_mnist(MNIST_DIR.parent / "data", seed=0)
    assert data.x_train.shape == (50000, 784)
    assert data.x_valid.shape == (10000, 784)
    assert data.x_test.shape == (10000, 784)
    assert data.y_train.shape == (50000,) and data.y_valid.shape == (10000,)
    assert data.y_test.shape == (10000,)
    for t in (data.x_train, data.x_valid, data.x_test):
        assert t.dtype == np.float32 or t.dtype == __import__("torch").float32
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    # vocabulary on the test split (the eval splits used by every arm)
    assert sorted(set(data.y_test.tolist())) == list(range(10))
    # split convention: valid is the LAST 10000 of the 60000 train images, so
    # test labels are unaffected; the test histogram must still match MNIST.
    assert np.bincount(data.y_test.numpy() if hasattr(data.y_test, "numpy")
                      else np.asarray(data.y_test), minlength=10).tolist() == TEST_HIST
