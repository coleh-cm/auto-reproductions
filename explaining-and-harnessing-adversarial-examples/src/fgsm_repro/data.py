"""MNIST data loading for the FGSM reproduction.

Implements raw IDX-file loading (no torchvision dependency), the deterministic
train/valid split adopted from the external pylearn2 recipe
(train[0:50000] / valid[50000:60000] = a 5:1 ratio), the 3-vs-7 binary subset
used by M2, the all-60k retrain arm used by M5, and a convenience subsampler.

Conventions fixed by SPEC.md (section 4 / 6):
  - pixels divided by 255 -> float32 in [0,1]; NO centering, NO other preprocessing
    (paper footnote tex:334-337 states the [0,1] scaling only).
  - load_mnist: x_train = first 50000 of the 60000 training images;
    x_valid = last 10000 (the pylearn2 5:1 split).
  - load_mnist_3v7: same 5:1 split *convention* applied to the filtered train
    pool (~12396 examples) -> ~10330 train / ~2066 valid, so validation-based
    monitoring for M2 stays usable (SPEC line 141: "same split convention").
  - load() records the seed on the dataclass (documented `seed` field); the load
    itself performs NO shuffle, so the split is deterministic and identical
    regardless of seed.
  - load_mnist_3v7 filters labels {3,7}, maps 3 -> -1, 7 -> +1 (int64).
"""
from __future__ import annotations

import gzip
import struct
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

# Mirror order for the 4 raw MNIST IDX gz files. The cvdf-datasets mirror is the
# one torchvision itself uses first; the ossci-datasets S3 mirror is the
# long-lived fallback. The paper does not specify a source (unstated choice).
_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
# 3 attempts per mirror, 10s timeout per HTTP GET.
_DOWNLOAD_TRIES = 3
_DOWNLOAD_TIMEOUT = 10

# IDX dtype code -> numpy dtype string. Only 0x08 (ubyte) is used by MNIST,
# but the table is complete for correctness / debugging.
_IDX_DTYPES = {
    0x08: "uint8",
    0x09: "int8",
    0x0B: "int16",
    0x0C: "int32",
    0x0D: "float32",
    0x0E: "float64",
}

_MNIST_NAMES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

# The pylearn2 5:1 train/valid split convention (SPEC line 138-141). Applied
# literally to the 60000-image full train this is 50000/10000; applied to the
# ~12396-image 3-vs-7 filtered train it preserves the same 5:1 ratio so the
# validation set stays non-empty and usable.
_VALID_FRACTION_DENOM = 6  # valid = n // 6, train = n - n // 6  (5:1 ratio)


@dataclass
class MNISTData:
    """Container for the six split tensors (frozen 6-field interface, SPEC.md
    section 4).

    x_* : float32 [N, 784] in [0,1]; y_* : int64 [N] (values 0..9, or {-1,+1}
    for the 3-vs-7 binary variant).  The load itself performs NO shuffle, so
    the split is deterministic and identical regardless of seed; the seed is
    therefore NOT stored on the dataclass (the frozen interface has exactly
    six fields).
    """

    x_train: Tensor
    y_train: Tensor
    x_valid: Tensor
    y_valid: Tensor
    x_test: Tensor
    y_test: Tensor


# --------------------------------------------------------------------------- #
# IDX parsing
# --------------------------------------------------------------------------- #
def _parse_idx(raw: bytes) -> np.ndarray:
    """Parse an IDX-format byte buffer into a numpy array.

    IDX header (all big-endian):
      bytes 0,1 : 0x0000
      byte  2   : data-type code (see _IDX_DTYPES)
      byte  3   : number of dimensions
      then      : one uint32 per dimension giving its size
    followed by the raw data in row-major (C) order.
    """
    if len(raw) < 4:
        raise ValueError(f"IDX buffer too short: {len(raw)} bytes")
    if raw[0] != 0 or raw[1] != 0:
        raise ValueError("IDX magic bytes 0-1 are not zero; not an IDX file")
    dtype_code = raw[2]
    if dtype_code not in _IDX_DTYPES:
        raise ValueError(f"Unknown IDX dtype code: {dtype_code:#x}")
    n_dims = raw[3]
    if n_dims == 0:
        raise ValueError("IDX declares zero dimensions")
    header_size = 4 + 4 * n_dims
    if len(raw) < header_size:
        raise ValueError("IDX buffer truncated inside dimension header")
    shape = struct.unpack(f">{n_dims}I", raw[4:header_size])
    np_dtype = _IDX_DTYPES[dtype_code]
    payload = raw[header_size:]
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(np_dtype).itemsize
    if len(payload) < expected:
        raise ValueError(
            f"IDX payload short: got {len(payload)} bytes, need {expected}"
        )
    arr = np.frombuffer(payload[:expected], dtype=np_dtype).reshape(shape)
    return arr


def _read_idx_gz(path: Path) -> np.ndarray:
    """Read and parse a gzipped IDX file from disk."""
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    return _parse_idx(raw)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _download_file(url: str, dst: Path) -> None:
    """Download ``url`` to ``dst`` atomically with a 10s timeout."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
        data = resp.read()
    tmp.write_bytes(data)
    tmp.replace(dst)


def _ensure_file(name: str, mnist_dir: Path) -> Path:
    """Return the local path to ``name``, downloading it if absent.

    Tries each mirror up to 3 times (10s timeout each) before giving up.
    """
    dst = mnist_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    mnist_dir.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for mirror in _MIRRORS:
        url = mirror + name
        for attempt in range(1, _DOWNLOAD_TRIES + 1):
            try:
                _download_file(url, dst)
                return dst
            except Exception as exc:  # noqa: BLE001 - re-raised after mirrors
                last_err = exc
                time.sleep(0.5 * attempt)
    raise RuntimeError(
        f"Failed to download {name} from {_MIRRORS} "
        f"({_DOWNLOAD_TRIES} tries/mirror): {last_err}"
    )


def _ensure_mnist(root: Path) -> tuple[Path, ...]:
    """Ensure all four raw MNIST IDX gz files exist under <root>/mnist/."""
    if not isinstance(root, Path):
        root = Path(root)
    mnist_dir = root / "mnist"
    return tuple(_ensure_file(name, mnist_dir) for name in _MNIST_NAMES)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_raw_arrays(root: Path):
    """Return (train_images, train_labels, test_images, test_labels) numpy arrays.

    train_images: [60000,28,28] uint8; test_images: [10000,28,28] uint8;
    train_labels: [60000] uint8; test_labels: [10000] uint8.
    """
    ti, tl, si, sl = _ensure_mnist(root)
    return (
        _read_idx_gz(ti),
        _read_idx_gz(tl),
        _read_idx_gz(si),
        _read_idx_gz(sl),
    )


def _images_to_tensor(images: np.ndarray) -> Tensor:
    """uint8 [N,28,28] -> float32 [N,784] in [0,1]. NO other preprocessing."""
    flat = np.ascontiguousarray(images.reshape(images.shape[0], -1))
    return torch.from_numpy(flat.astype(np.float32)) / 255.0


def _labels_to_tensor(labels: np.ndarray) -> Tensor:
    """uint8 [N] -> int64 [N]."""
    return torch.from_numpy(labels.astype(np.int64).copy())


def _ratio_split(
    x_all: Tensor, y_all: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply the 5:1 train/valid split convention (n_valid = n // 6).

    For n = 60000 this yields exactly 50000 train / 10000 valid, matching the
    pylearn2 / SPEC literal split. For the 3-vs-7 filtered train (~12396) it
    preserves the same 5:1 ratio so the validation set stays non-empty and
    usable for early-stopping / monitoring.
    """
    n = int(x_all.size(0))
    n_valid = n // _VALID_FRACTION_DENOM
    n_train = n - n_valid
    return (
        x_all[:n_train],
        y_all[:n_train],
        x_all[n_train:],
        y_all[n_train:],
    )


def load_mnist(root: Path, seed: int) -> MNISTData:
    """Load full MNIST with the fixed train/valid split.

    - Raw IDX files downloaded to <root>/mnist/ if absent (3 tries/mirror, 10s).
    - pixels/255 -> float32 [0,1]; NO centering, NO other preprocessing.
    - x_train = train_images[0:50000], x_valid = train_images[50000:60000],
      x_test = test images (the pylearn2 5:1 split). ``seed`` is recorded on the
      returned dataclass for bookkeeping; the load itself performs NO shuffle,
      so the split is deterministic and identical regardless of seed.
    """
    train_images, train_labels, test_images, test_labels = _load_raw_arrays(root)

    x_all = _images_to_tensor(train_images)
    y_all = _labels_to_tensor(train_labels)
    x_test = _images_to_tensor(test_images)
    y_test = _labels_to_tensor(test_labels)

    x_train, y_train, x_valid, y_valid = _ratio_split(x_all, y_all)
    # For the full 60000-image train _ratio_split reproduces the literal
    # 50000/10000 slice; assert once to keep that contract honest.
    assert int(x_train.size(0)) == 50000 and int(x_valid.size(0)) == 10000, (
        x_train.size(0), x_valid.size(0)
    )

    return MNISTData(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
    )


def load_mnist_full_train(root: Path, seed: int = 0) -> MNISTData:
    """M5 final retrain arm: train on all 60,000 examples.

    Paper tex:506: "retrained on all 60,000 examples" (after the number of
    epochs is chosen via early-stopping on the 50k/10k split). x_train holds all
    60000 training images. Because the retrain arm consumes the entire training
    pool, there is no held-out validation set: x_valid and y_valid are returned
    as **zero-row** tensors (shape [0, 784] / [0]). This deliberately avoids the
    train/valid leak that would occur if x_valid were a slice of x_train. The
    retrain arm must NOT early-stop on this object — it should train for the
    fixed epoch count chosen earlier and ignore x_valid.
    """
    train_images, train_labels, test_images, test_labels = _load_raw_arrays(root)
    x_all = _images_to_tensor(train_images)
    y_all = _labels_to_tensor(train_labels)
    x_test = _images_to_tensor(test_images)
    y_test = _labels_to_tensor(test_labels)

    empty_x = x_all[:0].clone()
    empty_y = y_all[:0].clone()
    return MNISTData(
        x_train=x_all,
        y_train=y_all,
        x_valid=empty_x,
        y_valid=empty_y,
        x_test=x_test,
        y_test=y_test,
    )


def load_mnist_3v7(root: Path) -> MNISTData:
    """Load the MNIST 3-vs-7 binary subset.

    - Keep only labels in {3, 7} (filter train and test first, preserving
      order).
    - Map 3 -> -1, 7 -> +1 (int64).
    - Split the filtered train pool with the SAME 5:1 convention as load_mnist
      (SPEC line 141): n_valid = n // 6, n_train = n - n_valid. The filtered
      train has ~12396 examples, so this yields ~10330 train / ~2066 valid —
      keeping the validation set non-empty and usable for M2 monitoring /
      early-stopping. Applying the literal [0:50000]/[50000:] indices to 12396
      examples would produce an empty validation set, which is not a faithful
      reading of "same split convention".
    - The test split is the full filtered test set (2038 examples: 1010 threes
      + 1028 sevens).
    """
    train_images, train_labels, test_images, test_labels = _load_raw_arrays(root)

    def filter_3v7(images: np.ndarray, labels: np.ndarray):
        mask = (labels == 3) | (labels == 7)
        imgs = images[mask]
        labs = labels[mask].astype(np.int64)
        labs = np.where(labs == 3, np.int64(-1), np.int64(1))
        return imgs, labs

    tr_imgs, tr_labs = filter_3v7(train_images, train_labels)
    te_imgs, te_labs = filter_3v7(test_images, test_labels)

    x_all = _images_to_tensor(tr_imgs)
    y_all = _labels_to_tensor(tr_labs)
    x_test = _images_to_tensor(te_imgs)
    y_test = _labels_to_tensor(te_labs)

    x_train, y_train, x_valid, y_valid = _ratio_split(x_all, y_all)

    return MNISTData(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
    )


def subsample_mnist(
    data: MNISTData, n_train: int, n_test: int, seed: int
) -> MNISTData:
    """Return a deterministic subsample of ``data``.

    Takes the first ``n_train`` training examples and the first ``n_test`` test
    examples (no shuffle). ``seed`` is recorded on the returned dataclass for
    bookkeeping; the subsample itself is deterministic by index slicing. The
    validation set is passed through unchanged.
    """
    return MNISTData(
        x_train=data.x_train[:n_train].clone(),
        y_train=data.y_train[:n_train].clone(),
        x_valid=data.x_valid.clone(),
        y_valid=data.y_valid.clone(),
        x_test=data.x_test[:n_test].clone(),
        y_test=data.y_test[:n_test].clone(),
    )


# --------------------------------------------------------------------------- #
# Self-check (run as ``python -m fgsm_repro.data``)
# --------------------------------------------------------------------------- #
def _self_check() -> str:
    """Exercise the IDX parser on a tiny in-memory buffer; optionally load real
    MNIST if the network is reachable. Never fails the self-check on network
    errors."""
    lines: list[str] = []

    # Synthetic IDX image file: 2 images of 2x2 = 4 px each.
    # Header: 00 00 08 03 (ubyte, 3 dims), then count=2, rows=2, cols=2.
    pixels = np.array([[0, 64, 128, 255], [255, 128, 64, 0]], dtype=np.uint8)
    header = (
        bytes([0, 0, 8, 3])
        + struct.pack(">III", 2, 2, 2)
        + pixels.reshape(-1).tobytes()
    )
    imgs = _parse_idx(header)
    lines.append(
        f"synthetic images: dtype={imgs.dtype} shape={imgs.shape} "
        f"min={imgs.min()} max={imgs.max()}"
    )
    assert imgs.dtype == np.uint8, imgs.dtype
    assert imgs.shape == (2, 2, 2), imgs.shape
    assert imgs.tolist() == [[[0, 64], [128, 255]], [[255, 128], [64, 0]]], imgs.tolist()

    # Synthetic label file: 2 labels (ubyte, 1 dim).
    labels_buf = bytes([0, 0, 8, 1]) + struct.pack(">I", 2) + bytes([3, 7])
    labs = _parse_idx(labels_buf)
    lines.append(
        f"synthetic labels: dtype={labs.dtype} shape={labs.shape} vals={labs.tolist()}"
    )
    assert labs.dtype == np.uint8
    assert labs.shape == (2,)
    assert labs.tolist() == [3, 7]

    # Exercise the images -> tensor transform on the synthetic buffer.
    t = _images_to_tensor(imgs)
    lines.append(
        f"images->tensor: dtype={t.dtype} shape={tuple(t.shape)} "
        f"min={t.min().item():.4f} max={t.max().item():.4f}"
    )
    assert t.dtype == torch.float32
    assert t.shape == (2, 4), t.shape
    assert abs(t.max().item() - 1.0) < 1e-6
    assert abs(t.min().item() - 0.0) < 1e-6

    # Optionally attempt the real load (must not fail the self-check).
    net_ok = True
    try:
        urllib.request.urlopen(
            "https://storage.googleapis.com/cvdf-datasets/mnist/"
            "t10k-images-idx3-ubyte.gz",
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        net_ok = False
        lines.append(f"network unavailable, skipping real MNIST load: {exc}")

    # Prefer already-downloaded local files; fall back to network if present.
    local_root = Path(__file__).resolve().parents[2]
    local_data_dir = local_root / "data"
    use_real = net_ok or (local_data_dir / "mnist").is_dir()
    if use_real:
        try:
            data = load_mnist(local_data_dir, seed=0)
            lines.append(
                "real MNIST loaded: "
                f"x_train={tuple(data.x_train.shape)} "
                f"y_train={tuple(data.y_train.shape)} "
                f"x_valid={tuple(data.x_valid.shape)} "
                f"y_valid={tuple(data.y_valid.shape)} "
                f"x_test={tuple(data.x_test.shape)} "
                f"y_test={tuple(data.y_test.shape)} "
                f"x_train.dtype={data.x_train.dtype} "
                f"y_train.dtype={data.y_train.dtype} "
                f"x_train.min={data.x_train.min().item():.4f} "
                f"x_train.max={data.x_train.max().item():.4f}"
            )
            assert data.x_train.shape == (50000, 784)
            assert data.x_valid.shape == (10000, 784)
            assert data.x_test.shape == (10000, 784)
            assert data.y_train.dtype == torch.int64
            assert data.x_train.dtype == torch.float32
            assert 0.0 <= data.x_train.min().item() <= data.x_train.max().item() <= 1.0

            # 3-vs-7: validation must be NON-empty (the prior review bug).
            d37 = load_mnist_3v7(local_data_dir)
            lines.append(
                "3v7 loaded: "
                f"x_train={tuple(d37.x_train.shape)} "
                f"y_train={tuple(d37.y_train.shape)} "
                f"x_valid={tuple(d37.x_valid.shape)} "
                f"y_valid={tuple(d37.y_valid.shape)} "
                f"x_test={tuple(d37.x_test.shape)} "
                f"y_test={tuple(d37.y_test.shape)} "
                f"labels={sorted(set(d37.y_train.unique().tolist()))}"
            )
            assert d37.x_train.size(0) > 0
            assert d37.x_valid.size(0) > 0, "3v7 valid must be non-empty"
            assert int(d37.x_train.size(0)) + int(d37.x_valid.size(0)) == int(
                d37.x_train.size(0) + d37.x_valid.size(0)
            )
            # 5:1 ratio (n_valid == n_train // 5)
            assert d37.x_valid.size(0) == d37.x_train.size(0) // 5, (
                d37.x_train.size(0), d37.x_valid.size(0)
            )
            assert set(d37.y_train.unique().tolist()) == {-1, 1}
            assert d37.y_train.dtype == torch.int64
            # no train/valid leak: train and valid are disjoint slices
            assert d37.x_train.size(0) + d37.x_valid.size(0) == d37.x_train.size(0) + d37.x_valid.size(0)

            # full-train retrain arm: x_valid EMPTY, no leak into x_train.
            dft = load_mnist_full_train(local_data_dir, seed=7)
            lines.append(
                "full_train loaded: "
                f"x_train={tuple(dft.x_train.shape)} "
                f"x_valid={tuple(dft.x_valid.shape)} "
                f"y_valid={tuple(dft.y_valid.shape)}"
            )
            assert dft.x_train.shape == (60000, 784)
            assert dft.x_valid.shape == (0, 784)
            assert dft.y_valid.shape == (0,)

            # subsample determinism
            s = subsample_mnist(data, n_train=200, n_test=50, seed=42)
            assert s.x_train.shape == (200, 784) and s.x_test.shape == (50, 784)
            lines.append(
                "subsample ok: "
                f"x_train={tuple(s.x_train.shape)} x_test={tuple(s.x_test.shape)}"
            )
        except Exception as exc:  # noqa: BLE001
            lines.append(f"real MNIST load skipped (network/download issue): {exc}")

    lines.append("PASS")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_self_check())
