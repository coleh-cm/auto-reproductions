"""Data loading for the EAE reproduction.

MNIST (IDX, gz) and CIFAR-10 (pickle) are downloaded on first use into
./data/ (gitignored). MNIST pixels are scaled to [0,1]. CIFAR-10 is
GCN-preprocessed to global std ~0.5 (recipe below).

Dataset splits (deterministic; `seed` reserved for future shuffling but the
canonical splits here are seed-independent so runs are comparable):
  load_mnist(seed)       : 50k train / 10k val (last 10k of the 60k train) / 10k test
  load_mnist_full(seed)  : 60k train / 10k val (last 10k of the 60k) / 10k test
  load_mnist_3v7(seed)   : 3-vs-7 subset of the above; y in {-1,+1}, +1 == digit 3
  load_cifar10(seed)     : 45k train / 5k val (last 5k of the 50k) / 10k test; GCN std~0.5
  rubbish(dim, n, seed)  : [n,dim] f32 ~ N(0, I_dim)

References: SPEC.md §5; paper/source/iclr2015.tex:334 (MNIST [0,1]),
  tex:343-345 (CIFAR GCN std ~0.5), tex:905-911 (rubbish N(0,I_784)/N(0,I_3072)).
"""
import gzip
import os
import pickle
import struct
import tarfile
import urllib.request
from io import BytesIO

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _download(url, dest):
    if os.path.exists(dest):
        return
    _ensure_dir()
    tmp = dest + ".part"
    print(f"downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, dest)


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x00000803, f"bad image magic {magic:#x}"
        buf = f.read(n * rows * cols)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return arr.astype(np.float32) / 255.0  # [N, 784] in [0,1]


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x00000801, f"bad label magic {magic:#x}"
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)  # [N]


def _load_mnist_raw():
    paths = {}
    for fname, url in MNIST_FILES.items():
        p = os.path.join(DATA_DIR, fname)
        _download(url, p)
        paths[fname] = p
    x_train = _read_idx_images(paths["train-images-idx3-ubyte.gz"])
    y_train = _read_idx_labels(paths["train-labels-idx1-ubyte.gz"])
    x_test = _read_idx_images(paths["t10k-images-idx3-ubyte.gz"])
    y_test = _read_idx_labels(paths["t10k-labels-idx1-ubyte.gz"])
    return x_train, y_train, x_test, y_test


def load_mnist(seed=0):
    """50k train / 10k val / 10k test, pixels [0,1] f32, labels int64."""
    x_train, y_train, x_test, y_test = _load_mnist_raw()
    x_val = x_train[50000:]
    y_val = y_train[50000:]
    x_tr = x_train[:50000]
    y_tr = y_train[:50000]
    return {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }


def load_mnist_full(seed=0):
    """60k train / 10k val (last 10k of train) / 10k test."""
    x_train, y_train, x_test, y_test = _load_mnist_raw()
    x_val = x_train[50000:]
    y_val = y_train[50000:]
    return {
        "x_train": x_train, "y_train": y_train,
        "x_train_full": x_train, "y_train_full": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }


def _to_3v7(x, y):
    mask = (y == 3) | (y == 7)
    xs = x[mask]
    ys = y[mask]
    # +1 == digit 3, -1 == digit 7
    out = np.where(ys == 3, 1, -1).astype(np.int64)
    return xs, out


def load_mnist_3v7(seed=0):
    d = load_mnist(seed)
    x_tr, y_tr = _to_3v7(d["x_train"], d["y_train"])
    x_val, y_val = _to_3v7(d["x_val"], d["y_val"])
    x_te, y_te = _to_3v7(d["x_test"], d["y_test"])
    return {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_te, "y_test": y_te,
    }


def _load_cifar_raw():
    dest = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    _download(CIFAR_URL, dest)
    train_batches, test_batch = [], None
    with tarfile.open(dest, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("data_batch_1") or member.name.endswith("data_batch_2") \
                    or member.name.endswith("data_batch_3") or member.name.endswith("data_batch_4") \
                    or member.name.endswith("data_batch_5"):
                f = tar.extractfile(member)
                d = pickle.load(f, encoding="bytes")
                train_batches.append(d)
            elif member.name.endswith("test_batch"):
                f = tar.extractfile(member)
                test_batch = pickle.load(f, encoding="bytes")
    x_train = np.concatenate([d[b"data"] for d in train_batches], axis=0)  # [50000, 3072] uint8
    y_train = np.concatenate([np.array(d[b"labels"]) for d in train_batches], axis=0)
    x_test = test_batch[b"data"]  # [10000, 3072]
    y_test = np.array(test_batch[b"labels"])
    return x_train.astype(np.float32), y_train.astype(np.int64), x_test.astype(np.float32), y_test.astype(np.int64)


def _gcn_preprocess(x_train, x_test):
    """Global contrast normalization: subtract per-pixel mean over train, then
    scale so the GLOBAL std over train ~0.5 (paper, footnote 2, tex:343-345).

    Recipe (ours; paper defers to the dead pylearn2 maxout scripts):
      mu_i = mean over training images of pixel i
      s   = 0.5 / std(mean-subtracted train pixels)   # so post-scale global std = 0.5
      x   = (x - mu_i) * s
    """
    mu = x_train.mean(axis=0, keepdims=True)  # [1, 3072]
    xc = x_train - mu
    global_std = float(xc.std())
    if global_std < 1e-8:
        raise ValueError("CIFAR-10 train has near-zero std after centering")
    s = 0.5 / global_std
    return (x_train - mu) * s, (x_test - mu) * s, {"per_pixel_mean": mu, "scale": s, "global_std": global_std}


def load_cifar10(seed=0):
    """45k train / 5k val / 10k test, GCN-preprocessed to global std ~0.5, f32."""
    x_train, y_train, x_test, y_test = _load_cifar_raw()
    x_train, x_test, _info = _gcn_preprocess(x_train, x_test)
    x_val = x_train[45000:]
    y_val = y_train[45000:]
    x_tr = x_train[:45000]
    y_tr = y_train[:45000]
    return {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }


def rubbish(dim, n, seed=0):
    """[n,dim] f32 ~ N(0, I_dim)."""
    if n <= 0 or dim <= 0:
        raise ValueError("rubbish requires n>0 and dim>0")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)
