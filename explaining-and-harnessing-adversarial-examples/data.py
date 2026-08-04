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


# Fingerprint checksums of the canonical MNIST IDX files from the
# ossci-datasets S3 mirror (the standard 60k/10k split). These are aggregate
# sums over the float32 [0,1] pixels, used by check_mnist_fingerprint to reject
# a synthetic / wrong corpus (the closed-book failure mode: a 65-token
# vocabulary corpus has a different shape AND a different pixel sum). Tolerances
# absorb float32 pairwise-summation order differences across numpy builds.
_MNIST_X_TRAIN_SUM = 5133683.0   # sum over x_train[:50000]
_MNIST_X_VAL_SUM = 1012586.25   # sum over x_train[50000:]
_MNIST_X_TEST_SUM = 1038914.5   # sum over x_test


def check_mnist_fingerprint(d):
    """Assert d is the real MNIST split by fingerprint: size (shapes), vocabulary
    (label set exactly {0..9}), pixel range ([0,1] f32), and a checksum (pixel
    sums) that a synthetic / all-zeros / wrong-corpus substitute cannot match.

    A closed-book run once fell back to a 65-token synthetic corpus and passed
    every downstream gate; this fingerprint (size + vocabulary + checksum) is
    what prevents that. Raises AssertionError on any mismatch -- never returns
    a boolean, so a loader that cannot verify its data fails loudly."""
    for k in ("x_train", "x_val", "x_test"):
        assert k in d, f"check_mnist_fingerprint: missing {k}"
        assert d[k].ndim == 2 and d[k].shape[1] == 784, f"{k} shape {d[k].shape} != [N,784]"
        assert d[k].dtype == np.float32, f"{k} dtype {d[k].dtype} != float32"
    assert d["x_train"].shape == (50000, 784), f"x_train shape {d['x_train'].shape}"
    assert d["x_val"].shape == (10000, 784), f"x_val shape {d['x_val'].shape}"
    assert d["x_test"].shape == (10000, 784), f"x_test shape {d['x_test'].shape}"
    for k in ("y_train", "y_val", "y_test"):
        assert d[k].dtype == np.int64, f"{k} dtype {d[k].dtype} != int64"
        labels = set(np.unique(d[k]).tolist())
        assert labels == set(range(10)), f"{k} labels {labels} != {{0..9}}"
    for k, arr in (("x_train", d["x_train"]), ("x_val", d["x_val"]), ("x_test", d["x_test"])):
        assert arr.min() >= 0.0, f"{k} min {arr.min()} < 0"
        assert arr.max() <= 1.0 + 1e-6, f"{k} max {arr.max()} > 1"
    # checksum: rejects an all-zeros / synthetic substitute that passed shape+range
    assert abs(float(d["x_train"].sum()) - _MNIST_X_TRAIN_SUM) < 200.0, \
        f"x_train sum {float(d['x_train'].sum())} != {_MNIST_X_TRAIN_SUM} (synthetic corpus?)"
    assert abs(float(d["x_val"].sum()) - _MNIST_X_VAL_SUM) < 50.0, \
        f"x_val sum {float(d['x_val'].sum())} != {_MNIST_X_VAL_SUM}"
    assert abs(float(d["x_test"].sum()) - _MNIST_X_TEST_SUM) < 50.0, \
        f"x_test sum {float(d['x_test'].sum())} != {_MNIST_X_TEST_SUM}"


def check_mnist_3v7_fingerprint(d3, d_full):
    """Assert d3 is the 3-vs-7 subset of d_full: only labels {-1,+1}, and the
    +1 count equals the digit-3 count in the full split (+1 == digit 3)."""
    for k in ("x_train", "x_val", "x_test"):
        assert k in d3, f"3v7 missing {k}"
        assert d3[k].dtype == np.float32
    ys = set(np.unique(np.concatenate([d3["y_train"], d3["y_val"], d3["y_test"]])).tolist())
    assert ys <= {-1, 1} and -1 in ys and 1 in ys, f"3v7 labels {ys} not subset of {{-1,+1}}"
    for k in ("y_train", "y_val", "y_test"):
        assert d3[k].dtype == np.int64
    # +1 == digit 3: the count of +1 in the 3v7 split equals the count of digit 3
    # in the corresponding full split (within the 3&7 mask).
    for k in ("y_train", "y_val", "y_test"):
        mask37 = (d_full[k] == 3) | (d_full[k] == 7)
        n3 = int((d_full[k] == 3).sum())
        n_pos = int((d3[k] == 1).sum())
        assert n3 == n_pos, f"+1 != digit 3 on {k}: {n3} vs {n_pos}"
        assert d3[k].shape[0] == int(mask37.sum()), f"3v7 {k} size != 3&7 count"


def load_mnist(seed=0):
    """50k train / 10k val / 10k test, pixels [0,1] f32, labels int64.

    Runs the fingerprint check on the loaded data so a synthetic substitute
    is rejected at the source rather than silently producing chance-level arms."""
    x_train, y_train, x_test, y_test = _load_mnist_raw()
    x_val = x_train[50000:]
    y_val = y_train[50000:]
    x_tr = x_train[:50000]
    y_tr = y_train[:50000]
    out = {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }
    check_mnist_fingerprint(out)
    return out


def load_mnist_full(seed=0):
    """50k train / 10k val / 10k test for Phase 1 early-stopping, PLUS the full
    60k (x_train_full) for the Phase 2 from-scratch retrain (paper tex:505-506:
    train on the train split to pick the epoch count, then RETRAIN on all 60k).
    x_train is the first 50k (val held out); x_train_full is all 60k."""
    x_train, y_train, x_test, y_test = _load_mnist_raw()
    x_val = x_train[50000:]
    y_val = y_train[50000:]
    x_tr = x_train[:50000]
    y_tr = y_train[:50000]
    return {
        "x_train": x_tr, "y_train": y_tr,
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
    out = {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_te, "y_test": y_te,
    }
    check_mnist_3v7_fingerprint(out, d)
    return out


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


def check_cifar10_fingerprint(d):
    """Assert d is the real CIFAR-10 split (post-GCN): size (45k/5k/10k x 3072),
    vocabulary (labels {0..9} int64), dtype float32, and GLOBAL std of x_train
    ~0.5 (within 0.04) -- the GCN preprocessing target (paper footnote 2,
    tex:343-345). A non-GCN array (std ~1.0) or a wrong-dim array (e.g. 1024)
    is rejected. Raises AssertionError on mismatch."""
    for k in ("x_train", "x_val", "x_test"):
        assert k in d, f"check_cifar10_fingerprint: missing {k}"
        assert d[k].ndim == 2 and d[k].shape[1] == 3072, f"{k} shape {d[k].shape} != [N,3072]"
        assert d[k].dtype == np.float32, f"{k} dtype {d[k].dtype} != float32"
    assert d["x_train"].shape == (45000, 3072), f"x_train shape {d['x_train'].shape}"
    assert d["x_val"].shape == (5000, 3072), f"x_val shape {d['x_val'].shape}"
    assert d["x_test"].shape == (10000, 3072), f"x_test shape {d['x_test'].shape}"
    for k in ("y_train", "y_val", "y_test"):
        assert d[k].dtype == np.int64, f"{k} dtype {d[k].dtype} != int64"
        labels = set(np.unique(d[k]).tolist())
        assert labels == set(range(10)), f"{k} labels {labels} != {{0..9}}"
    for k in ("x_train", "x_val", "x_test"):
        gstd_k = float(d[k].std())
        assert abs(gstd_k - 0.5) < 0.04, f"{k} global std {gstd_k} not ~0.5 (GCN not applied?)"
    # Structural check (not a rubber stamp): real CIFAR-10 images have highly
    # NON-UNIFORM per-pixel variance (sky pixels ~constant, object pixels vary),
    # while a std-matched SYNTHETIC iid corpus (e.g. N(0, 0.25*I_3072) GCN'd to
    # global std 0.5) has ~uniform per-pixel variance (std of per-pixel stds ~0).
    # This rejects the closed-book failure mode (a synthetic corpus that passes
    # shape + global-std + label-set) WITHOUT a precomputed checksum, which we
    # cannot populate here because the CIFAR download is blocked in this env.
    per_pix_std = d["x_train"].std(axis=0)  # [3072] std of each pixel across images
    pp_std_spread = float(per_pix_std.std())
    assert pp_std_spread > 0.01, (
        f"per-pixel-std spread {pp_std_spread} too uniform; real CIFAR-10 has "
        f"structured (non-uniform) per-pixel variance, a std-matched iid "
        f"synthetic corpus does not. (GCN pixel-sum is ~0 by construction for "
        f"ANY centered corpus, so a sum checksum cannot distinguish them.)")


def cifar10_available():
    """True iff the real CIFAR-10 tar is present and loadable, WITHOUT triggering
    a network download. A missing dataset is a blocked result (the contract);
    callers must not hang on a download that this environment throttles/drops, so
    this checks the local tar only and never reaches the network."""
    dest = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
        return False
    try:
        _load_cifar_raw()  # reads the local tar; no download (dest exists)
        return True
    except Exception:
        return False


def load_cifar10(seed=0):
    """45k train / 5k val / 10k test, GCN-preprocessed to global std ~0.5, f32.

    Runs the fingerprint check on the loaded data so a non-GCN or wrong-dim
    substitute is rejected at the source."""
    x_train, y_train, x_test, y_test = _load_cifar_raw()
    x_train, x_test, _info = _gcn_preprocess(x_train, x_test)
    x_val = x_train[45000:]
    y_val = y_train[45000:]
    x_tr = x_train[:45000]
    y_tr = y_train[:45000]
    out = {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }
    check_cifar10_fingerprint(out)
    return out


def rubbish(dim, n, seed=0):
    """[n,dim] f32 ~ N(0, I_dim)."""
    if n <= 0 or dim <= 0:
        raise ValueError("rubbish requires n>0 and dim>0")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)
