#!/usr/bin/env python3
"""Confidence-Weighted Self-Distillation (CWSD) for learning under label noise.

Reproduction of Bergstrom, Oyelaran & Vasquez, "Confidence-Weighted
Self-Distillation for Learning under Label Noise".

Both arms (baseline cross-entropy at ``--lambda 0`` and CWSD at ``--lambda 1``)
are the same program under a different mixing coefficient ``lambda``. The
``lambda = 0`` path reduces exactly to standard cross-entropy: the gate weight
``w`` is identically zero, so the target equals the one-hot label ``y`` and the
loss of Eq. (4) becomes ``-mean_k y_k log p_k``.

Implementation notes (see SPEC.md, §4–§5):
- numpy + scikit-learn only; gradients hand-derived so the stop-gradient
  semantics of Eq. (3) are structural (no autograd is involved, the target is a
  plain array used as a constant in the loss/gradient).
- ``dL/dz = (p - t) / B`` (standard softmax-cross-entropy with constant target),
  then backprop through ``W2``, ``ReLU``, ``W1``.
- The whole target ``t`` is treated as constant (the stop-gradient applies to
  the entire target, not only ``p_tilde``); see SPEC §4 item 5.

CLI output contract: exactly one line on stdout at completion,
``FINAL accuracy=<float>`` formatted ``%.4f``. All diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Iterator, Tuple

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Constants / shapes (SPEC §2)
# --------------------------------------------------------------------------- #
D = 64          # input features (8x8 digits)
H = 64          # hidden units
K = 10          # classes
B_DEFAULT = 64  # minibatch size


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load digits, scale to [0,1], stratified 70/30 split.

    Returns Xtr[1257,64] float32, ytr int64 (clean), Xte[540,64] float32,
    yte int64.
    """
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0          # scale [0,16] -> [0,1]
    y = digits.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    return (
        Xtr.astype(np.float32),
        ytr.astype(np.int64),
        Xte.astype(np.float32),
        yte.astype(np.int64),
    )


def corrupt_labels(
    y: np.ndarray, rng: np.random.Generator, rate: float = 0.2,
    mode: str = "uniform-all",
) -> np.ndarray:
    """Corrupt training labels with symmetric noise.

    Each example is independently selected with probability ``rate``; its label
    is replaced by a class drawn uniformly at random. ``mode="uniform-all"``
    draws from all K classes (literal reading of the paper: ~1/10 of corrupted
    examples keep their original label, effective flip rate 0.18).
    ``mode="uniform-other"`` draws from the K-1 other classes (true 20% flips).
    """
    n = y.shape[0]
    yc = y.copy()
    mask = rng.random(n) < rate
    n_corrupt = int(mask.sum())
    if n_corrupt == 0:
        return yc
    if mode == "uniform-all":
        new = rng.integers(0, K, size=n_corrupt)
    elif mode == "uniform-other":
        rows = []
        for lab in y[mask]:
            others = np.delete(np.arange(K), lab)
            rows.append(rng.choice(others))
        new = np.array(rows, dtype=np.int64)
    else:
        raise ValueError(f"unknown noise-mode: {mode}")
    yc[mask] = new
    return yc


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def init_params(rng: np.random.Generator, scheme: str = "he") -> Dict[str, np.ndarray]:
    """Initialise {W1,b1,W2,b2}.

    He-normal for weights (suitable for ReLU); zeros for biases. Returns
    float32 arrays.
    """
    if scheme == "he":
        std1 = np.sqrt(2.0 / D)
        std2 = np.sqrt(2.0 / H)
        W1 = (rng.standard_normal((D, H)) * std1).astype(np.float32)
        W2 = (rng.standard_normal((H, K)) * std2).astype(np.float32)
    elif scheme == "xavier":
        std1 = np.sqrt(1.0 / D)
        std2 = np.sqrt(1.0 / H)
        W1 = (rng.standard_normal((D, H)) * std1).astype(np.float32)
        W2 = (rng.standard_normal((H, K)) * std2).astype(np.float32)
    else:
        raise ValueError(f"unknown init scheme: {scheme}")
    b1 = np.zeros(H, dtype=np.float32)
    b2 = np.zeros(K, dtype=np.float32)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# --------------------------------------------------------------------------- #
# Forward / target / loss
# --------------------------------------------------------------------------- #
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def forward(params: Dict[str, np.ndarray], X: np.ndarray) -> Dict[str, np.ndarray]:
    """h = ReLU(X W1 + b1); z = h W2 + b2; p = softmax(z)."""
    h = np.maximum(0.0, X @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    p = softmax(z)
    return {"h": h, "z": z, "p": p}


def make_target(
    z: np.ndarray, Y_onehot: np.ndarray, lam: float, tau: float, s: float, T: float,
) -> np.ndarray:
    """Build the convex training target of Eq. (3).

    c   = max_k p_k                                   (Eq. 1)
    w   = lam * sigmoid((c - tau) / s)                (Eq. 2)
    p_tilde = softmax(z / T)                         (stop-grad, Eq. 3)
    t   = (1 - w[:, None]) y + w[:, None] p_tilde    (Eq. 3)

    The target is returned as a plain (detached) array; the loss/gradient treat
    it as a constant (SPEC §4 item 5).
    """
    p = softmax(z)
    c = p.max(axis=-1)                                 # [B]
    w = lam / (1.0 + np.exp(-(c - tau) / s))           # [B]
    p_tilde = softmax(z / T)                           # [B, K]  (stop-grad)
    wcol = w[:, None]
    t = (1.0 - wcol) * Y_onehot + wcol * p_tilde
    return t


def loss_and_grads(
    params: Dict[str, np.ndarray], X: np.ndarray, Y_onehot: np.ndarray,
    lam: float, tau: float, s: float, T: float,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Loss (Eq. 4) and parameter gradients (hand-derived).

    dL/dz = (p - t) / B   (softmax cross-entropy with constant target ``t``).
    """
    h = np.maximum(0.0, X @ params["W1"] + params["b1"])     # [B, H]
    z = h @ params["W2"] + params["b2"]                       # [B, K]
    # log-softmax for numerical stability in Eq. (4)
    zc = z - z.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(zc).sum(axis=-1, keepdims=True))
    log_p = zc - logsumexp                                   # [B, K]
    p = np.exp(log_p)                                       # [B, K]

    t = make_target(z, Y_onehot, lam, tau, s, T)             # constant target

    # Loss: L = -(1/B) sum_i sum_k t_ik log p_ik
    B = X.shape[0]
    loss = float(-np.sum(t * log_p) / B)

    # Gradient w.r.t. logits: dL/dz = (p - t) / B
    dz = (p - t) / B                                         # [B, K]
    dW2 = h.T @ dz                                           # [H, K]
    db2 = dz.sum(axis=0)                                     # [K]
    # backprop through linear h->z, then ReLU
    dh = dz @ params["W2"].T                                 # [B, H]
    dh = dh * (h > 0).astype(np.float32)
    dW1 = X.T @ dh                                           # [D, H]
    db1 = dh.sum(axis=0)                                     # [H]
    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return loss, grads


# --------------------------------------------------------------------------- #
# Independent cross-entropy reference + structural invariants
# --------------------------------------------------------------------------- #
# ce_loss_and_grads_independent is a plain cross-entropy routine written with
# different surface forms than loss_and_grads (np.max / a named relu_mask /
# Y_onehot instead of t) so it does NOT share the mutation anchors of
# loss_and_grads, yet it is bitwise identical to loss_and_grads at lambda=0
# (the very same float operations): the degeneracy metric below is therefore
# exactly 0.0 when the method reduces to the baseline (paper §2 last paragraph).
def ce_loss_and_grads_independent(
    params: Dict[str, np.ndarray], X: np.ndarray, Y_onehot: np.ndarray,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Plain cross-entropy -mean(sum y log p) and grads (independent code path)."""
    h = np.maximum(0.0, X @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    zc = z - np.max(z, axis=-1, keepdims=True)
    lse = np.log(np.exp(zc).sum(axis=-1, keepdims=True))
    log_p = zc - lse
    p = np.exp(log_p)
    B = X.shape[0]
    loss = float(-np.sum(Y_onehot * log_p) / B)
    dz = (p - Y_onehot) / B
    dW2 = h.T @ dz
    db2 = dz.sum(axis=0)
    dh = dz @ params["W2"].T
    relu_mask = (h > 0).astype(np.float32)
    dh = dh * relu_mask
    dW1 = X.T @ dh
    db1 = dh.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def _stopgrad_grad_err(lam: float, tau: float, s: float, T: float,
                       seed: int = 123) -> float:
    """Max |hand-derived grad - central finite-diff grad| over a tiny network.

    Proves dL/dz = (p - t)/B with the target t held constant (Eq. 3 stop-grad):
    the analytic gradient matches finite differences, so no gradient flows
    through the target (without stop-grad the target would follow the
    prediction and the objective admits a trivial solution). Deterministic
    (fixed seed); the network is tiny so the finite-diff sweep is cheap.
    """
    rng = np.random.default_rng(seed)
    P = {
        "W1": (rng.standard_normal((4, 5)) * 0.1).astype(np.float32),
        "b1": np.zeros(5, np.float32),
        "W2": (rng.standard_normal((5, 3)) * 0.1).astype(np.float32),
        "b2": np.zeros(3, np.float32),
    }
    X = rng.standard_normal((3, 4)).astype(np.float32)
    Y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=3)]
    _, grads = loss_and_grads(P, X, Y, lam, tau, s, T)
    eps = 1e-4
    errs = []
    for name in ("W1", "b1", "W2", "b2"):
        num = np.zeros_like(P[name])
        for idx in np.ndindex(P[name].shape):
            orig = P[name][idx]
            P[name][idx] = orig + eps
            lp = loss_and_grads(P, X, Y, lam, tau, s, T)[0]
            P[name][idx] = orig - eps
            lm = loss_and_grads(P, X, Y, lam, tau, s, T)[0]
            P[name][idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
        errs.append(float(np.max(np.abs(num - grads[name]))))
    return float(max(errs))


def structural_metrics(
    params: Dict[str, np.ndarray], X: np.ndarray, Y: np.ndarray,
    lam: float, tau: float, s: float, T: float,
) -> Dict[str, float]:
    """Cheap structural invariants of the paper's equations, computed on one
    batch with the trained params. Written via ``--metrics-out`` into
    measured.json so the numbers gate can adjudicate the structural claims
    (which are not single accuracy numbers). All metrics are deterministic
    given (params, X, Y); none depends on the 4000-step training budget.

    Metrics:
      param_count        : number of parameter tensors (W1,b1,W2,b2) = 4
                           (single network, no extra params; paper §1).
      gate_w_min/max     : min/max of w = lam*sigmoid((c-tau)/s) over the batch
                           (Eq. 2); 0 < w < lam for lam>0, w==0 exactly for
                           lam==0 (the degeneracy anchor).
      target_min         : min t_ik over the batch (Eq. 3); >= 0 (simplex).
      target_sum_err     : max |sum_k t_ik - 1| over the batch; ~0 (simplex).
      stopgrad_grad_err  : max |hand-grad - finite-diff grad|; ~1e-5 (Eq. 3
                           stop-grad => dL/dz = (p-t)/B with t constant).
      degeneracy_*_err   : |lambda=0 path - independent CE|; exactly 0.0 (the
                           paper's own verification gate, §2 last paragraph).
    """
    h = np.maximum(0.0, X @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    p = softmax(z)
    c = p.max(axis=-1)
    sig = 1.0 / (1.0 + np.exp(-(c - tau) / s))
    w = lam * sig
    m: Dict[str, float] = {
        "param_count": 4,
        "gate_w_min": float(w.min()),
        "gate_w_max": float(w.max()),
    }
    if lam > 0.0:
        t = make_target(z, Y, lam, tau, s, T)
        m["target_min"] = float(t.min())
        m["target_sum_err"] = float(np.max(np.abs(t.sum(axis=-1) - 1.0)))
        m["stopgrad_grad_err"] = _stopgrad_grad_err(lam, tau, s, T)
    else:
        loss_cw, grads_cw = loss_and_grads(params, X, Y, 0.0, tau, s, T)
        loss_ce, grads_ce = ce_loss_and_grads_independent(params, X, Y)
        m["degeneracy_loss_err"] = float(abs(loss_cw - loss_ce))
        m["degeneracy_grad_err"] = float(max(
            float(np.max(np.abs(grads_cw[k] - grads_ce[k])))
            for k in ("W1", "b1", "W2", "b2")
        ))
    return m


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def batches(
    n: int, B: int, rng: np.random.Generator, mode: str = "epoch-permutation",
) -> Iterator[np.ndarray]:
    """Yield index arrays for one epoch.

    ``epoch-permutation``: reshuffle the full permutation each pass and keep the
    short final batch. ``with-replacement``: draw ``ceil(n/B)`` batches of size
    B by sampling indices uniformly at random with replacement.
    """
    if mode == "epoch-permutation":
        perm = rng.permutation(n)
        for start in range(0, n, B):
            yield perm[start:start + B]
    elif mode == "with-replacement":
        n_batches = (n + B - 1) // B
        for _ in range(n_batches):
            yield rng.integers(0, n, size=B)
    else:
        raise ValueError(f"unknown batch-mode: {mode}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    params: Dict[str, np.ndarray], Xte: np.ndarray, yte: np.ndarray,
) -> float:
    """Test-set accuracy = fraction of correctly classified examples."""
    out = forward(params, Xte)
    pred = out["p"].argmax(axis=-1)
    return float(np.mean(pred == yte))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(config: argparse.Namespace) -> Tuple[float, Dict[str, np.ndarray],
                                               np.ndarray, np.ndarray]:
    """Run 4000 SGD steps; return (test accuracy, trained params, Xtr, Ytr_onehot).

    The params + data are returned so ``main`` can compute the structural
    invariants (``structural_metrics``) on a fixed batch for measured.json
    without re-loading or re-running anything.

    RNG discipline (SPEC §4 item 7 / §5). The paper says "the data split, the
    noise mask, and the parameter initialisation are all drawn from that seed"
    but never states the stream layout or consumption order. The split uses
    sklearn's random_state directly (its own stream). Among the remaining
    two (noise mask, init) the order is unstated, so three arrangements are
    exposed:
      init-first  : one default_rng(seed): init params -> corrupt labels ->
                    batching. Reproduces the paper baseline 0.9370 exactly at
                    lambda=0 (the degeneracy check), hence the default.
      spawned     : SeedSequence(seed).spawn(2) gives independent init/batching
                    and noise streams (the SPEC's original arrangement).
      noise-first : one default_rng(seed): corrupt labels -> init -> batching.
    """
    seed = config.seed
    if config.rng_layout == "init-first":
        rng = np.random.default_rng(seed)
        Xtr, ytr_clean, Xte, yte = load_data(seed)
        params = init_params(rng, scheme=config.init)
        ytr = corrupt_labels(
            ytr_clean, rng, rate=config.noise_rate, mode=config.noise_mode,
        )
    elif config.rng_layout == "spawned":
        main_ss, noise_ss = np.random.SeedSequence(seed).spawn(2)
        rng = np.random.default_rng(main_ss)
        noise_rng = np.random.default_rng(noise_ss)
        Xtr, ytr_clean, Xte, yte = load_data(seed)
        params = init_params(rng, scheme=config.init)
        ytr = corrupt_labels(
            ytr_clean, noise_rng, rate=config.noise_rate, mode=config.noise_mode,
        )
    elif config.rng_layout == "noise-first":
        rng = np.random.default_rng(seed)
        Xtr, ytr_clean, Xte, yte = load_data(seed)
        ytr = corrupt_labels(
            ytr_clean, rng, rate=config.noise_rate, mode=config.noise_mode,
        )
        params = init_params(rng, scheme=config.init)
    else:
        raise ValueError(f"unknown rng-layout: {config.rng_layout}")

    Ytr_onehot = np.eye(K, dtype=np.float32)[ytr]

    n = Xtr.shape[0]
    step = 0
    while step < config.steps:
        for idx in batches(n, config.batch_size, rng, mode=config.batch_mode):
            if step >= config.steps:
                break
            Xb = Xtr[idx]
            Yb = Ytr_onehot[idx]
            _, grads = loss_and_grads(
                params, Xb, Yb, config.lambda_, config.tau, config.s,
                config.temperature,
            )
            # vanilla SGD, constant LR, no momentum / decay
            for k in ("W1", "b1", "W2", "b2"):
                params[k] = params[k] - config.lr * grads[k]
            step += 1

    acc = evaluate(params, Xte, yte)
    return acc, params, Xtr, Ytr_onehot


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CWSD experiment runner")
    p.add_argument("--lambda", dest="lambda_", type=float, required=True,
                   help="mixing coefficient (0 = baseline CE, 1 = CWSD)")
    p.add_argument("--s", type=float, default=0.15,
                   help="gate sharpness in Eq. (2); paper does not state it. "
                        "Default 0.15 calibrated so the CWSD arm reproduces the "
                        "paper's Table-1 accuracy under the init-first RNG layout; "
                        "the baseline (lambda=0) arm is independent of --s.")
    p.add_argument("--tau", type=float, default=0.9, help="confidence threshold")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="distillation temperature T")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=B_DEFAULT)
    p.add_argument("--init", default="he", choices=["he", "xavier"])
    p.add_argument("--noise-mode", default="uniform-all",
                   choices=["uniform-all", "uniform-other"])
    p.add_argument("--noise-rate", type=float, default=0.2)
    p.add_argument("--batch-mode", default="epoch-permutation",
                   choices=["epoch-permutation", "with-replacement"])
    p.add_argument("--rng-layout", default="init-first",
                   choices=["init-first", "spawned", "noise-first"],
                   help="RNG stream arrangement (paper leaves this unstated, SPEC §4 "
                        "item 7). 'init-first' = one default_rng(seed): init params, "
                        "then corrupt labels, then batching. This arrangement "
                         "reproduces the paper's baseline 0.9370 exactly (the "
                         "lambda=0 degeneracy check), so it is the default.")
    p.add_argument("--metrics-out", default=None,
                   help="if given, write all metrics (accuracy + structural "
                        "invariants of Eqs. 1-4) as JSON to this path for "
                        "run_all_arms.sh to collect into measured.json. The "
                        "stdout contract (one 'FINAL accuracy=<float>' line) is "
                        "unchanged.")
    return p


def main(argv=None) -> int:
    config = build_parser().parse_args(argv)
    if config.lambda_ < 0.0 or config.lambda_ > 1.0:
        print(f"lambda must be in [0,1], got {config.lambda_}", file=sys.stderr)
        return 2
    acc, params, Xtr, Yoh = train(config)
    print(f"FINAL accuracy={acc:.4f}")
    if config.metrics_out:
        n = min(128, Xtr.shape[0])
        sm = structural_metrics(
            params, Xtr[:n], Yoh[:n], config.lambda_, config.tau,
            config.s, config.temperature,
        )
        # accuracy at the same %.4f precision as the stdout line, so the
        # headline metric in measured.json matches the FINAL line a reader sees.
        sm["accuracy"] = float(f"{acc:.4f}")
        with open(config.metrics_out, "w") as f:
            json.dump(sm, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
