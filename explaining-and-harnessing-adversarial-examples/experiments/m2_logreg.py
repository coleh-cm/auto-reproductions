#!/usr/bin/env python
"""experiments/m2_logreg.py — Milestone M2.

Logistic regression on the MNIST 3-vs-7 binary subset; FGSM eps=0.25.

Paper target (tex:454-456): clean test error 1.6% ("has a 1.6% error rate on the
3 versus 7 discrimination task"); FGSM adversarial error 99% (tex:456 "an error
rate of 99% on these examples").

For the linear model FGSM is EXACT (not an approximation): the per-example
input gradient of the softplus loss has sign ``-y*sign(w)`` (the sigmoid factor
is strictly positive), so the worst-case max-norm perturbation is
``eta = -eps*y*sign(w)`` (Algorithm C / E6, tex:407-411). We use this exact form
via ``attacks.fgsm_logreg``.

Training uses a plain SGD loop on the softplus loss E5 (tex:401-404) — NOT the
shared ``train()`` (which assumes softmax cross-entropy with 0..K-1 labels).

Writes ``results/m2_logreg.json``.

Runnable::

    python experiments/m2_logreg.py
    python experiments/m2_logreg.py --seed 0 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPRO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from fgsm_repro.data import load_mnist_3v7  # noqa: E402
from fgsm_repro.models import LogisticRegression  # noqa: E402
from fgsm_repro.objectives import softplus_logreg_cost  # noqa: E402
from fgsm_repro.attacks import fgsm_logreg  # noqa: E402
from fgsm_repro.eval import eval_clean  # noqa: E402

# Paper target (tex:454-456).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:454-456",
    "quote": "has a 1.6% error rate on the 3 versus 7 ... an error rate of 99%",
    "clean_error": 0.016,
    "adv_error": 0.99,
}

EPS = 0.25  # tex:333 (MNIST FGSM eps).


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M2: logistic regression 3-vs-7 + FGSM eps=0.25.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--epochs", dest="epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "m2_logreg.json"))
    return p


def _train_logreg(model: LogisticRegression, data, args) -> dict:
    """Plain SGD on the softplus loss E5 (binary y in {-1,+1})."""
    opt = torch.optim.SGD([model.w, model.b], lr=args.lr, momentum=args.momentum)
    gen = torch.Generator().manual_seed(args.seed)
    x_train, y_train = data.x_train, data.y_train
    x_valid, y_valid = data.x_valid, data.y_valid
    n = int(x_train.size(0))
    y_train_f = y_train.to(torch.float32)
    y_valid_f = y_valid.to(torch.float32)

    best_acc = -1.0
    best_state = None
    patience = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        loss_sum, loss_cnt = 0.0, 0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb = x_train[idx]
            yb = y_train_f[idx]
            loss = softplus_logreg_cost(model.w, model.b, xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * int(xb.size(0))
            loss_cnt += int(xb.size(0))
        train_loss = loss_sum / max(loss_cnt, 1)
        model.eval()
        valid_acc = float(eval_clean(model, x_valid, y_valid))
        history.append({"train_loss": train_loss, "valid_acc": valid_acc})
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_valid_acc": best_acc, "history": history}


def _binary_fgsm_eval(model: LogisticRegression, x: torch.Tensor, y: torch.Tensor,
                      eps: float) -> dict:
    """FGSM eval for binary logistic regression (y in {-1,+1}).

    Perturbation is the exact worst case eta = -eps*y*sign(w). Prediction is
    the logistic decision rule sign(score). Confidence is the predicted-class
    probability: for y_pred=+1, p = sigmoid(score); for y_pred=-1,
    p = 1 - sigmoid(score) = sigmoid(-score). Average confidence is over the
    misclassified subset only (paper convention, SPEC sec6 item 14).
    """
    y_f = y.to(torch.float32)
    x_adv = fgsm_logreg(model.w, model.b, x, y_f, eps)
    model.eval()
    with torch.no_grad():
        s = model.score(x_adv)                       # [B]
        pred = torch.where(s > 0, torch.ones_like(y_f), -torch.ones_like(y_f))
        p_pos = torch.sigmoid(s)
        conf = torch.where(pred > 0, p_pos, 1.0 - p_pos)
    wrong = pred != y_f
    error_rate = wrong.float().mean().item()
    n = int(pred.numel())
    if wrong.sum() > 0:
        mean_conf = conf[wrong].mean().item()
    else:
        mean_conf = 0.0
    return {"error_rate": error_rate, "mean_confidence_on_errors": mean_conf, "n": n}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    data = load_mnist_3v7(Path(args.data_root))

    torch.manual_seed(args.seed)
    model = LogisticRegression(in_dim=784)

    train_info = _train_logreg(model, data, args)

    clean_acc = float(eval_clean(model, data.x_test, data.y_test))
    adv_eval = _binary_fgsm_eval(model, data.x_test, data.y_test, EPS)

    record = {
        "milestone": "M2",
        "description": "Logistic regression on MNIST 3-vs-7; FGSM eps=0.25 (exact).",
        "seed": args.seed,
        "hyperparams": {
            "model": "LogisticRegression",
            "in_dim": 784,
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "eps": EPS,
            "fgsm": "exact closed-form eta=-eps*y*sign(w) (tex:407-411)",
        },
        "clean_accuracy": clean_acc,
        "clean_error": 1.0 - clean_acc,
        "fgsm_eval": adv_eval,
        "train_info": {"best_valid_acc": train_info["best_valid_acc"]},
        "paper_target": PAPER_TARGET,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"[M2] clean error={1.0 - clean_acc:.4f}  "
          f"adv error={adv_eval['error_rate']:.4f}  "
          f"adv conf={adv_eval['mean_confidence_on_errors']:.4f}  "
          f"(paper tex:454-456 target clean 1.6% / adv 99%)")
    print(f"[M2] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
