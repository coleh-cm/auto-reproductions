#!/usr/bin/env python
"""experiments/m1_softmax.py — Milestone M1.

Train softmax regression on MNIST with clean SGD and evaluate:
  * clean test accuracy (``eval_clean``), and
  * FGSM attack at eps=0.25 (``eval_fgsm``) — paper target tex:333:
    "an error rate of 99.9\\% with an average confidence of [79.3\\%]".

Writes ``results/m1_softmax.json`` recording the milestone id, all
hyperparameters, seed, and the four ``AttackEval`` fields for the FGSM
evaluation, plus the clean accuracy and the grep-able paper target line.

Runnable::

    python experiments/m1_softmax.py
    python experiments/m1_softmax.py --seed 0 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPRO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from fgsm_repro.data import load_mnist  # noqa: E402
from fgsm_repro.models import SoftmaxRegression  # noqa: E402
from fgsm_repro.train import TrainConfig, train  # noqa: E402
from fgsm_repro.eval import eval_clean, eval_fgsm  # noqa: E402

# Paper target (tex:333).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:333",
    "quote": "an error rate of 99.9\\% with an average confidence of",
    "error_rate": 0.999,
    "mean_confidence_on_errors": 0.793,
}

# FGSM epsilon for M1 (tex:333 "using $\\eps=.25$").
EPS = 0.25


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M1: softmax regression on MNIST + FGSM eps=0.25.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--epochs", dest="epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))  # load_mnist appends /mnist
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "m1_softmax.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    data = load_mnist(Path(args.data_root), args.seed)

    # SoftmaxRegression init (nn.Linear) draws from the GLOBAL torch RNG, so
    # seed it before construction for bit-for-bit determinism across runs.
    torch.manual_seed(args.seed)
    model = SoftmaxRegression(in_dim=784, n_classes=10)

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        max_epochs=args.epochs,
        seed=args.seed,
        adv_train=False,
        eps=EPS,
        alpha=0.5,
        early_stop="clean",
        patience=args.patience,
    )

    result = train(model, cfg, data)
    if getattr(result, "best_state_dict", None) is not None:
        model.load_state_dict(result.best_state_dict)

    clean_acc = float(eval_clean(model, data.x_test, data.y_test))
    adv_eval = eval_fgsm(model, data.x_test, data.y_test, EPS)

    record = {
        "milestone": "M1",
        "description": "Softmax regression on MNIST; FGSM eps=0.25.",
        "seed": args.seed,
        "hyperparams": {
            "model": "SoftmaxRegression",
            "in_dim": 784,
            "n_classes": 10,
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "adv_train": False,
        },
        "eps": EPS,
        "clean_accuracy": clean_acc,
        "clean_error": 1.0 - clean_acc,
        "fgsm_eval": asdict(adv_eval),
        "paper_target": PAPER_TARGET,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"[M1] clean accuracy={clean_acc:.4f}  "
          f"fgsm error_rate={adv_eval.error_rate:.4f}  "
          f"fgsm conf_on_errors={adv_eval.mean_confidence_on_errors:.4f}  "
          f"(paper tex:333 target error 0.999, conf 0.793)")
    print(f"[M1] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
