#!/usr/bin/env python
"""experiments/m3_maxout_fgsm.py — Milestone M3.

Maxout 240x2 + input dropout on MNIST, trained clean; FGSM eps=0.25 evaluation.

Paper target (tex:338-339): "a maxout network misclassifies 89.4% of our
adversarial examples with an average confidence of 97.6%".

Writes ``results/m3_maxout_fgsm.json``.

Runnable::

    python experiments/m3_maxout_fgsm.py
    python experiments/m3_maxout_fgsm.py --epochs 20
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
from fgsm_repro.models import MaxoutMLP  # noqa: E402
from fgsm_repro.train import TrainConfig, train  # noqa: E402
from fgsm_repro.eval import eval_clean, eval_fgsm  # noqa: E402

# Paper target (tex:338-339).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:338-339",
    "quote": "a maxout network misclassifies 89.4% of our adversarial examples "
             "with an average confidence of 97.6%",
    "error_rate": 0.894,
    "mean_confidence_on_errors": 0.976,
}

EPS = 0.25  # tex:333 (MNIST FGSM eps).

# External maxout defaults (pylearn2 mnist_pi.yaml; recipe dropout is INPUT
# only, include 0.8 -- no hidden dropout).
DEFAULT_UNITS = 240
DEFAULT_DROPOUT_INPUT = 0.8
DEFAULT_DROPOUT_HIDDEN = 1.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M3: maxout 240x2 + dropout, FGSM eps=0.25.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--units", type=int, default=DEFAULT_UNITS)
    p.add_argument("--pieces", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--epochs", dest="epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "m3_maxout_fgsm.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    data = load_mnist(Path(args.data_root), args.seed)

    torch.manual_seed(args.seed)
    model = MaxoutMLP(
        units=args.units, pieces=args.pieces, in_dim=784, n_classes=10,
        dropout_input_include=DEFAULT_DROPOUT_INPUT,
        dropout_hidden_include=DEFAULT_DROPOUT_HIDDEN, seed=args.seed,
    )
    cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        max_epochs=args.epochs, seed=args.seed, adv_train=False, eps=EPS,
        alpha=0.5, early_stop="clean", patience=args.patience,
    )
    result = train(model, cfg, data)
    if getattr(result, "best_state_dict", None) is not None:
        model.load_state_dict(result.best_state_dict)

    clean_acc = float(eval_clean(model, data.x_test, data.y_test))
    adv_eval = eval_fgsm(model, data.x_test, data.y_test, EPS)

    # The paper's number comes from a fully-converged net (patience 100).
    # Our default epochs/patience are sub-scale for CPU feasibility.
    sub_scale = args.patience < 100 or args.epochs < 100

    record = {
        "milestone": "M3",
        "description": "Maxout 240x2 + input dropout on MNIST; FGSM eps=0.25.",
        "seed": args.seed,
        "hyperparams": {
            "model": "MaxoutMLP", "units": args.units, "pieces": args.pieces,
            "in_dim": 784, "n_classes": 10,
            "dropout_input_include": DEFAULT_DROPOUT_INPUT,
            "dropout_hidden_include": DEFAULT_DROPOUT_HIDDEN,
            "lr": args.lr, "momentum": args.momentum, "batch_size": args.batch_size,
            "max_epochs": args.epochs, "patience": args.patience,
            "early_stop": "clean", "epochs_run": result.epochs_run,
        },
        "eps": EPS,
        "clean_accuracy": clean_acc,
        "clean_error": 1.0 - clean_acc,
        "fgsm_eval": asdict(adv_eval),
        "paper_target": PAPER_TARGET,
        "note": ("Sub-scale run (epochs=%d, patience=%d); the paper's 89.4 pct "
                 "comes from a fully-converged net (patience 100)."
                 % (args.epochs, args.patience)) if sub_scale else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"[M3] clean acc={clean_acc:.4f}  "
          f"fgsm error={adv_eval.error_rate:.4f}  "
          f"fgsm conf={adv_eval.mean_confidence_on_errors:.4f}  "
          f"(paper tex:338-339 target 89.4% / 97.6%)")
    print(f"[M3] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
