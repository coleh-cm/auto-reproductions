#!/usr/bin/env python
"""experiments/m4_adversarial.py — Milestone M4.

Train a MaxoutMLP(units=240, pieces=5) on MNIST under two arms:

  * **baseline**   (``adv_train=False``)                — clean SGD only.
  * **adversarial** (``adv_train=True, eps=0.25, alpha=0.5``) — FGSM adversarial
    training (Algorithm B, tex:486-488).

Report the clean test accuracy for both arms. Paper target tex:492-494:
"from 0.94\\% without adversarial training to 0.84\\%" — i.e. clean test error
should drop slightly (0.94% -> 0.84%) with adversarial training.

Writes ``results/m4_adversarial.json`` recording the milestone id, all
hyperparameters, seed, both arms' clean test accuracy/error, and the
grep-able paper target line.

Runnable::

    python experiments/m4_adversarial.py
    python experiments/m4_adversarial.py --seed 0 --epochs 20
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

from fgsm_repro.data import load_mnist  # noqa: E402
from fgsm_repro.models import MaxoutMLP  # noqa: E402
from fgsm_repro.train import TrainConfig, train  # noqa: E402
from fgsm_repro.eval import eval_clean  # noqa: E402

# Paper target (tex:492-494).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:492-494",
    "quote": "from 0.94\\% without adversarial training to 0.84\\%",
    "baseline_test_error": 0.0094,
    "adversarial_test_error": 0.0084,
}

# Adversarial-training hyperparameters (tex:488 alpha=0.5; tex:428-429 eps=0.25).
EPS = 0.25
ALPHA = 0.5

# External maxout defaults (lisa-lab/pylearn2 mnist_pi.yaml; documented as
# external, not paper-stated — see SPEC.md section 6). The recipe's dropout
# (input_include_probs: {h0: .8}, input_scales: {h0: 1.}) applies dropout to
# the INPUT only (include 0.8); there is NO hidden / readout-input dropout.
# We therefore set input include 0.8 and hidden include 1.0 (off) to match the
# recipe, rather than the earlier 0.8/0.5 which added an unrecipe'd hidden
# dropout site.
DEFAULT_UNITS = 240
DEFAULT_PIECES = 5
DEFAULT_DROPOUT_INPUT = 0.8
DEFAULT_DROPOUT_HIDDEN = 1.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M4: MaxoutMLP(240) baseline vs FGSM adversarial training.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--units", type=int, default=DEFAULT_UNITS)
    p.add_argument("--pieces", type=int, default=DEFAULT_PIECES)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--epochs", dest="epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--dropout-input", dest="dropout_input", type=float,
                   default=DEFAULT_DROPOUT_INPUT)
    p.add_argument("--dropout-hidden", dest="dropout_hidden", type=float,
                   default=DEFAULT_DROPOUT_HIDDEN)
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))  # load_mnist appends /mnist
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "m4_adversarial.json"))
    return p


def _train_arm(name: str, adv_train: bool, data, args) -> dict:
    # MaxoutMLP weight init draws from the GLOBAL torch RNG, so seed it before
    # constructing each arm. Re-seeding for every arm gives both arms identical
    # initial weights (a controlled comparison) and bit-for-bit reproducibility
    # across runs (train() uses a separate generator for shuffling/dropout, so
    # the global RNG is untouched during training).
    torch.manual_seed(args.seed)
    model = MaxoutMLP(
        units=args.units,
        pieces=args.pieces,
        in_dim=784,
        n_classes=10,
        dropout_input_include=args.dropout_input,
        dropout_hidden_include=args.dropout_hidden,
        seed=args.seed,
    )
    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        max_epochs=args.epochs,
        seed=args.seed,
        adv_train=adv_train,
        eps=EPS,
        alpha=ALPHA,
        early_stop="clean",
        patience=args.patience,
    )
    result = train(model, cfg, data)
    if getattr(result, "best_state_dict", None) is not None:
        model.load_state_dict(result.best_state_dict)
    acc = float(eval_clean(model, data.x_test, data.y_test))
    print(f"[M4/{name}] clean test accuracy={acc:.4f} "
          f"(error={1.0 - acc:.4f})")
    return {
        "arm": name,
        "adv_train": adv_train,
        "eps": EPS if adv_train else 0.0,
        "alpha": ALPHA,
        "clean_accuracy": acc,
        "clean_error": 1.0 - acc,
        "epochs_run": getattr(result, "epochs_run", None),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    data = load_mnist(Path(args.data_root), args.seed)

    baseline = _train_arm("baseline", False, data, args)
    adversarial = _train_arm("adversarial", True, data, args)

    record = {
        "milestone": "M4",
        "description": "MaxoutMLP(240) baseline vs FGSM adversarial training "
                       "(eps=0.25, alpha=0.5).",
        "seed": args.seed,
        "hyperparams": {
            "model": "MaxoutMLP",
            "units": args.units,
            "pieces": args.pieces,
            "in_dim": 784,
            "n_classes": 10,
            "dropout_input_include": args.dropout_input,
            "dropout_hidden_include": args.dropout_hidden,
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "early_stop": "clean",
        },
        "arms": {"baseline": baseline, "adversarial": adversarial},
        "paper_target": PAPER_TARGET,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"[M4] baseline error={baseline['clean_error']:.4f} "
          f"adversarial error={adversarial['clean_error']:.4f} "
          f"(paper tex:492-494 target 0.0094 -> 0.0084)")
    print(f"[M4] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
