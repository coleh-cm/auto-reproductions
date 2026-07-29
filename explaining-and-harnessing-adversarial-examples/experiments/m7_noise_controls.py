#!/usr/bin/env python
"""experiments/m7_noise_controls.py — Milestone M7.

Noise-training controls on maxout: the paper's CONTROL for FGSM adversarial
training (training on RANDOM additive noise instead of FGSM adversarials).

Paper target (tex:555-557): training with +/-eps pixel flips -> FGSM error 86.2%
(conf 97.3%); training with U(-eps,eps) noise -> FGSM error 90.4% (conf 97.8%).
(For comparison FGSM adversarial training drives this down to 17.9% — M6 — so
noise training is a much WEAKER regularizer, which is the paper's point.)

The noise arms train on NOISE-ONLY batches (every example perturbed by random
noise of max-norm <= eps, regenerated per batch) -- the prose reading of tex:555-
556 ("we trained a maxout network with noise based on randomly adding +/-eps
..."); NO clean/noise alpha-mixture is stated by the paper (SPEC §6 item 27).

Writes ``results/m7_noise_controls.json``.

Runnable::

    python experiments/m7_noise_controls.py
    python experiments/m7_noise_controls.py --epochs 20
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

# Paper target (tex:555-557).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:555-557",
    "quote": "randomly adding $\\pm\\eps$ to each pixel, or adding noise in "
             "$U(-\\eps, \\eps)$ ... 86.2% ... 97.3% ... 90.4% ... 97.8%",
    "bernoulli_fgsm_error": 0.862,
    "bernoulli_fgsm_conf": 0.973,
    "uniform_fgsm_error": 0.904,
    "uniform_fgsm_conf": 0.978,
}

EPS = 0.25
ALPHA = 0.5
DEFAULT_UNITS = 240
DEFAULT_DROPOUT_INPUT = 0.8
DEFAULT_DROPOUT_HIDDEN = 1.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M7: noise-training controls (bernoulli/uniform).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--units", type=int, default=DEFAULT_UNITS)
    p.add_argument("--pieces", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--epochs", dest="epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "m7_noise_controls.json"))
    return p


def _train_arm(noise_train, data, args) -> MaxoutMLP:
    torch.manual_seed(args.seed)
    model = MaxoutMLP(
        units=args.units, pieces=args.pieces, in_dim=784, n_classes=10,
        dropout_input_include=DEFAULT_DROPOUT_INPUT,
        dropout_hidden_include=DEFAULT_DROPOUT_HIDDEN, seed=args.seed,
    )
    cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        max_epochs=args.epochs, seed=args.seed, adv_train=False, eps=EPS,
        alpha=ALPHA, early_stop="clean", patience=args.patience,
        noise_train=noise_train,  # None -> clean baseline; "bernoulli"/"uniform" -> M7 noise-only
    )
    res = train(model, cfg, data)
    if getattr(res, "best_state_dict", None) is not None:
        model.load_state_dict(res.best_state_dict)
    return model


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = load_mnist(Path(args.data_root), args.seed)
    x_test, y_test = data.x_test, data.y_test

    arms = {}
    for noise in (None, "bernoulli", "uniform"):
        label = "baseline" if noise is None else noise
        model = _train_arm(noise, data, args)
        clean_acc = float(eval_clean(model, x_test, y_test))
        fgsm = eval_fgsm(model, x_test, y_test, EPS)
        arms[label] = {
            "clean_accuracy": clean_acc,
            "fgsm_eval": asdict(fgsm),
        }

    sub_scale = args.patience < 100 or args.epochs < 100
    record = {
        "milestone": "M7",
        "description": "Noise-training controls (+/-eps flips, U(-eps,eps)) on "
                       "maxout (NOISE-ONLY batches, no clean mixture, SPEC §6 item "
                       "27); FGSM eps=0.25 eval after each.",
        "seed": args.seed,
        "hyperparams": {
            "model": "MaxoutMLP", "units": args.units, "pieces": args.pieces,
            "lr": args.lr, "momentum": args.momentum, "batch_size": args.batch_size,
            "max_epochs": args.epochs, "patience": args.patience,
            "eps": EPS, "alpha": ALPHA, "dropout_input": DEFAULT_DROPOUT_INPUT,
            "noise_train_form": "noise-only batches (no clean mixture)",
        },
        "arms": arms,
        "paper_target": PAPER_TARGET,
        "note": ("Sub-scale run (epochs=%d, patience=%d). The paper's 86.2/90.4 pct "
                 "are from fully-converged maxout nets." % (args.epochs, args.patience))
                 if sub_scale else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    b = arms["bernoulli"]["fgsm_eval"]
    u = arms["uniform"]["fgsm_eval"]
    print(f"[M7] bernoulli fgsm err={b['error_rate']:.4f} conf={b['mean_confidence_on_errors']:.4f} ; "
          f"uniform fgsm err={u['error_rate']:.4f} conf={u['mean_confidence_on_errors']:.4f} "
          f"(paper tex:555-557 86.2/90.4%)")
    print(f"[M7] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
