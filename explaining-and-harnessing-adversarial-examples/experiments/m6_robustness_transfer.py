#!/usr/bin/env python
"""experiments/m6_robustness_transfer.py — Milestone M6.

Robustness & cross-transfer of an adversarially-trained maxout model.

Paper target (tex:514-523): own-FGSM error 17.9%; adversarial examples from the
ORIGINAL (non-adv-trained) model yield 19.6% on the adv-trained model; adv
examples from the NEW (adv-trained) model yield 40.9% on the original model;
mean confidence on misclassified (own-FGSM, adv model) 81.4%.

Writes ``results/m6_robustness_transfer.json``.

Runnable::

    python experiments/m6_robustness_transfer.py
    python experiments/m6_robustness_transfer.py --epochs 20
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
from fgsm_repro.eval import eval_clean, eval_fgsm, eval_transfer  # noqa: E402

# Paper target (tex:514-523).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:514-523",
    "quote": "fell to 17.9% ... 19.6% ... 40.9% ... average confidence ... 81.4%",
    "own_fgsm_error": 0.179,
    "orig_to_adv_transfer_error": 0.196,
    "adv_to_orig_transfer_error": 0.409,
    "mean_confidence_on_misclassified": 0.814,
}

EPS = 0.25
ALPHA = 0.5
DEFAULT_UNITS = 240
DEFAULT_DROPOUT_INPUT = 0.8
DEFAULT_DROPOUT_HIDDEN = 1.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M6: robustness & transfer of adv-trained maxout.")
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
                   default=str(REPRO_ROOT / "results" / "m6_robustness_transfer.json"))
    return p


def _train_arm(adv_train: bool, data, args) -> MaxoutMLP:
    torch.manual_seed(args.seed)
    model = MaxoutMLP(
        units=args.units, pieces=args.pieces, in_dim=784, n_classes=10,
        dropout_input_include=DEFAULT_DROPOUT_INPUT,
        dropout_hidden_include=DEFAULT_DROPOUT_HIDDEN, seed=args.seed,
    )
    cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        max_epochs=args.epochs, seed=args.seed, adv_train=adv_train, eps=EPS,
        alpha=ALPHA, early_stop="clean", patience=args.patience,
    )
    res = train(model, cfg, data)
    if getattr(res, "best_state_dict", None) is not None:
        model.load_state_dict(res.best_state_dict)
    return model


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = load_mnist(Path(args.data_root), args.seed)

    m_orig = _train_arm(False, data, args)   # original (no adv training)
    m_adv = _train_arm(True, data, args)      # new (adv trained)

    x_test, y_test = data.x_test, data.y_test

    # own-FGSM error & confidence on the adversarially-trained model (17.9% / 81.4%)
    own = eval_fgsm(m_adv, x_test, y_test, EPS)
    # original -> adversarial transfer (19.6%)
    orig_to_adv = eval_transfer(m_orig, m_adv, x_test, y_test, EPS)
    # adversarial -> original transfer (40.9%)
    adv_to_orig = eval_transfer(m_adv, m_orig, x_test, y_test, EPS)

    orig_clean = float(eval_clean(m_orig, x_test, y_test))
    adv_clean = float(eval_clean(m_adv, x_test, y_test))

    # Honest-reporting flags (same schema as e1_ensemble.json). The paper's §6
    # robustness claim (tex:514-523) is the ASYMMETRY orig->adv (19.6%) <
    # adv->orig (40.9%) -- adv-trained examples transfer HARDER to the naive
    # model than naive examples do to the adv model -- plus the adv model's own
    # FGSM error (17.9%) being far below the untrained ~89.4% (tex:514, 338-339).
    # `direction_matches_paper` is true iff that asymmetry holds AND the adv arm
    # actually trained (clean accuracy well above the 11.35% MNIST majority-class
    # constant predictor -- a collapsed arm gives a vacuous 1-majority-class
    # number that carries no signal about tex:514-523).
    MAJORITY_CLASS_RATE = 0.1135  # MNIST test P(label=1)=1135/10000
    adv_arm_trained = adv_clean > 0.5 and abs(adv_clean - MAJORITY_CLASS_RATE) > 0.05
    asymmetry_holds = orig_to_adv.error_rate < adv_to_orig.error_rate
    direction_matches = adv_arm_trained and asymmetry_holds
    training_note = (
        "adversarial arm trained healthily (clean_accuracy=%.4f, well above the "
        "11.35%% MNIST majority-class constant predictor); measurements are real, "
        "not a collapsed-arm artifact." % adv_clean
    ) if adv_arm_trained else (
        "adversarial arm COLLAPSED (clean_accuracy=%.4f ~ the 11.35%% majority-class "
        "rate): the own-FGSM / transfer numbers are properties of a constant "
        "predictor and carry NO signal about tex:514-523; re-run at more epochs."
        % adv_clean
    )

    sub_scale = args.patience < 100 or args.epochs < 100 or args.units < 1600
    record = {
        "milestone": "M6",
        "description": "Robustness & cross-transfer of an adv-trained maxout "
                       "(own-FGSM, orig->adv, adv->orig, confidence).",
        "seed": args.seed,
        "hyperparams": {
            "model": "MaxoutMLP", "units": args.units, "pieces": args.pieces,
            "lr": args.lr, "momentum": args.momentum, "batch_size": args.batch_size,
            "max_epochs": args.epochs, "patience": args.patience,
            "eps": EPS, "alpha": ALPHA, "dropout_input": DEFAULT_DROPOUT_INPUT,
        },
        "arms": {
            "original": {"clean_accuracy": orig_clean},
            "adversarial": {"clean_accuracy": adv_clean},
        },
        "own_fgsm_on_adv_model": asdict(own),
        "transfer_orig_to_adv": asdict(orig_to_adv),
        "transfer_adv_to_orig": asdict(adv_to_orig),
        "direction_matches_paper": direction_matches,
        "adv_arm_trained": adv_arm_trained,
        "transfer_asymmetry_holds": asymmetry_holds,
        "paper_target": PAPER_TARGET,
        "note": ("Sub-scale run (units=%d, epochs=%d, patience=%d). The paper's "
                 "17.9/19.6/40.9/81.4 pct are from a 1600-unit adv-trained model. "
                 "%s"
                 % (args.units, args.epochs, args.patience, training_note))
                 if sub_scale else training_note,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"[M6] own_fgsm={own.error_rate:.4f} (conf {own.mean_confidence_on_errors:.4f})  "
          f"orig->adv={orig_to_adv.error_rate:.4f}  "
          f"adv->orig={adv_to_orig.error_rate:.4f}  "
          f"(paper tex:514-523 17.9/19.6/40.9/81.4%)")
    print(f"[M6] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
