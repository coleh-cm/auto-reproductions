#!/usr/bin/env python
"""experiments/m5_large_advtrain.py — Milestone M5.

Large maxout 1600x2: baseline vs adversarial training, with epoch selection on
the ADVERSARIAL validation error, then retrain on all 60,000 examples, averaged
over 5 seeds.

Paper target (tex:497-512): without adversarial training 1.14% test error; with
adversarial training (adversarial-valid early stopping, retrain on 60k, 5 seeds)
-> 4x0.77% + 1x0.83%, mean 0.782%.

Protocol (faithful to tex:501-512):
  A. SELECT epochs on the 50k/10k split with early stopping on the
     ADVERSARIAL validation error (tex:503-505).
  B. RETRAIN on all 60,000 examples for the chosen epoch count (tex:506).
  C. Average test error over 5 seeds (tex:506-510).

The full paper scale (1600 units x patience-100 x 5 seeds x retrain) is
infeasible on this CPU. Defaults are a documented sub-scale; the CLI exposes
the full-scale knobs.

Writes ``results/m5_large_advtrain.json``.

Runnable::

    python experiments/m5_large_advtrain.py
    python experiments/m5_large_advtrain.py --seeds 0,1,2,3,4 --units 1600 --epochs 100 --patience 100
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

from fgsm_repro.data import load_mnist, load_mnist_full_train  # noqa: E402
from fgsm_repro.models import MaxoutMLP  # noqa: E402
from fgsm_repro.train import TrainConfig, train  # noqa: E402
from fgsm_repro.eval import eval_clean  # noqa: E402

# Paper target (tex:497-512).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:497-512",
    "quote": "1600 units per layer ... 0.94% without adversarial training to "
             "0.84% ... 1.14% ... 4 trials 0.77% + 1 trial 0.83% ... 0.782%",
    "baseline_test_error": 0.0114,
    "adversarial_test_error_mean": 0.00782,
}

EPS = 0.25
ALPHA = 0.5
DEFAULT_UNITS = 1600
DEFAULT_DROPOUT_INPUT = 0.8
DEFAULT_DROPOUT_HIDDEN = 1.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M5: large maxout adv-training (adversarial-valid early stop + 60k retrain + multi-seed).")
    p.add_argument("--seeds", type=str, default="0",
                   help="comma-separated seeds (paper: 5 seeds)")
    p.add_argument("--units", type=int, default=DEFAULT_UNITS)
    p.add_argument("--pieces", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--epochs", dest="epochs", type=int, default=10,
                   help="max epochs for the epoch-selection phase")
    p.add_argument("--patience", type=int, default=10,
                   help="early-stop patience (paper: 100)")
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "m5_large_advtrain.json"))
    return p


def _select_and_retrain(adv_train: bool, seed: int, args) -> dict:
    """Phase A: select epochs on 50k/10k. Phase B: retrain on 60k for that many."""
    # --- Phase A: epoch selection on the 50k/10k split ---
    data = load_mnist(Path(args.data_root), seed)
    torch.manual_seed(seed)
    model = MaxoutMLP(
        units=args.units, pieces=args.pieces, in_dim=784, n_classes=10,
        dropout_input_include=DEFAULT_DROPOUT_INPUT,
        dropout_hidden_include=DEFAULT_DROPOUT_HIDDEN, seed=seed,
    )
    # The adversarial arm selects on adversarial-valid error (tex:503-505);
    # the baseline arm selects on clean-valid error (the original maxout recipe,
    # tex:501-503). eps=0.25, alpha=0.5 for the adv arm.
    select_cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        max_epochs=args.epochs, seed=seed, adv_train=adv_train, eps=EPS,
        alpha=ALPHA, early_stop="adversarial" if adv_train else "clean",
        patience=args.patience,
    )
    res_sel = train(model, select_cfg, data)
    epochs_chosen = res_sel.epochs_run

    # --- Phase B: retrain on all 60,000 examples for epochs_chosen ---
    # (tex:506 "retrained on all 60,000 examples"). The full-train loader
    # returns an EMPTY valid set (no leak); we train for the fixed chosen epoch
    # count with NO early stopping and keep the FINAL state (no best_state).
    data_full = load_mnist_full_train(Path(args.data_root), seed)
    torch.manual_seed(seed)
    model2 = MaxoutMLP(
        units=args.units, pieces=args.pieces, in_dim=784, n_classes=10,
        dropout_input_include=DEFAULT_DROPOUT_INPUT,
        dropout_hidden_include=DEFAULT_DROPOUT_HIDDEN, seed=seed,
    )
    retrain_cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        max_epochs=epochs_chosen, seed=seed, adv_train=adv_train, eps=EPS,
        alpha=ALPHA, early_stop="clean", patience=10 ** 9,
    )
    res_re = train(model2, retrain_cfg, data_full)
    # With an empty valid set, best_state_dict is the final state (no epoch
    # ever improves on +inf except epoch 0; load it to be safe).
    if getattr(res_re, "best_state_dict", None) is not None:
        model2.load_state_dict(res_re.best_state_dict)
    test_acc = float(eval_clean(model2, data_full.x_test, data_full.y_test))
    return {
        "seed": seed,
        "epochs_chosen": epochs_chosen,
        "test_accuracy": test_acc,
        "test_error": 1.0 - test_acc,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    baseline_results = []
    adversarial_results = []
    for seed in seeds:
        print(f"[M5] seed={seed} baseline arm...", file=sys.stderr)
        baseline_results.append(_select_and_retrain(False, seed, args))
        print(f"[M5] seed={seed} adversarial arm...", file=sys.stderr)
        adversarial_results.append(_select_and_retrain(True, seed, args))

    def _mean(rs):
        errs = [r["test_error"] for r in rs]
        return sum(errs) / len(errs) if errs else 0.0

    baseline_mean = _mean(baseline_results)
    adv_mean = _mean(adversarial_results)
    sub_scale = (args.patience < 100) or (len(seeds) < 5) or (args.units < 1600)

    record = {
        "milestone": "M5",
        "description": "Large maxout 1600x2: adv-valid early stop + 60k retrain "
                       "+ multi-seed mean (baseline vs adversarial).",
        "seeds": seeds,
        "hyperparams": {
            "model": "MaxoutMLP", "units": args.units, "pieces": args.pieces,
            "in_dim": 784, "n_classes": 10,
            "dropout_input_include": DEFAULT_DROPOUT_INPUT,
            "dropout_hidden_include": DEFAULT_DROPOUT_HIDDEN,
            "lr": args.lr, "momentum": args.momentum, "batch_size": args.batch_size,
            "select_max_epochs": args.epochs, "patience": args.patience,
            "eps": EPS, "alpha": ALPHA,
            "select_criterion_adv": "adversarial-valid", "select_criterion_base": "clean-valid",
        },
        "arms": {
            "baseline": {"per_seed": baseline_results, "mean_test_error": baseline_mean},
            "adversarial": {"per_seed": adversarial_results, "mean_test_error": adv_mean},
        },
        "paper_target": PAPER_TARGET,
        "note": ("Sub-scale run (units=%d, patience=%d, seeds=%d). The paper's "
                 "0.782 pct mean uses 1600 units, patience-100, 5 seeds."
                 % (args.units, args.patience, len(seeds))) if sub_scale else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"[M5] baseline mean error={baseline_mean:.4f}  "
          f"adversarial mean error={adv_mean:.4f}  "
          f"(paper tex:497-512 target 1.14% -> 0.782%)")
    print(f"[M5] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
