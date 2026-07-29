#!/usr/bin/env python
"""experiments/m8_rbf.py — Milestone M8.

Shallow RBF network on MNIST; FGSM eps=0.25; cross-model class agreement
(section 8).

Paper target (tex:600-604): FGSM adv error 55.4%, confidence-on-mistakes 1.2%,
clean confidence 60.6%. Section 8 (tex:679-688): RBF predicts maxout's class
16.0% over m1's errors (54.3% over both-wrong); softmax predicts maxout's class
54.6% over m1's errors (84.6% over both-wrong); RBF predicts softmax's class
53.6% over m1's errors.

The paper notes quadratic models are hard to train with SGD (tex:616-617), so
the RBF net may underfit at sub-scale; the result is recorded with a note.

Writes ``results/m8_rbf.json``.

Runnable::

    python experiments/m8_rbf.py
    python experiments/m8_rbf.py --epochs 20
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
from fgsm_repro.models import MaxoutMLP, SoftmaxRegression, RBFNet  # noqa: E402
from fgsm_repro.train import TrainConfig, train  # noqa: E402
from fgsm_repro.eval import (  # noqa: E402
    eval_clean, eval_fgsm_rbf, eval_clean_confidence_rbf, class_agreement,
)

# Paper target (tex:600-604, 679-688).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:600-604,679-688",
    "quote": "55.4% ... 1.2% ... 60.6% ... 16.0% / 54.6% / 84.6% / 54.3% / 53.6%",
    "rbf_fgsm_error": 0.554,
    "rbf_conf_on_mistakes": 0.012,
    "rbf_clean_confidence": 0.606,
    "agreement": {"rbf_pred_maxout_over_m1_errors": 0.160,
                  "soft_pred_maxout_over_m1_errors": 0.546,
                  "soft_pred_maxout_over_both_wrong": 0.846,
                  "rbf_pred_maxout_over_both_wrong": 0.543,
                  "rbf_pred_soft_over_m1_errors": 0.536},
}

EPS = 0.25
DEFAULT_UNITS = 240
DEFAULT_DROPOUT_INPUT = 0.8
DEFAULT_DROPOUT_HIDDEN = 1.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M8: shallow RBF + cross-model class agreement.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--units", type=int, default=DEFAULT_UNITS)
    p.add_argument("--pieces", type=int, default=5)
    p.add_argument("--rbf-epochs", dest="rbf_epochs", type=int, default=10)
    p.add_argument("--maxout-epochs", dest="maxout_epochs", type=int, default=10)
    p.add_argument("--soft-epochs", dest="soft_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "m8_rbf.json"))
    return p


def _train(model, epochs, data, args):
    cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        max_epochs=epochs, seed=args.seed, adv_train=False, eps=EPS, alpha=0.5,
        early_stop="clean", patience=args.patience,
    )
    res = train(model, cfg, data)
    if getattr(res, "best_state_dict", None) is not None:
        model.load_state_dict(res.best_state_dict)
    return model


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = load_mnist(Path(args.data_root), args.seed)
    x_test, y_test = data.x_test, data.y_test

    # RBF network (M8). Note: quadratic models are hard to train with SGD
    # (tex:616-617); sub-scale is expected to underfit.
    torch.manual_seed(args.seed)
    rbf = _train(RBFNet(n_classes=10, in_dim=784), args.rbf_epochs, data, args)

    # Maxout (m1 in section 8) and softmax for the agreement numbers.
    torch.manual_seed(args.seed)
    m_max = _train(MaxoutMLP(units=args.units, pieces=args.pieces, in_dim=784,
                             n_classes=10, dropout_input_include=DEFAULT_DROPOUT_INPUT,
                             dropout_hidden_include=DEFAULT_DROPOUT_HIDDEN,
                             seed=args.seed), args.maxout_epochs, data, args)
    torch.manual_seed(args.seed)
    m_soft = _train(SoftmaxRegression(in_dim=784, n_classes=10),
                    args.soft_epochs, data, args)

    # RBF FGSM eval (55.4% / 1.2%). Uses the UNNORMALIZED per-class exp(q)
    # reading for confidence (SPEC §6 item 9); error rate uses argmax(q).
    rbf_fgsm = eval_fgsm_rbf(rbf, x_test, y_test, EPS)
    # RBF clean confidence = mean over ALL test of max_k exp(q_k) (60.6%).
    rbf_clean_conf = eval_clean_confidence_rbf(rbf, x_test)

    # Section 8 cross-model class agreement (tex:679-688).
    # maxout vs rbf: rbf predicts maxout's class 16.0% over m1 errors, 54.3% both-wrong.
    ag_mr = class_agreement(m_max, rbf, x_test, y_test, EPS)
    # maxout vs softmax: softmax predicts maxout's class 54.6% over m1 errors, 84.6% both-wrong.
    ag_ms = class_agreement(m_max, m_soft, x_test, y_test, EPS)
    # softmax vs rbf: rbf predicts softmax's class 53.6% over m1 errors.
    ag_sr = class_agreement(m_soft, rbf, x_test, y_test, EPS)

    # Sub-scale if ANY model is under-trained: the paper converges each net to
    # patience-100; we flag any epoch/patience knob below the paper scale.
    sub_scale = (
        args.patience < 100
        or args.rbf_epochs < 100
        or args.maxout_epochs < 100
        or args.soft_epochs < 100
    )
    record = {
        "milestone": "M8",
        "description": "Shallow RBF on MNIST; FGSM eps=0.25; cross-model class "
                       "agreement (section 8). RBF confidence uses the "
                       "UNNORMALIZED per-class exp(q) reading (SPEC §6 item 9).",
        "seed": args.seed,
        "hyperparams": {
            "models": ["RBFNet", "MaxoutMLP", "SoftmaxRegression"],
            "units": args.units, "pieces": args.pieces,
            "lr": args.lr, "momentum": args.momentum, "batch_size": args.batch_size,
            "rbf_epochs": args.rbf_epochs, "maxout_epochs": args.maxout_epochs,
            "soft_epochs": args.soft_epochs, "patience": args.patience, "eps": EPS,
            "rbf_confidence_metric": "unnorm exp(q_k) (paper binary form, tex:595)",
        },
        "rbf_fgsm_eval": asdict(rbf_fgsm),
        "rbf_clean_confidence": rbf_clean_conf,
        "rbf_clean_accuracy": float(eval_clean(rbf, x_test, y_test)),
        "agreement": {
            "maxout_vs_rbf": asdict(ag_mr),
            "maxout_vs_softmax": asdict(ag_ms),
            "softmax_vs_rbf": asdict(ag_sr),
        },
        "paper_target": PAPER_TARGET,
        "note": ("Sub-scale run (rbf_epochs=%d, maxout_epochs=%d, soft_epochs=%d, "
                 "patience=%d). The paper notes quadratic models are hard to train "
                 "with SGD (tex:616-617); the RBF net may underfit. RBF confidence "
                 "uses the unnormalized per-class exp(q) reading (SPEC §6 item 9) "
                 "so it CAN structurally reach the paper's 1.2%%/60.6%%/0%% (a softmax "
                 "reading is bounded below by 1/K and cannot)."
                 % (args.rbf_epochs, args.maxout_epochs, args.soft_epochs,
                    args.patience)) if sub_scale else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"[M8] rbf fgsm err={rbf_fgsm.error_rate:.4f} conf={rbf_fgsm.mean_confidence_on_errors:.4f} "
          f"clean_conf={rbf_clean_conf:.4f} ; "
          f"agreements maxout-rbf={ag_mr.p_pred_match_over_m1_errors:.3f}/{ag_mr.p_pred_match_over_both_wrong:.3f} "
          f"maxout-soft={ag_ms.p_pred_match_over_m1_errors:.3f}/{ag_ms.p_pred_match_over_both_wrong:.3f} "
          f"soft-rbf={ag_sr.p_pred_match_over_m1_errors:.3f} "
          f"(paper 55.4/1.2/60.6; 16.0/54.6/84.6/54.3/53.6%)")
    print(f"[M8] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
