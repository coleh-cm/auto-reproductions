#!/usr/bin/env python
"""experiments/m9_rubbish.py — Milestone M9.

Rubbish examples (appendix): 10,000 samples ~ N(0, I_784) fed to four models.

Paper target (tex:905-912): maxout+softmax 98.35% (conf 92.8%); sigmoid-top 68%
(conf 87.9%); softmax regression 59.8% (conf 70.8%); RBF 0%.

The sigmoid-top arm is "Changing the top layer to independent sigmoids"
(tex:908-909). The faithful reading is to TRAIN a maxout net with independent
per-class sigmoid outputs (per-class BCE cost), not to copy the softmax-trained
readout weights and apply sigmoids with no retraining (a frozen swap that
mechanically forces rubbish error -> 1.0 regardless of training, SPEC §6 item
25). ``SigmoidTopMLP`` shares the maxout trunk architecture (same init / SGD /
external recipe via ``train(cost="sigmoid_top")``); only the top activation and
training cost differ from the maxout+softmax net.

The RBF arm uses the UNNORMALIZED per-class exp(q) reading for the
any-class-p>0.5 error rule (SPEC §6 item 9); a softmax reading is bounded below
by 1/K and cannot reproduce the paper's 0%.

Writes ``results/m9_rubbish.json``.

Runnable::

    python experiments/m9_rubbish.py
    python experiments/m9_rubbish.py --epochs 20 --n 10000
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
from fgsm_repro.models import (  # noqa: E402
    MaxoutMLP, SigmoidTopMLP, SoftmaxRegression, RBFNet,
)
from fgsm_repro.train import TrainConfig, train  # noqa: E402
from fgsm_repro.eval import (  # noqa: E402
    eval_rubbish, eval_rubbish_sigmoid, eval_rubbish_rbf, eval_clean,
)

# Paper target (tex:905-912).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:905-912",
    "quote": "98.35% ... 92.8% ... 68% ... 87.9% ... 59.8% ... 70.8% ... RBF ... 0%",
    "maxout_softmax": {"error": 0.9835, "conf": 0.928},
    "sigmoid_top": {"error": 0.68, "conf": 0.879},
    "softmax_regression": {"error": 0.598, "conf": 0.708},
    "rbf": {"error": 0.0, "conf": None},
}

EPS = 0.25
DEFAULT_UNITS = 240
DEFAULT_DROPOUT_INPUT = 0.8
DEFAULT_DROPOUT_HIDDEN = 1.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M9: rubbish examples N(0,I_784) on 4 models.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--units", type=int, default=DEFAULT_UNITS)
    p.add_argument("--pieces", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--epochs", dest="epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--n", type=int, default=10000, help="rubbish samples (paper: 10000)")
    p.add_argument("--dim", type=int, default=784)
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "m9_rubbish.json"))
    return p


def _train(model, data, args, cost="softmax"):
    cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        max_epochs=args.epochs, seed=args.seed, adv_train=False, eps=EPS,
        alpha=0.5, early_stop="clean", patience=args.patience, cost=cost,
    )
    res = train(model, cfg, data)
    if getattr(res, "best_state_dict", None) is not None:
        model.load_state_dict(res.best_state_dict)
    return model


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = load_mnist(Path(args.data_root), args.seed)

    # Maxout + softmax top (tex:908 98.35%/92.8%).
    torch.manual_seed(args.seed)
    m_max = _train(MaxoutMLP(units=args.units, pieces=args.pieces, in_dim=784,
                             n_classes=10, dropout_input_include=DEFAULT_DROPOUT_INPUT,
                             dropout_hidden_include=DEFAULT_DROPOUT_HIDDEN,
                             seed=args.seed), data, args)
    # Sigmoid-top arm (tex:908-909): a maxout net with independent per-class
    # sigmoid outputs, TRAINED with the per-class BCE cost (not a frozen
    # softmax-to-sigmoid weight swap, SPEC §6 item 25). Same trunk architecture
    # / init / SGD / external recipe as m_max; only the top activation + cost
    # differ -- isolating the architecture change the paper names.
    torch.manual_seed(args.seed)
    m_sig = _train(SigmoidTopMLP(units=args.units, pieces=args.pieces, in_dim=784,
                                 n_classes=10, dropout_input_include=DEFAULT_DROPOUT_INPUT,
                                 dropout_hidden_include=DEFAULT_DROPOUT_HIDDEN,
                                 seed=args.seed), data, args, cost="sigmoid_top")
    # Softmax regression (tex:920 59.8%/70.8%).
    torch.manual_seed(args.seed)
    m_soft = _train(SoftmaxRegression(in_dim=784, n_classes=10), data, args)
    # RBF (tex: 0%).
    torch.manual_seed(args.seed)
    m_rbf = _train(RBFNet(n_classes=10, in_dim=784), data, args)

    ev_max = eval_rubbish(m_max, args.n, args.dim, args.seed)
    ev_sig = eval_rubbish_sigmoid(m_sig, args.n, args.dim, args.seed)
    ev_soft = eval_rubbish(m_soft, args.n, args.dim, args.seed)
    ev_rbf = eval_rubbish_rbf(m_rbf, args.n, args.dim, args.seed)

    sub_scale = args.patience < 100 or args.epochs < 100
    record = {
        "milestone": "M9",
        "description": "Rubbish examples N(0,I_784) on maxout+softmax, "
                       "sigmoid-top (TRAINED, per-class BCE), softmax-reg, RBF "
                       "(unnorm exp(q) metric).",
        "seed": args.seed,
        "hyperparams": {
            "models": ["MaxoutMLP(softmax)", "SigmoidTopMLP(trained,BCE)",
                       "SoftmaxRegression", "RBFNet(unnorm exp(q))"],
            "units": args.units, "pieces": args.pieces, "n": args.n, "dim": args.dim,
            "lr": args.lr, "momentum": args.momentum, "batch_size": args.batch_size,
            "max_epochs": args.epochs, "patience": args.patience,
            "sigmoid_top_note": "trained with per-class BCE (not a frozen swap)",
            "rbf_note": "unnormalized per-class exp(q) metric (SPEC §6 item 9)",
        },
        "evals": {
            "maxout_softmax": asdict(ev_max),
            "sigmoid_top": asdict(ev_sig),
            "softmax_regression": asdict(ev_soft),
            "rbf": asdict(ev_rbf),
        },
        "paper_target": PAPER_TARGET,
        "note": ("Sub-scale run (epochs=%d, patience=%d, n=%d). The paper's "
                 "98.35/68/59.8/0%% are from fully-converged nets."
                 % (args.epochs, args.patience, args.n))
                 if sub_scale else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"[M9] maxout={ev_max.error_rate:.4f} sigmoid={ev_sig.error_rate:.4f} "
          f"softmax_reg={ev_soft.error_rate:.4f} rbf={ev_rbf.error_rate:.4f} "
          f"(paper tex:905-912 98.35/68/59.8/0%)")
    print(f"[M9] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
