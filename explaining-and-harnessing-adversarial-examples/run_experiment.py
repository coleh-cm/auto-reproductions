#!/usr/bin/env python3
"""FGSM adversarial-training experiment runner (graded harness CLI).

Reproduces the core method of Goodfellow, Shlens & Szegedy (ICLR 2015):
the Fast Gradient Sign Method used as a training regularizer via
adversarial training (Algorithm B, tex:486-488):

    J~(theta, x, y) = alpha J(theta, x, y)
                   + (1 - alpha) J(theta, x + eps * sign(grad_x J))

with alpha = 0.5 (tex:488).  The model is a maxout MLP (the paper's MNIST
family, tex:492), trained with SGD + momentum.

Frozen CLI contract (SPEC.md section 6 item 22; tests/test_degeneracy.py):

    run_experiment.py --lambda EPS --steps N [--seed S] [--baseline]
      --lambda   : FGSM perturbation magnitude eps (the paper's epsilon,
                   the free hyperparameter).  eps=0 selects the baseline
                   arm and must reproduce --baseline exactly (degeneracy).
      --steps    : number of SGD training steps (the paper states no count).
      --seed     : RNG seed (default 0).
      --baseline : force the baseline (clean) arm; --lambda is ignored.

Additional knobs the paper leaves free (defaults from the external
pylearn2 mnist_pi.yaml, documented as external in SPEC.md section 6):
--units, --pieces, --batch-size, --lr, --alpha, --dropout-input,
--dropout-hidden, --data-root.

Output: EXACTLY one line  `FINAL accuracy=<float>`  = clean test accuracy
(matching /^FINAL accuracy=[0-9.]+$/).  All other output goes to stderr.

Determinism: MaxoutMLP weight init draws from the GLOBAL torch RNG
(``nn.init.uniform_`` / ``nn.Linear`` init use no generator), so we call
``torch.manual_seed(seed)`` BEFORE constructing the model.  Two runs with
the same --seed then produce bit-identical weights, dropout masks (the
per-module dropout generator is reseeded from ``seed`` inside ``train``),
and batch order, hence the same FINAL accuracy line.

Degeneracy: with ``adv_train = (--lambda > 0)``, ``--lambda 0`` selects the
baseline arm (adv_train=False) -- identical to ``--baseline``; ``--lambda > 0``
selects the adversarial arm (Algorithm B).  This makes ``--lambda 0`` the
degeneracy no-op that exactly reproduces the baseline arm (the adv branch
performs no extra forward at eps=0, so the RNG stays in lock-step).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
REPRO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO_ROOT / "src"))

import torch  # noqa: E402

from fgsm_repro.data import load_mnist  # noqa: E402
from fgsm_repro.eval import eval_clean  # noqa: E402
from fgsm_repro.models import MaxoutMLP  # noqa: E402
from fgsm_repro.train import TrainConfig, train  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FGSM adversarial-training experiment runner (graded harness)."
    )
    # --lambda is the primary knob but is given a default (not argparse-
    # required) so that --baseline can be invoked alone, per the frozen CLI
    # contract (SPEC.md section 6 item 22 and tests/test_degeneracy.py).
    p.add_argument("--lambda", dest="lam", type=float, default=0.25,
                   help="FGSM eps (perturbation magnitude); 0 = baseline arm")
    p.add_argument("--steps", type=int, required=True,
                   help="number of SGD training steps")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--units", type=int, default=240,
                   help="maxout units per hidden layer (paper M3 size 240)")
    p.add_argument("--pieces", type=int, default=5,
                   help="maxout pieces (external pylearn2 default)")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100,
                   help="SGD batch size (external pylearn2 default)")
    p.add_argument("--lr", type=float, default=0.1,
                   help="SGD learning rate (external pylearn2 default)")
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT),
                   help="MNIST root; IDX files are read from <data-root>/mnist/")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="adversarial-training mixing weight (tex:488 alpha=0.5)")
    p.add_argument("--dropout-input", dest="dropout_input", type=float,
                   default=0.8, help="input dropout include-prob (external)")
    p.add_argument("--dropout-hidden", dest="dropout_hidden", type=float,
                   default=0.5, help="hidden dropout include-prob (external)")
    p.add_argument("--baseline", action="store_true",
                   help="force the baseline (clean) arm; --lambda is ignored")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # MaxoutMLP weight init draws from the GLOBAL torch RNG, so seed it
    # BEFORE constructing the model for bit-for-bit determinism
    # (SPEC.md section 6 item 22; review fix for the determinism contract).
    torch.manual_seed(args.seed)

    data = load_mnist(Path(args.data_root), seed=args.seed)

    model = MaxoutMLP(
        units=args.units,
        pieces=args.pieces,
        in_dim=784,
        n_classes=10,
        dropout_input_include=args.dropout_input,
        dropout_hidden_include=args.dropout_hidden,
        seed=args.seed,
    )

    eps = max(0.0, args.lam)
    # --lambda 0  -> adv_train=False (baseline, the degeneracy no-op).
    # --lambda>0 -> adv_train=True (Algorithm B adversarial training).
    # --baseline -> adv_train=False regardless of --lambda.
    adv_train = (args.lam > 0.0) and (not args.baseline)

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=0.5,
        max_epochs=10**9,
        seed=args.seed,
        adv_train=adv_train,
        eps=eps,
        alpha=args.alpha,
        early_stop="clean",
        patience=10**9,
        max_steps=args.steps,
    )

    result = train(model, cfg, data)
    # Monitor-best checkpoint (the paper's protocol) for evaluation.
    model.load_state_dict(result.best_state_dict)
    model.eval()

    acc = eval_clean(model, data.x_test, data.y_test)
    print(f"FINAL accuracy={acc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
