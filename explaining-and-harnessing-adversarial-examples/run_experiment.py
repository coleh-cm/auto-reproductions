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

Output: EXACTLY one line  `FINAL <arm>=<float>`  where <arm> is
``baseline`` (for ``--baseline``) or ``adversarial`` (for the method arm,
``--lambda``), and <float> is the clean test accuracy.  The arm name matches
the keys in ``arms.json`` so the gate (``run_all_arms.sh``) can pair each
line with its arm.  All other output goes to stderr.

Determinism: MaxoutMLP weight init draws from the GLOBAL torch RNG
(``nn.init.uniform_`` / ``nn.Linear`` init use no generator), so we call
``torch.manual_seed(seed)`` BEFORE constructing the model.  Two runs with
the same --seed then produce bit-identical weights, dropout masks (the
per-module dropout generator is reseeded from ``seed`` inside ``train``),
and batch order, hence the same FINAL line.

Degeneracy: ``adv_train = not --baseline`` — the method (Algorithm B) runs
by default, including at ``--lambda 0`` (its no-op setting, where
``x + 0*sign(grad) == x`` so ``J~ == J`` exactly).  ``--baseline`` forces the
clean arm (``adv_train=False``) as the reference.  Dropout is DISABLED by
default (include-prob 1.0) so the method at ``--lambda 0`` reproduces the
``--baseline`` arm BIT-FOR-BIT: with dropout off, ``adversarial_train_cost``
at eps=0 performs no extra RNG draw and its second forward equals the first,
so the parameter update is identical to clean training.  The degeneracy test
(tests/test_degeneracy.py) asserts that ``--lambda 0`` (arm ``adversarial``)
and ``--baseline`` (arm ``baseline``) print the SAME accuracy VALUE (the arm
name differs by construction; the value must match bit-for-bit).

Thread tuning: on multi-core CPUs PyTorch's default thread pool over-spawns
and contends (observed ~7x slowdown at 1500 steps).  ``run_all_arms.sh`` and
``smoke.sh`` export ``OMP_NUM_THREADS=4`` / ``MKL_NUM_THREADS=4``; the CLI
itself leaves threading to the environment so the degeneracy/determinism
tests (tiny step counts) are unaffected.
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
                   default=1.0, help="input dropout include-prob (1.0=off; "
                   "the paper states no dropout rate -- tex:492 only says "
                   "'also regularized with dropout'; the external pylearn2 "
                   "recipe uses input include 0.8. Off here so the eps=0 "
                   "degeneracy holds bit-for-bit.)")
    p.add_argument("--dropout-hidden", dest="dropout_hidden", type=float,
                   default=1.0, help="hidden dropout include-prob (1.0=off; "
                   "the paper states no dropout rate; the external recipe has "
                   "NO hidden dropout. Off here for the degeneracy gate.)")
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
    # The method (Algorithm B) runs by default, including at --lambda 0 (its
    # no-op setting). --baseline forces the clean arm (adv_train=False) as the
    # degeneracy reference. With dropout off (defaults 1.0/1.0), the method at
    # eps=0 is bit-for-bit identical to the clean arm (adversarial_train_cost
    # at eps=0 == cross_entropy_cost; no extra RNG draw).
    adv_train = not args.baseline

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
    arm = "baseline" if args.baseline else "adversarial"
    print(f"FINAL {arm}={acc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
