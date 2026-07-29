#!/usr/bin/env python
"""experiments/e1_ensemble.py — Extended milestone E1.

Ensemble of 12 maxout nets; FGSM designed to perturb the WHOLE ensemble vs a
SINGLE member.

Paper target (tex:819-825): ensemble-targeted FGSM error 91.1%; single-member-
targeted FGSM error 87.9%.

ERROR METRIC (SPEC §6 item 11): the paper says "The ensemble gets an error rate
of 91.1%" (tex:822) without defining the aggregation. The natural reading is
the error of the ENSEMBLE'S aggregated prediction, not the mean of the members'
individual error rates. We form the ensemble prediction as argmax of the MEAN
of the members' softmax probabilities (mean-prob voting) and report that as the
headline ``ensemble_targeted_error`` / ``single_member_targeted_error``. The
single-member arm averages over which member is targeted (attack member j,
score the ensemble, average the error over j) -- the paper does not name a
target member. The prior per-member-mean statistic is kept as a secondary
diagnostic (``mean_per_member_*``).

Writes ``results/e1_ensemble.json``.

Runnable::

    python experiments/e1_ensemble.py
    python experiments/e1_ensemble.py --members 12 --epochs 5
    python experiments/e1_ensemble.py --members 4 --epochs 3   # fast smoke
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
import torch.nn.functional as F  # noqa: E402

from fgsm_repro.data import load_mnist  # noqa: E402
from fgsm_repro.models import MaxoutMLP  # noqa: E402
from fgsm_repro.train import TrainConfig, train  # noqa: E402
from fgsm_repro.eval import eval_clean  # noqa: E402
from fgsm_repro.attacks import fgsm, fgsm_ensemble  # noqa: E402

# Paper target (tex:819-825).
PAPER_TARGET = {
    "citation": "paper/source/iclr2015.tex:819-825",
    "quote": "error rate of 91.1% on adversarial examples designed to perturb "
             "the entire ensemble ... falls to 87.9%",
    "ensemble_targeted_error": 0.911,
    "single_member_targeted_error": 0.879,
}

EPS = 0.25
DEFAULT_UNITS = 240
DEFAULT_DROPOUT_INPUT = 0.8
DEFAULT_DROPOUT_HIDDEN = 1.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E1: 12-maxout ensemble FGSM (ensemble vs single target).")
    p.add_argument("--base-seed", dest="base_seed", type=int, default=0)
    p.add_argument("--members", type=int, default=12, help="paper: 12")
    p.add_argument("--units", type=int, default=DEFAULT_UNITS)
    p.add_argument("--pieces", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--epochs", dest="epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "e1_ensemble.json"))
    return p


def _train_member(seed: int, data, args) -> MaxoutMLP:
    torch.manual_seed(seed)
    model = MaxoutMLP(
        units=args.units, pieces=args.pieces, in_dim=784, n_classes=10,
        dropout_input_include=DEFAULT_DROPOUT_INPUT,
        dropout_hidden_include=DEFAULT_DROPOUT_HIDDEN, seed=seed,
    )
    cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        max_epochs=args.epochs, seed=seed, adv_train=False, eps=EPS, alpha=0.5,
        early_stop="clean", patience=args.patience,
    )
    res = train(model, cfg, data)
    if getattr(res, "best_state_dict", None) is not None:
        model.load_state_dict(res.best_state_dict)
    return model


def _ensemble_predict(models, x) -> torch.Tensor:
    """Ensemble prediction = argmax of the MEAN of members' softmax probs.

    The paper's "the ensemble gets an error rate of ..." (tex:822) most
    naturally means the error of the ensemble's combined prediction; mean-prob
    voting is the standard combination (SPEC §6 item 11).
    """
    for m in models:
        m.eval()
    with torch.no_grad():
        probs = torch.zeros((x.shape[0], 10), dtype=torch.float32)
        for m in models:
            probs = probs + F.softmax(m.logits(x), dim=-1)
        probs = probs / len(models)
    return probs.argmax(dim=1)


def _ensemble_error(models, x_adv, y) -> float:
    """Headline E1 metric: error rate of the ensemble's aggregated prediction
    on x_adv (mean-prob argmax)."""
    pred = _ensemble_predict(models, x_adv)
    return float((pred != y).float().mean().item())


def _mean_per_member_error(models, x_adv, y) -> float:
    """Secondary diagnostic: mean over members of each member's individual
    error rate on x_adv (the prior implementation's statistic)."""
    errs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            pred = m.logits(x_adv).argmax(dim=1)
        errs.append(float((pred != y).float().mean().item()))
    return sum(errs) / len(errs)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = load_mnist(Path(args.data_root), args.base_seed)
    x_test, y_test = data.x_test, data.y_test

    models = []
    for i in range(args.members):
        seed = args.base_seed + i
        print(f"[E1] training member {i} (seed={seed})...", file=sys.stderr)
        models.append(_train_member(seed, data, args))
        for m in models:
            m.eval()

    # Ensemble-targeted FGSM: gradient of mean CE over members (91.1%).
    x_adv_ens = fgsm_ensemble(models, x_test, y_test, EPS)
    ens_error = _ensemble_error(models, x_adv_ens, y_test)
    ens_mean_per_member = _mean_per_member_error(models, x_adv_ens, y_test)

    # Single-member-targeted FGSM: attack EACH member in turn, score the
    # ensemble (mean-prob argmax) on those adversarials, and AVERAGE the
    # ensemble error over the targeted member (the paper does not name a
    # target member; averaging avoids the arbitrary choice of member 0).
    single_errors = []
    single_mean_per_member = []
    for j in range(args.members):
        x_adv_single = fgsm(models[j], x_test, y_test, EPS)
        single_errors.append(_ensemble_error(models, x_adv_single, y_test))
        single_mean_per_member.append(
            _mean_per_member_error(models, x_adv_single, y_test))
    single_error = sum(single_errors) / len(single_errors)
    single_mean_per_member_avg = sum(single_mean_per_member) / len(single_mean_per_member)

    clean_accs = [float(eval_clean(m, x_test, y_test)) for m in models]

    sub_scale = args.members < 12 or args.patience < 100 or args.epochs < 100
    # Paper direction (tex:822-825): ensemble-targeted (91.1%) > single-targeted
    # (87.9%) -- attacking the whole ensemble fools the ensemble MORE than
    # attacking one member. At sub-scale (few undertrained members) both errors
    # saturate near 100% and the direction can reverse; record it honestly.
    direction_matches = ens_error >= single_error
    direction_note = (
        "direction matches paper (ensemble-targeted >= single-targeted)"
        if direction_matches else
        "direction REVERSED at sub-scale (single-targeted > ensemble-targeted); "
        "both errors saturate near 100% with few undertrained members -- the "
        "paper's 91.1>87.9 direction needs 12 converged nets.")
    record = {
        "milestone": "E1",
        "description": "Ensemble of N maxout nets; FGSM ensemble-targeted vs "
                       "single-member-targeted. Headline metric = error of the "
                       "ensemble's mean-prob prediction (SPEC §6 item 11).",
        "base_seed": args.base_seed,
        "hyperparams": {
            "members": args.members, "units": args.units, "pieces": args.pieces,
            "lr": args.lr, "momentum": args.momentum, "batch_size": args.batch_size,
            "max_epochs": args.epochs, "patience": args.patience, "eps": EPS,
            "ensemble_prediction": "argmax(mean member softmax probs)",
            "single_member_arm": "attack each member, average ensemble error",
        },
        "ensemble_targeted_error": ens_error,
        "single_member_targeted_error": single_error,
        "direction_matches_paper": direction_matches,
        # Secondary diagnostics (the prior per-member-mean statistic).
        "mean_per_member_error_ensemble_targeted": ens_mean_per_member,
        "mean_per_member_error_single_targeted": single_mean_per_member_avg,
        "per_member_single_targeted_errors": single_errors,
        "member_clean_accuracies": clean_accs,
        "paper_target": PAPER_TARGET,
        "note": ("Sub-scale run (members=%d, epochs=%d, patience=%d). The paper's "
                 "91.1/87.9%% use 12 fully-converged nets. %s"
                 % (args.members, args.epochs, args.patience, direction_note))
                 if sub_scale else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"[E1] ensemble-targeted={ens_error:.4f} single-targeted={single_error:.4f} "
          f"(paper tex:819-825 91.1/87.9%)")
    print(f"[E1] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
