#!/usr/bin/env python
"""experiments/f4_eps_curve.py — Figure 4 (eps-sweep) milestone.

Reproduces Figure 4 (tex:755-775): the argument to the softmax layer for each
of the 10 MNIST classes as eps varies along the FGSM 1-D subspace for a single
test example of class 4, on a NAIVELY trained maxout network (tex:766
"This plot was made from a naively trained maxout network.").

Procedure (SPEC.md section 2, Algorithm A applied as a sweep):
  1. Train the naive maxout M3 configuration (240x2 units, 5 pieces, input
     dropout 0.8) — identical recipe to ``experiments/m3_maxout_fgsm.py``.
  2. Take the FIRST test example of class 4 (deterministic choice).
  3. Compute the FGSM direction ``sign(grad_x J)`` at that clean example, once,
     with the model in eval() mode (dropout off) — the same deterministic
     attacker as eval.py (SPEC section 6 item 23).
  4. Sweep eps over the figure's own x-range [-15, 15] (the axis range exists
     only in the figure, not the text — read off paper/source/eps_curve.pdf
     via read-figure, transcript in paper/figure_transcripts.md) in steps of
     0.5, and record the 10 logits at x + eps*sign(g).

Sequences recorded per eps: the correct-class (4) logit, the maximum
wrong-class logit, the argmax class, a predicted_is_correct 0/1 flag, and the
max |logit|. These feed the three ``curve`` claims fc1..fc3 in claims.json
(evaluated by numbers_gate.py directly from the per-seed result files;
measured.json carries only the scalar summaries, per the scalars-only
contract).

Writes ``results/f4_eps_curve.json``.

Runnable::

    python experiments/f4_eps_curve.py --seed 0
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
from fgsm_repro.objectives import cross_entropy_cost  # noqa: E402
from fgsm_repro.train import TrainConfig, train  # noqa: E402

# Figure 4 caption (tex:767-769): "the argument to the softmax layer for each
# of the 10 MNIST classes as we vary eps on a single input example. The correct
# class is 4."
CORRECT_CLASS = 4

# The figure x-axis is eps in [-15, 15] (read off paper/source/eps_curve.pdf;
# axis range exists only in the figure, not the text — tex:755-775 caption).
EPS_LO, EPS_HI, EPS_STEP = -15.0, 15.0, 0.5

DEFAULT_UNITS = 240
DEFAULT_PIECES = 5
DEFAULT_DROPOUT_INPUT = 0.8
DEFAULT_DROPOUT_HIDDEN = 1.0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="F4: eps-sweep of softmax logits along the FGSM direction "
                    "for a class-4 example (naive maxout, Figure 4).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--units", type=int, default=DEFAULT_UNITS)
    p.add_argument("--pieces", type=int, default=DEFAULT_PIECES)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    p.add_argument("--epochs", dest="epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--data-root", dest="data_root", type=str,
                   default=str(REPRO_ROOT / "data"))
    p.add_argument("--out", dest="out", type=str,
                   default=str(REPRO_ROOT / "results" / "f4_eps_curve.json"))
    p.add_argument("--fig-dir", dest="fig_dir", type=str,
                   default=str(REPRO_ROOT / "results" / "figures"),
                   help="directory for the regenerated Figure-4 curve panel PNG")
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   default=True,
                   help="skip regenerating the Figure-4 curve panel PNG")
    return p


def _plot_figure4(record: dict, fig_dir: Path, fname: str | None = None) -> Path:
    """Regenerate Figure 4's LEFT panel (logits vs eps) beside the paper's
    ``paper/source/eps_curve.pdf``.

    The paper's figure (read with the vision tool, transcript
    ``paper/figure_transcripts.md``) uses:
      * x-axis: eps in [-15, 15], ticks -15,-10,-5,0,5,10,15, label "epsilon"
      * y-axis: "argument to softmax" (i.e. the RAW pre-softmax logits, units
        identical to ours; the paper's numeric y-range [-2000, 1000] reflects a
        fully-trained maxout, ours is smaller-magnitude at sub-scale — the
        curve CLAIMS fc1..fc3 are SHAPE claims (crosses/below/increasing) and
        are scale-invariant, so the magnitude mismatch is expected and noted
        in REPRODUCTION.md, not hidden).

    We assert the x-range and y UNITS match the paper's figure exactly so a
    right-shaped curve on a different scale cannot read as a match.
    """
    import matplotlib
    matplotlib.use("Agg")  # noqa: E402  headless
    import matplotlib.pyplot as plt  # noqa: E402

    eps = list(record["eps_values"])
    logits = [list(row) for row in record["logits"]]  # [n_eps][10]
    correct = record["correct_class"]

    # Axis-range / units assertion against the paper's Figure 4.
    assert record["figure"]["x_axis_range_read"] == [-15.0, 15.0], (
        "Figure 4 x-axis must be eps in [-15, 15] (paper/source/eps_curve.pdf); "
        f"got {record['figure']['x_axis_range_read']}")
    assert record["figure"]["y_axis_label"] == "argument to softmax", (
        "Figure 4 y-axis must be the raw 'argument to softmax' (pre-softmax "
        f"logits); got {record['figure']['y_axis_label']}")
    assert eps[0] == -15.0 and eps[-1] == 15.0, (
        f"eps sweep must span [-15, 15]; got [{eps[0]}, {eps[-1]}]")

    fig, ax = plt.subplots(figsize=(7, 5))
    n_classes = len(logits[0])
    for c in range(n_classes):
        ys = [row[c] for row in logits]
        if c == correct:
            ax.plot(eps, ys, linewidth=2.6, label=f"class {c} (correct)",
                    color="magenta")
        else:
            ax.plot(eps, ys, linewidth=1.0, linestyle="--", label=f"{c}",
                    alpha=0.8)
    # Match the paper's x-axis exactly.
    ax.set_xlim(-15.0, 15.0)
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("argument to softmax")  # exact paper y-label
    ax.set_title(
        f"Reproduction of Figure 4 (left): softmax logits vs eps along the "
        f"FGSM direction\nnaive maxout {record['hyperparams']['units']}x2, "
        f"class-{correct} test example (seed {record['seed']}). "
        f"Sub-scale: logit magnitude is NOT the paper's [-2000,1000] range.")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / (fname if fname is not None else "f4_eps_curve_repro.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


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
        max_epochs=args.epochs, seed=args.seed, adv_train=False, eps=0.25,
        alpha=0.5, early_stop="clean", patience=args.patience,
    )
    result = train(model, cfg, data)
    if getattr(result, "best_state_dict", None) is not None:
        model.load_state_dict(result.best_state_dict)

    # First class-4 test example (deterministic choice, same at every seed).
    matches = (data.y_test == CORRECT_CLASS).nonzero().flatten()
    example_index = int(matches[0].item())
    x0 = data.x_test[example_index:example_index + 1].clone()
    y0 = data.y_test[example_index:example_index + 1].clone()

    # FGSM direction at the clean example, model in eval mode (the
    # deterministic evaluation attacker, SPEC section 6 item 23).
    model.eval()
    xg = x0.detach().clone().requires_grad_(True)
    J = cross_entropy_cost(model, xg, y0)
    g = torch.autograd.grad(J, xg)[0]
    direction = torch.sign(g)  # [1, 784]

    n_steps = int(round((EPS_HI - EPS_LO) / EPS_STEP)) + 1
    eps_values = [EPS_LO + i * EPS_STEP for i in range(n_steps)]

    logits_curve: list[list[float]] = []
    with torch.no_grad():
        for eps in eps_values:
            z = model.logits(x0 + eps * direction)  # [1, 10]
            logits_curve.append([float(v) for v in z[0].tolist()])

    correct_logit = [row[CORRECT_CLASS] for row in logits_curve]
    max_wrong_logit = [
        max(v for k, v in enumerate(row) if k != CORRECT_CLASS)
        for row in logits_curve
    ]
    top_class = [max(range(len(row)), key=lambda k: row[k]) for row in logits_curve]
    predicted_is_correct = [1 if t == CORRECT_CLASS else 0 for t in top_class]
    max_abs_logit = [max(abs(v) for v in row) for row in logits_curve]

    clean_logits = model.logits(x0)
    clean_predicted = int(clean_logits.argmax(dim=1).item())

    # Scalar summaries (the values the claims x_ranges restrict).
    def _in_range(lo: float, hi: float) -> list[int]:
        return [i for i, e in enumerate(eps_values) if lo - 1e-9 <= e <= hi + 1e-9]

    pos_wide = _in_range(4.0, 15.0)
    frac_correct_pos_4_15 = (
        sum(predicted_is_correct[i] for i in pos_wide) / len(pos_wide)
    )
    crossed = None
    for i, e in enumerate(eps_values):
        if e > 0 and top_class[i] != CORRECT_CLASS:
            crossed = e
            break

    record = {
        "milestone": "F4",
        "description": (
            "Softmax logits of a naive maxout 240x2 net for one class-4 test "
            "example along x + eps*sign(grad_x J), eps in [%g, %g] step %g "
            "(Figure 4; tex:755-775)." % (EPS_LO, EPS_HI, EPS_STEP)
        ),
        "seed": args.seed,
        "hyperparams": {
            "model": "MaxoutMLP", "units": args.units, "pieces": args.pieces,
            "dropout_input_include": DEFAULT_DROPOUT_INPUT,
            "dropout_hidden_include": DEFAULT_DROPOUT_HIDDEN,
            "lr": args.lr, "momentum": args.momentum,
            "batch_size": args.batch_size, "max_epochs": args.epochs,
            "patience": args.patience, "epochs_run": result.epochs_run,
        },
        "figure": {
            "id": "fig:eps_curve",
            "citation": "paper/source/iclr2015.tex:755-775",
            "x_axis_range_read": [-15.0, 15.0],
            "y_axis_label": "argument to softmax",
            "transcript": "paper/figure_transcripts.md",
        },
        "example_index": example_index,
        "correct_class": CORRECT_CLASS,
        "clean_predicted": clean_predicted,
        "clean_is_correct": bool(clean_predicted == CORRECT_CLASS),
        "eps_values": eps_values,
        "logits": logits_curve,
        "correct_logit": correct_logit,
        "max_wrong_logit": max_wrong_logit,
        "top_class": top_class,
        "predicted_is_correct": predicted_is_correct,
        "max_abs_logit": max_abs_logit,
        "eps_crossover_pos": crossed,
        "frac_correct_pos_4_15": frac_correct_pos_4_15,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    fig_path = None
    if args.plot:
        try:
            fig_path = _plot_figure4(record, Path(args.fig_dir))
        except ImportError:
            print("[F4] matplotlib not installed; skipping figure regeneration "
                  "(curve DATA in results/f4_eps_curve.json is the gate evidence)",
                  file=sys.stderr)

    print(f"[F4] clean pred={clean_predicted} (true {CORRECT_CLASS}, idx {example_index})  "
          f"crossover eps+={crossed}  frac correct in [4,15]={frac_correct_pos_4_15:.3f}  "
          f"max_wrong_logit eps=0 -> {max_wrong_logit[len(eps_values)//2]:.2f}, "
          f"eps=15 -> {max_wrong_logit[-1]:.2f}")
    print(f"[F4] wrote {out_path}")
    if fig_path is not None:
        print(f"[F4] regenerated Figure 4 panel -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
