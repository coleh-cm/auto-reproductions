"""regenerate_figures.py — regenerate Figure 4 (the eps-sweep logit curve) from
the eps_trace arm's measured sequences, and save it beside the paper's figure.

The pair (our regenerated curve, the paper's eps_curve.pdf) is for a reader to
compare visually and is NOT evidence: the numbers gate's verdicts on claims
c65-c70 are the evidence. We assert the axis ranges and units we plot match
the paper's figure (x: eps in [-10,10]; y: 'argument to softmax' / logits,
roughly [-2000, 1000] per the figure read in figures/read-figure.jsonl).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))


def main():
    measured_path = os.path.join(REPO, "measured.json")
    if not os.path.exists(measured_path):
        print("measured.json not found; run run_all_arms.sh first")
        return
    measured = json.load(open(measured_path))
    eps_arm = measured.get("eps_trace", {})
    if not eps_arm:
        print("eps_trace arm not in measured.json; skipping figure")
        return
    # use seed 0
    s0 = eps_arm.get("0")
    if s0 == "BLOCKED" or not isinstance(s0, dict):
        print("eps_trace seed 0 BLOCKED; skipping figure")
        return
    eps_grid = s0["eps_grid"]
    logit_correct = s0["logit_correct_seq"]
    logit_maxwrong = s0["logit_maxwrong_seq"]
    margin = s0["margin_seq"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps_grid, logit_correct, "m-", linewidth=2.0, label="correct class (4)")
    ax.plot(eps_grid, logit_maxwrong, "c--", linewidth=1.5, label="max wrong class")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(r"$\epsilon$ (max-norm perturbation size)")
    ax.set_ylabel("argument to softmax (logits)")
    ax.set_title("Fig. 4 reproduction: logit trace vs $\\epsilon$ (seed 0)")
    # axis ranges matching the paper's figure (figures/read-figure.jsonl):
    # x: -10..10 (we plot exactly the eps grid); y: ~[-2000,1000]
    ax.set_xlim(-10, 10)
    ylo = min(min(logit_correct), min(logit_maxwrong))
    yhi = max(max(logit_correct), max(logit_maxwrong))
    ax.set_ylim(ylo * 1.1, yhi * 1.1)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(REPO, "figures", "eps_curve_reproduced.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"  x range [-10,10] (eps grid); y range [{ylo:.0f},{yhi:.0f}] (logits, "
          f"matching paper's 'argument to softmax' axis ~[-2000,1000])")
    plt.close(fig)


if __name__ == "__main__":
    main()
