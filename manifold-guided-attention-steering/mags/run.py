"""MAGS run CLI.

``python -m mags.run --benchmark MATH-500 --model <id> --arm <name> [--limit N]``

Runs one arm of the paper's comparison and prints exactly one line:
    FINAL <arm name>=<value>

where <value> is the primary metric for that arm (accuracy for reasoning/code;
validity_pct for molecular). For the molecular task the secondary metric
(autodock_gpu_affinity_kcal_mol) is also printed on a second line as
    FINAL <arm name>__binding_affinity=<value>

The arm names match arms.json. The shell commands that invoke this CLI for every
arm live in run_all_arms.sh; the small-size smoke path lives in smoke.sh.

This runner needs (a) the model to load and (b) a fitted ManifoldBank for steering
arms. In a no-GPU / no-gated-token sandbox the model load fails -> we print a
BLOCKED marker instead of a fabricated number, and exit non-zero so the gate sees
the failure rather than a silent pass.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np

from . import config


def _blocked(arm, reason, secondary=None):
    print(f"FINAL {arm}=BLOCKED")
    if secondary:
        print(f"FINAL {arm}__binding_affinity=BLOCKED")
    sys.stderr.write(f"BLOCKED[{arm}]: {reason}\n")
    # write a manifest for the publish step to read
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/BLOCKED__{arm}.json", "w") as f:
        json.dump({"arm": arm, "blocked_reason": reason}, f, indent=2)
    sys.exit(2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--arm-id", default="",
                    help="unique arm name (key of arms.json); printed in the FINAL line. "
                         "Defaults to --arm if unset.")
    ap.add_argument("--limit", type=int, default=0, help="0 = full eval set")
    ap.add_argument("--manifold", default="", help="path to ManifoldBank .npz (steering arms)")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--angle-deg", type=float, default=None)
    ap.add_argument("--iti-K", type=int, default=None)
    ap.add_argument("--iti-alpha", type=float, default=None)
    ap.add_argument("--cd-amateur", default="")
    ap.add_argument("--smoke", action="store_true", help="use a synthetic fixture (smoke.sh only)")
    args = ap.parse_args(argv)

    bench = args.benchmark
    model_id = args.model
    arm = args.arm
    arm_id = args.arm_id or arm

    def emit(value, secondary=None):
        print(f"FINAL {arm_id}={value}")
        if secondary is not None:
            print(f"FINAL {arm_id}__binding_affinity={secondary}")

    # ---- molecular task is a stretch target with unstated setup (SPEC §4.18) ----
    if bench == "SMILES-molecular-generation":
        _blocked(arm_id, "Molecular task (Table 3) target protein, prompt template, SMILES "
                          "contrastive corpus, affinity cutoff and AutoDock-GPU params are all "
                          "UNSTATED by the paper (SPEC §4.18); GPT-OSS-20B needs >=40GB VRAM. "
                          "Treated as a stretch target, not implemented for real data.")

    # ---- load the model (expected to fail in a no-GPU/no-token sandbox) ----
    try:
        from .models import load_model
        model, tok = load_model(model_id)
    except Exception as e:
        _blocked(arm_id, f"model load failed: {e!r}")

    # ---- load the eval problems ----
    from .data.loaders import EVAL_LOADERS
    if bench not in EVAL_LOADERS:
        _blocked(arm_id, f"no eval loader for benchmark {bench!r}")
    problems = EVAL_LOADERS[bench]()
    if args.limit and args.limit > 0:
        problems = problems[:args.limit]
    max_new = args.max_new_tokens or config.DEFAULT_MAX_NEW_TOKENS.get(
        "code" if bench in ("HumanEval", "MBPP") else "math", 512)

    # ---- build the controller for this arm ----
    from .steering import MAGSController, NoOpController
    from .baselines import ITIController, AngularSteeringController
    from .generation import generate, cd_generate
    from .grading import grade
    from .eval import run_arm, bootstrap_ci

    controller = None
    cd = None
    if arm == "unsteered":
        controller = NoOpController()
    elif arm in ("mags", "mags-u"):
        if not args.manifold or not os.path.exists(args.manifold):
            _blocked(arm_id, "steering arm needs --manifold <path.npz>; none provided "
                          "(manifold fit requires GPU traces from the paper's 8B/20B models, "
                          "blocked in this sandbox).")
        from .manifold import ManifoldBank
        bank = ManifoldBank.load(args.manifold)
        alpha = args.alpha if args.alpha is not None else bank.alpha
        controller = MAGSController(bank, alpha=alpha)
    elif arm == "iti":
        if not args.manifold or not os.path.exists(args.manifold):
            _blocked(arm_id, "iti arm needs --manifold (probe directions derived from the "
                          "contrastive activation set; fit blocked in this sandbox).")
        from .manifold import ManifoldBank
        bank = ManifoldBank.load(args.manifold)
        K = args.iti_K or config.ITI_DEFAULT_K
        alpha = args.iti_alpha if args.iti_alpha is not None else config.ITI_DEFAULT_ALPHA
        controller = ITIController(bank, alpha=alpha, K=K)
    elif arm == "angular-steering":
        if not args.manifold or not os.path.exists(args.manifold):
            _blocked(arm_id, "angular-steering needs --manifold (rotation plane from the "
                          "contrastive activation set; fit blocked in this sandbox).")
        from .manifold import ManifoldBank
        bank = ManifoldBank.load(args.manifold)
        ang = args.angle_deg if args.angle_deg is not None else config.AS_DEFAULT_ANGLE_DEG
        controller = AngularSteeringController(bank, angle_deg=ang)
    elif arm == "contrastive-decoding":
        amateur_id = args.cd_amateur or config.CD_AMATEUR.get(model_id, "")
        if not amateur_id:
            _blocked(arm_id, "contrastive-decoding needs an amateur model (SPEC §4.17); none "
                          "registered for this model and none provided.")
        from .models import load_model
        from .generation import cd_generate
        try:
            amateur, atok = load_model(amateur_id)
        except Exception as e:
            _blocked(arm_id, f"amateur model load failed: {e!r}")
        # CD uses a custom greedy loop, not the controller hook path
        corrects = []
        for prob in problems:
            completion, _ = cd_generate(
                model, amateur, tok, prob.prompt_text, max_new_tokens=max_new,
                alpha_plausibility=config.CD_DEFAULT_ALPHA_PLAUS,
                beta=config.CD_DEFAULT_BETA,
            )
            corrects.append(int(grade(bench, completion, prob)))
        acc, ci = bootstrap_ci(corrects)
        emit(f"{acc:.4f}")
        return
    else:
        _blocked(arm_id, f"unknown arm {arm!r}")

    # ---- run the arm ----
    result = run_arm(model, tok, model_id, controller, bench, problems,
                     max_new_tokens=max_new, grading=grade)
    # commit results to disk (NOT gitignored)
    os.makedirs("runs", exist_ok=True)
    out_path = f"runs/{bench.replace('/','_')}__{model_id.replace('/','_')}__{arm}.json"
    with open(out_path, "w") as f:
        json.dump({"arm": arm, "benchmark": bench, "model": model_id,
                   "n": result["n"], "acc": result["acc"], "ci95": result["ci95"],
                   "ppl": result["ppl"], "per_problem": result["per_problem"]}, f, indent=2)
    emit(f"{result['acc']:.4f}")


if __name__ == "__main__":
    main()
