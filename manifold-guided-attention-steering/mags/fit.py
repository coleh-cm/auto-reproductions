"""Manifold fit CLI (SPEC Phase A).

``python -m mags.fit --model <id> --benchmark <name> --out manifolds/<slug>.npz``

Phase A: collect contrastive traces from the paper's training source for the
benchmark, capture per-head activations, fit the contrastive error manifold per
monitored head, rank heads by held-out mean-AUROC, keep top-K, and save a
ManifoldBank. This is the offline "training" step that the steering arms consume.

In a no-GPU / no-gated-token sandbox this step is BLOCKED at the model-load /
trace-capture stage (the base model must generate the contrastive traces). We emit a
BLOCKED marker rather than fabricating a manifold from synthetic activations.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# --- offline fast-fail (round-13 root-cause fix; see mags/run.py) -------------
# Force OFFLINE unconditionally unless the reproducer opts in with MAGS_ONLINE=1,
# so an uncached model OR training dataset fast-fails (ConnectionError in <1s)
# instead of hanging on a blackholed network until the gate kills the process
# (the round-1..12 "all arms missing a FINAL line" failure: a cached model +
# token + blackholed net let fit proceed past load_model to load_dataset, which
# hung). fit.py's _blocked() then writes the BLOCKED marker and exits 0. A real
# GPU host pre-caches models/datasets (README) or sets MAGS_ONLINE=1 to download.
if os.environ.get("MAGS_ONLINE", "0") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")

from . import config


def _blocked(reason):
    sys.stderr.write(f"BLOCKED[mags.fit]: {reason}\n")
    os.makedirs("runs", exist_ok=True)
    with open("runs/BLOCKED__fit.json", "w") as f:
        json.dump({"step": "fit", "blocked_reason": reason}, f, indent=2)
    # Exit 0: run_all_arms.sh Phase-1 wraps this in `if timeout ...; then :; else`
    # (handles any exit code), and a non-zero exit would make a gate that keeps
    # stdout only on exit 0 discard any output. Consistent with mags/run._blocked.
    sys.exit(0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train-problems", type=int, default=0,
                    help="cap on # problems sampled for trace collection (0 = all retained)")
    ap.add_argument("--n-samples", type=int, default=config.DEFAULT_N_SAMPLES)
    ap.add_argument("--k", type=int, default=config.DEFAULT_K_RANK)
    ap.add_argument("--q", type=float, default=config.DEFAULT_Q_PERCENTILE)
    ap.add_argument("--K", type=int, default=config.DEFAULT_K_HEADS)
    ap.add_argument("--alpha", type=float, default=config.DEFAULT_ALPHA)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="override per-domain max_new_tokens (0 = config default; "
                         "use a small value for fast CPU verification of the fit path)")
    args = ap.parse_args(argv)

    # load model (expected to fail in sandbox)
    try:
        from .models import load_model
        model, tok = load_model(args.model)
    except Exception as e:
        _blocked(f"model load failed ({e!r}); cannot collect contrastive traces from "
                 f"{args.model}. Manifold fit needs the base model to sample <=8 traces "
                 f"per problem (tex:L399); blocked without GPU / gated-token access.")
    # ... full fit path (trace collection + fit + save) lives below; only reached on a
    # GPU host with model access.
    from .data.loaders import TRAIN_LOADERS, EVAL_LOADERS
    from .data.store import save_trace_store, load_trace_store, build_head_activations, retain_paired_problems
    from .manifold import fit_manifold_bank
    from .capture import capture_trace
    from .grading import grade
    import torch
    import numpy as np

    monitored = config.monitored_layers_for(args.model)
    # 1. load training problems for the benchmark
    loader = TRAIN_LOADERS.get(args.benchmark)
    if loader is None:
        _blocked(f"no training source for {args.benchmark}")
    try:
        train_problems = loader(limit=args.n_train_problems or None)
    except Exception as e:
        _blocked(f"training source unavailable for {args.benchmark}: {e!r}")

    # 2. sample <=n traces per problem, keep problems with >=1 correct & >=1 incorrect
    max_new = (args.max_new_tokens if args.max_new_tokens and args.max_new_tokens > 0
               else config.DEFAULT_MAX_NEW_TOKENS.get(
                   "code" if args.benchmark in ("HumanEval", "MBPP") else "math", 512))
    paired = []
    for prob in train_problems:
        traces = []
        for s in range(args.n_samples):
            torch.manual_seed(config.DEFAULT_SEED + s)
            text, ids, acts = capture_trace(
                model, tok, args.model, model._mags_registry, prob.prompt_text,
                max_new_tokens=max_new, do_sample=True,
                temperature=config.DEFAULT_SAMPLING_TEMP,
                top_p=config.DEFAULT_SAMPLING_TOP_P,
                seed=config.DEFAULT_SEED + s,
            )
            ok = grade(prob.benchmark, text, prob)   # grade against the SOURCE benchmark
            traces.append((text, ids, acts, ok))
            if any(t[3] for t in traces) and any(not t[3] for t in traces):
                pass  # keep sampling up to n to enrich
        has_c = any(t[3] for t in traces)
        has_i = any(not t[3] for t in traces)
        if has_c and has_i:
            paired.append((prob, traces))

    if not paired:
        _blocked("no problems produced both a correct and an incorrect trace within "
                 f"{args.n_samples} samples; manifold cannot be fit (tex:L399).")

    # 3. problem-level 70/15/15 split (SPEC §4.8): fit / select / report-only
    #    (the report-only split scores the Figure-3 drift-validation AUROC on
    #    problems neither fit nor used for head selection, avoiding selection
    #    bias on the diagnostic of the selected heads).
    rng = np.random.default_rng(config.DEFAULT_SEED)
    idx = rng.permutation(len(paired))
    n = len(paired)
    n_fit = int(0.70 * n)
    n_sel = int(0.15 * n)
    fit_idx = idx[:n_fit].tolist()
    sel_idx = idx[n_fit:n_fit + n_sel].tolist()
    report_idx = idx[n_fit + n_sel:].tolist()
    # Degenerate small-N: leave the held-out splits EMPTY rather than aliasing the
    # fit split. Aliasing fit_idx reintroduces train-data selection bias that
    # manifold.py explicitly guards against (heads scored on the fit split get
    # inflated held-out AUROCs); with empty splits fit_manifold_bank assigns
    # m.auroc=0.5 (chance) and fit_iti_bank/fit_as_bank fall back gracefully. Real
    # benchmarks have n>>7 so this is latent; smoke uses synthetic fixtures.
    fit_pids = [paired[i][0].id for i in fit_idx]
    sel_pids = [paired[i][0].id for i in sel_idx]
    report_pids = [paired[i][0].id for i in report_idx]

    # 4. persist TraceStore and build head activations
    act_dir = args.out + ".acts"
    save_trace_store(args.out + ".traces.jsonl", paired, act_dir)
    records = load_trace_store(args.out + ".traces.jsonl")
    records = retain_paired_problems(records)
    all_acts = build_head_activations(records, monitored)

    # 5. fit + select
    git_sha = _git_sha()
    bank = fit_manifold_bank(
        all_acts, fit_pids, sel_pids, model_id=args.model, benchmark=args.benchmark,
        k=args.k, q=args.q, K=args.K, alpha=args.alpha,
        layers_monitored=monitored, split_seed=config.DEFAULT_SEED, git_sha=git_sha,
        report_pids=report_pids,
    )
    bank.save(args.out)
    print(f"OK mags fit -> {args.out} ({len(bank.selected_heads)} heads selected, "
          f"{bank.n_problems_fit} fit / {bank.n_problems_select} select problems)")

    # 6. fit the baseline banks from the SAME contrastive activation set:
    #    ITI: per-head logistic probes for ALL monitored heads (K reachable in
    #         {24,48,96}); AS: per-layer (d_feat, d_PC0) rotation planes.
    from .baselines import fit_iti_bank, fit_as_bank
    iti_bank = fit_iti_bank(
        all_acts, fit_pids, sel_pids, model_id=args.model, benchmark=args.benchmark,
        K=config.ITI_DEFAULT_K, alpha=config.ITI_DEFAULT_ALPHA,
        layers_monitored=monitored,
    )
    iti_path = args.out.replace(".npz", ".iti.npz")
    iti_bank.save(iti_path)
    print(f"OK iti fit -> {iti_path} ({len(iti_bank.selected_heads)} heads, "
          f"{len(iti_bank.heads)} monitored)")

    as_bank = fit_as_bank(
        all_acts, fit_pids, model_id=args.model, benchmark=args.benchmark,
        angle_deg=config.AS_DEFAULT_ANGLE_DEG, layers_monitored=monitored,
    )
    as_path = args.out.replace(".npz", ".as.npz")
    as_bank.save(as_path)
    print(f"OK as fit -> {as_path} (global plane over {len(as_bank.layers_monitored)} "
          f"monitored layers, applied at all layers)")


def _git_sha():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip()[:12]
    except Exception:
        return ""


if __name__ == "__main__":
    main()
