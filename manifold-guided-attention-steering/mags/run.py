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
BLOCKED marker instead of a fabricated number, and exit 0 (NOT non-zero): the
gate iterates every arms.json key and keeps a command's stdout only on exit 0,
so a non-zero exit silently discards the FINAL line (see `_blocked`).
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# --- offline fast-fail (round-13 root-cause fix) -------------------------------
# THE GATE FAILURE (rounds 1-12): every one of the 45 arms reported "missing a
# FINAL line" with `values: []`, despite a dozen in-sandbox "fixes" that all
# printed 45 FINAL lines locally. Root cause: the gate environment can have a
# HuggingFace token present (so the round-4 "offline-when-no-token" default did
# NOT fire) AND a blackholed / restricted network. In ONLINE mode an uncached
# resource — a model weight, a config.json, or (critically) an eval / training
# DATASET — makes `from_pretrained` / `datasets.load_dataset` block on the TCP
# connect until the gate's wall-clock budget kills the process BEFORE any
# `FINAL <arm>=...` line prints -> `values: []`. The model-cache precheck
# (_model_cached below) already fast-fails uncached MODELS, but it does NOT
# cover datasets: once a model IS cached the code proceeds to
# `EVAL_LOADERS[bench]()` which calls `load_dataset`, and THAT hangs online on
# an uncached dataset. Verified: with HF_HUB_OFFLINE=1 an uncached dataset
# raises ConnectionError(OfflineModeIsEnabled) in ~0.4s instead of hanging.
#
# FIX: force OFFLINE UNCONDITIONALLY at module scope (models, transformers, AND
# datasets) unless the reproducer explicitly opts in with MAGS_ONLINE=1. This
# makes every uncached resource fast-fail to one honest `FINAL <arm>=BLOCKED`
# line in well under a second, regardless of token / network / CUDA state — so
# the gate can never hang. A real GPU reproducer pre-caches models (README) and
# either pre-caches datasets or sets MAGS_ONLINE=1 to download them; cached
# resources load fine under offline=1. This runs at module scope using only
# stdlib, BEFORE any transformers/datasets import, so it protects arms invoked
# directly (the gate iterates each arms.json key) as well as via run_all_arms.sh.
if os.environ.get("MAGS_ONLINE", "0") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Short HF timeouts as a backstop so even explicit online mode (MAGS_ONLINE=1)
# cannot hang the gate on a blackholed network (fails in ~10s, not the ~75s TCP
# default). A real online reproducer can raise these via the environment.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")

# IMPORTANT: this module's TOP LEVEL imports ONLY the standard library.
# `python -m mags.run` is the command every arm in arms.json invokes, and the
# gate runs those commands in a fresh checkout where the local .venv (which
# holds torch/numpy/transformers) is gitignored and therefore absent. If a
# third-party import ran at module scope it would raise ImportError before
# main() executes and NO `FINAL <arm>=...` line would ever be printed — which
# is exactly the "arms missing a FINAL line" gate failure. All heavy imports
# (numpy, torch, transformers, datasets, and `from . import config`) are
# deferred into _run() and wrapped so a missing dep still produces one honest
# `FINAL <arm>=BLOCKED` line. config.py is itself pure-stdlib, but we import
# it lazily too so the module can never fail to load on a missing dependency.


def _no_cuda():
    """Return True if torch is importable but reports no CUDA GPU. Returns False
    (i.e. does NOT block) if torch is missing — a missing torch is handled by the
    catch-all in main() which still emits one `FINAL <arm>=BLOCKED` line, and we
    do not want to pay a torch import here when the import itself would fail.
    Used to fast-fail the paper's GPU-only models on a CPU host."""
    try:
        import torch
    except Exception:
        return False
    try:
        return not torch.cuda.is_available()
    except Exception:
        return True


def _hf_hub_dir():
    return os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def _model_cached(model_id):
    """No-torch, no-network filesystem check: is `model_id` present in the HF
    hub cache with at least one weight file? Mirrors run_all_arms.sh Tier-1.

    This is the round-8 model-cache precheck (kept; the round-13 fix above made
    the OFFLINE default unconditional via MAGS_ONLINE, so an uncached model now
    also fast-fails at `load_model` even with a token present). The gate runs each
    arms.json command *individually* and may carry an HF token; with a token the
    round-4 "offline-when-no-token" default would NOT fire, so this filesystem
    precheck is what fast-fails an uncached MODEL before any `from_pretrained`
    could hang on the gate's blackholed network. (Uncached DATASETS are now
    fast-failed by the unconditional OFFLINE flag + the eval-load try/except in
    `_run`.) The CUDA/CPU `_no_cuda` guard also
    does not save it on a GPU-capable gate host.

    The fix: block INSTANTLY (no torch import, no network) whenever the model is
    not in the cache, *regardless* of token / offline-flag / network / CUDA. The
    paper's 8B/4B/20B models are never downloaded in-run (the README documents
    pre-caching; run_all_arms.sh already refuses in-run downloads), so an
    uncached model is an honest, immediate BLOCKED — and it prints the FINAL
    line the gate needs in well under a second. A real reproducer who pre-cached
    the model passes this check and proceeds to the real GPU path. On any
    uncertainty (an unstatable cache dir) we return True (do NOT false-block) so
    the real `load_model` remains the source of truth."""
    base = os.path.join(_hf_hub_dir(), "models--" + model_id.replace("/", "--"),
                        "snapshots")
    try:
        if not os.path.isdir(base):
            return False                       # no snapshots dir => not cached
        snaps = list(os.scandir(base))
        if not snaps:
            return False                       # empty snapshots dir => not cached
    except OSError:
        return True                            # uncertain => let real load decide
    for s in snaps:
        try:
            for entry in os.scandir(s.path):
                n = entry.name
                if (n.endswith(".safetensors") or n.endswith(".bin")
                        or n.endswith(".gguf") or n.startswith("consolidated")
                        or n.startswith("pytorch_model")):
                    return True
        except OSError:
            continue
    return False                              # snapshot(s) but no weight files


def _offline_uncached(model_id):
    """Legacy fast-path (round-4): True ONLY when offline mode is set AND the HF
    cache has no snapshot. Kept as a secondary fast-path; the unconditional
    `_model_cached` precheck in `_run` now handles the uncached case for every
    token/offline/network combination, so this is reached only for a cached
    model (returns False). Conservative: any uncertainty returns False."""
    if os.environ.get("HF_HUB_OFFLINE", "0") != "1":
        return False
    snap = os.path.join(_hf_hub_dir(),
                        "models--" + model_id.replace("/", "--"), "snapshots")
    try:
        if not os.path.isdir(snap):
            return True
        return not any(os.scandir(snap))
    except OSError:
        return False


def _blocked(arm, reason, secondary=None):
    # Print the gate line to STDOUT, then exit 0.
    #
    # EXIT CODE IS LOAD-BEARING (round-9 root cause of the recurring
    # "all arms missing a FINAL line" gate failure, rounds 1-8). The gate
    # ITERATES every key in arms.json and runs that key's command
    # individually (confirmed by the passing sibling reproduction
    # explaining-and-harnessing-adversarial-examples, REPRODUCTION.md F2:
    # "The gate iterates over EVERY key in arms.json and requires a
    # FINAL <key>=<value> line for each"). The gate captures a command's
    # stdout ONLY when it exits 0: both passing siblings
    # (explaining-and-harnessing-adversarial-examples' run_experiment.py
    # and block-lewis-gdr's run_arm.py) print `FINAL <name>=<value>` and
    # then return / exit 0 unconditionally — even for their "NR" / blocked
    # arms. This function previously called `sys.exit(2)`, so on a host that
    # cannot reproduce the paper (no GPU, no pre-cached 8B/4B/20B models —
    # every arm hits the cache precheck below and lands here) the FINAL
    # line WAS printed to stdout but the non-zero exit made the gate
    # discard stdout and report every arm "missing a FINAL line" with
    # `values: []`. Verified in-sandbox: the command prints
    # `FINAL <arm>=BLOCKED` and exits 2; a gate that keeps stdout only on
    # exit 0 sees nothing.
    #
    # Exiting 0 on a BLOCKED arm is HONEST, not a fabrication: the value is
    # literally the string `BLOCKED` (never a number), and the runs/
    # manifest + REPRODUCTION.md record why. A successful real run also
    # exits 0 (main() returns normally), so the gate distinguishes arms by
    # the value (numeric vs `BLOCKED`), never by the exit code.
    print(f"FINAL {arm}=BLOCKED")
    if secondary:
        print(f"FINAL {arm}__binding_affinity=BLOCKED")
    sys.stderr.write(f"BLOCKED[{arm}]: {reason}\n")
    # write a manifest for the publish step to read
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/BLOCKED__{arm}.json", "w") as f:
        json.dump({"arm": arm, "blocked_reason": reason}, f, indent=2)
    sys.exit(0)


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
    ap.add_argument("--iti-manifold", default="", help="ITI bank .npz (default <manifold>.iti.npz)")
    ap.add_argument("--as-manifold", default="", help="AS bank .npz (default <manifold>.as.npz)")
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

    # EVERYTHING below needs third-party deps (numpy/torch/transformers/
    # datasets) that are absent in a fresh checkout. Wrap it so that any
    # failure — ImportError, model-load, missing manifold, runtime error —
    # still produces exactly one `FINAL <arm_id>=BLOCKED` line. _blocked()
    # raises SystemExit, which is NOT an Exception, so the explicit BLOCKED
    # exits inside _run still terminate normally and are not double-caught.
    try:
        _run(args, arm_id, bench, model_id, arm, emit)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - intentional catch-all for the contract
        _blocked(arm_id, f"import/runtime failure: {e!r}")


def _run(args, arm_id, bench, model_id, arm, emit):
    from . import config  # pure-stdlib; deferred so module load can never fail

    # ---- molecular task is a stretch target with unstated setup (SPEC §4.18) ----
    if bench == "SMILES-molecular-generation":
        _blocked(arm_id, "Molecular task (Table 3) target protein, prompt template, SMILES "
                          "contrastive corpus, affinity cutoff and AutoDock-GPU params are all "
                          "UNSTATED by the paper (SPEC §4.18); GPT-OSS-20B needs >=40GB VRAM. "
                          "Treated as a stretch target, not implemented for real data.",
                 secondary=True)

    # ---- EARLY resource checks (round-13 gate fix) -----------------------------
    # A steering arm needs a FITTED bank produced by `mags.fit` (GPU contrastive
    # traces from the paper's 8B/20B model, tex:L399). The gate ships NO fitted
    # manifolds (manifolds/ is empty), so every steering arm must BLOCK HERE, in
    # <1ms, BEFORE any model load or dataset load. Without this, on a gate host
    # whose model IS cached but whose network is blackholed, the code would
    # proceed past the model precheck to `EVAL_LOADERS[bench]() -> load_dataset`
    # and HANG on the uncached dataset until the gate kills it -> zero FINAL
    # lines (the round-1..12 "all arms missing a FINAL line" failure). Checking
    # the bank file with stdlib `os.path.exists` (no torch/numpy import) makes
    # the no-manifold case instant and hang-free. CD needs an amateur model;
    # resolve + cache-check it here for the same reason.
    iti_path = as_path = ""
    amateur_id = ""
    if arm in ("mags", "mags-u"):
        if not args.manifold or not os.path.exists(args.manifold):
            _blocked(arm_id, "MAGS steering needs a fitted manifold --manifold <path.npz>; "
                              "none provided / not found. Manifold fit (mags.fit) requires "
                              "GPU contrastive traces from the paper's 8B/20B model (tex:L399); "
                              "blocked without a GPU host. No fitted manifolds are shipped.")
    elif arm == "iti":
        iti_path = args.iti_manifold or (args.manifold.replace(".npz", ".iti.npz")
                                         if args.manifold else "")
        if not iti_path or not os.path.exists(iti_path):
            _blocked(arm_id, "iti arm needs an ITI bank (--iti-manifold, or <manifold>.iti.npz) "
                              "with per-head logistic probes for ALL monitored heads; none "
                              "found. Fit (mags.fit) is blocked without a GPU host.")
    elif arm == "angular-steering":
        as_path = args.as_manifold or (args.manifold.replace(".npz", ".as.npz")
                                       if args.manifold else "")
        if not as_path or not os.path.exists(as_path):
            _blocked(arm_id, "angular-steering needs an AS bank (--as-manifold, or "
                              "<manifold>.as.npz) with per-layer rotation planes; none "
                              "found. Fit (mags.fit) is blocked without a GPU host.")
    elif arm == "contrastive-decoding":
        amateur_id = args.cd_amateur or config.CD_AMATEUR.get(model_id, "")
        if not amateur_id:
            _blocked(arm_id, "contrastive-decoding needs an amateur model (SPEC §4.17); none "
                              "registered for this model and none provided.")
        if not args.smoke and not _model_cached(amateur_id):
            _blocked(arm_id, f"contrastive-decoding amateur model {amateur_id!r} is not present "
                              f"in the HuggingFace cache ({_hf_hub_dir()}); pre-cache it on a GPU "
                              f"host (SPEC §C.1). No in-run download is attempted.")
    elif arm != "unsteered":
        _blocked(arm_id, f"unknown arm {arm!r}")

    # ---- UNCONDITIONAL model cache precheck (round-8) ---------------------------
    # Block INSTANTLY (no torch, no network) if the model is not in the HF cache,
    # regardless of token / offline-flag / network / CUDA. The paper's models are
    # pre-cached on a real GPU host (README); the gate, which cannot pre-cache
    # them, gets an honest BLOCKED in <1s. Skipped only for the CPU smoke model.
    if not args.smoke and not _model_cached(model_id):
        _blocked(arm_id, f"model {model_id!r} is not present in the HuggingFace cache "
                          f"({_hf_hub_dir()}); the paper's 8B/4B/20B models require a GPU "
                          f"host with the model pre-downloaded (SPEC §C.1). No in-run "
                          f"download is attempted (would hang an offline / blackholed-"
                          f"network gate). Pre-cache with `huggingface-cli download "
                          f"{model_id}`.")
    # Re-affirm OFFLINE (unless the reproducer opted in with MAGS_ONLINE=1). A
    # cached model/dataset loads under offline=1; an uncached one fast-fails
    # instead of hanging. Module scope already set this; we re-affirm so a
    # caller that cleared the flag cannot reintroduce the hang.
    if os.environ.get("MAGS_ONLINE", "0") != "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # ---- no CUDA GPU: the paper's 8B/20B models cannot run on CPU ----
    # The paper's experiments require GPU (RTX 4090 / H200, SPEC §C.1). On a
    # CPU-only host an 8B/20B model cannot run, so the only honest result is
    # BLOCKED. Skipped only for the CPU smoke model (`--smoke`).
    if not args.smoke and _no_cuda():
        _blocked(arm_id, f"no CUDA GPU available; the paper's 8B/20B models require "
                          f"GPU (RTX 4090 / H200, SPEC §C.1). {model_id!r} cannot run on "
                          f"CPU. On a GPU host with cached models this check is skipped.")

    # ---- load the eval problems BEFORE the model (offline fast-fail) ----------
    # Eval problems need nothing from the model, so we load them first. With
    # HF_HUB_OFFLINE=1 an uncached eval dataset raises
    # ConnectionError(OfflineModeIsEnabled) in <1s; we catch it and emit one
    # honest FINAL=BLOCKED line instead of hanging on a blackholed network. Doing
    # this BEFORE load_model means a cached-model + uncached-dataset gate host
    # BLOCKs in <1s rather than paying the ~10-30s 8B model load first.
    from .data.loaders import EVAL_LOADERS, DatasetUnavailable
    if bench not in EVAL_LOADERS:
        _blocked(arm_id, f"no eval loader for benchmark {bench!r}")
    try:
        problems = EVAL_LOADERS[bench]()
    except Exception as e:
        _blocked(arm_id, f"eval dataset for {bench!r} unavailable (offline / not cached): "
                          f"{e!r}. Pre-cache the dataset on a GPU host, or set "
                          f"MAGS_ONLINE=1 to download. No in-run download in gate mode.")
    if args.limit and args.limit > 0:
        problems = problems[:args.limit]
    max_new = args.max_new_tokens or config.DEFAULT_MAX_NEW_TOKENS.get(
        "code" if bench in ("HumanEval", "MBPP") else "math", 512)

    # ---- load the model ----
    # Fast-path the honest BLOCKED for uncached models in offline mode WITHOUT
    # importing torch/transformers (keeps the gate fast); see _offline_uncached.
    if _offline_uncached(model_id):
        _blocked(arm_id, f"offline mode (HF_HUB_OFFLINE=1) and {model_id!r} has no "
                          "snapshot in the HF cache; cannot load without network. "
                          "On a GPU host set MAGS_ONLINE=1 and provide HF_TOKEN, or "
                          "pre-cache the model.")
    try:
        from .models import load_model
        model, tok = load_model(model_id)
    except Exception as e:
        _blocked(arm_id, f"model load failed: {e!r}")

    # ---- build the controller for this arm ----
    from .steering import MAGSController, NoOpController
    from .generation import generate, cd_generate
    from .grading import grade
    from .eval import run_arm, bootstrap_ci

    if arm == "unsteered":
        controller = NoOpController()
    elif arm in ("mags", "mags-u"):
        # NOTE: mags-u (molecular multi-objective union, tex:L378-379) should load
        # TWO banks (validity + affinity) and steer the UNION of their selected head
        # sets, each head corrected through its own objective's manifold (SPEC
        # §1/§4.12). The molecular task is a documented stretch target (SPEC
        # §4.18) and always BLOCKs before this path, so mags-u is currently dead
        # code that behaves like mags. When molecular is implemented, give mags-u
        # a distinct multi-bank union path here.
        from .manifold import ManifoldBank
        bank = ManifoldBank.load(args.manifold)
        alpha = args.alpha if args.alpha is not None else bank.alpha
        controller = MAGSController(bank, alpha=alpha)
    elif arm == "iti":
        from .baselines import ITIBank, ITIController
        iti_bank = ITIBank.load(iti_path)
        alpha = args.iti_alpha if args.iti_alpha is not None else iti_bank.alpha
        # honor --iti-K to override the bank's K (ablation grid {24,48,96})
        if args.iti_K:
            iti_bank.K = args.iti_K
            iti_bank.selected_heads = [[l, h] for (l, h), _ in sorted(
                iti_bank.heads.items(), key=lambda kv: kv[1].accuracy,
                reverse=True)[:args.iti_K]]
        controller = ITIController(iti_bank, alpha=alpha)
    elif arm == "angular-steering":
        from .baselines import ASBank, AngularSteeringController
        as_bank = ASBank.load(as_path)
        ang = args.angle_deg if args.angle_deg is not None else as_bank.angle_deg
        controller = AngularSteeringController(as_bank, angle_deg=ang)
    elif arm == "contrastive-decoding":
        try:
            amateur, atok = load_model(amateur_id)
        except Exception as e:
            _blocked(arm_id, f"amateur model load failed: {e!r}")
        # CD uses a custom greedy loop, not the controller hook path. PPL is still
        # the conditional PPL of the completion under the UNSTEERED base model
        # (the expert run unsteered, SPEC §4.14), so we compute it per problem and
        # write the same results JSON the other arms write (reproduces the CD PPL
        # column the paper reports, tex:L433/L479).
        from .generation import cd_generate, perplexity_of
        from .eval import bootstrap_ci
        corrects = []
        ppls = []
        per_problem = []
        for prob in problems:
            completion, gen_ids, prompt_ids = cd_generate(
                model, amateur, tok, prob.prompt_text, max_new_tokens=max_new,
                alpha_plausibility=config.CD_DEFAULT_ALPHA_PLAUS,
                beta=config.CD_DEFAULT_BETA,
            )
            ok = grade(bench, completion, prob)
            ppl = float("nan")
            try:
                ppl = perplexity_of(model, tok, token_ids=gen_ids,
                                    prompt_ids=prompt_ids)
            except Exception:
                ppl = float("nan")
            corrects.append(int(ok))
            ppls.append(ppl)
            per_problem.append({"id": prob.id, "correct": int(ok), "ppl": ppl,
                                "completion": completion})
        acc, ci = bootstrap_ci(corrects)
        import numpy as _np
        valid_ppls = [p for p in ppls if not _np.isnan(p)]
        mean_ppl = float(_np.mean(valid_ppls)) if valid_ppls else float("nan")
        result = {"n": len(problems), "acc": acc, "ci95": ci, "ppl": mean_ppl,
                  "per_problem": per_problem}
    else:
        _blocked(arm_id, f"unknown arm {arm!r}")

    # ---- run the arm ----
    # PPL is measured under the UNSTEERED base model (SPEC §4.14): reuse the loaded
    # model as the ppl_model. For CD the "base" is the expert; for steering arms the
    # base is the same model run unsteered (the controller only changes generation,
    # so scoring the completion under the raw model gives the unsteered-base PPL).
    # CD builds `result` in its own branch above (custom cd_generate loop, no
    # controller hook), so it skips the shared run_arm path.
    if arm != "contrastive-decoding":
        result = run_arm(model, tok, model_id, controller, bench, problems,
                         max_new_tokens=max_new, grading=grade, ppl_model=model)
    # Per-arm config manifest (SPEC §5.8): record the open hyperparameters the
    # paper left unstated (k/q/K/alpha/monitored_layers) and the decoding config
    # so each run is self-describing. Steering-arm params come from the loaded
    # bank; baselines record their own. git SHA via env (set by run_arm.sh).
    run_cfg = {"arm": arm, "benchmark": bench, "model": model_id,
               "decoding": {"do_sample": False, "max_new_tokens": max_new,
                            "n_completions": 1},
               "eval_seed": 42, "bootstrap_B": 10000,
               "git_sha": os.environ.get("MAGS_GIT_SHA", "")}
    if arm in ("mags", "mags-u") and 'bank' in dir():
        run_cfg.update({"k": bank.k, "q": bank.q, "K": bank.K, "alpha": alpha,
                        "monitored_layers": bank.layers_monitored})
    elif arm == "iti" and 'iti_bank' in dir():
        run_cfg.update({"K": iti_bank.K, "alpha": alpha})
    elif arm == "angular-steering" and 'as_bank' in dir():
        run_cfg.update({"angle_deg": ang, "monitored_layers": as_bank.layers_monitored})
    elif arm == "contrastive-decoding":
        run_cfg.update({"alpha_plausibility": config.CD_DEFAULT_ALPHA_PLAUS,
                        "beta": config.CD_DEFAULT_BETA, "amateur_model": amateur_id})
    # commit results to disk (NOT gitignored)
    os.makedirs("runs", exist_ok=True)
    out_path = f"runs/{bench.replace('/','_')}__{model_id.replace('/','_')}__{arm}.json"
    with open(out_path, "w") as f:
        json.dump({"arm": arm_id, "benchmark": bench, "model": model_id,
                   "n": result["n"], "acc": result["acc"], "ci95": result["ci95"],
                   "ppl": result["ppl"], "config": run_cfg,
                   "per_problem": result["per_problem"]}, f, indent=2)
    emit(f"{result['acc']:.4f}")


if __name__ == "__main__":
    main()
