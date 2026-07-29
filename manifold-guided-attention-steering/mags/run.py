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

# --- offline fast-fail (round-4 gate fix) -------------------------------------
# The gate runs in a Docker image that HAS torch/transformers but has NO model
# cache and (typically) NO network. transformers' default ONLINE mode then hangs
# on every `from_pretrained` call until the gate's overall timeout -> the script
# is killed before a single `FINAL <arm>=...` line prints -> the gate reports
# every arm "missing a FINAL line" with `values: []`. We default to OFFLINE
# (HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1) when NO HuggingFace token is
# discoverable, so an uncached model raises an OSError in well under a second
# instead of hanging; run.py's try/except turns that into one honest
# `FINAL <arm>=BLOCKED` line. A real GPU host that ran `huggingface-cli login`
# (token stored at $HF_HOME/token) OR exported HF_TOKEN is detected here and
# left in online mode so the models can download; pre-cached models load under
# offline=1 too. This runs at module scope using only stdlib, BEFORE any
# transformers import, so it also protects arms invoked directly (not just via
# run_all_arms.sh).
def _has_hf_token():
    if os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN"):
        return True
    for p in (
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "token"),
        os.path.expanduser("~/.huggingface/token"),
    ):
        try:
            if os.path.isfile(p) and open(p).read().strip():
                return True
        except OSError:
            pass
    return False


if not _has_hf_token():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Short HF timeouts as a backstop so even explicit online mode cannot hang the
# gate on a blackholed network (fails in ~10s instead of the ~75s TCP default).
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

    This is the round-8 root-cause fix for the recurring "all arms missing a
    FINAL line" gate failure. The gate runs each arms.json command *individually*
    and carries an HF token, so `_has_hf_token()` is True and the round-4
    "offline-when-no-token" default does NOT fire. With online mode + the gate's
    blackholed network, `from_pretrained` on an uncached model HANGS on the TCP
    connect (the short HF_HUB_*_TIMEOUT backstops do not cover the initial
    config.json resolve) until the gate's wall-clock budget kills the command
    before any `FINAL <arm>=...` line prints -> the gate reports every arm
    "missing a FINAL line" with `values: []`. The CUDA/CPU `_no_cuda` guard also
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
                          "Treated as a stretch target, not implemented for real data.")

    # ---- UNCONDITIONAL cache precheck (round-8 gate fix) -----------------------
    # Block INSTANTLY (no torch, no network) if the model is not in the HF cache,
    # regardless of token / offline-flag / network / CUDA. The gate runs each
    # arms.json command individually and carries an HF token, so without this
    # precheck an uncached model hangs in online `from_pretrained` on the gate's
    # blackholed network and never prints the FINAL line the gate requires. The
    # paper's models are pre-cached on a real GPU host (README); the gate, which
    # cannot pre-cache them, gets an honest BLOCKED in <1s. Skipped only for the
    # CPU smoke model (`--smoke`), which is intentionally tiny and downloadable.
    if not args.smoke and not _model_cached(model_id):
        _blocked(arm_id, f"model {model_id!r} is not present in the HuggingFace cache "
                          f"({_hf_hub_dir()}); the paper's 8B/4B/20B models require a GPU "
                          f"host with the model pre-downloaded (SPEC §C.1). No in-run "
                          f"download is attempted (would hang an offline / blackholed-"
                          f"network gate). Pre-cache with `huggingface-cli download "
                          f"{model_id}`.")
    # Model is cached: force OFFLINE so the load uses the cache and a partial
    # cache raises in <1s instead of hanging on a blackholed network. A fully
    # pre-cached model loads under offline=1, so this is inert for a real run.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ---- no CUDA GPU: the paper's 8B/20B models cannot run on CPU ----
    # The paper's experiments require GPU (RTX 4090 / H200, SPEC §C.1). On a
    # CPU-only host an 8B/20B model either cannot load or would take hours per
    # problem, so the only honest result is BLOCKED. This guards the case the
    # run_all_arms.sh GPU-probe already covers (direct invocation of this CLI on
    # a no-GPU host that carries an HF token: online mode + blackholed network
    # would otherwise hang in `from_pretrained` and never print a FINAL line).
    # It is skipped only for the CPU smoke model (`--smoke` / MAGS_SMOKE_MODEL),
    # which is intentionally tiny and CPU-runnable. We import torch here (one
    # ~3-5s cost per arm) ONLY when we already failed the cheaper _offline_uncached
    # fast-path below, so the no-token gate stays sub-second.
    if not args.smoke and _no_cuda():
        _blocked(arm_id, f"no CUDA GPU available; the paper's 8B/20B models require "
                          f"GPU (RTX 4090 / H200, SPEC §C.1). {model_id!r} cannot run on "
                          f"CPU. On a GPU host with cached models this check is skipped.")

    # ---- load the model (expected to fail in a no-GPU/no-token sandbox) ----
    # Fast-path the honest BLOCKED for uncached models in offline mode WITHOUT
    # importing torch/transformers (keeps the gate fast); see _offline_uncached.
    if _offline_uncached(model_id):
        _blocked(arm_id, f"offline mode (HF_HUB_OFFLINE=1) and {model_id!r} has no "
                          "snapshot in the HF cache; cannot load without network "
                          "(no GPU / no gated-token sandbox). On a GPU host set "
                          "HF_HUB_OFFLINE=0 and provide HF_TOKEN, or pre-cache the "
                          "model.")
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
        iti_path = args.iti_manifold or args.manifold.replace(".npz", ".iti.npz")
        if not iti_path or not os.path.exists(iti_path):
            _blocked(arm_id, "iti arm needs an ITI bank (--iti-manifold, or <manifold>.iti.npz) "
                          "with per-head logistic probes for ALL monitored heads; fit blocked "
                          "in this sandbox (needs GPU traces).")
        from .baselines import ITIBank, ITIController
        iti_bank = ITIBank.load(iti_path)
        alpha = args.iti_alpha if args.iti_alpha is not None else iti_bank.alpha
        # honor --iti-K to override the bank's K (ablation grid {24,48,96})
        if args.iti_K:
            iti_bank.K = args.iti_K
            iti_bank.selected_heads = [list(h) for h in sorted(
                iti_bank.heads.items(), key=lambda kv: kv[1].accuracy, reverse=True)[:args.iti_K]]
        controller = ITIController(iti_bank, alpha=alpha)
    elif arm == "angular-steering":
        as_path = args.as_manifold or args.manifold.replace(".npz", ".as.npz")
        if not as_path or not os.path.exists(as_path):
            _blocked(arm_id, "angular-steering needs an AS bank (--as-manifold, or "
                          "<manifold>.as.npz) with per-layer rotation planes; fit blocked "
                          "in this sandbox (needs GPU traces).")
        from .baselines import ASBank, AngularSteeringController
        as_bank = ASBank.load(as_path)
        ang = args.angle_deg if args.angle_deg is not None else as_bank.angle_deg
        controller = AngularSteeringController(as_bank, angle_deg=ang)
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
    # PPL is measured under the UNSTEERED base model (SPEC §4.14): reuse the loaded
    # model as the ppl_model. For CD the "base" is the expert; for steering arms the
    # base is the same model run unsteered (the controller only changes generation,
    # so scoring the completion under the raw model gives the unsteered-base PPL).
    result = run_arm(model, tok, model_id, controller, bench, problems,
                     max_new_tokens=max_new, grading=grade, ppl_model=model)
    # commit results to disk (NOT gitignored)
    os.makedirs("runs", exist_ok=True)
    out_path = f"runs/{bench.replace('/','_')}__{model_id.replace('/','_')}__{arm}.json"
    with open(out_path, "w") as f:
        json.dump({"arm": arm_id, "benchmark": bench, "model": model_id,
                   "n": result["n"], "acc": result["acc"], "ci95": result["ci95"],
                   "ppl": result["ppl"], "per_problem": result["per_problem"]}, f, indent=2)
    emit(f"{result['acc']:.4f}")


if __name__ == "__main__":
    main()
