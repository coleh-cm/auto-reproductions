#!/usr/bin/env python3
"""make_measured.py — run every arm of claims.json at every seed, write measured.json.

This is the harness the numbers gate consumes. For each arm in ``claims.json``
and each seed in ``claims.json['seeds']`` it:

  1. runs the arm's ``command_per_seed`` (with ``{seed}`` substituted), writing
     that run's result to a per-seed file via the experiment script's ``--out``
     flag (every experiment script supports ``--out``);
  2. resolves each of the arm's metrics from that result file using the
     ``<results json>:<json path>`` pointer in ``claims.json['arms'][arm]['metrics']``;
  3. records the value in ``measured[arm][seed][metric]``.

It then writes ``measured.json`` as ``{arm: {seed: {metric: value}}}`` and prints
exactly one ``FINAL <arm>=<headline value>`` line per arm to STDOUT (the value a
reader sees in the log; the gate pairs lines to arms by name). A metric the
environment cannot produce is the string ``"BLOCKED"``.

Sub-scale overrides. The paper's full configuration for ``m5_large_advtrain`` is
1600 units x patience-100 x 5 seeds (tex:497-512), which is infeasible on this
CPU (a single seed at 1600 units did not finish within the 5-minute budget; see
REPRODUCTION.md). The reproduction's established sub-scale is 240 units / 12
select-epochs / patience-10 (the committed ``results/m5_large_advtrain.json``
was produced at exactly this scale). ``make_measured`` appends
``--units 240 --epochs 12`` to the m5 command so the arm actually runs here;
every other arm's ``command_per_seed`` is already sub-scale (DEFAULT_UNITS=240).
This override is recorded in ``measured.json['_meta']['subscale_overrides']``
and in REPRODUCTION.md. The m5 headline magnitude (0.782%) is rated
``compute_invariance=low`` in claims.json precisely because it needs the paper's
full scale; the HIGH m5 claim is the *direction* (c11, advtrain <= baseline),
which the sub-scale reproduces.

Real data, no synthetic fallback. Every arm loads real MNIST via
``fgsm_repro.data.load_mnist`` (raw IDX files under ``mnist/``); the data loader
is fingerprinted in ``instruments.json``. If MNIST is unavailable a run fails
loudly (nonzero exit) and its metrics are recorded BLOCKED — never substituted
with a synthetic corpus.

Runnable::

    python make_measured.py                 # all arms, all seeds -> measured.json
    python make_measured.py --arms m1_softmax_regression,m2_logistic_3v7
    python make_measured.py --dry-run        # print the plan, run nothing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parent
CLAIMS_PATH = REPRO_ROOT / "claims.json"
MEASURED_PATH = REPRO_ROOT / "measured.json"
PER_SEED_DIR = REPRO_ROOT / "results" / "_per_seed"

# Sub-scale overrides appended to a command_per_seed, keyed by the experiment
# script basename. Only m5's default (1600 units) is infeasible on this CPU.
SUBSCALE_OVERRIDES: dict[str, list[str]] = {
    "m5_large_advtrain.py": ["--units", "240", "--epochs", "12"],
}

# The single headline metric per arm for the FINAL log line (mean over seeds
# that produced a number; BLOCKED if any seed blocked). Chosen as the most
# paper-salient metric per arm; the gate settles values from measured.json, not
# these lines, so this is a human-readable summary only.
HEADLINE_METRIC: dict[str, str] = {
    "m1_softmax_regression": "fgsm_error_rate",
    "m2_logistic_3v7": "fgsm_error_rate",
    "m3_maxout240_clean": "fgsm_error_rate",
    "m4_maxout240_advtrain": "adversarial_clean_error",
    "m5_maxout1600_clean": "mean_test_error",
    "m5_maxout1600_advtrain": "mean_test_error",
    "m6_robustness_transfer_eval": "own_fgsm_error",
    "m7_maxout_noise_sign": "fgsm_error_rate",
    "m7_maxout_noise_uniform": "fgsm_error_rate",
    "m8_rbf_shallow": "rbf_fgsm_error_rate",
    "m9_rubbish_evals": "maxout_softmax_rubbish_error",
    "e1_ensemble12_maxout": "ensemble_targeted_error",
    "m_l1_weight_decay": "l1_coeff0025_train_error",
    "f4_eps_curve": "eps_crossover_pos",
}

BLOCKED = "BLOCKED"
# Placeholder for ``derived:`` metric pointers; overwritten by _compute_derived.
# Never reaches measured.json: the post-step replaces it (or BLOCKED on failure).
DERIVED_PLACEHOLDER = "__DERIVED__"


# Derived metrics: cross-arm or cross-seed aggregates the numbers gate cannot
# express inline (its expression evaluator exposes no min/max/mean builtins, so
# a claim that needs ``min(a, b)`` or ``max-min`` across seeds must reference a
# precomputed scalar metric instead). Each entry computes one metric per seed
# from already-assembled per-seed scalars; the value is stored on the named arm
# at every seed (cross-seed aggregates are constant across seeds by design).
#
# ``spec`` strings (after ``derived:``) are matched here:
#   spread_of_per_seed_test_errors          max-min of m5 advtrain per-seed errors
#   min_of_m7_noise_sign_uniform_fgsm_error_rate   per-seed min of the two m7 controls
#   min_of_maxout_softmax_softmax_regression_rubbish_error  per-seed min of the two linear arms
def _compute_derived(measured: dict, seeds: list[int]) -> None:
    def _num(v):
        return v if isinstance(v, (int, float)) and v != DERIVED_PLACEHOLDER else None

    # m5_maxout1600_advtrain.per_seed_spread = max-min across seeds of
    # per_seed_test_errors (c13: paper's 5-seed spread 0.77%..0.83% = 6e-4).
    errs = [
        _num(measured.get("m5_maxout1600_advtrain", {}).get(s, {}).get("per_seed_test_errors"))
        for s in seeds
    ]
    errs = [e for e in errs if e is not None]
    spread = (max(errs) - min(errs)) if errs else BLOCKED
    for s in seeds:
        measured.setdefault("m5_maxout1600_advtrain", {}).setdefault(s, {})["per_seed_spread"] = spread

    # m6.min_noise_control_fgsm_error = min(m7_sign.fgsm_error_rate,
    # m7_uniform.fgsm_error_rate) at each seed (c17: noise controls weaker
    # than adversarial training).
    for s in seeds:
        a = _num(measured.get("m7_maxout_noise_sign", {}).get(s, {}).get("fgsm_error_rate"))
        b = _num(measured.get("m7_maxout_noise_uniform", {}).get(s, {}).get("fgsm_error_rate"))
        vals = [v for v in (a, b) if v is not None]
        measured.setdefault("m6_robustness_transfer_eval", {}).setdefault(s, {})["min_noise_control_fgsm_error"] = (
            min(vals) if vals else BLOCKED
        )

    # m9.min_linear_rubbish_error = min(maxout_softmax_rubbish_error,
    # softmax_regression_rubbish_error) at each seed (c34: RBF less fooled than
    # every linear-built model on identical rubbish draws).
    for s in seeds:
        a = _num(measured.get("m9_rubbish_evals", {}).get(s, {}).get("maxout_softmax_rubbish_error"))
        b = _num(measured.get("m9_rubbish_evals", {}).get(s, {}).get("softmax_regression_rubbish_error"))
        vals = [v for v in (a, b) if v is not None]
        measured.setdefault("m9_rubbish_evals", {}).setdefault(s, {})["min_linear_rubbish_error"] = (
            min(vals) if vals else BLOCKED
        )


def _resolve_pointer(blob: dict, json_path: str):
    """Resolve a dotted json path with a single ``[*]`` array wildcard AND keys
    that themselves contain dots (e.g. ``arms.l1_0.0025.clean_train_error``,
    where ``l1_0.0025`` is one key).

    Peels ONE key off the front of the path at each dict level, greedily
    matching the longest actual key (so a dotted key like ``l1_0.0025`` is
    matched whole before its dot-fragments). ``per_seed[*].test_error`` indexes
    every list element and reads ``test_error`` from each. Returns the leaf
    value, a list (for ``[*]``), or raises KeyError if any step is absent.
    """
    return _resolve_step(blob, json_path)


def _resolve_step(cur, path: str):
    if path == "":
        return cur
    if not isinstance(cur, dict):
        raise KeyError(f"expected dict at {path!r}")
    # Find the longest actual key K that prefixes `path` as either an exact
    # match, a dotted descent (K + '.'), or an array wildcard (K + '[').
    best_key = None
    for k in cur:
        if path == k or path.startswith(k + ".") or path.startswith(k + "["):
            if best_key is None or len(k) > len(best_key):
                best_key = k
    if best_key is None:
        raise KeyError(path)
    remainder = path[len(best_key):]
    if remainder == "":
        return cur[best_key]
    if remainder.startswith("[*]"):
        arr = cur[best_key]
        if not isinstance(arr, list):
            raise KeyError(f"{best_key} is not a list")
        rest = remainder[3:]
        if rest.startswith("."):
            rest = rest[1:]
        return [_resolve_step(el, rest) for el in arr]
    if remainder.startswith("."):
        return _resolve_step(cur[best_key], remainder[1:])
    raise KeyError(f"unparseable remainder {remainder!r} at {path!r}")


def _run_one(cmd_template: str, seed: int, out_file: Path, env: dict) -> tuple[bool, str]:
    """Run one (command, seed) job. Returns (ok, stderr_tail)."""
    # Substitute {seed}; append --out; append sub-scale overrides if any.
    script_match = re.search(r"experiments/(\S+\.py)", cmd_template)
    overrides: list[str] = []
    if script_match:
        overrides = SUBSCALE_OVERRIDES.get(script_match.group(1), [])
    cmd = cmd_template.replace("{seed}", str(seed))
    # run_experiment.py-style scripts (run_experiment.py) take no --out; only
    # experiments/*.py scripts do. The template always names an experiments/ py.
    cmd_list = cmd.split() + ["--out", str(out_file)] + overrides
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd_list, cwd=str(REPRO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except FileNotFoundError as exc:
        return False, f"FileNotFoundError: {exc}"
    dt = time.monotonic() - t0
    ok = (proc.returncode == 0) and out_file.exists()
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    tag = "OK" if ok else "FAIL(rc=%d)" % proc.returncode
    return ok, f"[{tag} {dt:.0f}s] {' '.join(cmd_list)} :: " + " | ".join(tail)


def _metrics_for_arm(blob: dict, metrics: dict[str, str]) -> dict[str, object]:
    """Resolve every metric pointer for one arm from one result blob.

    A metric value stored in ``measured.json`` must be a scalar the gate can
    format (``f"{v:.4f}"``) and aggregate across seeds; a list value crashes the
    gate's evaluator with ``unsupported format string passed to list.__format__``.
    The one pointer that uses ``[*]`` — ``arms.adversarial.per_seed[*].test_error``
    (metric ``per_seed_test_errors``) — indexes a per-seed result file, where the
    ``per_seed`` array contains exactly ONE element (the seed this run trained
    under). So the ``[*]`` resolution yields a one-element list ``[x]``. We
    collapse it to the scalar ``x``: that scalar is THIS seed's test error, and
    when the gate gathers the metric across seeds it gets a flat list of scalars
    (``[x0, x1, x2]``), which ``mean(...)`` / ``max(...)`` / ``min(...)`` in
    claims c12/c13 operate on as the paper intends (the spread across the
    per-seed training runs). A multi-element list would be a genuine ambiguity
    we cannot resolve to one number, so it is left as BLOCKED rather than
    silently flattened.
    """
    out: dict[str, object] = {}
    for metric, pointer in metrics.items():
        # ``derived:<spec>`` pointers are NOT read from the per-seed result
        # file; they name a cross-arm / cross-seed aggregate computed by
        # ``_compute_derived`` after every arm has been assembled. Mark them
        # with a placeholder so the key is present (the post-step overwrites
        # it; if the post-step never runs the gate sees BLOCKED, never a
        # stale or missing value).
        if pointer.startswith("derived:"):
            out[metric] = DERIVED_PLACEHOLDER
            continue
        _file, _, json_path = pointer.partition(":")
        try:
            val = _resolve_pointer(blob, json_path)
        except (KeyError, TypeError, IndexError):
            out[metric] = BLOCKED
            continue
        if isinstance(val, list):
            if len(val) == 1:
                out[metric] = val[0]
            else:
                out[metric] = BLOCKED
        else:
            out[metric] = val
    return out


def _curve_metrics_for_arm(blob: dict, curve_metrics: dict[str, str]) -> dict[str, object]:
    """Resolve every curve_metric pointer for one arm from one result blob.

    Curve metrics are SEQUENCES (one value per x in the figure's sample grid);
    they are stored verbatim as lists in ``measured.json`` so a ``curve`` claim
    can read ``measured.<arm>.<curve_metric>`` directly. The numbers gate applies
    its noise floor point by point, so the list must be the same length and
    order at every seed. ``derived:`` pointers are not permitted for curve
    metrics.
    """
    out: dict[str, object] = {}
    for metric, pointer in curve_metrics.items():
        if pointer.startswith("derived:"):
            out[metric] = BLOCKED
            continue
        _file, _, json_path = pointer.partition(":")
        try:
            val = _resolve_pointer(blob, json_path)
        except (KeyError, TypeError, IndexError):
            out[metric] = BLOCKED
            continue
        if isinstance(val, list) and all(isinstance(v, (int, float)) for v in val):
            out[metric] = [float(v) for v in val]
        elif isinstance(val, (int, float)):
            out[metric] = float(val)
        else:
            out[metric] = BLOCKED
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run all arms x seeds -> measured.json.")
    ap.add_argument("--arms", default="", help="comma-subset of arms to run (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 3),
                    help="concurrent arm runs (default: cpu//3)")
    ap.add_argument("--omp", type=int, default=2,
                    help="OMP_NUM_THREADS/MKL_NUM_THREADS per arm process")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip running; assemble measured.json from existing results/_per_seed files")
    args = ap.parse_args(argv)

    claims = json.loads(CLAIMS_PATH.read_text())
    seeds: list[int] = list(claims["seeds"])
    arms: dict[str, dict] = claims["arms"]
    if args.arms.strip():
        want = {a.strip() for a in args.arms.split(",")}
        arms = {a: v for a, v in arms.items() if a in want}

    # Group arms by command_per_seed template so a shared command (m5 clean/adv,
    # m7 sign/uniform) runs ONCE per seed and feeds both arms.
    cmd_to_arms: dict[str, list[str]] = {}
    for arm, info in arms.items():
        cmd_to_arms.setdefault(info["command_per_seed"], []).append(arm)

    PER_SEED_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp)
    env["MKL_NUM_THREADS"] = str(args.omp)
    # Put the pinned venv first on PATH so the bare ``python`` in each
    # command_per_seed resolves to the venv interpreter (torch is installed
    # only there). If the venv is absent, fall back to the caller's PATH.
    venv_bin = REPRO_ROOT / ".venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
    # Make src importable for any in-process fallback (not used for runs, which
    # are subprocesses, but harmless).
    env["PYTHONPATH"] = str(REPRO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    # Build the job list: (command_template, seed) -> out_file.
    jobs = []
    for cmd_tmpl in cmd_to_arms:
        script_match = re.search(r"experiments/(\S+\.py)", cmd_tmpl)
        stem = script_match.group(1).replace(".py", "") if script_match else "arm"
        for seed in seeds:
            out_file = PER_SEED_DIR / f"{stem}__seed{seed}.json"
            jobs.append((cmd_tmpl, seed, out_file, stem))

    plan_lines = [f"plan: {len(jobs)} runs over {len(cmd_to_arms)} commands x {len(seeds)} seeds, jobs={args.jobs}, omp={args.omp}"]
    for cmd_tmpl, arm_list in cmd_to_arms.items():
        sm = re.search(r"experiments/(\S+\.py)", cmd_tmpl)
        ov = SUBSCALE_OVERRIDES.get(sm.group(1), []) if sm else []
        ovstr = (" +subscale " + " ".join(ov)) if ov else ""
        plan_lines.append(f"  {arm_list}{ovstr}")
    print("\n".join(plan_lines), file=sys.stderr)
    if args.dry_run:
        return 0

    run_log: list[str] = []
    if not args.assemble_only:
        # Run all jobs concurrently. Each writes a unique --out file, so no
        # cross-job file contention.
        done_idx = 0
        t_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_run_one, cmd, seed, outf, env): (cmd, seed, outf, stem)
                       for cmd, seed, outf, stem in jobs}
            for fut in as_completed(futures):
                cmd, seed, outf, stem = futures[fut]
                ok, msg = fut.result()
                done_idx += 1
                run_log.append(msg)
                print(f"[{done_idx}/{len(jobs)} {time.monotonic()-t_start:.0f}s] {msg}", file=sys.stderr)
        print(f"all runs done in {time.monotonic()-t_start:.0f}s", file=sys.stderr)

    # Assemble measured.json from the per-seed result files.
    measured: dict[str, dict[int, dict[str, object]]] = {}
    meta_overrides = {}
    for cmd_tmpl, arm_list in cmd_to_arms.items():
        sm = re.search(r"experiments/(\S+\.py)", cmd_tmpl)
        stem = sm.group(1).replace(".py", "") if sm else "arm"
        if sm and SUBSCALE_OVERRIDES.get(sm.group(1)):
            meta_overrides[sm.group(1)] = SUBSCALE_OVERRIDES[sm.group(1)]
        for seed in seeds:
            out_file = PER_SEED_DIR / f"{stem}__seed{seed}.json"
            if not out_file.exists():
                blob = None
            else:
                try:
                    blob = json.loads(out_file.read_text())
                except json.JSONDecodeError:
                    blob = None
            for arm in arm_list:
                metrics = arms[arm]["metrics"]
                seed_block = (
                    _metrics_for_arm(blob, metrics) if blob is not None
                    else {m: BLOCKED for m in metrics}
                )
                # Curve metrics (sequences) are resolved alongside scalars and
                # stored verbatim as lists so a ``curve`` claim can read its
                # quantity/x/against sequences straight from measured.json.
                curve_metrics = arms[arm].get("curve_metrics", {})
                if curve_metrics:
                    seed_block.update(
                        _curve_metrics_for_arm(blob, curve_metrics)
                        if blob is not None
                        else {m: BLOCKED for m in curve_metrics}
                    )
                measured.setdefault(arm, {})[seed] = seed_block
            # Mirror the seed-0 result into the canonical results/ path so the
            # committed results/*.json files stay in sync with measured.json and
            # remain valid as the gate's <results json> pointers.
            if seed == seeds[0] and blob is not None:
                canonical = REPRO_ROOT / arms[arm_list[0]]["results"]
                canonical.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out_file, canonical)

    # Fill the ``derived:`` metric placeholders now that every arm's scalars
    # are assembled (cross-arm / cross-seed aggregates the gate cannot express
    # inline because its expression evaluator has no min/max/mean builtins).
    _compute_derived(measured, seeds)
    # Any surviving placeholder (a derived metric whose inputs were missing)
    # becomes BLOCKED so the gate treats the claim as blocked, not stale.
    for arm in measured:
        for s in measured[arm]:
            for m, v in list(measured[arm][s].items()):
                if v == DERIVED_PLACEHOLDER:
                    measured[arm][s][m] = BLOCKED

    # Emit one FINAL line per arm (headline metric, mean over non-blocked seeds).
    final_lines = []
    for arm in arms:
        head = HEADLINE_METRIC.get(arm, next(iter(arms[arm]["metrics"])))
        vals = []
        for seed in seeds:
            v = measured.get(arm, {}).get(seed, {}).get(head, BLOCKED)
            if v is not BLOCKED and not isinstance(v, list):
                vals.append(float(v))
        if vals:
            final_val = sum(vals) / len(vals)
            final_lines.append(f"FINAL {arm}={final_val}")
        else:
            final_lines.append(f"FINAL {arm}={BLOCKED}")
    # Print FINAL lines to stdout (the reader-visible gate lines).
    print("\n".join(final_lines))

    out = {
        "_meta": {
            "schema": "measured.<arm>.<seed>.<metric>; arm keys are exactly claims.json['arms']; this _meta key is reserved and ignored by the gate.",
            "seeds": seeds,
            "subscale_overrides": meta_overrides,
            "blocked_sentinel": BLOCKED,
            "headline_metric": HEADLINE_METRIC,
            "run_log": run_log,
        },
    }
    # Arm values live at the TOP LEVEL (measured.<arm>.<seed>.<metric>), exactly
    # the claims.json contract; `_meta` is a single reserved key the gate skips.
    for arm, per_seed in measured.items():
        out[arm] = {str(seed): per_seed[seed] for seed in seeds}
    MEASURED_PATH.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {MEASURED_PATH}", file=sys.stderr)
    # Fail loudly if EVERY arm is fully blocked (method never applied).
    all_blocked = all(
        all(measured[arm][seed][m] == BLOCKED for m in arms[arm]["metrics"] for seed in seeds)
        for arm in arms
    )
    if all_blocked:
        print("ERROR: every arm is fully BLOCKED — method was never applied.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
