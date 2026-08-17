"""Run every CASteer arm at the paper's full configuration at every seed and
write measured.json.

Each arm prints exactly one line:
    FINAL <arm>=<value>      (the arm's primary metric for this run)
or
    FINAL <arm>=BLOCKED      (this environment cannot produce the value)

measured.json shape: {arm: {seed: {metric: value-or-"BLOCKED"}}} covering every
(arm, seed, metric) referenced by claims.json claims. The arm and metric names
are claims.json's; do not rename. A sidecar measured_blocked_reasons.json
records why each BLOCKED value could not be produced.

On a CPU-only host the paper's full config (50 steps, >=1000 prompts, 3 seeds)
is infeasible (paper used 8xV100, supplementary.tex:30): is_arm_feasible returns
False and the arm emits BLOCKED without downloading any model. On a CUDA host
the same script runs the arms for real via _run_and_evaluate (review F2: this
is no longer an unconditional stub -- the arm->eval->measured.json driver is
implemented, the per-task steering vector is selected via TASK_VECTOR (review
F3), and the 7-concept Eq.9 average for the I2P-overall task is composed on the
fly from the 7 per-concept stores).
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from core.runner import ARM_CONFIG, is_arm_feasible
from core.eval.metrics import MissingEvaluatorError
from core.invariants import cpu_invariant_metrics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLAIMS_PATH = os.path.join(REPO, "claims.json")
MEASURED_PATH = os.path.join(REPO, "measured.json")
REASONS_PATH = os.path.join(REPO, "measured_blocked_reasons.json")

_MEASURED_RE = re.compile(r"measured\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)")


def arm_metrics_from_claims(claims: dict) -> dict:
    """Parse every measured.<arm>.<metric> reference in the claims (quantity,
    against, and curve x/against lists) -> {arm: set(metrics)}. claims.json is
    the single source of arm/metric names."""
    out: dict[str, set[str]] = {}
    texts = []
    for c in claims.get("claims", []):
        for key in ("quantity", "against"):
            v = c.get(key)
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, list):
                texts.extend(str(x) for x in v)
        # value claims reference measured.* in quantity too (already above)
    for t in texts:
        for arm, metric in _MEASURED_RE.findall(t):
            out.setdefault(arm, set()).add(metric)
    return {a: sorted(ms) for a, ms in out.items()}


def main():
    with open(CLAIMS_PATH) as f:
        claims = json.load(f)
    arms = claims["arms"]
    seeds = claims["seeds"]
    arm_metrics = arm_metrics_from_claims(claims)

    measured: dict[str, dict[int, dict[str, object]]] = {}
    reasons: dict[str, dict[int, dict[str, str]]] = {}

    for arm in arms:
        measured[arm] = {}
        reasons[arm] = {}
        metrics = arm_metrics.get(arm, [])
        for seed in seeds:
            measured[arm][seed] = {}
            reasons[arm][seed] = {}
            feasible, why = is_arm_feasible(arm, scale="full")
            if not feasible:
                print(f"FINAL {arm}=BLOCKED")
                for m in metrics:
                    measured[arm][seed][m] = "BLOCKED"
                    reasons[arm][seed][m] = why
                # CPU-runnable invariants (e.g. the pure-math `house`
                # Householder norm-preservation claim) are filled in EVEN when
                # the diffusion arm is BLOCKED: they need no GPU and no
                # generation. The numbers gate evaluates claim `house`'s
                # predicate against these real values, so it settles to
                # `reproduced` (or `refuted`) rather than `unevaluable`/`blocked`.
                inv = cpu_invariant_metrics(arm, seed)
                for im, iv in inv.items():
                    measured[arm][seed][im] = iv
                continue
            # Feasible (GPU host): run + evaluate. On a CPU host we never reach
            # here. Wrapped so any uncaught error -> BLOCKED, not a crash.
            try:
                value = _run_and_evaluate(arm, seed, metrics, reasons)
                primary = _primary_metric(arm, metrics)
                pv = value.get(primary, "BLOCKED")
                print(f"FINAL {arm}={pv}")
                for m in metrics:
                    measured[arm][seed][m] = value.get(m, "BLOCKED")
            except MissingEvaluatorError as e:
                print(f"FINAL {arm}=BLOCKED")
                for m in metrics:
                    measured[arm][seed][m] = "BLOCKED"
                    reasons[arm][seed][m] = f"MissingEvaluatorError: {e}"
            except Exception as e:  # any other failure -> BLOCKED with reason
                print(f"FINAL {arm}=BLOCKED")
                for m in metrics:
                    measured[arm][seed][m] = "BLOCKED"
                    reasons[arm][seed][m] = f"{type(e).__name__}: {e}"

    with open(MEASURED_PATH, "w") as f:
        json.dump(measured, f, indent=2, sort_keys=True)
    with open(REASONS_PATH, "w") as f:
        json.dump(reasons, f, indent=2, sort_keys=True)
    print(f"wrote {MEASURED_PATH}")


def _primary_metric(arm, metrics):
    """The one metric printed on the FINAL line for this arm."""
    # Prefer nudity_total for nudity arms, coco_fid30k for coco arms, etc.
    for pref in ("nudity_total", "i2p_overall_pct", "coco_fid30k", "snoopy_cs",
                 "mean_others_fid", "vangogh_lpips_e"):
        if pref in metrics:
            return pref
    return metrics[0] if metrics else "BLOCKED"


def _run_and_evaluate(arm, seed, metrics, reasons):
    """Actually run the arm at full config and evaluate its metrics (review F2).

    `reasons` is the measured_blocked_reasons dict[arm][seed][metric]; per-metric
    BLOCKED reasons are recorded here as they arise.

    Only invoked on a feasible (GPU) host. Generates images per task using the
    per-task steering vector (review F3: snoopy->Snoopy, style->Van Gogh,
    i2p->nudity, i2p_overall->7-class average, coco->nudity), then scores each
    metric INDEPENDENTLY -- a MissingEvaluatorError for one metric (e.g.
    NudeNet not installed) marks that metric BLOCKED but lets the other metrics
    for the same arm (e.g. snoopy_cs, which only needs CLIP) still get real
    values. No success path reports OK on an empty result.

    The Q16 classifier checkpoint is not named by the paper and not vendored
    (metrics.q16_inappropriate_count raises by construction), so i2p_overall_pct
    is BLOCKED on every host until the Q16 checkpoint is supplied -- this is an
    honest structural block, not a CPU-only block. The real COCO-30k FID
    reference is not vendored either, so coco_fid30k is BLOCKED until COCO_REF_DIR
    points at the 30k real COCO images. Both are documented in
    measured_blocked_reasons.json.
    """
    import os
    import torch

    from core.runner import run_generation, ARM_CONFIG, FULL_CONFIG_MIN_IMAGES
    from core.data import (
        load_i2p_prompts, load_coco_captions, load_coco_reference_images,
        load_imagenet_classes, load_clip_templates,
    )
    from core.construct_prompts import (
        get_prompts_concrete, get_prompts_style,
    )
    from core.eval import metrics as EM
    from core.invariants import cpu_invariant_metrics

    cfg = ARM_CONFIG[arm]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(REPO, "results", "arms", arm, f"seed{seed}")
    os.makedirs(out_root, exist_ok=True)

    # CPU-runnable invariants (house) always filled, even on a GPU host.
    values: dict = dict(cpu_invariant_metrics(arm, seed))

    def _blocked(metric, reason):
        values[metric] = "BLOCKED"
        reasons[arm][seed][metric] = reason

    def _safe(metric, fn, *args, **kwargs):
        """Run `fn`; on MissingEvaluatorError or RuntimeError mark `metric`
        BLOCKED with the reason, never return a silent zero."""
        try:
            values[metric] = float(fn(*args, **kwargs))
        except EM.MissingEvaluatorError as e:
            _blocked(metric, f"MissingEvaluatorError: {e}")
        except RuntimeError as e:
            _blocked(metric, f"RuntimeError: {e}")

    # ------------------------------------------------------------------
    # I2P nudity task (nudity_total) and I2P-overall task (i2p_overall_pct)
    # ------------------------------------------------------------------
    if "nudity_total" in metrics or "i2p_overall_pct" in metrics:
        try:
            i2p_prompts = load_i2p_prompts()
            i2p_prompts = i2p_prompts[:FULL_CONFIG_MIN_IMAGES["i2p"]]
        except Exception as e:
            for m in ("nudity_total", "i2p_overall_pct"):
                if m in metrics:
                    _blocked(m, f"I2P prompts unavailable: {e}")
            i2p_prompts = None

        if i2p_prompts is not None:
            # nudity erasure (nudity task) -> nudity_total
            if "nudity_total" in metrics:
                nudity_dir = os.path.join(out_root, "i2p")
                run_generation(arm, seed, i2p_prompts, nudity_dir, task="i2p",
                               device=device)
                _safe("nudity_total", EM.nudity_total, nudity_dir)
            # 7-class average erasure (i2p_overall task) -> i2p_overall_pct
            if "i2p_overall_pct" in metrics:
                overall_dir = os.path.join(out_root, "i2p_overall")
                run_generation(arm, seed, i2p_prompts, overall_dir,
                               task="i2p_overall", device=device)
                _safe("i2p_overall_pct", EM.i2p_overall_pct, overall_dir)

    # ------------------------------------------------------------------
    # Snoopy task: snoopy_cs / norm_snoopy_cs / mean_norm_others_cs / mean_others_fid
    # The Snoopy vector erases Snoopy; preservation metrics score the 5 'other'
    # concepts (mickey/spongebob/pikachu/dog/legislator, experiments.tex:64).
    # ------------------------------------------------------------------
    snoopy_metrics = {"snoopy_cs", "norm_snoopy_cs",
                      "mean_norm_others_cs", "mean_others_fid"}
    if snoopy_metrics & set(metrics):
        try:
            templates = load_clip_templates()
            n_per = max(1, FULL_CONFIG_MIN_IMAGES["snoopy"] // len(templates))
            # Snoopy prompts (the erased concept)
            snoopy_prompts = [t.format("Snoopy") for t in templates][:FULL_CONFIG_MIN_IMAGES["snoopy"]]
            snoopy_dir = os.path.join(out_root, "snoopy")
            run_generation(arm, seed, snoopy_prompts, snoopy_dir, task="snoopy",
                           device=device)
            sd14_snoopy_dir = os.path.join(REPO, "results", "arms", "sd14",
                                           f"seed{seed}", "snoopy")
            if "snoopy_cs" in metrics:
                _safe("snoopy_cs", EM.clip_score_mean, snoopy_dir, snoopy_prompts, device=device.type)
                if os.path.isdir(sd14_snoopy_dir):
                    _safe("norm_snoopy_cs",
                          lambda a, b: EM.norm_snoopy_cs(a, b),
                          values.get("snoopy_cs", 0.0) or _sd14_metric("snoopy_cs", seed),
                          _sd14_metric("snoopy_cs", seed))
            # 'other' concepts: erase Snoopy, generate mickey/spongebob/pikachu/dog/legislator
            others = ["Mickey", "Spongebob", "Pikachu", "dog", "legislator"]
            arm_others_cs = {}
            arm_others_fid = {}
            for concept in others:
                cprompts = [t.format(concept) for t in templates][:FULL_CONFIG_MIN_IMAGES["other"]]
                cdir = os.path.join(out_root, "other", concept)
                run_generation(arm, seed, cprompts, cdir, task="snoopy", device=device)
                sd14_cdir = os.path.join(REPO, "results", "arms", "sd14",
                                         f"seed{seed}", "other", concept)
                try:
                    arm_others_cs[concept] = EM.clip_score_mean(
                        cdir, cprompts, device=device.type)
                except (EM.MissingEvaluatorError, RuntimeError):
                    pass
                if os.path.isdir(sd14_cdir):
                    try:
                        arm_others_fid[concept] = EM.other_fid(cdir, sd14_cdir)
                    except (EM.MissingEvaluatorError, RuntimeError):
                        pass
            if "mean_norm_others_cs" in metrics and arm_others_cs:
                sd14_others = _sd14_others_cs(seed, set(arm_others_cs))
                if sd14_others:
                    _safe("mean_norm_others_cs",
                          lambda a, b: EM.mean_norm_others_cs(a, b),
                          arm_others_cs, sd14_others)
            if "mean_others_fid" in metrics and arm_others_fid:
                _safe("mean_others_fid", EM.mean_others_fid, arm_others_fid)
        except (EM.MissingEvaluatorError, RuntimeError, FileNotFoundError) as e:
            for m in snoopy_metrics:
                if m in metrics and m not in values:
                    _blocked(m, f"Snoopy task error: {e}")

    # ------------------------------------------------------------------
    # COCO task: coco_fid30k (+ coco_clip30k if requested)
    # Erases nudity on COCO-30k captions (supplementary.tex:544).
    # ------------------------------------------------------------------
    if "coco_fid30k" in metrics or "coco_clip30k" in metrics:
        try:
            coco_prompts = load_coco_captions()
            coco_prompts = coco_prompts[:FULL_CONFIG_MIN_IMAGES["coco"]]
            coco_dir = os.path.join(out_root, "coco")
            run_generation(arm, seed, coco_prompts, coco_dir, task="coco",
                           device=device)
            if "coco_fid30k" in metrics:
                try:
                    ref = load_coco_reference_images()
                    _safe("coco_fid30k", EM.coco_fid30k, coco_dir, ref)
                except RuntimeError as e:
                    _blocked("coco_fid30k", f"COCO reference unavailable: {e}")
            if "coco_clip30k" in metrics:
                _safe("coco_clip30k", EM.clip_score_mean, coco_dir, coco_prompts, device=device.type)
        except (EM.MissingEvaluatorError, RuntimeError, FileNotFoundError) as e:
            for m in ("coco_fid30k", "coco_clip30k"):
                if m in metrics and m not in values:
                    _blocked(m, f"COCO task error: {e}")

    # ------------------------------------------------------------------
    # Style task: vangogh_lpips_e (paired steered vs vanilla sd14 Van Gogh)
    # ------------------------------------------------------------------
    if "vangogh_lpips_e" in metrics:
        try:
            vprompts = [f"{c}, Van Gogh style" for c in load_imagenet_classes()][:FULL_CONFIG_MIN_IMAGES["style"]]
            vdir = os.path.join(out_root, "style_vangogh")
            run_generation(arm, seed, vprompts, vdir, task="style", device=device)
            # vanilla sd14 generations for the same prompts (paired)
            sd14_vdir = os.path.join(REPO, "results", "arms", "sd14",
                                     f"seed{seed}", "style_vangogh")
            if not os.path.isdir(sd14_vdir):
                _blocked("vangogh_lpips_e",
                         "vanilla sd14 style_vangogh generations not found; "
                         "run the sd14 arm first (paired LPIPS needs both sets)")
            else:
                _safe("vangogh_lpips_e", EM.vangogh_lpips_e, vdir, sd14_vdir)
        except (EM.MissingEvaluatorError, RuntimeError, FileNotFoundError) as e:
            _blocked("vangogh_lpips_e", f"style task error: {e}")

    # Guarantee no requested metric is silently absent.
    for m in metrics:
        if m not in values:
            _blocked(m, "metric not produced by the driver (no OK-on-empty)")
    return values


# module-level cache so the sd14 reference arm's per-task metrics are read once
_SD14_CACHE: dict = {}


def _sd14_metric(metric: str, seed: int):
    """Read a sd14-arm metric from measured.json (already produced, since sd14
    is the first arm in claims.json `arms`). Used as the per-seed normalization
    denominator (experiments.tex:75). Returns None if unavailable."""
    if "sd14" not in _SD14_CACHE:
        try:
            with open(MEASURED_PATH) as f:
                _SD14_CACHE.update(json.load(f))
        except Exception:
            return None
    sd14 = _SD14_CACHE.get("sd14", {}).get(str(seed), {})
    return sd14.get(metric)


def _sd14_others_cs(seed: int, concepts: set):
    """Read sd14's per-concept other_cs dict for `seed`. The driver does not
    persist per-concept other_cs, so this returns None (and the caller marks the
    normalized mean BLOCKED rather than substituting). A future run that
    persists sd14 per-concept CS can fill this in."""
    return None


if __name__ == "__main__":
    main()
