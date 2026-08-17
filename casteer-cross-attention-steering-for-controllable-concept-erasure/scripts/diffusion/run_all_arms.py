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

On a CPU-only host the paper's full config (50 steps, 4,703 I2P / 800-per-
concept snoopy / 3,000 COCO prompts, 3 seeds) is infeasible (paper used 8xV100,
supplementary.tex:30): is_arm_feasible returns False and the arm emits BLOCKED
without downloading any model. On a CUDA host the same script runs the arms for
real via _run_and_evaluate -- the arm->eval->measured.json driver is
implemented, the per-task steering vector is selected via TASK_VECTOR, the
7-concept Eq.9 average for the I2P-overall task is composed on the fly from the
7 per-concept stores, and the sd14 normalization reference for snoopy/other/
style is generated on demand per steered arm (review-driven fixes this pass:
nudity_total full-set scaling + inconclusive floor, per-prompt sd_seed for I2P,
declared image counts via n_per, bare-concept CS reference text, in-memory sd14
references so the normalized Snoopy claims are reachable on a GPU host).
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
    against, curve x/against lists, AND predicate) -> {arm: set(metrics)}.
    claims.json is the single source of arm/metric names. `predicate` is parsed
    too so an invariant-only claim (e.g. `house`, which has no quantity/against)
    still registers the metrics its predicate references (review F-13: a
    predicate-only metric would otherwise silently never be produced)."""
    out: dict[str, set[str]] = {}
    texts = []
    for c in claims.get("claims", []):
        for key in ("quantity", "against", "predicate"):
            v = c.get(key)
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, list):
                texts.extend(str(x) for x in v)
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
                # Record companion (non-gated) keys the driver produced, e.g.
                # nudity_total_raw recorded beside the gated (scaled)
                # nudity_total so a reader can see both bases (claims.json
                # metrics.nudity_total). Not referenced by any claim.
                for k, v in value.items():
                    if k not in measured[arm][seed]:
                        measured[arm][seed][k] = v
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
    per-task steering vector (review F3), then scores each metric INDEPENDENTLY
    -- a MissingEvaluatorError for one metric (e.g. NudeNet not installed) marks
    that metric BLOCKED but lets the other metrics for the same arm still get
    real values. No success path reports OK on an empty result.

    Review-driven corrections applied here (this pass):
      - I2P uses the per-prompt curated `sd_seed` (vendored run_i2p_eval.py:71),
        not a fresh continuous generator (review: I2P seed-protocol divergence).
      - nudity_total is recorded on the FULL-SET basis (scaled; raw recorded
        under nudity_total_raw) and reports inconclusive (BLOCKED) below the
        declared 2000-prompt floor (review: nudity_total never scaled / below
        the declared floor).
      - snoopy/other/style generate the DECLARED image count (80 templates x
        n_per, 50 classes x n_per), not 80/50 (review: dead n_per under-generated
        below the repo's own >=200 restriction).
      - snoopy_cs / other_cs score against the BARE concept string (vendored
        produce_scores.py:43 -> clip.py:191 `[concept]*N`), not the full
        template (review: CLIP-score reference-text divergence).
      - the sd14 normalization reference for snoopy/other/style is generated
        ON DEMAND per steered arm at the same seed (lazily cached on disk) and
        scored in-memory -- the sd14 arm no longer needs to pre-generate those
        reference sets, and norm_snoopy_cs / mean_norm_others_cs / mean_others_fid
        / vangogh_lpips_e are reachable on a GPU host (review: 6-7 claims
        structurally unsettleable on any host).

    The Q16 classifier checkpoint is not named by the paper and not vendored
    (metrics.q16_inappropriate_count raises by construction), so i2p_overall_pct
    is BLOCKED on every host until the Q16 checkpoint is supplied -- this is an
    honest structural block, not a CPU-only block. The real COCO-30k FID
    reference is not vendored either, so coco_fid30k is BLOCKED until COCO_REF_DIR
    points at the 30k real COCO images. Both are documented in
    measured_blocked_reasons.json.
    """
    import os
    import glob as _glob
    import torch

    from core.runner import (
        run_generation, FULL_CONFIG_MIN_IMAGES, expand_template_prompts,
    )
    from core.data import (
        load_i2p_prompts, load_i2p_seeds, load_coco_captions,
        load_coco_reference_images, load_imagenet_classes, load_clip_templates,
    )
    from core.eval import metrics as EM
    from core.invariants import cpu_invariant_metrics

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

    def _ensure_sd14_reference(rel_dir, prompts, task):
        """Generate vanilla sd14 images for `prompts` at this seed into
        results/arms/sd14/seed{seed}/<rel_dir> if not already cached, return the
        dir. The sd14 arm has beta=None so build_controller returns None (no
        steering) regardless of `task`; the SAME generation config (50 steps,
        512^2, gs 7.5) and the SAME continuous seed stream as the steered arm are
        used, so the steered-vs-vanilla comparison is exactly latent-paired per
        prompt index (experiments.tex:72-73). Reused across steered arms (same
        task+seed+prompts -> same cache dir), so the sd14 reference is generated
        once per seed."""
        ref_dir = os.path.join(REPO, "results", "arms", "sd14",
                               f"seed{seed}", rel_dir)
        existing = sum(len(_glob.glob(os.path.join(ref_dir, f"*.{e}")))
                       for e in EM.EXTENSIONS) if os.path.isdir(ref_dir) else 0
        if existing < len(prompts):
            run_generation("sd14", seed, prompts, ref_dir, task=task, device=device)
        return ref_dir

    # ------------------------------------------------------------------
    # I2P nudity task (nudity_total) and I2P-overall task (i2p_overall_pct)
    # Both use the per-prompt curated sd_seed (review: I2P seed protocol).
    # ------------------------------------------------------------------
    if "nudity_total" in metrics or "i2p_overall_pct" in metrics:
        try:
            i2p_prompts_all = load_i2p_prompts()
            i2p_seeds_all = load_i2p_seeds()
        except Exception as e:
            for m in ("nudity_total", "i2p_overall_pct", "nudity_total_raw"):
                if m in metrics or m == "nudity_total_raw":
                    _blocked(m, f"I2P dataset unavailable: {e}")
            i2p_prompts_all = None

        if i2p_prompts_all is not None:
            n_i2p = FULL_CONFIG_MIN_IMAGES["i2p"]
            i2p_prompts = i2p_prompts_all[:n_i2p]
            i2p_seeds = i2p_seeds_all[:n_i2p]
            n_subset = len(i2p_prompts)

            # nudity erasure (nudity task) -> nudity_total (full-set basis)
            if "nudity_total" in metrics:
                nudity_dir = os.path.join(out_root, "i2p")
                run_generation(arm, seed, i2p_prompts, nudity_dir, task="i2p",
                               device=device, prompt_seeds=i2p_seeds)
                # raw count over the subset
                raw = None
                try:
                    raw = float(EM.nudity_total(nudity_dir))
                    values["nudity_total_raw"] = raw
                except EM.MissingEvaluatorError as e:
                    _blocked("nudity_total", f"MissingEvaluatorError: {e}")
                    _blocked("nudity_total_raw", f"MissingEvaluatorError: {e}")
                except RuntimeError as e:
                    _blocked("nudity_total", f"RuntimeError: {e}")
                    _blocked("nudity_total_raw", f"RuntimeError: {e}")
                if raw is not None:
                    scaled, reason = EM.nudity_scaled_or_inconclusive(
                        int(raw), n_subset)
                    if reason is None:
                        values["nudity_total"] = float(scaled)
                    else:
                        _blocked("nudity_total", reason)
            # 7-class average erasure (i2p_overall task) -> i2p_overall_pct
            if "i2p_overall_pct" in metrics:
                overall_dir = os.path.join(out_root, "i2p_overall")
                run_generation(arm, seed, i2p_prompts, overall_dir,
                                task="i2p_overall", device=device,
                                prompt_seeds=i2p_seeds)
                _safe("i2p_overall_pct", EM.i2p_overall_pct, overall_dir)

    # ------------------------------------------------------------------
    # Snoopy task: snoopy_cs / norm_snoopy_cs / mean_norm_others_cs / mean_others_fid
    # The Snoopy vector erases Snoopy; preservation metrics score the 5 'other'
    # concepts (mickey/spongebob/pikachu/dog/legislator, experiments.tex:64).
    # CLIP reference text = the BARE concept string (vendored produce_scores.py:43
    # -> clip.py:191 `[concept]*N`), matching the paper's CS scale (review: CS
    # reference-text divergence). The sd14 normalization reference is generated
    # on demand (review: structurally-unsettleable claims).
    # ------------------------------------------------------------------
    snoopy_metrics = {"snoopy_cs", "norm_snoopy_cs",
                      "mean_norm_others_cs", "mean_others_fid"}
    if snoopy_metrics & set(metrics):
        try:
            templates = load_clip_templates()
            # Snoopy prompts (the erased concept): each template repeated n_per
            # times to reach the declared count (80 templates x 10 = 800,
            # experiments.tex:67). review: the dead `n_per` previously left this
            # at 80, below the >=200 restriction.
            snoopy_prompts = expand_template_prompts(
                templates, "Snoopy", FULL_CONFIG_MIN_IMAGES["snoopy"])
            snoopy_dir = os.path.join(out_root, "snoopy")
            run_generation(arm, seed, snoopy_prompts, snoopy_dir, task="snoopy",
                           device=device)
            sd14_snoopy_dir = _ensure_sd14_reference("snoopy", snoopy_prompts, "snoopy")
            n_snoop = len(snoopy_prompts)
            # snoopy_cs (arm) -- bare "Snoopy" reference text (paper pipeline)
            if "snoopy_cs" in metrics:
                _safe("snoopy_cs", EM.clip_score_mean, snoopy_dir,
                      EM.cs_reference_prompts("Snoopy", n_snoop), device=device.type)
            # norm_snoopy_cs = snoopy_cs(arm)/snoopy_cs(sd14) at the same seed
            # (experiments.tex:75). Computed from IN-MEMORY sd14 CS, not a disk
            # read of a previous run (review: _sd14_metric read stale measured.json).
            if "norm_snoopy_cs" in metrics:
                try:
                    sd14_cs = float(EM.clip_score_mean(sd14_snoopy_dir,
                                                      EM.cs_reference_prompts("Snoopy", n_snoop),
                                                      device=device.type))
                    arm_cs = values.get("snoopy_cs")
                    if arm_cs is None:
                        arm_cs = float(EM.clip_score_mean(snoopy_dir,
                                                          EM.cs_reference_prompts("Snoopy", n_snoop),
                                                          device=device.type))
                    _safe("norm_snoopy_cs", EM.norm_snoopy_cs, arm_cs, sd14_cs)
                except (EM.MissingEvaluatorError, RuntimeError) as e:
                    _blocked("norm_snoopy_cs", f"Snoopy CS reference error: {e}")
            # 'other' concepts: erase Snoopy, generate mickey/spongebob/pikachu/
            # dog/legislator; score CS (bare concept) and FID vs the sd14 ref.
            others = ["Mickey", "Spongebob", "Pikachu", "dog", "legislator"]
            arm_others_cs = {}
            sd14_others_cs = {}
            arm_others_fid = {}
            for concept in others:
                cprompts = expand_template_prompts(
                    templates, concept, FULL_CONFIG_MIN_IMAGES["other"])
                cdir = os.path.join(out_root, "other", concept)
                run_generation(arm, seed, cprompts, cdir, task="snoopy", device=device)
                sd14_cdir = _ensure_sd14_reference(f"other/{concept}", cprompts, "snoopy")
                nc = len(cprompts)
                try:
                    arm_others_cs[concept] = float(
                        EM.clip_score_mean(cdir, EM.cs_reference_prompts(concept, nc), device=device.type))
                except (EM.MissingEvaluatorError, RuntimeError):
                    pass
                try:
                    sd14_others_cs[concept] = float(
                        EM.clip_score_mean(sd14_cdir, EM.cs_reference_prompts(concept, nc), device=device.type))
                except (EM.MissingEvaluatorError, RuntimeError):
                    pass
                try:
                    arm_others_fid[concept] = float(EM.other_fid(cdir, sd14_cdir))
                except (EM.MissingEvaluatorError, RuntimeError):
                    pass
            if "mean_norm_others_cs" in metrics and arm_others_cs and sd14_others_cs:
                common = {c: arm_others_cs[c] for c in arm_others_cs
                          if c in sd14_others_cs}
                if common:
                    _safe("mean_norm_others_cs", EM.mean_norm_others_cs,
                          common, {c: sd14_others_cs[c] for c in common})
                else:
                    _blocked("mean_norm_others_cs",
                             "no common other-concept CS pairs with sd14 reference")
            elif "mean_norm_others_cs" in metrics:
                _blocked("mean_norm_others_cs",
                         "other-concept CS not produced (CLIP unavailable?)")
            if "mean_others_fid" in metrics and arm_others_fid:
                _safe("mean_others_fid", EM.mean_others_fid, arm_others_fid)
            elif "mean_others_fid" in metrics:
                _blocked("mean_others_fid",
                         "other-concept FID not produced (FID/reference unavailable?)")
        except (EM.MissingEvaluatorError, RuntimeError, FileNotFoundError) as e:
            for m in snoopy_metrics:
                if m in metrics and m not in values:
                    _blocked(m, f"Snoopy task error: {e}")

    # ------------------------------------------------------------------
    # COCO task: coco_fid30k (+ coco_clip30k if requested)
    # Erases nudity on COCO-30k captions (supplementary.tex:544).
    # coco_fid30k compares against the REAL COCO-30k reference (SPEC U10); the
    # sd14 arm computes its OWN coco_fid30k the same way, so coco_fid_vs_vanilla
    # = FID(arm,real) - FID(sd14,real) (both vs the same reference).
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
    # 50 ImageNet classes x n_per to reach the declared >=200 (review: the prior
    # 50-image style set was below the lpips_eval_images survives band [200,1000]).
    # ------------------------------------------------------------------
    if "vangogh_lpips_e" in metrics:
        try:
            classes = load_imagenet_classes()
            # 50 ImageNet classes x n_per to reach the declared >=200 (review:
            # the prior 50-image style set was below the lpips_eval_images
            # survives band [200,1000]). The paper does not state a style-eval
            # prompt count (it defers to SAFREE, experiments.tex:127); we use
            # "{class}, Van Gogh style" prompts.
            n_style = FULL_CONFIG_MIN_IMAGES["style"]
            n_per_s = max(1, (n_style + len(classes) - 1) // len(classes))
            vprompts = [f"{c}, Van Gogh style" for c in classes
                        for _ in range(n_per_s)][:n_style]
            vdir = os.path.join(out_root, "style_vangogh")
            run_generation(arm, seed, vprompts, vdir, task="style", device=device)
            sd14_vdir = _ensure_sd14_reference("style_vangogh", vprompts, "style")
            _safe("vangogh_lpips_e", EM.vangogh_lpips_e, vdir, sd14_vdir)
        except (EM.MissingEvaluatorError, RuntimeError, FileNotFoundError) as e:
            _blocked("vangogh_lpips_e", f"style task error: {e}")

    # Guarantee no requested metric is silently absent.
    for m in metrics:
        if m not in values:
            _blocked(m, "metric not produced by the driver (no OK-on-empty)")
    return values


if __name__ == "__main__":
    main()
