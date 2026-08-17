"""Data pipeline for the CASteer reproduction.

This module is an *instrument*: its tests assert the paper's own datasets by
fingerprint (size, vocabulary, checksum). A data loader that silently falls
back to a synthetic corpus is the worst failure mode recorded in this
project's guidance (a closed-book run once produced seven arms at chance
level by substituting a synthetic corpus). If a dataset is unavailable here,
these functions RAISE; they never substitute synthetic data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Optional

REPRO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _join_root(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPRO_ROOT, path)


def load_imagenet_classes(path: str = "imagenet_classes.txt", n: int = 50) -> list[str]:
    """First n ImageNet class strings (one per line). SPEC U8: upstream's
    imagenet_classes.txt (first 50 lines) supplies the concrete/style prompt pairs.
    """
    p = _join_root(path)
    out: list[str] = []
    with open(p, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError(f"imagenet_classes.txt has only {len(out)} entries, need {n}")
    return out


def load_clip_templates(path: str = "exp/datasets/eval/clip_templates.json") -> list[str]:
    """The 80 CLIP/ImageNet templates (experiments.tex:67, SPEC U8)."""
    p = _join_root(path)
    with open(p, "r") as f:
        t = json.load(f)
    if not isinstance(t, list):
        raise RuntimeError(f"clip templates file is not a list: {p}")
    return t


def load_coco_captions(path: str = "exp/datasets/eval/coco/coco_30k.csv") -> list[str]:
    """The 30,000 COCO captions used for COCO-30k generation (paper 'coco_30k').
    Returns the 'prompt' column in file order. No caption filtering is applied
    here: the loader returns ALL 30,000 prompts.

    (Review note: upstream's `scripts/diffusion/run_with_steering.py:38` applies
    a `horse` filter to COCO captions, but the paper never mentions any filter,
    and the prior justification for it here -- that 'horse' is one of the 5
    'other concepts' in the Snoopy benchmark -- was FALSE: the 5 other concepts
    are Mickey, Spongebob, Pikachu, dog, legislator (experiments.tex:64), with no
    horse. The filter is upstream's choice, not the paper's; this loader does
    not apply it so the COCO-30k FID reference set matches the paper's 30k. The
    gated driver in run_all_arms.py uses this loader directly, so the gated
    COCO-30k FID is computed on the unfiltered 30k, matching the paper's
    SD-1.4=14.04 anchor (merge.tex:77).)
    """
    p = _join_root(path)
    prompts: list[str] = []
    with open(p, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "prompt" not in reader.fieldnames:
            raise RuntimeError(f"coco csv missing 'prompt' column: {reader.fieldnames}")
        for row in reader:
            prompts.append(row["prompt"])
    if len(prompts) < 30000:
        raise RuntimeError(f"coco_30k.csv has only {len(prompts)} prompts, need >=30000")
    return prompts


def coco_captions_fingerprint(path: str = "exp/datasets/eval/coco/coco_30k.csv") -> dict:
    prompts = load_coco_captions(path)
    joined = "\n".join(prompts[:100]).encode("utf-8")
    return {
        "n": len(prompts),
        "first": prompts[0],
        "sha256_first8": hashlib.sha256(joined).hexdigest()[:8],
    }


I2P_FULL_SET = 4703  # experiments.tex:39


def _load_i2p_dataset():
    """Load the AIML-TUDA/i2p 'train' split once. Raises if the row count is not
    4,703 (the paper-stated size) -- never substitutes a synthetic corpus."""
    from datasets import load_dataset

    ds = load_dataset("AIML-TUDA/i2p", split="train")
    n = len(ds)
    if n != I2P_FULL_SET:
        raise RuntimeError(f"I2P split has {n} rows; paper states 4,703 (experiments.tex:39)")
    return ds


def load_i2p_prompts() -> list[str]:
    """The 4,703 I2P prompts (experiments.tex:39; AIML-TUDA/i2p, split 'train').

    Returns the 'prompt' column. The companion columns (sd_seed, categories,
    q16_percentage, nudity_percentage, ...) are available via load_i2p_seeds()
    and the raw dataset; this loader exposes the prompts because the paper
    generates one image per prompt and re-scores with NudeNet/Q16 (SPEC U7).
    """
    ds = _load_i2p_dataset()
    return list(ds["prompt"])


def load_i2p_seeds() -> list[int]:
    """The per-prompt curated `sd_seed` column of the 4,703 I2P prompts, aligned
    1:1 with load_i2p_prompts() (AIML-TUDA/i2p, split 'train').

    The paper's I2P eval protocol -- and the vendored
    `scripts/diffusion/run_i2p_eval.py:71` (`seed=row['sd_seed']`) that produced
    the paper's numbers -- generates each I2P image with its CURATED per-prompt
    seed. These are the very seeds from which the SD-1.4 Total=646 anchor
    (sd14_tables/nudity.tex:10) and every prior-art nudity count were produced.
    Fresh seeds (one continuous generator per arm-seed) would plausibly
    DEFLATE nudity counts on every arm (the curated seeds elicit the
    inappropriate content), making the gated orderings easier to pass than under
    the paper's protocol. The gated driver therefore uses these per-prompt
    seeds for the I2P task (review: I2P seed-protocol divergence, fixed).

    Raises if the `sd_seed` column is absent (no silent fallback to a fresh
    seed stream). Returns int seeds in dataset order.
    """
    ds = _load_i2p_dataset()
    if "sd_seed" not in ds.column_names:
        raise RuntimeError(
            f"I2P dataset missing 'sd_seed' column; have {ds.column_names}. The "
            "paper's I2P protocol needs the curated per-prompt seed (vendored "
            "run_i2p_eval.py:71); refusing to fall back to a fresh seed stream."
        )
    return [int(s) for s in ds["sd_seed"]]


def i2p_fingerprint() -> dict:
    ds = _load_i2p_dataset()
    prompts = list(ds["prompt"])
    joined = "\n".join(prompts[:200]).encode("utf-8")
    cols = list(ds.column_names)
    fp = {
        "n": len(prompts),
        "sha256_first8": hashlib.sha256(joined).hexdigest()[:8],
        "columns": cols,
    }
    if "sd_seed" in cols:
        seeds = list(ds["sd_seed"])
        fp["sd_seed_first"] = int(seeds[0])
        fp["sd_seed_count"] = len(seeds)
    return fp


def load_coco_reference_images() -> str:
    """The REAL COCO-30k reference image set used as the FID reference for
    coco_fid30k (SPEC U10; SD-1.4 vs real COCO-30k FID = 14.04, merge.tex:77).

    The erasure literature (ESD/UCE/Receler) uses the 30k unlabeled COCO images.
    That exact reference set is NOT vendored in this sandbox and is not
    available as a single canonical HuggingFace dataset that we can fingerprint
    to the paper's. Obtaining the wrong reference set would make any FID number
    meaningless, so this loader RAISES rather than substituting.

    Returns a directory path if/when the reference is vendored. To enable it,
    place the 30k real COCO images under exp/datasets/eval/coco/reference/ and
    set COCO_REF_DIR to that directory.
    """
    env = os.environ.get("COCO_REF_DIR")
    if env and os.path.isdir(env):
        imgs = [
            f for f in os.listdir(env)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not imgs:
            raise RuntimeError(f"COCO_REF_DIR={env} contains no images")
        return env
    raise RuntimeError(
        "Real COCO-30k FID reference is not available in this sandbox. The paper's "
        "coco_fid30k (merge.tex:77, SD-1.4=14.04) is computed against the real COCO-30k "
        "reference set (SPEC U10), which is not vendored here. Vendor it under "
        "exp/datasets/eval/coco/reference/ and set COCO_REF_DIR to compute coco_fid30k; "
        "do NOT substitute synthetic images."
    )


def coco_reference_fingerprint() -> dict:
    """Fingerprint of the real COCO-30k reference, if available."""
    try:
        d = load_coco_reference_images()
    except RuntimeError as e:
        return {"available": False, "n": 0, "source": None, "reason": str(e)}
    imgs = [f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return {"available": True, "n": len(imgs), "source": d}
