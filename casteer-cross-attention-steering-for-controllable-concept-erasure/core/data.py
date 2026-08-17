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
    Returns the 'prompt' column in file order. The horse-filter that upstream's
    run_with_steering.py applies is left to the caller (it removes 'horse'
    captions because 'horse' is one of the 5 'other concepts' in the Snoopy
    benchmark; experiments.tex:64).
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


def load_i2p_prompts() -> list[str]:
    """The 4,703 I2P prompts (experiments.tex:39; AIML-TUDA/i2p, split 'train').

    Returns the 'prompt' column. The companion columns (sd_seed, categories,
    q16_percentage, nudity_percentage, ...) are available via the raw dataset;
    this loader exposes only the prompts because the paper generates one
    image per prompt and re-scores with NudeNet/Q16 (SPEC U7).
    """
    from datasets import load_dataset

    ds = load_dataset("AIML-TUDA/i2p", split="train")
    n = len(ds)
    if n != 4703:
        raise RuntimeError(f"I2P split has {n} rows; paper states 4,703 (experiments.tex:39)")
    return list(ds["prompt"])


def i2p_fingerprint() -> dict:
    from datasets import load_dataset

    ds = load_dataset("AIML-TUDA/i2p", split="train")
    prompts = list(ds["prompt"])
    joined = "\n".join(prompts[:200]).encode("utf-8")
    return {
        "n": len(prompts),
        "sha256_first8": hashlib.sha256(joined).hexdigest()[:8],
        "columns": list(ds.column_names),
    }


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
