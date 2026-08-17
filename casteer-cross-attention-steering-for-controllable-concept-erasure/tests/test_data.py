"""Fingerprint tests for the data pipeline (it is an instrument).

Local-file fingerprints (ImageNet classes, CLIP templates, COCO captions) MUST
pass for real. The I2P fingerprint needs network (HuggingFace) and the COCO
reference needs a vendored image set; those skip with a reason if unavailable
but never substitute synthetic data.
"""
import os
import sys

import pytest

from core.data import (
    load_imagenet_classes,
    load_clip_templates,
    load_coco_captions,
    coco_captions_fingerprint,
)


def test_imagenet_classes_fingerprint():
    cls = load_imagenet_classes(n=50)
    assert len(cls) == 50
    assert cls[0] == "tench", f"first ImageNet class must be 'tench', got {cls[0]!r}"


def test_clip_templates_fingerprint():
    t = load_clip_templates()
    assert len(t) == 80, f"paper states 80 CLIP templates (experiments.tex:67), got {len(t)}"
    assert t[0] == "a bad photo of a {}"


def test_coco_captions_fingerprint():
    prompts = load_coco_captions()
    assert len(prompts) == 30000, f"coco_30k must have 30000 captions, got {len(prompts)}"
    assert prompts[0] == "A bicycle replica with a clock as the front wheel."
    fp = coco_captions_fingerprint()
    assert fp["n"] == 30000
    assert len(fp["sha256_first8"]) == 8


def test_i2p_prompts_fingerprint():
    try:
        from core.data import load_i2p_prompts, i2p_fingerprint
    except Exception as e:  # pragma: no cover
        pytest.skip(f"i2p import failed: {e}")
    try:
        prompts = load_i2p_prompts()
    except Exception as e:
        pytest.skip(f"I2P dataset unavailable in sandbox: {e}")
    assert len(prompts) == 4703, f"paper states 4,703 I2P prompts (experiments.tex:39), got {len(prompts)}"
    fp = i2p_fingerprint()
    assert fp["n"] == 4703
    assert "prompt" in fp["columns"]


def test_i2p_seeds_fingerprint():
    """The paper's I2P protocol uses the dataset's per-prompt curated sd_seed
    (vendored run_i2p_eval.py:71). The loader must expose it aligned 1:1 with
    the prompts and assert the `sd_seed` column exists (review: I2P seed
    protocol divergence). Skips if the dataset is unavailable here."""
    try:
        from core.data import load_i2p_prompts, load_i2p_seeds, i2p_fingerprint
    except Exception as e:  # pragma: no cover
        pytest.skip(f"i2p import failed: {e}")
    try:
        prompts = load_i2p_prompts()
        seeds = load_i2p_seeds()
    except Exception as e:
        pytest.skip(f"I2P dataset unavailable in sandbox: {e}")
    assert len(seeds) == 4703
    assert len(seeds) == len(prompts), "sd_seed must align 1:1 with prompts"
    assert all(isinstance(s, int) for s in seeds)
    fp = i2p_fingerprint()
    assert "sd_seed" in fp["columns"], "I2P must carry the sd_seed column"
    assert fp.get("sd_seed_count") == 4703


def test_coco_reference_fingerprint_blocks_without_vendor():
    """The real COCO-30k FID reference is not vendored here; the loader must
    RAISE (never substitute synthetic images)."""
    from core.data import load_coco_reference_images, coco_reference_fingerprint
    if os.environ.get("COCO_REF_DIR"):
        # if a real reference is vendored, just fingerprint it
        fp = coco_reference_fingerprint()
        assert fp["available"] and fp["n"] > 0
        return
    with pytest.raises(RuntimeError):
        load_coco_reference_images()
    fp = coco_reference_fingerprint()
    assert fp["available"] is False
