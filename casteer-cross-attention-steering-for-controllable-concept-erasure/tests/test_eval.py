"""Instrument tests for the evaluation metrics.

Every metric that decides whether an output is correct gets a positive test
(accepts a known-correct input) and a negative test (rejects a known-wrong
input). Evaluators whose backing tool is not installed in this CPU venv
(NudeNet, Q16, LPIPS) are exercised by asserting they RAISE
MissingEvaluatorError rather than returning a verdict -- a grader that cannot
run must raise, never return a negative result.
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

from core.eval import metrics as M


def _make_imgs(dirpath, n, seed, color_fn):
    os.makedirs(dirpath, exist_ok=True)
    rng = np.random.RandomState(seed)
    for i in range(n):
        arr = (rng.rand(64, 64, 3) * 255).astype("uint8")
        base = np.array(color_fn(), dtype="uint8")
        arr[:, :] = base
        Image.fromarray(arr).save(os.path.join(dirpath, f"{i:03d}.png"))


# ---------------------------------------------------------------------------
# CLIP score (device-agnostic; ViT-B/32, SPEC U11)
# ---------------------------------------------------------------------------

def test_clip_score_positive_known_correct(tmp_path):
    """Positive: a real (solid blue) image with a matching prompt yields a
    finite, non-negative CLIP score -- the scorer accepts a known-correct pair."""
    d = tmp_path / "imgs"
    _make_imgs(str(d), 1, 0, lambda: (60, 90, 200))
    imgs = [os.path.join(str(d), "000.png")]
    s = M.clip_score_mean(imgs, ["a blue sky"], device="cpu")
    assert np.isfinite(s) and s >= 0.0


def test_clip_score_negative_rejects_empty(tmp_path):
    """Negative: empty image list raises (no OK-on-empty)."""
    with pytest.raises(ValueError):
        M.clip_score_mean([], ["a blue sky"], device="cpu")


def test_clip_score_negative_mismatched_lengths(tmp_path):
    """Negative: mismatched image/prompt counts raise."""
    d = tmp_path / "imgs"
    _make_imgs(str(d), 2, 0, lambda: (60, 90, 200))
    with pytest.raises(ValueError):
        M.clip_score_mean(os.path.join(str(d)), ["one prompt"], device="cpu")


# ---------------------------------------------------------------------------
# FID via clean-fid (CPU-runnable)
# ---------------------------------------------------------------------------

def test_fid_positive_identical_dirs(tmp_path):
    """Positive: FID(dir, dir) == 0 (identical sets)."""
    a = tmp_path / "a"
    _make_imgs(str(a), 6, 0, lambda: (120, 60, 30))
    v = M._fid(str(a), str(a))
    assert abs(v) < 1e-2, f"identical-dir FID should be ~0, got {v}"


def test_fid_negative_different_dirs(tmp_path):
    """Negative: FID between two clearly different image sets is > 0."""
    a = tmp_path / "a"; b = tmp_path / "b"
    _make_imgs(str(a), 6, 0, lambda: (200, 20, 20))
    _make_imgs(str(b), 6, 1, lambda: (20, 20, 200))
    v = M._fid(str(a), str(b))
    assert v > 1.0, f"different-dir FID should be large, got {v}"


def test_fid_negative_rejects_empty(tmp_path):
    """Negative: empty directory raises."""
    empty = tmp_path / "empty"
    os.makedirs(empty)
    a = tmp_path / "a"
    _make_imgs(str(a), 6, 0, lambda: (120, 60, 30))
    with pytest.raises(ValueError):
        M._fid(str(a), str(empty))


# ---------------------------------------------------------------------------
# Normalization helpers (pure arithmetic)
# ---------------------------------------------------------------------------

def test_norm_snoopy_cs_and_mean():
    assert M.norm_snoopy_cs(45.8, 78.5) == pytest.approx(0.5834, abs=1e-4)
    arm = {"mickey": 70.0, "spongebob": 68.0, "pikachu": 71.0, "dog": 72.0, "legislator": 69.0}
    sd14 = {"mickey": 71.0, "spongebob": 70.0, "pikachu": 71.0, "dog": 72.0, "legislator": 70.0}
    assert M.mean_norm_others_cs(arm, sd14) == pytest.approx(np.mean([70/71, 68/70, 71/71, 72/72, 69/70]))
    assert M.mean_others_fid({"a": 50.0, "b": 60.0, "c": 55.0}) == 55.0
    with pytest.raises(ValueError):
        M.mean_norm_others_cs({}, sd14)
    with pytest.raises(ValueError):
        M.mean_others_fid({})


# ---------------------------------------------------------------------------
# External-tool evaluators: must RAISE MissingEvaluatorError when unavailable
# ---------------------------------------------------------------------------

def _pkg_available(modname):
    import importlib.util
    return importlib.util.find_spec(modname) is not None


def test_nudity_total_raises_when_unavailable(tmp_path):
    d = tmp_path / "imgs"
    _make_imgs(str(d), 2, 0, lambda: (100, 100, 100))
    if _pkg_available("nudenet"):
        pytest.skip("nudenet installed; MissingEvaluatorError path not applicable")
    with pytest.raises(M.MissingEvaluatorError):
        M.nudity_total(str(d))


def test_q16_raises_when_unavailable(tmp_path):
    d = tmp_path / "imgs"
    _make_imgs(str(d), 2, 0, lambda: (100, 100, 100))
    # Q16 checkpoint is not vendored/named by the paper -> always raises here.
    with pytest.raises(M.MissingEvaluatorError):
        M.q16_inappropriate_count(str(d))


def test_lpips_raises_when_unavailable(tmp_path):
    d = tmp_path / "imgs"; e = tmp_path / "vanilla"
    _make_imgs(str(d), 2, 0, lambda: (100, 100, 100))
    _make_imgs(str(e), 2, 0, lambda: (100, 100, 100))
    if _pkg_available("lpips"):
        pytest.skip("lpips installed; MissingEvaluatorError path not applicable")
    with pytest.raises(M.MissingEvaluatorError):
        M.vangogh_lpips_e(str(d), str(e))


def test_metrics_no_ok_on_empty_dirs(tmp_path):
    """No metric may report OK on an empty image directory."""
    empty = tmp_path / "empty"
    os.makedirs(empty)
    for fn in [lambda: M.nudity_total(str(empty)),
               lambda: M.q16_inappropriate_count(str(empty))]:
        with pytest.raises((M.MissingEvaluatorError, ValueError)):
            fn()
