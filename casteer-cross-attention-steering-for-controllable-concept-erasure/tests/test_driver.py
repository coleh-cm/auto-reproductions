"""Tests for the run_all_arms driver's pure logic (no GPU, no generation).

Covers:
  - arm_metrics_from_claims parses `predicate` (review F-13: a predicate-only
    metric like `house`'s must be registered, not silently dropped);
  - expand_template_prompts reaches the declared image count (review: the dead
    `n_per` previously left snoopy/style at 80/50 images, below the >=200
    restriction).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.runner import expand_template_prompts, FULL_CONFIG_MIN_IMAGES
from scripts.diffusion.run_all_arms import arm_metrics_from_claims


def test_arm_metrics_from_claims_parses_predicate():
    """A claim with only a `predicate` (no quantity/against) must still
    register the metrics its predicate references. `house` is exactly this case
    (kind=ordering-invariant, predicate references house_pass +
    house_max_norm_err under arm casteer_noclip); a predicate-only metric would
    otherwise silently never be produced (review F-13)."""
    claims = {
        "arms": ["casteer_noclip"],
        "claims": [
            {"id": "house", "kind": "invariant",
             "predicate": "measured.casteer_noclip.house_pass == 1 and "
                           "measured.casteer_noclip.house_max_norm_err < 1e-5"},
            {"id": "x", "kind": "ordering",
             "quantity": "measured.casteer_clip.snoopy_cs - 0.5",
             "against": "[0.6968]"},
            {"id": "y", "kind": "curve",
             "quantity": "[measured.casteer_clip.mean_others_fid]",
             "against": "[77.5]", "x": ["Receler(1.0)"]},
        ],
    }
    out = arm_metrics_from_claims(claims)
    # house metrics registered via predicate
    assert "house_pass" in out["casteer_noclip"]
    assert "house_max_norm_err" in out["casteer_noclip"]
    # quantity/against registered as before
    assert "snoopy_cs" in out["casteer_clip"]
    assert "mean_others_fid" in out["casteer_clip"]


def test_arm_metrics_from_claims_matches_real_claims_json():
    """The real claims.json must register a metric set for every arm that
    appears in a measured.* reference, and `house`'s predicate-only metrics
    must be present under casteer_noclip."""
    import json
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "claims.json")) as f:
        claims = json.load(f)
    out = arm_metrics_from_claims(claims)
    assert "house_pass" in out.get("casteer_noclip", [])
    assert "house_max_norm_err" in out.get("casteer_noclip", [])
    # nudity_total_raw is NOT referenced by any claim -> not registered (it is
    # a driver-produced companion, not a gated metric).
    for arm_metrics in out.values():
        assert "nudity_total_raw" not in arm_metrics


def test_expand_template_prompts_reaches_declared_count():
    """80 templates x n_per must reach the declared snoopy count (800,
    experiments.tex:67), not 80 (review: dead n_per under-generated)."""
    templates = ["a photo of a {}"] * 80
    p = expand_template_prompts(templates, "Snoopy", FULL_CONFIG_MIN_IMAGES["snoopy"])
    assert len(p) == FULL_CONFIG_MIN_IMAGES["snoopy"]  # 800
    assert p[0] == "a photo of a Snoopy"
    assert p[-1] == "a photo of a Snoopy"
    # 50 classes x n_per reaches the style count (200)
    classes = [f"cls{i}" for i in range(50)]
    # style uses "{c}, Van Gogh style" not expand_template_prompts, but the
    # same arithmetic: ceil(200/50)=4 -> 200 prompts.
    n_per = max(1, (200 + len(classes) - 1) // len(classes))
    sp = [f"{c}, Van Gogh style" for c in classes for _ in range(n_per)][:200]
    assert len(sp) == 200


def test_expand_template_prompts_above_template_count_repeats():
    """When n_target > len(templates), each template is repeated so the total
    reaches n_target (not capped at len(templates))."""
    templates = ["t{}-a", "t{}-b"]  # 2 templates
    p = expand_template_prompts(templates, "X", 5)
    assert len(p) == 5
    assert all("X" in s for s in p)


def test_expand_template_prompts_rejects_empty_and_nonpositive():
    import pytest
    with pytest.raises(ValueError):
        expand_template_prompts([], "X", 5)
    with pytest.raises(ValueError):
        expand_template_prompts(["a {}"], "X", 0)
