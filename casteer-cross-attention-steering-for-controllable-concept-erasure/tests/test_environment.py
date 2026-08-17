"""Environment + method-invariant smoke tests for the CASteer reproduction.

These tests prove the pinned environment resolves every dependency the
vendored upstream code (commit 135912a, see UPSTREAM_COMMIT.txt) needs, and
that the core method equations behave as the paper claims. They run CPU-only
and need no GPU, no model downloads, and no network.
"""
import importlib.util
import os
import pathlib

import numpy as np
import torch

REPRO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_third_party_dependencies_import():
    import diffusers, accelerate, datasets, transformers, huggingface_hub, safetensors
    import scipy, pandas, PIL, tqdm, matplotlib, ftfy, regex, setuptools
    import cleanfid
    import clip  # provided by clip-anytorch; upstream's base.txt omits it
    assert hasattr(diffusers, "SanaPipeline")
    assert hasattr(diffusers, "SanaSprintPipeline")


def test_core_modules_import():
    from core.controller import CrossAttentionOutputSteering, VectorControl, EPS
    from core.diffusion_steering import (
        DiffusionModelType, diffusion_register_vector_controls_with_hooks,
    )
    from core.vector_dump import (
        CrossAttentionOutputStatsCollector, TokenAggregationMode,
    )
    from core.construct_prompts import (
        get_prompts_concrete, get_prompts_style, get_prompts_human_related,
    )
    from core.utils import (
        SUPPORTED_DIFFUSION_MODELS, get_num_denoising_steps,
    )
    from core.pickle import pickle_stats, unpickle, unpickle_pack
    from core.eval.clip import compute_clip, clip_score
    from core.eval.fid import compute_fid
    assert DiffusionModelType.from_model("sd14") == DiffusionModelType.SD
    assert DiffusionModelType.from_model("sana-sprint") == DiffusionModelType.SANA
    assert get_num_denoising_steps("sd14") == 50  # U4: upstream default


def test_scripts_import():
    for name, path in [
        ("estimate_steering_vectors", "scripts/diffusion/estimate_steering_vectors.py"),
        ("run_with_steering", "scripts/diffusion/run_with_steering.py"),
        ("produce_scores", "scripts/diffusion/produce_scores.py"),
        ("run_i2p_eval", "scripts/diffusion/run_i2p_eval.py"),
    ]:
        spec = importlib.util.spec_from_file_location(name, REPRO_ROOT / path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def test_construct_prompts_shapes():
    # SPEC U8 / construct_prompts.py: human-related builds 15x14 = 210 pairs
    # (docstring stale says 104; code matches supplementary.tex:120-123).
    from core.construct_prompts import (
        get_prompts_concrete, get_prompts_human_related,
    )
    pos, neg = get_prompts_human_related(concept_pos="nudity")
    assert len(pos) == 210 and len(neg) == 210
    assert all(p.endswith(", nudity") for p in pos)
    assert all(not p.endswith(", nudity") for p in neg)
    pos_c, neg_c = get_prompts_concrete(num=50, concept_pos="Snoopy")
    assert len(pos_c) == 50 and len(neg_c) == 50
    # imagenet_classes.txt first line is "tench"
    assert pos_c[0] == "tench with Snoopy" and neg_c[0] == "tench"


def test_householder_invariant():
    """SPEC claim `house` (experiments.tex:21-22): with beta=2 and unit s,
    (I - 2 s s^T) is a Householder reflection preserving ||c||_2."""
    torch.manual_seed(0)
    for d in (320, 640, 1280):
        for _ in range(100):
            s = torch.randn(d)
            s = s / s.norm()
            c = torch.randn(d)
            c_new = c - 2.0 * (s @ c) * s  # (I - 2 s s^T) c
            assert abs(c_new.norm().item() - c.norm().item()) < 1e-4
    # steering-vector construction is unit-norm (U1): f_norm(v) = v/||v||
    raw = torch.randn(640)
    sv = raw / raw.norm().clamp(min=1e-8)
    assert abs(sv.norm().item() - 1.0) < 1e-6


def test_controller_erasure_matches_householder():
    """CrossAttentionOutputSteering (no clip) implements c <- c - beta*<s,c>*s,
    i.e. (I - beta s s^T) c. With beta=2 and unit s this preserves L2 norm."""
    from core.controller import CrossAttentionOutputSteering
    d = 320
    s = torch.randn(d)
    s = s / s.norm()
    # steering store: dict[step][place][block] -> Tensor[1,1,d], unit norm
    store = {0: {"down": [s.view(1, 1, d)]}}
    device = torch.device("cpu")
    ctrl = CrossAttentionOutputSteering(
        source_concepts=[store],
        target_concepts=[None],
        strength=2.0,
        device=device,
        intermediate_clipping=False,
        use_first_diffusion_step=True,
        num_layers=1,
    )
    # hook view: [B, seq, 1, d] with B=2 (CFG uncond+cond); only cond half steered
    c = torch.randn(2, 4, 1, d)
    c_in = c.clone()
    out = ctrl.forward(c_in, diffusion_step=0, place_in_unet="down", block_index=0)
    # controller.forward() returns vector.half(); cast back for comparison
    out = out.float()
    # conditional half (index 1) should be the Householder reflection of input
    cond_in = c[1, :, 0, :]
    cond_out = out[1, :, 0, :]
    expected = cond_in - 2.0 * (cond_in @ s).unsqueeze(-1) * s
    assert torch.allclose(cond_out, expected, atol=1e-3)
    # norm preserved on the steered half
    assert torch.allclose(cond_out.norm(dim=-1), cond_in.norm(dim=-1), atol=1e-3)
    # unconditional half (index 0) untouched (values preserved up to half cast)
    assert torch.allclose(out[0], c_in[0], atol=1e-3)
