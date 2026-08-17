"""Smoke test: run the SAME CASteer code path at a size that finishes in a
couple of minutes on CPU.

Pipeline exercised:
  1. Load SD-1.4 (CPU fp32).
  2. Estimate a steering vector from 2 concrete prompt pairs at 4 denoising
     steps, 256x256 (Algorithm 1: shared seed 0, patch-mean, pair-mean,
     pos-neg, unit-norm).
  3. Generate 1 steered image (casteer_clip, beta=2, Eq.7 clip) and 1 vanilla
     image (sd14) at the same seed/prompt.
  4. Save both under results/smoke/ and print exactly one FINAL line.

This proves the path runs. It is NOT evidence about the paper (subset is tiny,
resolution and step count are reduced). Never report its output as a result.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from core.runner import init_pipeline, detect_gpu
from core.construct_prompts import get_prompts_concrete
from core.diffusion_steering import (
    DiffusionModelType, diffusion_register_vector_controls_with_hooks,
)
from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode
from core.controller import CrossAttentionOutputSteering, EPS

RES = 256
N_STEPS = 4
SEED = 42
PROMPT = "a photo of Snoopy"


def _collect_means(pipe, model, prompts, device):
    stats = CrossAttentionOutputStatsCollector(
        token_aggregation_mode=TokenAggregationMode.AVERAGE, normalize=False)
    component = getattr(pipe, "transformer", None) or pipe.unet
    hm = diffusion_register_vector_controls_with_hooks(
        component, stats, model_type=DiffusionModelType.from_model(model))
    for p in prompts:
        gen = torch.Generator(device=device).manual_seed(0)  # shared seed 0 (Algorithm 1)
        pipe(prompt=p, num_inference_steps=N_STEPS, guidance_scale=7.5,
             height=RES, width=RES, generator=gen, num_images_per_prompt=1)
        stats.reset()
    hm.remove_hooks()
    return stats.means


def _compute_vectors(pos_means, neg_means):
    sv = {}
    for step in pos_means:
        sv[step] = {}
        for place in pos_means[step]:
            sv[step][place] = []
            for i in range(len(pos_means[step][place])):
                v = pos_means[step][place][i] - neg_means[step][place][i]
                v = v / v.norm(dim=-1, keepdim=True).clamp(min=EPS)
                sv[step][place].append(v)
    return sv


def main():
    t0 = time.time()
    device = torch.device("cuda" if detect_gpu() else "cpu")
    pos_p, neg_p = get_prompts_concrete(num=2, concept_pos="Snoopy")
    pipe = init_pipeline("sd14", device)
    pos = _collect_means(pipe, "sd14", pos_p, device)
    neg = _collect_means(pipe, "sd14", neg_p, device)
    vec = _compute_vectors(pos, neg)
    n_vecs = sum(len(vec[s][p]) for s in vec for p in vec[s])
    assert n_vecs > 0, "smoke estimated 0 steering vectors (no OK-on-empty)"

    # steered generation (casteer_clip: beta=2, clip, per-step vectors)
    control = CrossAttentionOutputSteering(
        source_concepts=[vec], target_concepts=[None], strength=2.0, device=device,
        intermediate_clipping=True, use_first_diffusion_step=False,
        steering_mode="dotproduct")
    component = getattr(pipe, "transformer", None) or pipe.unet
    hm = diffusion_register_vector_controls_with_hooks(
        component, control, model_type=DiffusionModelType.from_model("sd14"))
    gen = torch.Generator(device=device).manual_seed(SEED)
    steered = pipe(prompt=PROMPT, num_inference_steps=N_STEPS, guidance_scale=7.5,
                   height=RES, width=RES, generator=gen, num_images_per_prompt=1).images[0]
    control.reset()
    hm.remove_hooks()

    # vanilla generation (sd14: no hook)
    gen2 = torch.Generator(device=device).manual_seed(SEED)
    vanilla = pipe(prompt=PROMPT, num_inference_steps=N_STEPS, guidance_scale=7.5,
                   height=RES, width=RES, generator=gen2, num_images_per_prompt=1).images[0]

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "smoke")
    os.makedirs(out_dir, exist_ok=True)
    steered.save(os.path.join(out_dir, "casteer_clip.png"))
    vanilla.save(os.path.join(out_dir, "sd14.png"))
    print(f"FINAL smoke=sd14=1img casteer_clip=1img vectors={n_vecs} steps={N_STEPS} res={RES} "
          f"device={device.type} t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
