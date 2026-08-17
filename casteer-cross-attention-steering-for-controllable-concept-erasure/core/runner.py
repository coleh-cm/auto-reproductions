"""Generation runner for the CASteer reproduction.

Provides a CPU-friendly pipeline loader (the vendored core/utils.py uses
torch_dtype=float16 + device_map='balanced', which crashes on CPU-only hosts),
the per-arm configuration mirrored from claims.json, and a single
run_generation entry that loads the model, optionally hooks the
CrossAttentionOutputSteering controller, generates one image per prompt with a
fixed seed, and saves PNGs. It RAISES on an empty result (no OK-on-empty).

The paper's full config (SD-1.4: 50 PNDM steps, guidance 7.5, 512x512, fp16)
requires a GPU; on a CPU-only host generating thousands of images per arm
across 3 seeds is infeasible (paper used 8xV100, supplementary.tex:30).
is_arm_feasible() exposes that decision so run_all_arms can emit BLOCKED rather
than pretending.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# claims.json arm -> exact config (mirrors claims.json `arm_configs` + SPEC sec 5/6).
# `use_first_diffusion_step` is False for SD-1.4 per-step vectors (experiments.tex:19)
# and True for the distilled SDXL arm (method_2.tex:191-197).
ARM_CONFIG = {
    "sd14": dict(model="sd14", beta=None, clip=False, mode=None,
                 use_first=False, vector=None, tasks=["coco", "i2p", "snoopy", "style"]),
    "casteer_noclip": dict(model="sd14", beta=2.0, clip=False, mode="dotproduct",
                           use_first=False, vector="nudity", tasks=["i2p", "snoopy", "style"]),
    "casteer_clip": dict(model="sd14", beta=2.0, clip=True, mode="dotproduct",
                         use_first=False, vector="nudity", tasks=["i2p", "snoopy", "coco", "style"]),
    "const_a2_clip": dict(model="sd14", beta=2.0, clip=True, mode="constant",
                         use_first=False, vector="nudity", tasks=["i2p", "snoopy", "coco"]),
    "const_a2_noclip": dict(model="sd14", beta=2.0, clip=False, mode="constant",
                           use_first=False, vector="nudity", tasks=["i2p", "snoopy", "coco"]),
    "const_a1_clip": dict(model="sd14", beta=1.0, clip=True, mode="constant",
                         use_first=False, vector="nudity", tasks=["i2p", "snoopy"]),
    "const_a1_noclip": dict(model="sd14", beta=1.0, clip=False, mode="constant",
                           use_first=False, vector="nudity", tasks=["i2p", "snoopy"]),
    "sdxl": dict(model="sdxl", beta=None, clip=False, mode=None,
                 use_first=False, vector=None, tasks=["i2p"]),
    "sdxl_casteer_clip": dict(model="sdxl", beta=2.0, clip=True, mode="dotproduct",
                              use_first=True, vector="nudity", tasks=["i2p"]),
}

# Paper full-config image counts (per arm, per seed) -- the situations the
# claims narrow to (claims.json `restrictions`): I2P >=1000 of 4703, COCO
# >=3000 of 30000, concrete 200 of 800 per concept (5 concepts), snoopy 200 of
# 800, style 200 of 1000.
FULL_CONFIG_MIN_IMAGES = {
    "i2p": 1000,
    "coco": 3000,
    "snoopy": 200,      # gated subset (>=200 of 800)
    "other": 200,       # per concept, 5 concepts
    "style": 200,
}

# Stepping at the paper's full config (experiments.tex:19; core/utils.get_num_denoising_steps).
FULL_STEPS = {"sd14": 50, "sdxl": 30, "sdxl-turbo": 1}
FULL_RES = {"sd14": 512, "sdxl": 1024}


def detect_gpu() -> bool:
    return bool(torch.cuda.is_available())


def is_arm_feasible(arm: str, scale: str = "full") -> tuple[bool, str]:
    """Is running this arm at the given scale feasible in this environment?

    scale='full' -> the paper's full config (50 steps, >=1000 prompts, 3 seeds).
    scale='smoke' -> a tiny 1-2 image, few-step run that finishes in minutes.
    On a CPU-only host, full-scale SD generation is infeasible (paper used
    8xV100, supplementary.tex:30) -> returns (False, reason). Smoke-scale is
    always feasible on CPU (~7s/step @512).
    """
    if arm not in ARM_CONFIG:
        return False, f"unknown arm {arm!r}"
    if scale == "smoke":
        return True, ""
    if scale != "full":
        return False, f"unknown scale {scale!r}"
    if detect_gpu():
        return True, ""
    cfg = ARM_CONFIG[arm]
    # Any arm that must generate >=1000 images at 50 steps on CPU is infeasible
    # (>=1000 imgs x ~350s/img >> any reasonable wall budget). The reference arm
    # sd14 likewise needs >=3000 COCO images for the FID reference.
    return (False,
            "CPU-only host; SD-1.4/SDXL at the paper full config "
            f"({FULL_STEPS[cfg['model']]} steps, >={FULL_CONFIG_MIN_IMAGES['i2p']} "
            "prompts x 3 seeds) is infeasible without a GPU. The paper used 8xV100 "
            "(supplementary.tex:30). Run on a CUDA host or mark BLOCKED.")


def init_pipeline(model: str, device: torch.device, cache_dir: str = "./cache"):
    """CPU-friendly pipeline loader. Avoids device_map='balanced' (fails on CPU)
    and uses float32 on CPU (fp16 ops missing on CPU). On CUDA uses fp16."""
    from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image

    fp16 = device.type == "cuda"
    dtype = torch.float16 if fp16 else torch.float32
    common = dict(torch_dtype=dtype, cache_dir=cache_dir, safety_checker=None,
                  requires_safety_checker=False)
    if model == "sd14":
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", **common)
    elif model == "sdxl":
        kwargs = dict(common)
        if fp16:
            kwargs["variant"] = "fp16"
            kwargs["use_safetensors"] = True
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **kwargs)
    elif model == "sdxl-turbo":
        kwargs = dict(common)
        if fp16:
            kwargs["variant"] = "fp16"
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", **kwargs)
    else:
        raise ValueError(f"runner.init_pipeline: unknown model {model!r}")
    pipe = pipe.to(device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def _load_vector_store(path: Optional[str], device: torch.device):
    if path is None:
        return None
    from core.pickle import unpickle
    return unpickle(path)


def build_controller(arm: str, device: torch.device, vector_dir: str = "./steering_vectors"):
    """Construct the CrossAttentionOutputSteering for an arm, or None for vanilla arms."""
    from core.controller import CrossAttentionOutputSteering
    cfg = ARM_CONFIG[arm]
    if cfg["beta"] is None:
        return None
    concept_path = os.path.join(vector_dir, f"{cfg['vector']}.pt")
    if not os.path.exists(concept_path):
        raise FileNotFoundError(
            f"steering vector for arm {arm!r} not found at {concept_path}; "
            "run estimate_steering_vectors first (no OK-on-empty)."
        )
    store = _load_vector_store(concept_path, device)
    return CrossAttentionOutputSteering(
        source_concepts=[store], target_concepts=[None],
        strength=float(cfg["beta"]), device=device,
        intermediate_clipping=bool(cfg["clip"]),
        use_first_diffusion_step=bool(cfg["use_first"]),
        steering_mode=cfg["mode"],
    )


def run_generation(
    arm: str,
    seed: int,
    prompts,
    output_dir: str,
    device: Optional[torch.device] = None,
    n_steps: Optional[int] = None,
    resolution: Optional[int] = None,
    model: Optional[str] = None,
    guidance_scale: Optional[float] = None,
    cache_dir: str = "./cache",
    vector_dir: str = "./steering_vectors",
):
    """Generate one image per prompt for `arm` at `seed`, save PNGs to
    output_dir/{i:05d}.png, return the list of saved paths. RAISES if zero
    images are generated (no OK-on-empty)."""
    cfg = ARM_CONFIG[arm]
    mdl = model or cfg["model"]
    if device is None:
        device = torch.device("cuda" if detect_gpu() else "cpu")
    if n_steps is None:
        n_steps = FULL_STEPS[mdl]
    if resolution is None:
        resolution = FULL_RES[mdl]
    if guidance_scale is None:
        guidance_scale = 0.0 if mdl in ("sdxl-turbo",) else 7.5

    if not prompts:
        raise ValueError("run_generation: empty prompt list (no OK-on-empty)")

    pipe = init_pipeline(mdl, device, cache_dir=cache_dir)
    control = build_controller(arm, device, vector_dir=vector_dir)
    hook_manager = None
    if control is not None:
        from core.diffusion_steering import (
            DiffusionModelType, diffusion_register_vector_controls_with_hooks,
        )
        component = getattr(pipe, "transformer", None) or pipe.unet
        hook_manager = diffusion_register_vector_controls_with_hooks(
            component, control, model_type=DiffusionModelType.from_model(mdl),
        )

    os.makedirs(output_dir, exist_ok=True)
    saved = []
    gen = torch.Generator(device=device).manual_seed(int(seed))
    for i, prompt in enumerate(prompts):
        out = pipe(
            prompt=prompt, num_inference_steps=int(n_steps),
            guidance_scale=float(guidance_scale),
            height=int(resolution), width=int(resolution),
            generator=gen, num_images_per_prompt=1,
        )
        img = out.images[0]
        path = os.path.join(output_dir, f"{i:05d}.png")
        img.save(path)
        saved.append(path)
        if control is not None:
            control.reset()
    if hook_manager is not None:
        hook_manager.remove_hooks()
    if not saved:
        raise RuntimeError(f"run_generation produced 0 images for arm {arm!r} (no OK-on-empty)")
    return saved
