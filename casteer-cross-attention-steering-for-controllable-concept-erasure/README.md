# CASteer: Cross-Attention Steering for Controllable Concept Erasure — reproduction

Reproduction of **"CASteer: Cross-Attention Steering for Controllable Concept
Erasure"** (Gaintseva, Oncescu, Ma, Liu, Benning, Slabaugh, Deng, Elezi;
ICLR 2026; arXiv:2503.09630).

- paper_ref: `d0ab97b5-e9bf-41af-9846-d19f846eef80`
- project_id: `06910d54-1d99-4864-8bff-ab3007a0c70e`
- arXiv: https://arxiv.org/abs/2503.09630
- Upstream code (vendored here): https://github.com/Atmyre/CASteer (pinned at
  commit `135912a555a8606c01f55843c738cf8062319ed7`; see `UPSTREAM_COMMIT.txt`).
- Authoritative paper LaTeX source: `paper/iclr2026_conference.tex`,
  `paper/content/*.tex`, `paper/math_commands.tex` (arXiv e-print 2503.09630).
  Reading surface: `paper/paper.html` (MathML rendering of the same source).
- Full method spec, gap list, arms, and claims: `SPEC.md`, `claims.json`.
- Reproduction log: `REPRODUCTION.md`.

This repo **adopts the authors' own implementation** (the paper links it from
`paper/content/abstract.tex:36`). The upstream tree (`core/`, `scripts/`,
`exp/`, `imagenet_classes.txt`, `assets/`, `UPSTREAM_README.md`) is vendored
verbatim at the repo root so `from core...` imports resolve with no path
configuration. The pinned environment lives in `requirements.txt`.

## What is in this folder

| Path | Purpose |
| --- | --- |
| `requirements.txt` | Pinned dependencies for the method + eval + pytest (Python 3.13). |
| `Dockerfile` | Builds the same environment from scratch (CUDA 12.4 base). |
| `pyproject.toml` | Adds the repo root to `sys.path` so `import core` works under pytest. |
| `tests/test_environment.py` | Import gate: every dependency + every `core` module must import; plus the Householder / unit-norm invariants (SPEC claim `house`, `experiments.tex:21-22`) and a closed-form check that `CrossAttentionOutputSteering` (no clip) implements `c <- (I - 2 s s^T) c`. |
| `core/` | Upstream library: `controller.py` (CASteer Eq. 5/6, `CrossAttentionOutputSteering`), `diffusion_steering.py` (attn2 forward hooks), `vector_dump.py` (per-patch/per-pair mean stats, Algorithm 1), `construct_prompts.py` (concrete/style/human-related prompt pairs), `utils.py` (pipeline init for SD-1.4/SDXL/SANA), `pickle.py`, `eval/{clip,fid}.py`. |
| `scripts/diffusion/` | Upstream entrypoints: `estimate_steering_vectors.py` (Step 1), `run_with_steering.py` (Step 2), `produce_scores.py` (Step 3), `run_i2p_eval.py` (I2P benchmark). |
| `exp/datasets/eval/` | Shipped eval templates: `clip_templates.json` (80 CLIP/ImageNet templates), `imagenet/template.json`, `coco/coco_30k.csv`. |
| `imagenet_classes.txt` | 50 ImageNet classes used for concrete/style prompt pairs (SPEC U8). |
| `paper/` | arXiv 2503.09630 LaTeX source (authoritative) + rendered HTML. |
| `SPEC.md`, `claims.json`, `figure_transcript.md` | Reproduction spec, claims, figure reads. |

## Quickstart

You need Python 3.13 and [uv](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/uv-installer.sh | sh`). The import gate runs
CPU-only; the model arms (SD-1.4 / SDXL / SANA) additionally need a CUDA GPU
and the `HF_TOKEN` env var (HuggingFace token, set for `CompVis/stable-diffusion-v1-4`
etc. — some checkpoints are gated).

```bash
# 1. Clone this reproduction branch.
git clone -b repro/casteer-cross-attention-steering-for-controllable-concept-erasure \
    https://github.com/coleh-cm/auto-reproductions.git
cd auto-reproductions/casteer-cross-attention-steering-for-controllable-concept-erasure

# 2. Create the virtual environment from the pinned requirements.
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt

# 3. Prove the environment resolves (the import gate) and the Householder
#    invariant (SPEC claim `house`) holds.
source .venv/bin/activate
python -m pytest tests/ -q          # 6 passed

# 4. Smoke-test the CASteer erasure operator directly (no network, no weights).
python - <<'PY'
import torch
from core.controller import CrossAttentionOutputSteering        # Eq. 5/6
d = 320
s = torch.randn(d); s = s / s.norm()                             # unit steering vector (U1)
store = {0: {"down": [s.view(1, 1, d)]}}                          # dict[step][place][block]->[1,1,d]
ctrl = CrossAttentionOutputSteering(
    source_concepts=[store], target_concepts=[None], strength=2.0,
    device=torch.device("cpu"), intermediate_clipping=False,
    use_first_diffusion_step=True, num_layers=1,
)
c = torch.randn(2, 4, 1, d)                                      # B=2 (CFG uncond+cond)
out = ctrl.forward(c.clone(), diffusion_step=0, place_in_unet="down", block_index=0).float()
# conditional half (index 1) is the Householder reflection (I - 2 s s^T) c -> norm preserved
assert torch.allclose(out[1,:,0,:].norm(dim=-1), c[1,:,0,:].norm(dim=-1), atol=1e-3)
print("CASteer Householder erasure OK")
PY
```

### Reproduce with Docker (CUDA 12.4 base, builds the same environment)

```bash
docker build -t casteer-repro .
docker run --rm casteer-repro python -m pytest tests/ -q   # 6 passed
# GPU + HF_TOKEN required for the model arms (steering-vector estimation,
# steered generation, I2P, CLIP/FID scoring):
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN casteer-repro \
    python scripts/diffusion/estimate_steering_vectors.py \
        --model_name sdxl-turbo --concept snoopy --mode concrete \
        --num_prompts 50 --output_dir ./results/sdxl/steering_vectors
```

## Running the method (GPU + HF_TOKEN required)

The three steps below mirror the upstream README (`UPSTREAM_README.md`). All
gated claims in `claims.json` are on **SD-1.4** (50 PNDM steps, guidance 7.5,
512x512, fp16; SPEC U4); SDXL/SANA arms are optional/budget. Default steering
strength `beta = 2` (Householder reflection, `experiments.tex:21`).

```bash
# Step 1 — compute steering vectors from paired prompts (Algorithm 1).
python scripts/diffusion/estimate_steering_vectors.py \
    --model_name sd14 --concept snoopy --mode concrete --num_prompts 50 \
    --output_dir ./results/sd14/steering_vectors

# Step 2 — generate with steering (Algorithm 2; Eq. 6 no clip, or --intermediate_clipping for Eq. 7).
python scripts/diffusion/run_with_steering.py \
    --model_name sd14 --generate_concept snoopy \
    --template_path exp/datasets/eval/clip_templates.json \
    --num_images_per_prompt 10 --output_dir ./results/sd14/eval_snoopy/casteer-2.0 \
    --steering_strength 2.0 --intermediate_clipping \
    erase --concept_path ./results/sd14/steering_vectors/snoopy.pt

# Step 3 — score (CLIP score + FID vs the un-steered baseline).
python scripts/diffusion/produce_scores.py \
    --concept snoopy mickey --dir ./results/sd14/eval_snoopy \
    --num_workers 4 --batch_size 32
# -> ./results/sd14/eval_snoopy/clip_score.tsv and ./results/sd14/eval_snoopy/fid.tsv

# I2P benchmark (4,703 prompts; NudeNet + Q16 via the Receler eval code,
# https://github.com/jasper0314-huang/Receler, threshold 0.6 — SPEC U7).
python scripts/diffusion/run_i2p_eval.py \
    --model_name sd14 --output_dir ./results/i2p/casteer-2.0 \
    --concept_path ./results/sd14/steering_vectors/nudity.pt \
    --steering_strength 2.0 --intermediate_clipping
```

## Environment notes

- `clip` (imported by `core/eval/clip.py` as `import clip`) is provided by
  **`clip-anytorch`** — an API-compatible, maintained fork of OpenAI CLIP that
  ships the `clip` module on Python 3.13. Upstream's own `requirements/base.txt`
  omits this dependency; it is added here. `setuptools` is pinned because
  `clip-anytorch` imports `pkg_resources` at module load.
- `diffusers==0.33.1` is the earliest pin that exposes both `SanaPipeline` and
  `SanaSprintPipeline` used in `core/utils.py`; `SanaSprintPipeline` is absent
  in earlier diffusers and would print a benign "SANA-Sprint is not available"
  warning. Both are present here (verified by `test_third_party_dependencies_import`).
- The full set of pinned packages and the rationale per group are documented
  inline in `requirements.txt`.

## Status

Rung reached: **environment**. The pinned environment builds from scratch
(`requirements.txt` + `Dockerfile`), every third-party dependency and every
vendored `core`/`scripts` module imports cleanly under Python 3.13, and the
SPEC `house` invariant (Householder norm-preservation under `beta=2`, unit
steering vector) plus the `CrossAttentionOutputSteering` Eq. 5/6 closed-form
match pass in `tests/` (6 passed). The model arms (steering-vector estimation,
steered generation, I2P, CLIP/FID scoring) need a CUDA GPU and `HF_TOKEN`;
this sandbox is CPU-only, so the paper's empirical claims (the 17 claims in
`claims.json`) are not yet tested by this run. See `REPRODUCTION.md` for the
full log and `SPEC.md` for the arms, restrictions, and unstated-items analysis
(U1-U14).
