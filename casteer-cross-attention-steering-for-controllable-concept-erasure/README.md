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
| `tests/test_method_core.py` | Invariants from the paper's maths: Householder norm preservation (claim `house`), Eq.6 matrix==dot-product, Eq.7 clip-only-positive, Eq.4 constant-α, Eq.9 multi-concept average, CFG conditional-half-only. |
| `tests/test_degeneracy.py` | Degeneracy test: the method at β=0 reproduces the baseline EXACTLY (bit-identical) on both dotproduct and constant modes. |
| `tests/test_data.py` | Fingerprint tests for the data loader (ImageNet 50/`tench`, 80 CLIP templates, 30,000 COCO captions, 4,703 I2P prompts); COCO reference raises when not vendored. |
| `tests/test_eval.py` | Instrument tests for the eval metrics (CLIP positive/negative, FID identical/different/empty, NudeNet/Q16/LPIPS raise-when-missing). |
| `core/` | Upstream library + reproduction additions: `controller.py` (Eq. 5/6/7 + Eq.4 constant mode + Eq.9 `average_concept_vectors`), `diffusion_steering.py` (attn2 hooks), `vector_dump.py` (Algorithm 1 stats), `construct_prompts.py`, `utils.py` (vendored pipeline init), `pickle.py`, `data.py` (fingerprinted data loader), `runner.py` (CPU-friendly runner + per-arm config), `eval/{clip.py (device-agnostic), metrics.py (all claims metrics), fid.py}`. |
| `scripts/diffusion/` | Upstream entrypoints + `run_all_arms.py` (runs every arm/seed, writes measured.json), `smoke.py` (CPU smoke run). |
| `run_all_arms.sh`, `smoke.sh` | Entry points for the full arm sweep (BLOCKED on CPU) and the CPU smoke run. |
| `selfcheck_claims.py` → `selfcheck.json` | Our own evaluator (NOT `claims_result.json`): `house` PASS, 16 diffusion claims BLOCKED. |
| `measured.json`, `measured_blocked_reasons.json` | Per-arm/seed/metric values (BLOCKED on CPU) + reason sidecar. |
| `instruments.json` | Every correctness-deciding instrument (data loader, CLIP, FID, degeneracy, Householder, NudeNet, Q16, LPIPS, per-task/per-model vector mapping, driver dispatch) with positive/negative tests. |
| `mutations.json` | 16 deliberate defects, each caught by its `must_fail` test. |
| `exp/datasets/eval/` | Shipped eval templates: `clip_templates.json` (80 CLIP/ImageNet templates), `imagenet/template.json`, `coco/coco_30k.csv`. |
| `imagenet_classes.txt` | 50 ImageNet classes used for concrete/style prompt pairs (SPEC U8). |
| `paper/` | arXiv 2503.09630 LaTeX source (authoritative) + rendered HTML. |
| `SPEC.md`, `claims.json`, `figure_transcript.md` | Reproduction spec (incl. §11 Constructed truth, §12 sweeps, §13 environment constraint), claims, figure reads. |

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

# 3. Prove the environment resolves (import gate), the Householder invariant
#    (SPEC claim `house`), the degeneracy test (beta=0 == baseline), the
#    invariants from the paper's maths, and the fingerprinted data + eval
#    instruments.
source .venv/bin/activate
python -m pytest tests/ -q          # 53 passed (CPU-only, no model download)

# 4. Smoke-test the FULL code path on CPU: load SD-1.4, estimate steering
#    vectors (Algorithm 1), steer (Eq. 7) and generate 1 image each for
#    casteer_clip and sd14 at 4 steps / 256x256 / seed 42. ~1-2 min once
#    SD-1.4 is cached. Prints one FINAL line; NOT evidence about the paper.
bash smoke.sh
# -> results/smoke/{casteer_clip,sd14}.png , FINAL casteer_clip=...

# 5. Run every arm at the paper's full config across all 3 seeds.
#    On this CPU-only host every diffusion arm is BLOCKED (paper used 8xV100,
#    supplementary.tex:30). On a CUDA host the same script runs them for real
#    via the implemented arm->eval->measured.json driver: per-task steering
#    vector via TASK_VECTOR, 7-concept Eq.9 average for I2P-overall, I2P
#    per-prompt curated sd_seed, nudity_total on the full-set basis (raw under
#    nudity_total_raw; inconclusive below 2000 prompts), declared image counts
#    (snoopy/other 800/concept, style 200) via n_per, bare-concept CLIP
#    reference text, and on-demand sd14 references so the normalized Snoopy
#    claims are reachable. Two metrics remain BLOCKED on every host until
#    their references are supplied: i2p_overall_pct (the Q16 classifier
#    checkpoint is not named by the paper and not vendored) and coco_fid30k
#    (the real COCO-30k FID reference is not vendored; set COCO_REF_DIR).
bash run_all_arms.sh
# -> one "FINAL <arm>=BLOCKED" line per arm/seed on CPU; measured.json

# 6. Self-check the claims (our own evaluator; writes selfcheck.json).
#    `house` PASSES (pure math); the 16 diffusion claims are BLOCKED on CPU.
python selfcheck_claims.py
```

### Reproduce with Docker (CUDA 12.4 base, builds the same environment)

```bash
docker build -t casteer-repro .
docker run --rm casteer-repro python -m pytest tests/ -q   # 33 passed
# GPU + HF_TOKEN required for the model arms (steering-vector estimation,
# steered generation, I2P, CLIP/FID scoring):
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN casteer-repro \
    bash run_all_arms.sh   # runs every arm at full config on CUDA; writes measured.json
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
#    NOTE: --use_all_diffusion_steps is REQUIRED for SD-1.4 per-step steering
#    vectors (experiments.tex:19); omitting it silently runs the appendix
#    single-step (sd14_0) ablation (supplementary.tex:1098-1103), which is NOT
#    the main config. The gated run_all_arms driver sets use_first=False for
#    sd14 arms (core/runner.py), but this vendored entrypoint defaults to
#    first-step-only, so pass the flag explicitly when invoking it by hand.
python scripts/diffusion/run_with_steering.py \
    --model_name sd14 --generate_concept snoopy \
    --template_path exp/datasets/eval/clip_templates.json \
    --num_images_per_prompt 10 --output_dir ./results/sd14/eval_snoopy/casteer-2.0 \
    --steering_strength 2.0 --intermediate_clipping --use_all_diffusion_steps \
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

Rung reached: **implementation + correctness** (the `numbers` rung is BLOCKED on this CPU-only host).

- The pinned environment builds from scratch (`requirements.txt` + `Dockerfile`), every third-party dependency and every vendored `core`/`scripts` module imports cleanly under Python 3.13, and `tests/` passes (65 tests): the import gate, the SPEC `house` Householder invariant, the degeneracy test (β=0 == baseline, bit-exact), the Eq.6/7/4/9 invariants, the fingerprinted data loader (incl. I2P `sd_seed`), the eval-metric instruments (CLIP bare-concept reference, FID, NudeNet 8-exposed-class filter + full-set scaling + inconclusive floor, NudeNet/Q16/LPIPS raise-when-missing), the per-task/per-model steering-vector mapping, and the driver dispatch (predicate parse, `n_per` expansion to the declared counts).
- The full code path runs end-to-end on CPU via `smoke.sh` (load SD-1.4 → estimate 64 steering vectors (16 CA blocks × 4 steps) → steer → generate), proving the implementation works on the real model.
- `mutations.json` (16 deliberate defects) are each caught by their `must_fail` test (verified); `instruments.json` records every correctness-deciding instrument with positive/negative tests.
- **`house` is reproduced** (`selfcheck.json`: PASS, max norm error 3.5e-14). It is the one high-invariance claim that needs no generation.
- **The 16 diffusion-number claims are BLOCKED**: this sandbox is CPU-only and the paper's full config (50 steps × 4,703 I2P / 800-per-concept snoopy / 3,000 COCO prompts × 3 seeds × multiple arms, on 8×V100, `supplementary.tex:30`) is infeasible on CPU. `measured.json` records `BLOCKED` for every diffusion metric with a reason sidecar; `run_all_arms.sh` emits `FINAL <arm>=BLOCKED` per arm/seed. The external evaluators (NudeNet, Q16, LPIPS) are not installed and the real COCO-30k FID reference is not vendored — documented blockers, not synthetic substitutes. The driver fixes (full-set scaling, per-prompt sd_seed, declared image counts, bare-concept CS, on-demand sd14 references) make the normalized Snoopy claims reachable on a GPU host; on CPU the BLOCKED reason is honestly "CPU-only host", no longer a structural block.
- `claims_result.json` is NOT produced here; it is written by the workflow's numbers gate. Our own evaluator is `selfcheck_claims.py` → `selfcheck.json`.

See `REPRODUCTION.md` for the full log and `SPEC.md` (§11 Constructed truth, §12 unstated-value sweeps incl. the Eq.9 renormalization fork, §13 environment constraint, §14–§15 review-driven fixes) for the arms, restrictions, and unstated-items analysis (U1-U14).
