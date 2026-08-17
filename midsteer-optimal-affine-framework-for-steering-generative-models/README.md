# MidSteer: Optimal Affine Framework for Steering Generative Models — reproduction

Reproduction of **"MidSteer: Optimal Affine Framework for Steering Generative
Models"** (Gaintseva, Stepanov, Liu, Benning, Slabaugh, Deng, Elezi; ICML 2026;
arXiv:2605.05220).

- paper_ref: `0e6756c9-d827-4749-a8d7-3140a8c98a25`
- project_id: `06910d54-1d99-4864-8bff-ab3007a0c70e`
- arXiv: https://arxiv.org/abs/2605.05220
- Upstream code (vendored here): https://github.com/Atmyre/MidSteer (HEAD `0f3b31e`)
- Authoritative paper LaTeX source: `paper/main.tex`, `paper/content/*.tex`,
  `paper/flipping_main.tex`, `paper/artefacts/**/*.tex` (arXiv v3).
- Full method spec, gap list, arms, and claims: `SPEC.md`, `claims.json`.
- Reproduction log: `REPRODUCTION.md`.

This repo **adopts the authors' own implementation** (the paper links it from
`paper/main.tex:160`). The upstream tree (`core/`, `scripts/`, `helpers/`,
`exp/`, `notebooks/`, `imagenet_classes.txt`, `.env.example`, `UPSTREAM_README.md`)
is vendored verbatim at the repo root so `from core...` imports resolve with
no path configuration. The pinned environment lives in `requirements.txt`.

## What is in this folder

| Path | Purpose |
| --- | --- |
| `requirements.txt` | Pinned dependencies for the method + eval + pytest (Python 3.13). |
| `Dockerfile` | Builds the same environment from scratch (CUDA 12.4 base). |
| `tests/test_environment.py` | Import gate: every dependency + every `core` module must import; plus a closed-form whitening smoke test (SPEC Eq. 7). |
| `core/` | Upstream library: `math.py` (whitening, `fractional_matrix_power_cov_torch`), `controller.py` (LEACE / LEACE-Switch / MidSteer closed-form affine maps, Eqs. 6/13/19/22/23), `vector_dump.py` (per-head Welford, Algorithm 1), `llm_steering.py`, `diffusion_steering.py`, `eval/{clip,fid}.py`. |
| `scripts/llm/`, `scripts/diffusion/` | Upstream entrypoints (covariance estimation, steering-vector generation, generation with steering, scoring). |
| `paper/` | arXiv 2605.05220 v3 LaTeX source (authoritative) + rendered HTML + extracted text. |
| `SPEC.md`, `claims.json`, `figure_reads/` | Reproduction spec, claims, figure reads. |

## Quickstart

You need Python 3.13 and [uv](https://docs.astral.sh/uv/) (`curl -LsSf
https://astral.sh/uv/uv-installer.sh | sh`). Everything below runs CPU-only;
the model arms (Llama-2-7B-chat, SDXL) additionally need a CUDA GPU and the
`HF_TOKEN` env var (copy `.env.example` to `.env`).

```bash
# 1. Clone this reproduction branch.
git clone https://github.com/coleh-cm/auto-reproductions.git
cd auto-reproductions/midsteer-optimal-affine-framework-for-steering-generative-models

# 2. Create the virtual environment from the pinned requirements.
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt

# 3. Prove the environment resolves (the import gate).
source .venv/bin/activate
python -m pytest tests/ -q          # 40 passed

# 4. Smoke-test the closed-form method directly (no network, no weights).
python - <<'PY'
import torch
from core.math import fractional_matrix_power_cov_torch        # SPEC Eq. 7
from core.controller import ModelToSteer, SteeringVectors       # Eqs. 6/13/19/22/23
d, H = 8, 2
A = torch.randn(H, d, d, dtype=torch.float64)
S = A @ A.mT + torch.eye(d, dtype=torch.float64) * 0.1           # PSD covariance / head
W  = fractional_matrix_power_cov_torch(S,  0.5)                  # Sigma^{1/2}
Wp = fractional_matrix_power_cov_torch(S, -0.5)                  # (Sigma^{1/2})^+
assert torch.allclose((Wp @ W)[0], torch.eye(d, dtype=torch.float64), atol=1e-6)
print("MidSteer whitening identity OK")
PY
```

### Reproduce with Docker (CUDA 12.4 base, builds the same environment)

```bash
docker build -t midsteer-repro .
docker run --rm midsteer-repro python -m pytest tests/ -q   # 95 passed
# GPU + HF_TOKEN required for the model arms (E2-E5); without them those arms BLOCKED:
docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN midsteer-repro bash run_all_arms.sh
```

## What runs in this repository

| Path | Runs here? | What it does |
| --- | --- | --- |
| `midsteer_core/` (stats, crosscov, affine, data) | ✅ CPU | closed-form affine maps (Eqs. 6/13/19/22/23 + vanilla 21/24/25), Welford, the data loader (raises `BlockedException` w/o CUDA/HF_TOKEN) |
| `midsteer_core/eval/*` | ✅ CPU (mock logic) | judge_cs / clip_cs / fid / detoxify / armorm / bertscore instruments; raise `BlockedException` on the real backbone here; scoring logic unit-tested |
| `experiments/run_e1_synth.py` | ✅ CPU, REAL | claims C1–C3 closed-form invariant checks on synthetic Gaussian data of known covariance, seeds {0,1,2}. **All PASS.** |
| `experiments/run_e{2..5}_*.py` | ❌ BLOCKED | Llama-2-7B-chat / SDXL arms; print `FINAL <arm>=BLOCKED`, write BLOCKED partials (no synthetic fallback) |
| `run_all_arms.sh` | ✅ | runs E1 real + E2–E5 BLOCKED → `measured.json` + `results/e1_synth.json`; prints one `FINAL <arm>=...` per arm |
| `smoke.sh` | ✅ | tiny E1 path, one FINAL line (NOT evidence about the paper) |
| `evaluate_claims.py` → `claims_result.json` | ✅ | verdict table: pass=3 (C1–C3), fail=0, blocked=19 (model arms) |
| `selfcheck_claims.py` → `selfcheck.json` | ✅ | the agent's own redundant check (different filename), agrees |
| `tests/` | ✅ 95 passed | environment import gate, core/degeneracy/invariants, data, eval instruments, claims eval, mutations (5 defects, all caught) |
| `instruments.json`, `mutations.json` | ✅ | every output-deciding instrument + positive/negative tests; 5 deliberate defects with `must_fail` nodes |
| `core/`, `scripts/`, `helpers/`, `exp/`, `notebooks/` | vendored upstream | the authors' own implementation (HEAD `0f3b31e`), unchanged; `midsteer_core/` is the readable closed form beside it |

**Blocker (honest):** this sandbox is CPU-only with no `HF_TOKEN`, so the model arms
(Llama-2-7B-chat, SDXL) cannot run. Every model metric in `measured.json` is the string
`"BLOCKED"`; every model-arm claim (C4–C22) is verdict `blocked`. The only real numbers
produced here are the E1 closed-form invariant checks (C1, C2, C3), all PASS. See
`REPRODUCTION.md` § Blockers and `SPEC.md` §12–14.

## Environment notes

- `clip` is provided by **`clip-anytorch`** (an OpenAI-CLIP fork with relaxed
  torch pins), not the PyPI `clip` package (which is an unrelated clipboard
  tool). `tests/test_environment.py::test_clip_is_openai_clip_not_clipboard`
  guards this. `setuptools` is pinned because `clip-anytorch` imports
  `pkg_resources` at module load.
- The `SANA-Sprint is not available` warning printed on importing `core.utils`
  is benign: `SanaSprintPipeline` is absent in `diffusers==0.32.2` and is only
  used for the SANA arms, which are **out of scope** for this reproduction
  (SPEC §8 — only Llama-2-7B-chat and SDXL are reproduced).
- The full set of pinned packages and the rationale per group are documented
  inline in `requirements.txt`.

## Status

Implementation rung complete. The closed-form core, E1 synthetic invariant checks
(C1–C3, all PASS), eval instruments, run scripts, claims evaluator, mutation suite,
`measured.json`, and `claims_result.json` (pass=3, fail=0, blocked=19) are committed
and pushed. The model arms are BLOCKED (no CUDA / no HF_TOKEN). See `REPRODUCTION.md`
for the full log, decisions, and blockers; `SPEC.md` §12–14 for constructed truth,
sweep ranges, and every choice the paper left open.
