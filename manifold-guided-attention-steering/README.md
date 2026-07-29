# Manifold-Guided Attention Steering (MAGS) — Reproduction

Reproduction of **"Manifold-Guided Attention Steering"** (Ian Li, Kapilesh
Guruprasad, Raunak Sengupta, Ninad Satish, Loris D'Antoni, Rose Yu; UCSD, 2026;
arXiv:2605.21770).

MAGS is a training-free, inference-time intervention for LLM reasoning. For a
set of attention heads it learns a low-dimensional **contrastive error
manifold** from paired correct/incorrect reasoning traces, monitors each head's
proximity to that manifold at every decode step, and — only when the proximity
score exceeds a calibrated threshold — projects the head output back onto the
correct subspace before the attention output projection `W_O` consumes it.

There is **no upstream code** (no link in the paper, no public repo found on
GitHub); the method is implemented from scratch against the LaTeX source
(`paper/latex_src/neurips_2026.tex`), which is the authoritative reference for
every equation, table and reported number. The full method spec, symbol/shape
table, equation citations, and the register of parameters the paper leaves
unstated live in `SPEC.md`; the numbers-gate contract (45 arms with claimed
values and bootstrap CIs) lives in `arms.json`.

## Repository layout

```
manifold-guided-attention-steering/
├── requirements.txt      # fully-pinned dependency set (Python 3.13)
├── Dockerfile            # reproducible environment image
├── README.md             # this file
├── SPEC.md               # method spec, equations, shapes, gaps/defaults
├── arms.json             # 45 arms: claimed accuracy/PPL/validity/affinity + CIs
├── paper/
│   ├── latex_src/        # authoritative LaTeX (neurips_2026.tex, refs.bib)
│   └── paper_pdf_extracted.txt   # prose-only PDF text (maths NOT trusted)
└── REPRODUCTION.md        # reproduction log & status
```

## Hardware

The paper loads three model families:

| Model | Layers × Heads × d_h | Precision | Hardware (paper) |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | 32 × 32 × 128 | float16 | RTX 4090 24 GB |
| `google/gemma-4-E4B-it` | 42 × 8 × 256 | bf16 | RTX 4090 24 GB |
| `openai/gpt-oss-20b` | 24 × 64 × 64 | native mxfp4 | H200 141 GB |

The 8B-class models fit on a single 24 GB GPU in fp16/bf16; GPT-OSS-20B needs
~40 GB (paper used an H200). `Llama-3.1-8B-Instruct` is a **gated** HF repo —
`huggingface-cli login` with an accepted access token before first run.

## Quickstart (host, with `uv`)

These commands are verified to work from a clean checkout.

```bash
# 1. Create a Python 3.13 virtual environment.
uv venv --python 3.13 .venv

# 2. Install the fully-pinned dependency set.
uv pip install --python .venv -r requirements.txt

# 3. Verify every method-critical import resolves.
.venv/bin/python - <<'PY'
import torch, transformers, datasets, sklearn, numpy, accelerate, safetensors, sentencepiece
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import math_verify
from human_eval.execution import check_correctness
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib
print("OK", torch.__version__, transformers.__version__)
PY

# 4. Run the test suite (smoke tests for the env once the implementation lands).
.venv/bin/pytest -q
```

> If `uv` is not installed: `pip install uv`, or use plain `python -m venv .venv`
> and `.venv/bin/pip install -r requirements.txt`.

## Quickstart (Docker)

```bash
docker build -t mags-repro .
# GPU run (NVIDIA Container Toolkit):
docker run --rm --gpus all -v "$(pwd):/work" -w /work mags-repro bash -lc \
  'python -c "import torch,transformers; print(torch.__version__, torch.cuda.is_available())"'
# CPU-only sanity run:
docker run --rm -v "$(pwd):/work" -w /work mags-repro bash
```

The image bakes the same pinned environment as the `uv` flow and runs an import
smoke test at build time. Repository source is bind-mounted at `/work`.

## Dependencies (why each is needed)

| Package | Role in the method |
|---|---|
| `torch` | tensors, compact SVD of the difference matrix (Eq. 5), forward passes, pre-hooks on `W_O` |
| `transformers` | load the three model families; chat templates; `generate` decode loop |
| `accelerate` | `device_map="auto"` for the 8B–20B models |
| `safetensors`, `huggingface-hub` | weight loading + model/dataset revision pinning |
| `sentencepiece`, `tokenizers` | Llama/Gemma tokenizer backends |
| `datasets` | MATH-500, GSM8K, MBPP, APPS, MathInstruct |
| `scikit-learn` (`scipy`) | `roc_auc_score` (per-head AUROC, head selection), `LogisticRegression` (ITI probes), PCA |
| `numpy` | `.npz` activation + manifold storage, array math |
| `math-verify` (`sympy`, `latex2sympy2-extended`) | boxed/number answer grading for MATH-500 & GSM8K |
| `human-eval` | HumanEval pass@1 execution harness |
| `rdkit` (`pillow`) | SMILES validity for molecular generation (Table 3) |
| `matplotlib` | latent-trajectory (Fig. 4) and attention-shift (Fig. 5) figure reproduction |
| `pytest` | test runner |

`requirements.txt` pins **all** transitive dependencies too, so the rebuild is
bit-for-bit identical. Verified resolvable and import-clean on
cpython-3.13 / linux-aarch64; the same pins resolve on linux-x86_64 (where the
default `torch==2.7.1` wheel is the CUDA 12 build).

## Molecular generation (Table 3)

SMILES **validity** is computed with RDKit (included). Binding **affinity** is
scored with AutoDock-GPU, a native C++/CUDA binary that is **not** a pip
package. Install it out-of-band on the GPU host:

```bash
# clone & build AutoDock-GPU (requires a CUDA toolkit + CMake)
git clone https://gitlab.com/autodock/autodock-gpu.git
cd autodock-gpu && mkdir build && cd build
cmake .. -DCUDA_INC_PATH=/usr/local/cuda/include -DCUDA_LIB_PATH=/usr/local/cuda/lib64
make -j
# the `autodock_gpu_128wi` binary is then invoked by the affinity scorer
```

Per `SPEC.md §4.18`, the molecular task's target protein, prompt template,
affinity cutoff and docking params are unspecified by the paper; Table 3 is
treated as a stretch target, resolved against AutoDock-GPU defaults at
implementation time.

## Reproduction status

See `REPRODUCTION.md` for the running log. The headline numbers-gate contract
is the MAGS-vs-unsteered gap on MATH-500: Llama **0.530 vs 0.478** (Δ +5.2),
Gemma **0.648 vs 0.614** (Δ +3.4), with 95% bootstrap CIs recorded in
`arms.json`.
