# Dockerfile — Manifold-Guided Attention Steering (MAGS) reproduction environment
# Builds a reproducible Python 3.13 environment with every dependency pinned in
# requirements.txt. Verified to resolve & import on linux-aarch64 and linux-x86_64.
#
# The reproduction runs Llama-3.1-8B-Instruct, Gemma-4-E4B-it and GPT-OSS-20b on
# GPU. On x86_64 the default PyPI `torch==2.7.1` wheel is the CUDA 12 build (ships
# its own CUDA runtime libs), so `pip install -r requirements.txt` yields a
# GPU-capable torch automatically. Run the container with `--gpus all`
# (NVIDIA Container Toolkit) to use the GPUs.
#
# Build:  docker build -t mags-repro .
# Run:    docker run --rm --gpus all -v $(pwd):/work -w /work mags-repro bash
#
# AutoDock-GPU (binding-affinity scoring for the molecular task, Table 3) is a
# native C++/CUDA binary, not a pip package; install it separately on the host
# (see README.md "Molecular generation" notes). RDKit (pip, included) covers
# SMILES validity on its own.

FROM python:3.13-slim AS base

# System build/runtime dependencies:
#   - build-essential / git : needed if any wheel must be built from source
#                             (sentencepiece, tokenizers compile fallbacks)
#   - libgomp1               : OpenMP runtime for torch / scikit-learn / rdkit
#   - libgl1, libxrender1    : rdkit rendering backends
#   - ca-certificates, curl  : HuggingFace Hub downloads
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
        libgl1 \
        libxrender1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Upgrade pip first so pinned-wheel resolution is deterministic.
RUN pip install --no-cache-dir --upgrade pip

# Install the fully-pinned dependency set.
COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -r /work/requirements.txt

# Smoke-test that the core imports resolve inside the image.
RUN python -c "import torch, transformers, datasets, sklearn, numpy, accelerate, safetensors, sentencepiece, pytest; \
from sklearn.metrics import roc_auc_score; \
from sklearn.linear_model import LogisticRegression; \
import math_verify; from human_eval.execution import check_correctness; \
from rdkit import Chem; from rdkit.Chem import AllChem; \
import matplotlib; \
print('MAGS environment imports OK; torch', torch.__version__, 'transformers', transformers.__version__)"

# The repository source (implementation, tests, configs) is expected to be
# bind-mounted at /work at run time, so nothing is COPYied here beyond the
# requirements used to bake the image.
CMD ["bash"]
