# Dockerfile — CASteer reproduction environment
# Reproduces the pinned environment from requirements.txt from scratch.
#
# The paper's GPU experiments (SD-1.4 / SDXL / SANA, "developed and tested using
# 8 V100 GPUs", paper/content/supplementary.tex:30) need CUDA; this image uses
# the CUDA 12.4 / cu124 base so the SAME requirements.txt builds the full stack.
# CPU-only import gate: `docker run --rm casteer-repro python -m pytest tests/ -q`.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps: Python 3.13 (deadsnakes), build tools, git, libsndfile (torchaudio).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
        software-properties-common \
        libsndfile1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, deterministic installs of the pinned requirements.
RUN curl -LsSf https://astral.sh/uv/uv-installer.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /casteer

# Install dependencies first (better layer caching).
COPY requirements.txt ./
RUN uv venv --python 3.13 .venv \
    && uv pip install --python .venv -r requirements.txt

# Copy the vendored upstream implementation + tests + paper/spec artefacts.
COPY . .

# Make the venv the default Python on PATH.
ENV PATH="/casteer/.venv/bin:$PATH" \
    VIRTUAL_ENV="/casteer/.venv" \
    HF_HOME="/casteer/.cache/huggingface" \
    MPLBACKEND="Agg"

# Import gate: prove the environment resolves the method's dependencies and
# the Householder / construction invariants (SPEC claim `house`) hold.
RUN python -m pytest tests/ -q

CMD ["bash"]
