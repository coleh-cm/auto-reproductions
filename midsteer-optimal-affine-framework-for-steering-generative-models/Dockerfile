# Dockerfile — MidSteer reproduction environment
# Reproduces the pinned environment from requirements.txt from scratch.
# The paper's GPU experiments (Llama-2-7B-chat, SDXL) need CUDA; this image uses
# the CUDA 12.4 / cu124 base so the same requirements.txt builds the full stack.
# CPU-only import gate: `docker run --rm midsteer-repro python -m pytest tests/ -q`.

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

WORKDIR /midsteer

# Install dependencies first (better layer caching).
COPY requirements.txt ./
RUN uv venv --python 3.13 .venv \
    && uv pip install --python .venv -r requirements.txt

# Copy the vendored implementation + tests + paper/spec artefacts.
COPY . .

# Make the venv the default Python on PATH.
ENV PATH="/midsteer/.venv/bin:$PATH" \
    VIRTUAL_ENV="/midsteer/.venv" \
    HF_HOME="/midsteer/.cache/huggingface" \
    MPLBACKEND="Agg"

# Import gate: prove the environment resolves the method's dependencies.
RUN python -m pytest tests/ -q

CMD ["bash"]
