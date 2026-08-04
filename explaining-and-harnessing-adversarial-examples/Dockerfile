# Reproducible build for the reproduction of
# "Explaining and Harnessing Adversarial Examples"
# (Goodfellow, Shlens, Szegedy; ICLR 2015, arXiv:1412.6572v3).
#
# CPU-only PyTorch environment. The paper's MNIST / CIFAR-10 experiments are
# small enough to run on CPU; this image has no CUDA runtime.
#
# Build:  docker build -t eae-repro .
# Run:    docker run --rm -it eae-repro
# Tests: docker run --rm -it eae-repro pytest -q
#
FROM python:3.13-slim AS base

# uv is used as the installer so the in-image install matches the documented
# quickstart exactly (same resolver, same pins).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned) — a single static binary.
RUN curl -fsSL https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /repro

# Install dependencies first (better layer caching than copying source).
COPY requirements.txt ./
RUN uv venv --python 3.13 .venv \
    && uv pip install --python .venv -r requirements.txt

# Copy the rest of the reproduction (implementation, tests, paper sources).
COPY . .

# Put the venv on PATH so `python` and `pytest` resolve to the pinned env.
ENV PATH="/repro/.venv/bin:${PATH}"

# MNIST / CIFAR-10 are downloaded on first run into ./data; keep it on a volume
# so repeated runs don't re-download.
VOLUME ["/repro/data"]

# Default: run the test suite. Override with `docker run eae-repro <cmd>`.
CMD ["pytest", "-q"]
