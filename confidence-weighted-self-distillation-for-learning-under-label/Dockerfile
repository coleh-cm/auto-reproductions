# Reproducible environment for
# "Confidence-Weighted Self-Distillation for Learning under Label Noise"
# (Bergstrom, Oyelaran, Vasquez).
#
# Builds from scratch: a clean Python 3.13 base, installs the pinned
# requirements, and copies the experiment in. The implementation uses only
# numpy + scikit-learn (gradients hand-derived), so there is no CUDA/GPU need.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /repro

# System deps: none beyond the slim base (scikit-learn ships its own wheels).

# Install pinned Python dependencies first (better layer caching).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the experiment, tests, and supporting docs.
COPY run_experiment.py conftest.py ./
COPY tests ./tests
COPY README.md SPEC.md REPRODUCTION.md ./
COPY paper ./paper

# Smoke-test that the implementation imports and the degeneracy gate holds:
# the lambda=0 path must be bitwise identical to an independent cross-entropy
# routine (per-step grads and a 300-step SGD loop), plus all equation
# invariants. Cheap; no 4000-step training.
RUN python -m pytest -q tests && python -c "import run_experiment; print('import OK')"

# Default command: run the baseline arm. Override with `--lambda 1.0` for CWSD.
ENTRYPOINT ["python", "run_experiment.py"]
CMD ["--lambda", "0.0"]
