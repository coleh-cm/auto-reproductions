# Reproducible environment for
# "Explaining and Harnessing Adversarial Examples"
# (Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy; ICLR 2015).
#
# Builds from scratch: a clean Python 3.13 base, installs the pinned
# requirements, copies the reproduction in, and smoke-tests that the
# dependencies import and that the core FGSM autograd path works
# (||η||_∞ == ε exactly, the paper's max-norm constraint).
#
# CPU-only: the reproduced MNIST experiments need no CUDA/GPU. torch is the
# +cpu wheel, so this image runs on any amd64 machine with no GPU driver.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /repro

# No extra system packages are required: torch + numpy ship manylinux/aarch64
# CPU wheels that need only the slim base's glibc.

# Install pinned Python dependencies first (better layer caching).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the reproduction: implementation, experiments, tests, and docs.
# (The implementation package and experiment scripts are added by the
# implementation step; this Dockerfile is written for the committed end
# state so a human can `docker build` after cloning.)
COPY . /repro

# Build-time smoke test: the only thing guaranteed to exist at every step is
# the dependency set, so we verify it imports AND that the FGSM invariant
# (η = ε·sign(∇ₓJ) ⇒ ||η||_∞ == ε) holds on a tiny synthetic batch. This is
# the gate the paper's method correctness depends on.
RUN python -c "import torch, numpy, pytest; \
import torch.nn.functional as F; \
x = torch.randn(8,784, requires_grad=True); y = torch.randint(0,10,(8,)); \
W = torch.randn(784,10, requires_grad=True); b = torch.randn(10, requires_grad=True); \
J = F.cross_entropy(x @ W + b, y); g = torch.autograd.grad(J, x)[0]; \
eta = 0.25 * g.sign(); \
assert (x + eta).shape == x.shape and abs(float(eta.abs().max()) - 0.25) < 1e-6; \
print('deps import OK; ||eta||_inf == eps == 0.25')"

# When the implementation step lands the package under src/fgsm_repro and the
# experiment entry scripts, run the degeneracy/shape tests as a build gate.
# `tests/` may be absent in intermediate commits, so guard with a check.
RUN if [ -d tests ]; then python -m pytest -q tests; else echo "no tests/ yet; skipping"; fi

# Default command: smoke-import the environment. The implementation step
# replaces this with the experiment runner (e.g. `python -m fgsm_repro.cli`).
CMD ["python", "-c", "import torch, numpy, pytest; print('fgsm-repro env OK', torch.__version__, numpy.__version__, pytest.__version__)"]
