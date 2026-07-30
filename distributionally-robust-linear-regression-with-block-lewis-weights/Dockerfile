# Reproducible environment for:
# "Distributionally Robust Linear Regression With Block Lewis Weights"
# (Manoj & Patel, 2026; arXiv 2607.00252)
#
# Builds the exact Python 3.13 + pinned-dependency environment described in
# requirements.txt from a clean base image.  This sandbox has no Docker daemon,
# so this file is not built here; it is written so a human can reproduce the
# result later with `docker build`.
#
# The reproduction itself is CPU-only on small problems (d=10, m<=100, n<=10200),
# so no GPU/CUDA is needed.

FROM python:3.13-slim

# Build/runtime prerequisites: a C toolchain (for any source builds of numpy /
# scipy / solvers if a wheel is unavailable on the target arch) plus curl for
# fetching ACS data via folktables at run time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

# Copy the lock and install first so dependency install is layer-cached.
COPY requirements.txt ./

# uv is used as the installer (fast, exact).  Install uv via the official
# standalone binary so we do not depend on a system Python pip.
RUN pip install --no-cache-dir uv==0.11.31

# Create an isolated venv and install the pinned environment from the lock.
RUN uv venv --python 3.13 .venv \
    && uv pip install --python .venv -r requirements.txt

# Copy the rest of the reproduction (source, SPEC, arms.json, README, paper/).
COPY . .

# Make the venv the default Python on PATH for interactive shells.
ENV PATH="/repo/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/repo/.venv"
ENV PYTHONUNBUFFERED=1

# Smoke test at build time: confirm every method import resolves and the
# CVXPY conic solver loads, so a broken dependency fails the build, not a run.
RUN python -c "import numpy, scipy, cvxpy, pandas, matplotlib, folktables, sklearn, requests" \
    && python -c "import cvxpy as cp; x=cp.Variable(2); p=cp.Problem(cp.Minimize(cp.sum_squares(x)),[x>=1]); assert abs(p.solve(solver=cp.CLARABEL)-2.0)<1e-6; print('build smoke test OK')"

# Default entrypoint leaves the user in the repo with the venv active.
CMD ["/bin/bash"]
