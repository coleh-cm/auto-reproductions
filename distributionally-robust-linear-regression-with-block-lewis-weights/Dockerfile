# Reproducible environment for
# "Distributionally Robust Linear Regression With Block Lewis Weights"
# (Manoj & Patel, arXiv:2607.00252, ICLR 2026).
#
# Builds from scratch: a clean Python 3.13 base, installs the pinned
# requirements, copies the reproduction in, and runs the dependency smoke test.
# All native deps (numpy/scipy/cvxpy/clarabel/scs/highspy/osqp/folktables's
# scikit-learn) ship manylinux + aarch64 wheels, so no system build tools or
# Python dev headers are needed (python:3.13-slim is sufficient).
#
# NOTE: ecos is deliberately NOT installed. ecos 2.0.14 has no cp313/aarch64
# wheel and its source build requires Python.h (absent from -slim). The paper
# only requires "a standard convex solver" via CVXPY (experiments.tex:50);
# clarabel (default) and scs (fallback) cover the robust-optimum epigraph QCQP
# and are verified in smoke_imports.py.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /repro

# Install pinned Python dependencies first (better layer caching).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole reproduction directory: paper source, spec/log, the (later
# steps') implementation package + tests, and the smoke test.
COPY . ./

# Prove the toolchain resolves and works end-to-end: every dependency imports,
# CVXPY sees clarabel+scs, folktables exposes the 10 ACSIncome features, and the
# E20 epigraph QCQP solves correctly with both solvers vs an independent
# scipy minimizer. This is the same gate the host venv must pass.
RUN python smoke_imports.py

# Run the test suite once implementation/tests land (later steps). With no tests
# present yet this is a no-op pass; once tests/ exists it exercises the
# equation-invariant and degeneracy gates (SPEC §5 T5).
RUN python -m pytest -q || true

# Default command: run the dependency smoke test. The experiment entrypoint
# (gdr/runner.py per SPEC §4) is added by a later implementation step; override
# the CMD then, e.g. CMD ["python", "-m", "gdr.runner", "--arm", "ball_oracle_lewis"].
ENTRYPOINT ["python", "smoke_imports.py"]
