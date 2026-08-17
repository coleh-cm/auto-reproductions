# Dockerfile — reproducible build for the LEACE reproduction environment.
# Paper: Belrose et al., "LEACE: Perfect linear concept erasure in closed form",
#        NeurIPS 2023, arXiv:2306.03819.
#
# Builds from scratch: Python 3.13 + uv, then installs requirements.txt (which pins the
# authors' concept-erasure package, torch CPU, transformers, datasets, scikit-learn,
# spacy + en_core_web_sm, tqdm, pytest). CPU-only; no GPU required.
#
# Build:
#   docker build -t leace-repro .
# Run (mount the repo so the paper/SPEC/claims and any experiment code are visible):
#   docker run --rm -it -v "$PWD":/work -w /work leace-repro bash
# Quickstart inside the container:
#   uv pip install --system -r requirements.txt
#   python -c "from concept_erasure import LeaceEraser; import torch; print('ok')"
#   pytest

FROM python:3.13-slim-bookworm

# uv needs a working C/C++ toolchain only for source builds; all our wheels are prebuilt
# (torch CPU wheels for linux/amd64 are on PyPI; concept-erasure is pure Python). We keep
# the image minimal but install ca-certificates and git (uv fetches concept-erasure from
# git by commit) and curl for the uv bootstrap.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned) globally.
ARG UV_VERSION=0.12.5
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx || true

WORKDIR /work

# Copy the pinned requirements first so this layer is cacheable across code changes.
COPY requirements.txt /work/requirements.txt

# Install into the system interpreter so `python`/`pytest` Just Work.
# torch==2.6.0 from PyPI is the CPU build on linux/arm64 and the CUDA build on linux/amd64;
# the CUDA build runs fine on CPU (it just does not use a GPU). To force the smaller CPU
# build on amd64, add: --extra-index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system --no-cache -r requirements.txt

# Smoke-test that the method core imports and the LEACE fit erases to machine precision.
RUN python -c "import torch, numpy, sklearn, transformers, datasets, spacy, tqdm, pytest; \
from concept_erasure import LeaceEraser, LeaceFitter, OracleEraser, ConceptScrubber; \
from concept_erasure.scrubbing import scrub, patch_attention_, random_scrub; \
import en_core_web_sm; \
torch.manual_seed(0); X=torch.randn(2000,16,dtype=torch.float64); \
Z=torch.nn.functional.one_hot(torch.randint(0,3,(2000,)),3).double(); \
e=LeaceEraser.fit(X,Z,affine=True,shrinkage=False,svd_tol=1e-9,constrain_cov_trace=False); \
Y=e(X); Xc=X-X.mean(0); Yc=Y-Y.mean(0); Zc=Z-Z.mean(0); \
cov=(Yc.t()@Zc)/(X.shape[0]-1); \
assert cov.abs().max().item() < 1e-12, cov.abs().max().item(); \
print('LEACE import + fit + guardedness OK; |Cov(Y,Z)|_inf =', cov.abs().max().item())"

# Bring in the rest of the repo (SPEC.md, claims.json, paper/, experiment code, tests).
COPY . /work

CMD ["bash"]
