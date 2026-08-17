"""Environment smoke tests for the MidSteer reproduction.

These tests pin the import gate: every third-party dependency the vendored
upstream implementation (https://github.com/Atmyre/MidSteer, HEAD 0f3b31e)
needs, plus every ``core`` module, must import cleanly under the pinned
environment in ``requirements.txt``. They are intentionally CPU-only and do
not touch the network or load model weights.
"""

import importlib

import pytest

# Third-party packages the method and its eval scripts import at module level.
THIRD_PARTY = [
    "torch", "torchvision", "torchaudio",
    "transformers", "diffusers", "accelerate", "datasets",
    "numpy", "scipy", "pandas", "PIL",
    "steering_vectors", "cleanfid", "cleanfid.fid", "clip",
    "detoxify", "bert_score", "nltk", "openai",
    "tqdm", "pydantic", "dotenv", "matplotlib", "pytest",
]

# Vendored upstream library modules (the actual MidSteer implementation).
CORE_MODULES = [
    "core", "core.math", "core.controller", "core.dataset",
    "core.llm_steering", "core.diffusion_steering", "core.vector_dump",
    "core.utils", "core.pickle", "core.prompt_utils", "core.prompts",
    "core.eval", "core.eval.clip", "core.eval.fid",
]


@pytest.mark.parametrize("module", THIRD_PARTY, ids=THIRD_PARTY)
def test_third_party_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", CORE_MODULES, ids=CORE_MODULES)
def test_core_imports(module):
    importlib.import_module(module)


def test_clip_is_openai_clip_not_clipboard():
    """Guard against the wrong `clip` PyPI package (a clipboard tool)."""
    import clip
    assert hasattr(clip, "load"), "clip.load missing — wrong `clip` package installed"
    assert hasattr(clip, "tokenize"), "clip.tokenize missing — wrong `clip` package installed"


def test_midsteer_whitening_identity():
    """Smoke-test the closed-form whitening W = (Sigma_XX^{1/2})^+ (SPEC Eq. 7).

    For a symmetric PSD covariance S, W^{+} @ W must equal the identity on the
    support of S. This is the numerical backbone of LEACE / LEACE-Switch /
    MidSteer (SPEC Eqs. 6/13/19, paper main.tex:445-448).
    """
    import torch
    from core.math import fractional_matrix_power_cov_torch

    torch.manual_seed(0)
    d, H = 8, 2
    A = torch.randn(H, d, d, dtype=torch.float64)
    S = A @ A.mT + torch.eye(d, dtype=torch.float64) * 0.1  # PSD per head
    W = fractional_matrix_power_cov_torch(S, 0.5)      # Sigma^{1/2}
    Wp = fractional_matrix_power_cov_torch(S, -0.5)   # (Sigma^{1/2})^+
    prod = (Wp @ W)[0]
    assert torch.allclose(prod, torch.eye(d, dtype=torch.float64), atol=1e-6)
