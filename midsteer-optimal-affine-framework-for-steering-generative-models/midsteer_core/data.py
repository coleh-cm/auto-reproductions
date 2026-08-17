"""Data pipeline for the MidSteer reproduction.

Two surfaces:
  - synthetic_gaussian / synthetic_pair: deterministic synthetic joint Gaussian data for the
    E1 closed-form invariant checks (claims C1-C3). These are the ONLY real numbers this
    CPU sandbox can produce; they are construction truth, not a substitute for the paper's
    datasets.
  - load_model_activations: REAL data loader for the model arms (Llama-2-7B-chat, SDXL).
    This is an INSTRUMENT (decides whether the paper's own dataset is loaded). It REQUIRES a
    GPU and HF_TOKEN; when either is missing it RAISES BlockedException and NEVER falls back to
    synthetic data (a closed-book run that fell back to a synthetic corpus produced seven arms
    at chance level and meant nothing).

BlockedException is the only signal the experiment entrypoints use to mark an arm BLOCKED.
"""
from __future__ import annotations
import os
import torch


class BlockedException(Exception):
    """Raised when a model arm cannot run (no CUDA and/or no HF_TOKEN)."""


# Fingerprints asserted when a real checkpoint DOES load (data-loader instrument).
# These pin the paper's own datasets by their public checkpoint identity (SPEC gap G10/G11).
DATA_FINGERPRINTS = {
    'meta-llama/Llama-2-7b-chat-hf': {'vocab': 32000, 'params_B': 7.0, 'family': 'llama2-chat'},
    'meta-llama/Llama-2-7b-hf':       {'vocab': 32000, 'params_B': 7.0, 'family': 'llama2'},
    'stabilityai/stable-diffusion-xl-base-1.0': {'unet': 'UNet2DConditionModel', 'res': 1024},
    'openai/clip-vit-base-patch32':  {'d_embed': 512, 'patches': 16},
}


def is_model_arm_blocked() -> bool:
    """True iff this sandbox cannot run the model arms (no CUDA OR no HF_TOKEN)."""
    no_cuda = not torch.cuda.is_available()
    no_token = os.environ.get('HF_TOKEN') is None
    return no_cuda or no_token


def _psd_cov(d: int, gen: torch.Generator, jitter: float = 0.1) -> torch.Tensor:
    A = torch.randn(d, d, generator=gen, dtype=torch.float64)
    return A @ A.mT + jitter * torch.eye(d, dtype=torch.float64)


def synthetic_gaussian(seed: int, d: int = 32, n: int = 200000) -> dict:
    """Deterministic joint Gaussian for the C1/C2 LEACE closed-form checks.

    Builds Sigma_XX (PSD), X ~ N(0, Sigma_XX) [n, d], and a column Sigma_XZ in Im(Sigma_XX)
    by taking Sigma_XZ = Sigma_XX @ v for a random v in R^d. Z is drawn from the LINEAR data
    model Z = X @ Sigma_XZ + eps (eps ~ N(0, tau)), so Cov(X, Z) = Sigma_XZ by construction
    (up to the chosen scale; we keep Sigma_XZ as the population cross-covariance).

    Returns dict with keys: X [n,d], Z [n,1], sigma_xx [d,d], sigma_xz [d,1], mu [d].
    """
    g = torch.Generator().manual_seed(int(seed))
    sigma_xx = _psd_cov(d, g)
    L = torch.linalg.cholesky(sigma_xx)
    X = torch.randn(n, d, generator=g, dtype=torch.float64) @ L.mT   # [n, d], Cov = Sigma_XX
    v = torch.randn(d, 1, generator=g, dtype=torch.float64)
    sigma_xz = sigma_xx @ v                                            # in Im(Sigma_XX) by construction
    # Z = X @ sigma_xz + eps; Cov(X, Z) = Sigma_XX @ sigma_xz = Sigma_XX @ (Sigma_XX v)
    # We instead want Cov(X, Z) = sigma_xz exactly. Use Z = X @ (sigma_xx^{-1} sigma_xz) + eps,
    # i.e. the linear combination w = Sigma_XX^{-1} sigma_xz = v (since sigma_xz = Sigma_XX v).
    w = v                                                              # = Sigma_XX^{-1} sigma_xz
    eps = torch.randn(n, 1, generator=g, dtype=torch.float64) * 1e-3
    Z = X @ w + eps                                                   # [n, 1]; Cov(X,Z) = Sigma_XX w = sigma_xz
    mu = X.mean(dim=0)                                               # [d] (sample mean; ~0)
    return {
        'X': X, 'Z': Z, 'sigma_xx': sigma_xx, 'sigma_xz': sigma_xz, 'mu': mu,
        'w': w,
    }


def synthetic_pair(seed: int, d: int = 32, n: int = 200000) -> dict:
    """Deterministic joint Gaussian for the C3 MidSteer two-concept check.

    Builds Sigma_XX, two columns Sigma_XZ1 (nonzero) and Sigma_XZ2 in Im(Sigma_XX),
    and Z1, Z2 via the linear trick. Returns dict with X, Z1, Z2, sigma_xx, sigma_xz1,
    sigma_xz2, mu.
    """
    g = torch.Generator().manual_seed(int(seed) + 7)
    sigma_xx = _psd_cov(d, g)
    L = torch.linalg.cholesky(sigma_xx)
    X = torch.randn(n, d, generator=g, dtype=torch.float64) @ L.mT
    v1 = torch.randn(d, 1, generator=g, dtype=torch.float64)
    v2 = torch.randn(d, 1, generator=g, dtype=torch.float64)
    sigma_xz1 = sigma_xx @ v1                                          # nonzero (v1 random, a.s. != 0)
    sigma_xz2 = sigma_xx @ v2
    eps1 = torch.randn(n, 1, generator=g, dtype=torch.float64) * 1e-3
    eps2 = torch.randn(n, 1, generator=g, dtype=torch.float64) * 1e-3
    Z1 = X @ v1 + eps1                                                 # Cov(X, Z1) = sigma_xz1
    Z2 = X @ v2 + eps2                                                 # Cov(X, Z2) = sigma_xz2
    mu = X.mean(dim=0)
    return {
        'X': X, 'Z1': Z1, 'Z2': Z2, 'sigma_xx': sigma_xx,
        'sigma_xz1': sigma_xz1, 'sigma_xz2': sigma_xz2, 'mu': mu,
        'v1': v1, 'v2': v2,
    }


def load_model_activations(model_id: str, concept: str, split: str, n: int,
                           hf_token: str | None) -> torch.Tensor:
    """REAL data loader for model arms. Raises BlockedException in this CPU sandbox.

    When CUDA + HF_TOKEN are available this would: download the named checkpoint,
    assert DATA_FINGERPRINTS[model_id], run the concept/background prompts through the
    model, and return the SA/CA block output activations (per head, last-token for LLMs;
    all patches for diffusion). Here it always raises.
    """
    if is_model_arm_blocked():
        raise BlockedException(
            f"model arm blocked: no CUDA and/or no HF_TOKEN (model_id={model_id}). "
            "This sandbox is CPU-only and has no HF_TOKEN; the model arms cannot run and "
            "do NOT fall back to synthetic data.")
    if model_id not in DATA_FINGERPRINTS:
        raise BlockedException(f"unknown model_id {model_id}; no fingerprint recorded")
    if hf_token is None:
        raise BlockedException("HF_TOKEN required to load model activations")
    # If we ever reach here, a real load would proceed; fingerprint assert is the gate.
    fp = DATA_FINGERPRINTS[model_id]
    # Placeholder: real implementation would assert fp against the loaded checkpoint.
    raise BlockedException(
        "real model load path not wired in this sandbox; fingerprints recorded for "
        f"{model_id}: {fp}")
