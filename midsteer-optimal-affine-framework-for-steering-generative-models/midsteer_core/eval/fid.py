"""FID between steered and vanilla image sets, SPEC section 4.4 item 15
(paper/content/experiments.tex:103)."""
from __future__ import annotations
import torch
from midsteer_core.data import BlockedException, is_model_arm_blocked


def _fid_from_stats(mu_a, sigma_a, mu_b, sigma_b) -> float:
    """Frechet distance: ||mu_a-mu_b||^2 + Tr(Sigma_a + Sigma_b - 2 sqrt(Sigma_a Sigma_b))."""
    diff = mu_a - mu_b
    sr = sigma_a @ sigma_b
    # matrix square root via eigh (symmetric PSD product approximated symmetric)
    evals, evecs = torch.linalg.eigh(sr)
    evals = torch.clamp(evals, min=0.0)
    sqrt_sr = (evecs * torch.sqrt(evals)) @ evecs.mT
    return float((diff @ diff) + torch.trace(sigma_a + sigma_b - 2 * sqrt_sr))


def fid(images_a, images_b):
    """FID between two image sets. Blocked (clean-fid backbone) in this sandbox."""
    if is_model_arm_blocked():
        raise BlockedException("fid blocked: clean-fid backbone needs CUDA + HF_TOKEN")
    raise BlockedException("fid real path not wired")
