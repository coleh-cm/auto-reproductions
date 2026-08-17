"""Cross-covariance under the balanced-prior mean-difference reading (SPEC gap G3,
reading (b); paper/content/experiments.tex:63).

The paper prints Sigma_XZi = Cov(X, Z_i) (paper/main.tex:422) but never states the
class priors or the sample mix. Under the balanced-prior mean-difference reading
(c_1 = c_2, absorbed into beta), Sigma_XZi = mu_i - mu_bg, which is exactly what the
upstream code computes (core/controller.py:157-176, verified SPEC section 1 item 2).
At beta = 1 (the MidSteer default in every claim) this coincides exactly with the
literal balanced Cov(X, Z_i).
"""
from __future__ import annotations
import torch


def mean_diff(x_concept: torch.Tensor, mu_bg: torch.Tensor) -> torch.Tensor:
    """Sigma_XZi = (mean over concept samples) - mu_bg, shaped [H, d, 1].

    x_concept: [H, n, d]; mu_bg: [H, d]. Raises if n == 0 (degenerate estimate).
    """
    if x_concept.dim() != 3:
        raise ValueError(f"x_concept must be [H, n, d], got {tuple(x_concept.shape)}")
    H, n, d = x_concept.shape
    if n == 0:
        raise ValueError("mean_diff: empty concept sample (n=0); cannot estimate Sigma_XZi")
    mu_concept = x_concept.mean(dim=1)                # [H, d]
    diff = mu_concept - mu_bg.to(mu_concept.dtype).to(mu_concept.device)  # [H, d]
    return diff.unsqueeze(-1)                         # [H, d, 1]
