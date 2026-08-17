"""Closed-form affine maps (SPEC section 4.3; Eqs. 6/13/19/22/23 + vanilla 21/24/25).

Sign conventions pinned to the paper:
  W   = (Sigma_XX^{1/2})^+  = Sigma_XX^{-1/2} on the support        (paper/content/guardedness.tex:85)
  Wp  =  Sigma_XX^{1/2}      (= W^+ on the support)
  u_i = W Sigma_XZi = Sigma_WX,Zi                                  (paper/main.tex:429)

  LEACE        A = I - beta * Wp u (u)^+ W                          (Eq.22, paper/main.tex:471-474)
  LEACE-Switch A = I - beta * Wp u (u)^+ W   with beta = 2          (Eq.13, paper/main.tex:363-367)
  MidSteer     A = I + beta * Wp (u2 - u1) (u1)^+ W                 (Eq.23, paper/main.tex:478-481)
                 = I - beta * Wp (u1 - u2) (u1)^+ W   (the form we compute)
  Vanilla      A = I - beta * s s^T,  ||s|| = 1                     (Eqs.21/24/25)
  b_hat        = mu - A mu                                          (paper/main.tex:366-367 / :448)

Erasure special case (paper/main.tex:461): with u2 = 0 (Z2 constant),
  MidSteer A = I - beta Wp u1 (u1)^+ W = LEACE A  exactly.
"""
from __future__ import annotations
import torch
from core.math import fractional_matrix_power_cov_torch


def whiten(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """W = (Sigma^{1/2})^+ = Sigma^{-1/2} (support), Wp = Sigma^{1/2}.

    cov: [H, d, d] symmetric PSD (float64). Returns (W, Wp), each [H, d, d].
    Tolerance is the host library default (SPEC gap G7; matches core/math.py:13).
    """
    if cov.dim() != 3:
        raise ValueError(f"cov must be [H, d, d], got {tuple(cov.shape)}")
    W = fractional_matrix_power_cov_torch(cov, -0.5)   # (Sigma^{1/2})^+
    Wp = fractional_matrix_power_cov_torch(cov, 0.5)  #  Sigma^{1/2}
    return W, Wp


def leace_update(W: torch.Tensor, Wp: torch.Tensor, sxz: torch.Tensor) -> torch.Tensor:
    """Q = Wp (W Sxz) (W Sxz)^+ W;  A_hat = I - beta * Q   (Eq.22)."""
    u = W @ sxz                                       # [H, d, 1]  == Sigma_WX,Z
    u_pinv = torch.linalg.pinv(u)                      # [H, 1, d]  (Moore-Penrose of a column)
    Q = Wp @ u @ u_pinv @ W                            # [H, d, d]
    return Q


def midsteer_update(W: torch.Tensor, Wp: torch.Tensor,
                    sxz1: torch.Tensor, sxz2: torch.Tensor) -> torch.Tensor:
    """Q2 = Wp (u1 - u2) (u1)^+ W  so that  A = I - beta*Q2 == I + beta*Wp (u2-u1) (u1)^+ W (Eq.23)."""
    u1 = W @ sxz1                                     # [H, d, 1]
    u2 = W @ sxz2                                     # [H, d, 1]
    u1_pinv = torch.linalg.pinv(u1)                   # [H, 1, d]
    Q2 = Wp @ (u1 - u2) @ u1_pinv @ W                  # [H, d, d]
    return Q2


def vanilla_update(s: torch.Tensor) -> torch.Tensor:
    """Returns s s^T (per head), so A = I - beta * s s^T. s must be unit-norm per head."""
    if s.dim() == 2:                                  # [H, d]
        s = s
    elif s.dim() == 3 and s.shape[-1] == 1:
        s = s.squeeze(-1)
    else:
        raise ValueError(f"s must be [H, d] or [H, d, 1], got {tuple(s.shape)}")
    return s.unsqueeze(-1) @ s.unsqueeze(-2)          # [H, d, d]


def compose_A(update: torch.Tensor, beta: float) -> torch.Tensor:
    """A = I - beta * update  (per-head block-diagonal; eye per head)."""
    H, d, _ = update.shape
    I = torch.eye(d, dtype=update.dtype, device=update.device).unsqueeze(0).expand_as(update)
    return I - beta * update


def bias(mu: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """b_hat = mu - A mu  (paper/main.tex:366-367 / :448). mu: [H, d]; A: [H, d, d]."""
    return mu - (A @ mu.unsqueeze(-1)).squeeze(-1)   # [H, d]


def apply_affine(h: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h' = A h + b per head, broadcasting over all leading dims.

    Accepts [B, T, H, d], [B, P, H, d], [H, d], or any [..., H, d]. A: [H, d, d], b: [H, d].
    """
    if h.shape[-1] != A.shape[-1]:
        raise ValueError(f"last dim mismatch: h {tuple(h.shape)} vs A {tuple(A.shape)}")
    if h.shape[-2] != A.shape[0]:
        raise ValueError(f"head dim mismatch: h {tuple(h.shape)} vs A {tuple(A.shape)}")
    lead = h.shape[:-2]                               # may be ()
    H, d = h.shape[-2], h.shape[-1]
    A = A.to(h.dtype).to(h.device)
    b = b.to(h.dtype).to(h.device)
    # einsum over head + feature: A[h, i, j] * h[..., h, j] -> [..., h, i]
    out = torch.einsum('hij,...hj->...hi', A, h)
    out = out + b                                     # broadcast b [H, d] over leading dims
    return out


def vanilla_steering_vector(mu_s: torch.Tensor, mu_t: torch.Tensor) -> torch.Tensor:
    """s = mu_s - mu_t, unit-normalised per head (SPEC gap G8). Raises on a zero head diff."""
    diff = mu_s - mu_t                                # [H, d]
    norms = torch.linalg.norm(diff, dim=-1)          # [H]
    if torch.any(norms == 0):
        raise ValueError("vanilla_steering_vector: a head has zero source-target mean diff "
                         "(degenerate; MidSteer not applicable there, paper/main.tex:456-457)")
    return diff / norms.unsqueeze(-1)                 # [H, d], unit-norm per head
