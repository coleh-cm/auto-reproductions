"""Welford online mean/covariance estimator (Algorithm 1,
paper/content/suppl.tex:11-52).

Per-head statistics: input batch X_k in R^{h x m_k x d}; running state
(n, M in R^{h x d}, S in R^{h x d x d}); finalization
  mu = M / n ; Sigma = (S + S^H) / (2 (n-1))
with the Hermitian symmetrisation and the (n-1) denominator made explicit
by the paper's pseudocode (paper/content/suppl.tex:48).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class WelfordState:
    n: int
    M: torch.Tensor   # [H, d]
    S: torch.Tensor   # [H, d, d]


def _empty_state(H: int, d: int, dtype: torch.dtype, device: torch.device) -> WelfordState:
    return WelfordState(n=0, M=torch.zeros(H, d, dtype=dtype, device=device),
                        S=torch.zeros(H, d, d, dtype=dtype, device=device))


def welford_update(state: WelfordState, batch: torch.Tensor) -> WelfordState:
    """Incorporate one batch X_k in R^{H, m, d} into the running statistics.

    Implements the first-batch initialisation branch and the subsequent-batch
    Welford update (Delta_old, Delta_new) of Algorithm 1.
    """
    if batch.dim() != 3:
        raise ValueError(f"batch must be [H, m, d], got shape {tuple(batch.shape)}")
    H, m, d = batch.shape
    dtype = batch.dtype
    device = batch.device
    if state is None:
        state = _empty_state(H, d, dtype, device)
    if state.M.shape != (H, d) or state.S.shape != (H, d, d):
        raise ValueError("WelfordState shape mismatch with batch")

    # sum over the m samples of this batch, per head: [H, d]
    batch_sum = batch.sum(dim=1)

    if state.n == 0:
        n = m
        M = batch_sum.clone()
        mu = M / n                                   # [H, d]
        # centre each sample: X_k - mu (broadcast over m): [H, m, d]
        delta = batch - mu.unsqueeze(1)
        # S = delta^H @ delta  (per head): [H, d, d]
        S = torch.matmul(delta.transpose(1, 2), delta)
        return WelfordState(n=n, M=M, S=S)

    mu_old = state.M / state.n                        # [H, d]
    n = state.n + m
    M = state.M + batch_sum
    mu_new = M / n                                    # [H, d]
    delta_old = batch - mu_old.unsqueeze(1)           # [H, m, d]
    delta_new = batch - mu_new.unsqueeze(1)           # [H, m, d]
    # S <- S + delta_old^H @ delta_new  (per head): [H, d, d]
    S = state.S + torch.matmul(delta_old.transpose(1, 2), delta_new)
    return WelfordState(n=n, M=M, S=S)


def welford_finalize(state: WelfordState) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mu, Sigma). Raises on empty (n < 2) — never silently return."""
    if state is None or state.n < 2:
        raise ValueError(f"Welford finalize requires n >= 2 samples, got n={getattr(state, 'n', None)}")
    mu = state.M / state.n                            # [H, d]
    # paper/content/suppl.tex:48: Sigma = (S + S^H) / (2 (n-1))
    S_sym = (state.S + state.S.transpose(-1, -2)) / 2.0
    cov = S_sym / (state.n - 1)
    return mu, cov
