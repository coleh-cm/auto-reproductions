"""Degeneracy test: the method at its no-op setting (beta=0) must reproduce the
baseline EXACTLY. This is the cheapest real correctness evidence (research-code skill)."""
import torch
from midsteer_core import affine


def _psd(H, d, gen, jitter=0.1):
    A = torch.randn(H, d, d, generator=gen, dtype=torch.float64)
    return A @ A.mT + jitter * torch.eye(d, dtype=torch.float64)


def test_beta_zero_affine_is_identity_exactly():
    """beta=0 => A = I, b = 0, h' = h EXACTLY for LEACE / MidSteer / vanilla."""
    H, d = 3, 6
    g = torch.Generator().manual_seed(10)
    cov = _psd(H, d, g)
    W, Wp = affine.whiten(cov)
    sxz = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    sxz2 = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    mu = torch.randn(H, d, dtype=torch.float64)
    h = torch.randn(4, 9, H, d, dtype=torch.float64)

    for update in (
        affine.leace_update(W, Wp, sxz),
        affine.midsteer_update(W, Wp, sxz, sxz2),
    ):
        A0 = affine.compose_A(update, 0.0)
        b0 = affine.bias(mu, A0)
        assert torch.equal(A0, torch.eye(d, dtype=torch.float64).unsqueeze(0).expand_as(A0))
        assert torch.equal(b0, torch.zeros_like(b0))
        assert torch.equal(affine.apply_affine(h, A0, b0), h)


def test_beta_zero_vanilla_is_identity_exactly():
    """(I - 0 * s s^T) == I EXACTLY."""
    H, d = 2, 5
    g = torch.Generator().manual_seed(11)
    mu_s = torch.randn(H, d, generator=g, dtype=torch.float64)
    mu_t = torch.randn(H, d, generator=g, dtype=torch.float64)
    s = affine.vanilla_steering_vector(mu_s, mu_t)
    A0 = affine.compose_A(affine.vanilla_update(s), 0.0)
    I = torch.eye(d, dtype=torch.float64).unsqueeze(0).expand_as(A0)
    assert torch.equal(A0, I)


def test_beta_one_leace_switch_equals_beta2_leace_form():
    """LEACE-Switch (Eq.13) is the LEACE beta-form (Eq.22) at beta=2."""
    H, d = 2, 6
    g = torch.Generator().manual_seed(12)
    cov = _psd(H, d, g)
    W, Wp = affine.whiten(cov)
    sxz = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    A_switch = affine.compose_A(affine.leace_update(W, Wp, sxz), 2.0)
    # Eq.13 literal: I - 2 Wp u pinv(u) W
    u = W @ sxz
    A_literal = (torch.eye(d, dtype=torch.float64).unsqueeze(0)
                 - 2.0 * Wp @ u @ torch.linalg.pinv(u) @ W)
    assert torch.allclose(A_switch, A_literal, atol=1e-9)
