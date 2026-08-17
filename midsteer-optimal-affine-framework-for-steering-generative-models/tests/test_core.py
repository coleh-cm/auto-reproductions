"""Unit tests for the closed-form affine core (midsteer_core.affine / stats / crosscov)."""
import torch
import pytest

from midsteer_core.stats import WelfordState, welford_update, welford_finalize
from midsteer_core.crosscov import mean_diff
from midsteer_core import affine


def _psd(d, H, gen, jitter=0.1):
    A = torch.randn(H, d, d, generator=gen, dtype=torch.float64)
    return A @ A.mT + jitter * torch.eye(d, dtype=torch.float64)


def test_whiten_roundtrip_on_support():
    H, d = 2, 8
    g = torch.Generator().manual_seed(0)
    cov = _psd(d, H, g)
    W, Wp = affine.whiten(cov)
    # W @ Wp == I on the support (W = Sigma^{-1/2}, Wp = Sigma^{1/2})
    prod = W @ Wp
    I = torch.eye(d, dtype=torch.float64).unsqueeze(0)
    assert torch.allclose(prod, I.expand_as(prod), atol=1e-7)


def test_welford_matches_torch_cov():
    H, m, d = 3, 256, 5
    g = torch.Generator().manual_seed(1)
    x = torch.randn(H, m, d, generator=g, dtype=torch.float64)
    # feed in two batches
    st = welford_update(None, x[:, :100, :])
    st = welford_update(st, x[:, 100:, :])
    mu, cov = welford_finalize(st)
    for h in range(H):
        mu_t = x[h].mean(dim=0)
        cov_t = torch.cov(x[h].T)  # unbiased (n-1)
        assert torch.allclose(mu[h], mu_t, atol=1e-9)
        assert torch.allclose(cov[h], cov_t, atol=1e-9)


def test_welford_finalize_raises_on_empty():
    with pytest.raises(ValueError):
        welford_finalize(WelfordState(n=0, M=torch.zeros(1, 3), S=torch.zeros(1, 3, 3)))
    with pytest.raises(ValueError):
        welford_finalize(None)


def test_mean_diff_shape_and_raise():
    H, n, d = 2, 10, 4
    x = torch.randn(H, n, d, dtype=torch.float64)
    mu_bg = torch.randn(H, d, dtype=torch.float64)
    md = mean_diff(x, mu_bg)
    assert md.shape == (H, d, 1)
    with pytest.raises(ValueError):
        mean_diff(torch.zeros(H, 0, d), mu_bg)


def test_affine_shapes():
    H, d = 2, 8
    g = torch.Generator().manual_seed(2)
    cov = _psd(d, H, g)
    W, Wp = affine.whiten(cov)
    sxz = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    Q = affine.leace_update(W, Wp, sxz)
    assert Q.shape == (H, d, d)
    sxz1 = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    sxz2 = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    Q2 = affine.midsteer_update(W, Wp, sxz1, sxz2)
    assert Q2.shape == (H, d, d)
    A = affine.compose_A(Q, 1.0)
    assert A.shape == (H, d, d)
    mu = torch.randn(H, d, dtype=torch.float64)
    b = affine.bias(mu, A)
    assert b.shape == (H, d)
    h = torch.randn(4, 7, H, d, dtype=torch.float64)
    out = affine.apply_affine(h, A, b)
    assert out.shape == h.shape
    h1 = torch.randn(H, d, dtype=torch.float64)
    assert affine.apply_affine(h1, A, b).shape == h1.shape


def test_degeneracy_beta_zero_is_identity_exactly():
    H, d = 2, 6
    g = torch.Generator().manual_seed(3)
    cov = _psd(d, H, g)
    W, Wp = affine.whiten(cov)
    sxz = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    Q = affine.leace_update(W, Wp, sxz)
    A0 = affine.compose_A(Q, 0.0)
    I = torch.eye(d, dtype=torch.float64).unsqueeze(0).expand_as(A0)
    assert torch.equal(A0, I)  # EXACT
    mu = torch.randn(H, d, dtype=torch.float64)
    b0 = affine.bias(mu, A0)
    assert torch.equal(b0, torch.zeros_like(b0))  # EXACT
    h = torch.randn(5, H, d, dtype=torch.float64)
    assert torch.equal(affine.apply_affine(h, A0, b0), h)  # EXACT


def test_midsteer_sign_matches_eq23_beta1():
    # compose_A(midsteer_update(...), 1) == I + Wp (u2 - u1) pinv(u1) W  (Eq.23, beta=1)
    H, d = 2, 8
    g = torch.Generator().manual_seed(4)
    cov = _psd(d, H, g)
    W, Wp = affine.whiten(cov)
    sxz1 = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    sxz2 = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    A = affine.compose_A(affine.midsteer_update(W, Wp, sxz1, sxz2), 1.0)
    u1 = W @ sxz1
    u2 = W @ sxz2
    u1_pinv = torch.linalg.pinv(u1)
    A_literal = (torch.eye(d, dtype=torch.float64).unsqueeze(0)
                 + Wp @ (u2 - u1) @ u1_pinv @ W)
    assert torch.allclose(A, A_literal, atol=1e-9)


def test_erasure_special_case_midsteer_eq_leace():
    # with sxz2 == 0, compose_A(midsteer_update(W,Wp,sxz1,0),1) == compose_A(leace_update(W,Wp,sxz1),1)
    H, d = 2, 8
    g = torch.Generator().manual_seed(5)
    cov = _psd(d, H, g)
    W, Wp = affine.whiten(cov)
    sxz1 = torch.randn(H, d, 1, generator=g, dtype=torch.float64)
    sxz2_zero = torch.zeros(H, d, 1, dtype=torch.float64)
    A_mid = affine.compose_A(affine.midsteer_update(W, Wp, sxz1, sxz2_zero), 1.0)
    A_leace = affine.compose_A(affine.leace_update(W, Wp, sxz1), 1.0)
    assert torch.allclose(A_mid, A_leace, atol=1e-9)


def test_vanilla_steering_vector_unit_norm_and_raise():
    H, d = 2, 4
    mu_s = torch.randn(H, d, dtype=torch.float64)
    mu_t = torch.randn(H, d, dtype=torch.float64)
    s = affine.vanilla_steering_vector(mu_s, mu_t)
    assert torch.allclose(torch.linalg.norm(s, dim=-1), torch.ones(H, dtype=torch.float64), atol=1e-12)
    with pytest.raises(ValueError):
        affine.vanilla_steering_vector(mu_t, mu_t)


def test_leace_constraint_algebra():
    # Population constraint: ||A Sigma_XZ||_F ~ 0 (A annihilates Sigma_XZ on the support).
    H, d = 2, 8
    g = torch.Generator().manual_seed(6)
    cov = _psd(d, H, g)
    W, Wp = affine.whiten(cov)
    sxz = cov @ torch.randn(d, 1, generator=g, dtype=torch.float64)  # in Im(Sigma_XX)
    A = affine.compose_A(affine.leace_update(W, Wp, sxz), 1.0)
    mu = torch.zeros(H, d, dtype=torch.float64)
    b = affine.bias(mu, A)
    # Cov(A X + b, Z) = A Cov(X, Z) = A sigma_xz  (population; mu=0, b constant)
    cov_after = (A @ sxz).squeeze(-1)
    assert torch.linalg.norm(cov_after) < 1e-8


def test_bias_exact_for_nontrivial_A():
    """b_hat = mu - A mu exactly (paper/main.tex:366-367 / :448) for a non-identity A."""
    H, d = 2, 5
    g = torch.Generator().manual_seed(31)
    A = torch.randn(H, d, d, generator=g, dtype=torch.float64)
    mu = torch.randn(H, d, generator=g, dtype=torch.float64)
    b = affine.bias(mu, A)
    assert torch.allclose(b, mu - (A @ mu.unsqueeze(-1)).squeeze(-1), atol=1e-12)
