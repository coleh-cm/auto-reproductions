"""Invariant tests: the paper's covariance constraints (Eqs. 5/13/19) hold on
synthetic Gaussian data, evaluated in exact population arithmetic.

  C1 LEACE:        ||Cov(A_hat X + b_hat, Z)||_F           < 1e-9   (Eq.5 constraint)
  C2 LEACE-Switch: ||Cov(A_hat X + b_hat, Z) + Cov(X, Z)||_F < 1e-9 (Eq.13 constraint)
  C3 MidSteer:     ||Cov(A_hat X + b_hat, Z1) - Cov(X, Z2)||_F < 1e-9 (Eq.19 constraint)

These are the closed-form algebra checks behind claims C1-C3 (population covariance,
exact up to float64 round-off). The sample-based predicate (finite-n objective, vanilla
special case) is exercised by experiments/run_e1_synth.py.
"""
import torch
from midsteer_core import affine


def _psd(H, d, gen, jitter=0.1):
    A = torch.randn(H, d, d, generator=gen, dtype=torch.float64)
    return A @ A.mT + jitter * torch.eye(d, dtype=torch.float64)


def test_c1_leace_constraint_zero_cov():
    H, d = 2, 8
    g = torch.Generator().manual_seed(20)
    cov = _psd(H, d, g)
    W, Wp = affine.whiten(cov)
    v = torch.randn(d, 1, generator=g, dtype=torch.float64)
    sxz = cov @ v                                  # in Im(Sigma_XX)
    mu = torch.zeros(H, d, dtype=torch.float64)
    A = affine.compose_A(affine.leace_update(W, Wp, sxz), 1.0)
    b = affine.bias(mu, A)
    # Cov(A X + b, Z) = A Sigma_XZ  (b constant; mu=0)
    cov_after = A @ sxz
    assert torch.linalg.norm(cov_after) < 1e-9


def test_c2_leace_switch_flips_sign_of_cov():
    H, d = 2, 8
    g = torch.Generator().manual_seed(21)
    cov = _psd(H, d, g)
    W, Wp = affine.whiten(cov)
    v = torch.randn(d, 1, generator=g, dtype=torch.float64)
    sxz = cov @ v
    mu = torch.zeros(H, d, dtype=torch.float64)
    A = affine.compose_A(affine.leace_update(W, Wp, sxz), 2.0)   # beta=2 switch
    b = affine.bias(mu, A)
    cov_after = A @ sxz
    # Cov(f(X), Z) + Cov(X, Z) = A Sxz + Sxz == 0
    assert torch.linalg.norm(cov_after + sxz) < 1e-9


def test_c3_midsteer_matches_target_cov():
    H, d = 2, 8
    g = torch.Generator().manual_seed(22)
    cov = _psd(H, d, g)
    W, Wp = affine.whiten(cov)
    v1 = torch.randn(d, 1, generator=g, dtype=torch.float64)
    v2 = torch.randn(d, 1, generator=g, dtype=torch.float64)
    sxz1 = cov @ v1                                 # nonzero
    sxz2 = cov @ v2
    mu = torch.zeros(H, d, dtype=torch.float64)
    A = affine.compose_A(affine.midsteer_update(W, Wp, sxz1, sxz2), 1.0)  # beta=1
    b = affine.bias(mu, A)
    cov_after_z1 = A @ sxz1
    # Cov(f(X), Z1) - Cov(X, Z2) = A Sxz1 - Sxz2 == 0
    assert torch.linalg.norm(cov_after_z1 - sxz2) < 1e-9


def test_c3_midsteer_erasure_special_case_equals_leace():
    """With Z2 constant (sxz2=0), MidSteer A == LEACE A (paper/main.tex:461)."""
    H, d = 2, 8
    g = torch.Generator().manual_seed(23)
    cov = _psd(H, d, g)
    W, Wp = affine.whiten(cov)
    v1 = torch.randn(d, 1, generator=g, dtype=torch.float64)
    sxz1 = cov @ v1
    sxz2_zero = torch.zeros(H, d, 1, dtype=torch.float64)
    A_mid = affine.compose_A(affine.midsteer_update(W, Wp, sxz1, sxz2_zero), 1.0)
    A_leace = affine.compose_A(affine.leace_update(W, Wp, sxz1), 1.0)
    assert torch.allclose(A_mid, A_leace, atol=1e-9)
