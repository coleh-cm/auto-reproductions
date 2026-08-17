"""Tests for the data pipeline (midsteer_core.data)."""
import torch
import pytest
from midsteer_core import data
from midsteer_core.data import BlockedException, is_model_arm_blocked, synthetic_gaussian, synthetic_pair


def test_synthetic_gaussian_deterministic():
    a = synthetic_gaussian(seed=0, d=6, n=200)
    b = synthetic_gaussian(seed=0, d=6, n=200)
    assert torch.equal(a['X'], b['X'])


def test_synthetic_gaussian_psd_and_im_condition():
    d = 6
    s = synthetic_gaussian(seed=1, d=d, n=500)
    evals = torch.linalg.eigvalsh(s['sigma_xx'])
    assert torch.all(evals >= -1e-9)
    # sigma_xz in Im(sigma_xx): project onto Im equals itself
    sxz = s['sigma_xz']
    proj = s['sigma_xx'] @ torch.linalg.pinv(s['sigma_xx']) @ sxz
    assert torch.allclose(proj, sxz, atol=1e-7)


def test_synthetic_gaussian_empirical_cov_matches():
    s = synthetic_gaussian(seed=2, d=6, n=200000)
    X = s['X']; Z = s['Z']
    # Cov(X, Z) empirical ~ sigma_xz
    Xc = X - X.mean(dim=0, keepdim=True)
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov_emp = (Xc.mT @ Zc) / (X.shape[0] - 1)   # [d, 1]
    # finite-sample noise ~ O(|sigma_xz|/sqrt(n)); generous absolute tolerance, and
    # the construction guarantees equality in expectation (Z = X w + eps, Cov = Sigma_XX w).
    assert torch.allclose(cov_emp, s['sigma_xz'], atol=0.1)


def test_synthetic_pair_keys_and_im():
    s = synthetic_pair(seed=3, d=6, n=1000)
    for k in ('X', 'Z1', 'Z2', 'sigma_xx', 'sigma_xz1', 'sigma_xz2', 'mu'):
        assert k in s
    # sigma_xz1 nonzero
    assert torch.linalg.norm(s['sigma_xz1']) > 0
    # both in Im(Sigma_XX)
    P = s['sigma_xx'] @ torch.linalg.pinv(s['sigma_xx'])
    assert torch.allclose(P @ s['sigma_xz1'], s['sigma_xz1'], atol=1e-7)
    assert torch.allclose(P @ s['sigma_xz2'], s['sigma_xz2'], atol=1e-7)


def test_load_model_activations_raises_blocked_here():
    with pytest.raises(BlockedException):
        data.load_model_activations('meta-llama/Llama-2-7b-chat-hf', 'horse', 'train',
                                    n=10, hf_token=None)


def test_is_model_arm_blocked_true_in_sandbox():
    assert is_model_arm_blocked() is True
