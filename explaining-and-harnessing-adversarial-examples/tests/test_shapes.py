"""Shape assertions at function boundaries (research-code skill: shapes written down).

Broadcasting silently does something plausible and wrong.  These assert the
shapes the SPEC (section 3) freezes for every array operation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT / "src"))

from fgsm_repro.models import SoftmaxRegression, LogisticRegression, MaxoutMLP, RBFNet
from fgsm_repro.attacks import fgsm, sample_rubbish, fooling_sign_step, fgsm_ensemble
from fgsm_repro.objectives import cross_entropy_cost, adversarial_train_cost
from fgsm_repro.eval import eval_clean, eval_fgsm, eval_rubbish, class_agreement

B, F, K = 8, 784, 10


def test_logits_shapes():
    x = torch.rand(B, F)
    assert SoftmaxRegression(F, K).logits(x).shape == (B, K)
    assert LogisticRegression(F).logits(x).shape == (B, 1)
    assert MaxoutMLP(units=16, pieces=2, dropout_input_include=1.0,
                     dropout_hidden_include=1.0).logits(x).shape == (B, K)
    assert RBFNet(K, F).logits(x).shape == (B, K)


def test_fgsm_shape_matches_input():
    m = SoftmaxRegression(F, K)
    x = torch.rand(B, F)
    y = torch.randint(0, K, (B,))
    xt = fgsm(m, x, y, 0.25)
    assert xt.shape == x.shape == (B, F)
    assert xt.dtype == x.dtype


def test_fgsm_ensemble_shape():
    models = [SoftmaxRegression(F, K) for _ in range(3)]
    x = torch.rand(B, F)
    y = torch.randint(0, K, (B,))
    assert fgsm_ensemble(models, x, y, 0.25).shape == (B, F)


def test_fooling_sign_step_shape():
    m = SoftmaxRegression(F, K)
    x = torch.rand(B, F)
    assert fooling_sign_step(m, x, cls=3, eps=0.25).shape == (B, F)


def test_sample_rubbish_shape():
    gen = torch.Generator().manual_seed(0)
    assert sample_rubbish(100, F, gen).shape == (100, F)


def test_cost_returns_scalar():
    m = SoftmaxRegression(F, K)
    x = torch.rand(B, F)
    y = torch.randint(0, K, (B,))
    assert cross_entropy_cost(m, x, y).shape == ()
    assert adversarial_train_cost(m, x, y, 0.25, 0.5).shape == ()


def test_maxout_layer_shapes():
    """Maxout layer: W [in, U*P], maxout output [B, U]."""
    m = MaxoutMLP(units=240, pieces=5, dropout_input_include=1.0,
                  dropout_hidden_include=1.0)
    assert m.layer0.W.shape == (F, 240 * 5)
    assert m.layer1.W.shape == (240, 240 * 5)
    assert m.readout.weight.shape == (K, 240)
    x = torch.rand(B, F)
    assert m.layer0(x).shape == (B, 240)
    assert m.layer1(m.layer0(x)).shape == (B, 240)


def test_eval_return_types():
    m = SoftmaxRegression(F, K)
    x = torch.rand(B, F)
    y = torch.randint(0, K, (B,))
    assert isinstance(eval_clean(m, x, y), float)
    e = eval_fgsm(m, x, y, 0.25)
    assert isinstance(e.error_rate, float) and isinstance(e.n, int) and e.n == B
    r = eval_rubbish(m, n=50, dim=F, seed=0)
    assert r.n == 50
    ag = class_agreement(m, m, x, y, 0.25)
    assert isinstance(ag.n_m1_errors, int) and isinstance(ag.n_both_wrong, int)
