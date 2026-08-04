"""Degeneracy test (research-code skill): the method at its no-op setting must
reproduce the baseline EXACTLY.

For FGSM adversarial training the no-op setting is eps=0: the adversarial
example x_adv = x + 0*sign(g) = x, so the adversarial training loss
J_tilde = alpha*J(x) + (1-alpha)*J(x) = J(x) — identical to plain training.
A model trained with adversarial eps=0 must therefore match a model trained
without adversarial training, bit-identically, given the same seed and data
order.

This is the cheapest real correctness evidence: it catches a mis-wired
adversarial-training loop (e.g. gradients leaking through sign, or the
adversarial half using the wrong batch) in seconds, without a full run.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data, models, train


def _tiny_data():
    d = data.load_mnist(0)
    t = {k: torch.from_numpy(v) for k, v in d.items()}
    t["x_train"] = t["x_train"][:600]
    t["y_train"] = t["y_train"][:600]
    t["x_val"] = t["x_val"][:300]
    t["y_val"] = t["y_val"][:300]
    return t


def _fresh(seed):
    torch.manual_seed(seed)
    return models.SoftmaxRegression()


def test_adversarial_eps0_equals_baseline_exact():
    """eps=0 adversarial training == plain training, bit-identical weights."""
    d = _tiny_data()
    cfg_base = {"lr": 0.1, "max_epochs": 3, "batch_size": 64, "seed": 0, "momentum": 0.0}
    m_base = _fresh(0)
    h_base = train.train(m_base, d, dict(cfg_base))
    m_adv = _fresh(0)
    cfg_adv = dict(cfg_base)
    cfg_adv["adversarial"] = {"alpha": 0.5, "eps": 0.0}
    h_adv = train.train(m_adv, d, cfg_adv)
    # weights must match exactly
    for (k1, v1), (k2, v2) in zip(m_base.state_dict().items(), m_adv.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2), f"weights differ at {k1} under eps=0 (degeneracy broken)"
    # and the loss curves match
    assert h_base["val_err"] == h_adv["val_err"], "val_err curves differ under eps=0"


def test_noise_eps0_equals_baseline_exact():
    """eps=0 noise (rademacher/uniform) == plain training, bit-identical."""
    d = _tiny_data()
    cfg_base = {"lr": 0.1, "max_epochs": 3, "batch_size": 64, "seed": 1, "momentum": 0.0}
    m_base = _fresh(1)
    train.train(m_base, d, dict(cfg_base))
    for ntype in ("rademacher", "uniform"):
        m_n = _fresh(1)
        cfg_n = dict(cfg_base)
        cfg_n["noise"] = {"type": ntype, "eps": 0.0}
        train.train(m_n, d, cfg_n)
        for (k1, v1), (k2, v2) in zip(m_base.state_dict().items(), m_n.state_dict().items()):
            assert torch.equal(v1, v2), f"weights differ under {ntype} eps=0 at {k1}"


def test_l1_coef0_equals_baseline_exact():
    """L1 coef=0 == plain training, bit-identical."""
    d = _tiny_data()
    cfg_base = {"lr": 0.1, "max_epochs": 3, "batch_size": 64, "seed": 2, "momentum": 0.0}
    m_base = _fresh(2)
    train.train(m_base, d, dict(cfg_base))
    m_l1 = _fresh(2)
    cfg_l1 = dict(cfg_base)
    cfg_l1["l1_first_layer"] = 0.0
    train.train(m_l1, d, cfg_l1)
    for (k1, v1), (k2, v2) in zip(m_base.state_dict().items(), m_l1.state_dict().items()):
        assert torch.equal(v1, v2), f"weights differ under L1 coef=0 at {k1}"
