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


def test_degeneracy_detects_nonzero_eps():
    """negative (degeneracy_check instrument): with eps>0 the adversarial loop
    is NOT a no-op (x_adv != x), so the trained weights MUST differ from the
    baseline -- the degeneracy assertion (torch.equal) would reject it. Proves
    the check is sensitive to a genuinely non-degenerate run (a loop that leaks
    a perturbation even at its claimed no-op setting), not just a rubber stamp
    that passes everything."""
    d = _tiny_data()
    cfg = {"lr": 0.1, "max_epochs": 3, "batch_size": 64, "seed": 0, "momentum": 0.0}
    m_base = _fresh(0)
    train.train(m_base, d, dict(cfg))
    m_adv = _fresh(0)
    cfg_adv = dict(cfg)
    cfg_adv["adversarial"] = {"alpha": 0.5, "eps": 0.05}  # genuinely perturbs
    train.train(m_adv, d, cfg_adv)
    differs = any(not torch.equal(v1, v2)
                  for (k1, v1), (k2, v2) in zip(m_base.state_dict().items(), m_adv.state_dict().items()))
    assert differs, "eps>0 adversarial training gave bit-identical weights -- degeneracy check cannot detect a leak"


def test_retrain_full_60k_is_from_scratch():
    """positive (retrain protocol invariant, paper tex:505-506): the Phase 2
    retrain-on-full-60k must restart from the INITIAL weights with a FRESH
    optimizer, NOT continue from the Phase-1 state. We assert the DIRECT
    property: the Phase-2 starting state (the model state right after Phase 2
    reloads the init weights, before any Phase-2 step) is BIT-IDENTICAL to the
    captured init state. If Phase 2 continued from the Phase-1 best_state (the
    old bug), phase2_start_state would be the Phase-1 weights, NOT init, and
    this assertion would fail. We also assert the final weights DIFFER from a
    Phase-1-only run (Phase 2 actually retrains — a no-op Phase 2 would leave
    the init weights and trivially "differ" from Phase-1, but then phase2_start
    == final == init, which the first assertion plus a non-trivial best_epoch
    rules out: if Phase 2 ran >=1 step from init it cannot stay at init).

    The earlier guard only asserted final != Phase-1-only, which is true under
    ANY continuation too — a vacuous check that could not detect the defect it
    named. This version directly compares the Phase-2 start to init."""
    d = _tiny_data()
    # give the retrain a full split to use
    d_full = dict(d)
    d_full["x_train_full"] = d_full["x_train"]  # same small set; exercises the path
    d_full["y_train_full"] = d_full["y_train"]
    cfg = {"lr": 0.1, "max_epochs": 4, "batch_size": 64, "seed": 3,
           "momentum": 0.9, "retrain_full_60k": True}
    m_full = _fresh(3)
    h_full = train.train(m_full, d_full, dict(cfg))
    # Phase 1 must select an epoch count
    assert h_full["best_epoch"] >= 1, "Phase 1 must select an epoch count"
    # DIRECT from-scratch check: Phase 2 starts from the init weights
    assert "init_state" in h_full and "phase2_start_state" in h_full, \
        "retrain run did not expose init_state/phase2_start_state for guarding"
    init_state = h_full["init_state"]
    p2_start = h_full["phase2_start_state"]
    for k in init_state:
        assert torch.equal(init_state[k], p2_start[k]), \
            f"Phase 2 did NOT start from init weights at {k} (continuation bug)"
    # Phase 1-only run (no retrain) for comparison
    m_p1 = _fresh(3)
    cfg_p1 = dict(cfg)
    cfg_p1.pop("retrain_full_60k")
    h_p1 = train.train(m_p1, d_full, cfg_p1)
    # final weights must DIFFER from Phase-1-only (Phase 2 actually retrains;
    # a from-scratch retrain for best_epoch>=1 epochs diverges from the
    # Phase-1-continued trajectory, and also from the init it started from)
    differs_from_p1 = any(not torch.equal(v1, v2)
                          for (k1, v1), (k2, v2) in zip(m_full.state_dict().items(),
                                                        m_p1.state_dict().items()))
    assert differs_from_p1, "retrain_full_60k produced bit-identical weights to Phase-1-only"
    # and Phase 2 must have moved OFF the init weights (>=1 step ran)
    final_state = m_full.state_dict()
    moved_off_init = any(not torch.equal(init_state[k], final_state[k]) for k in init_state)
    assert moved_off_init, "Phase 2 never moved off the init weights (no-op retrain?)"


def test_retrain_full_60k_detects_continuation_bug():
    """negative (retrain protocol invariant): simulate the OLD continuation
    bug — Phase 2 resumes from the Phase-1 best_state instead of the init
    weights — and show the guard above rejects it. We do this by checking that
    the Phase-1 best_state is NOT equal to the init state (so a Phase-2 start
    from Phase-1 would fail the `phase2_start == init` assertion). This proves
    the guard has discriminating power: a continuation Phase 2 (phase2_start =
    Phase-1 best_state) cannot pass as from-scratch (phase2_start = init)."""
    d = _tiny_data()
    d_full = dict(d)
    d_full["x_train_full"] = d_full["x_train"]
    d_full["y_train_full"] = d_full["y_train"]
    cfg = {"lr": 0.1, "max_epochs": 4, "batch_size": 64, "seed": 3, "momentum": 0.9}
    m = _fresh(3)
    h = train.train(m, d_full, dict(cfg))
    # Phase-1 best_state (the weights the old continuation bug would resume from)
    best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    # a fresh init for the same seed
    m_init = _fresh(3)
    init_state = {k: v.detach().clone() for k, v in m_init.state_dict().items()}
    # After >=1 epoch of training the best_state differs from the init (otherwise
    # training is a no-op). If they differ, a phase2_start == best_state would
    # fail the guard's `phase2_start == init` check — i.e. the guard catches the
    # continuation bug.
    differs = any(not torch.equal(best_state[k], init_state[k]) for k in init_state)
    assert differs, "Phase-1 best_state == init (training was a no-op); guard cannot be tested"
