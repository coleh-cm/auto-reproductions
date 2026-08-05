"""Invariants the paper's equations imply (research-code skill).

FGSM / adversarial-training invariants (paper/source/iclr2015.tex):
  - eq 309:  eta = eps*sign(grad_x J);  ||eta||_inf == eps exactly
  - sign(0) := 0 (SPEC §4.22)
  - no clipping (SPEC §4.9): x_adv can exceed [0,1]
  - eq 407 (logistic): sign(grad_x J) = -sign(w); w.sign(w) = ||w||_1
  - eq 411 (adversarial logistic): the closed form equals the FGSM form
  - softmax prob rows sum to 1; RBF prob rows need NOT sum to 1
  - a non-negative loss never goes negative
  - eps trace direction fixed at eps=0 => logits exactly piecewise-linear in eps
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data, models, attack, eval as ev
import train


def test_fgsm_inf_norm_equals_eps():
    m = models.SoftmaxRegression()
    x = torch.rand(32, 784)
    y = torch.randint(0, 10, (32,))
    for eps in (0.1, 0.25, 0.5):
        x_adv = attack.fgsm(m, x, y, eps)
        eta = (x_adv - x).abs()
        assert torch.allclose(eta.max(), torch.tensor(eps)), f"||eta||_inf != eps at eps={eps}"
        # every perturbed pixel is exactly eps (sign is +/-1, never fractional)
        nz = eta[eta > 0]
        assert torch.allclose(nz, torch.full_like(nz, eps)), "perturbation magnitudes not exactly eps"


def test_fgsm_no_clipping():
    """x_adv may exceed [0,1]; SPEC §4.9 mandates no clipping."""
    m = models.SoftmaxRegression()
    torch.manual_seed(0)
    m.linear.weight.data = torch.sign(torch.randn_like(m.linear.weight))  # large-ish grads
    x = torch.ones(8, 784)  # at the boundary
    y = torch.zeros(8, dtype=torch.long)
    x_adv = attack.fgsm(m, x, y, 0.25)
    assert (x_adv > 1.0).any() or (x_adv < 0.0).any(), "FGSM clipped x_adv (SPEC §4.9 violated)"


def test_sign_zero_is_zero():
    """sign(0) := 0 (SPEC §4.22). torch.sign already does this; assert it holds."""
    assert torch.sign(torch.tensor(0.0)).item() == 0


def test_softmax_prob_rows_sum_to_one():
    m = models.SoftmaxRegression()
    x = torch.randn(16, 784)
    p = m.prob(x)
    assert torch.allclose(p.sum(-1), torch.ones(16), atol=1e-5)


def test_rbf_prob_rows_need_not_sum_to_one():
    """SPEC §4.4 / tex:595: RBF per-class probs are independent; rows need NOT
    sum to 1 (this is what makes 'confidence on mistakes 1.2%' possible, which
    is impossible under a 10-class softmax floor of 10%)."""
    m = models.RBFNet()
    x = torch.randn(16, 784)
    p = m.prob(x)
    assert p.shape == (16, 10)
    # at least one row should not sum to 1 (the whole point of independent RBF units)
    sums = p.sum(-1)
    assert not torch.allclose(sums, torch.ones(16), atol=1e-3), "RBF rows sum to 1 — should be independent"


def test_logreg_sign_grad_equals_neg_sign_w():
    """tex:407: sign(grad_x J) = -sign(w) for logistic regression (modulo the y
    factor). On a batch with y=+1 the input gradient sign is -sign(w)."""
    torch.manual_seed(0)
    m = models.LogisticRegression3v7()
    x = torch.randn(8, 784)
    y = torch.ones(8, dtype=torch.long)  # y = +1
    x_req = x.clone().requires_grad_(True)
    loss = m.loss(x_req, y)
    g = torch.autograd.grad(loss, x_req)[0]
    sign_g = torch.sign(g)
    w = m.linear.weight.detach().squeeze(0)
    neg_sign_w = -torch.sign(w)
    # compare on pixels where w != 0 (everywhere here)
    assert torch.equal(sign_g[0], neg_sign_w), "sign(grad) != -sign(w) for logreg y=+1"


def test_logreg_w_dot_sign_w_equals_l1():
    """tex:407: w^T sign(w) = ||w||_1."""
    torch.manual_seed(0)
    m = models.LogisticRegression3v7()
    w = m.linear.weight.detach().squeeze(0)
    assert torch.allclose(w @ torch.sign(w), w.abs().sum(), atol=1e-5)


def test_logreg_fgsm_equals_analytic_form():
    """c07 invariant: for logistic regression FGSM is EXACT, so the analytic
    closed form equals the loss under the REAL gradient-based per-example FGSM.

    sign(grad_x J) = -y*sign(w) for logreg (tex:407 drops the y factor — a sign
    slip; the true sign is -y*sign(w)), so attack.fgsm gives x_adv_i = x_i -
    eps*y_i*sign(w) and the loss is zeta(eps*||w||_1 - y_i*(w.x_i+b)). The
    CORRECT worst-case closed form is zeta(eps*||w||_1 - y*(w.x+b)). We check
    the identity with MIXED labels (y in {-1,+1}) using the real attack.fgsm,
    so a sign bug in attack.fgsm OR the closed form breaks the invariant.

    The paper's tex:411 form zeta(y*(eps*||w||_1 - w.x - b)) equals this only
    for y=+1; for y=-1 it gives zeta(m - eps*||w||_1) which DECREASES the loss
    (not the worst case) — see test_logreg_paper_tex411_form_fails_for_yneg."""
    torch.manual_seed(0)
    m = models.LogisticRegression3v7()
    # train a little so w is non-trivial (sign(w) meaningful, mixed margins)
    x_tr = torch.randn(256, 784)
    y_tr = torch.where(torch.rand(256) > 0.5, 1, -1).long()
    train.train(m, {"x_train": x_tr, "y_train": y_tr, "x_val": x_tr[:64],
                    "y_val": y_tr[:64]},
                {"lr": 0.1, "max_epochs": 3, "batch_size": 64, "seed": 0,
                 "momentum": 0.0})
    x = torch.randn(64, 784)
    y = torch.where(torch.rand(64) > 0.5, 1, -1).long()  # MIXED labels
    eps = 0.25
    w = m.linear.weight.detach().squeeze(0)
    b = m.linear.bias.detach()
    w1 = w.abs().sum()
    yf = y.float()
    # tex:407: sign(w) @ w == ||w||_1
    assert torch.allclose(torch.sign(w) @ w, w1, atol=1e-5)
    # REAL gradient-based per-example FGSM
    x_adv = attack.fgsm(m, x, y, eps)
    fgsm_loss = float(m.loss(x_adv, y).item())
    # CORRECT worst-case closed form
    margin = eps * w1 - yf * (x @ w + b)
    analytic = float(torch.nn.functional.softplus(margin).mean().item())
    assert abs(fgsm_loss - analytic) < 1e-4, \
        f"FGSM loss {fgsm_loss} != correct analytic {analytic} (diff {abs(fgsm_loss-analytic):.2e})"


def test_logreg_paper_tex411_form_fails_for_yneg():
    """negative / discriminating (c07): the paper's OWN tex:411 closed form
    zeta(y*(eps*||w||_1 - w.x - b)) does NOT match the real per-example FGSM
    loss for mixed labels — it has a sign slip for y=-1 (it decreases the loss
    instead of maximizing it). Asserting the paper's form would fail, which is
    WHY c07 checks the corrected form zeta(eps*||w||_1 - y*(w.x+b)). This proves
    the invariant is not a rubber stamp: it would reject the paper's own
    (slipped) closed form on mixed labels."""
    torch.manual_seed(0)
    m = models.LogisticRegression3v7()
    x_tr = torch.randn(256, 784)
    y_tr = torch.where(torch.rand(256) > 0.5, 1, -1).long()
    train.train(m, {"x_train": x_tr, "y_train": y_tr, "x_val": x_tr[:64],
                    "y_val": y_tr[:64]},
                {"lr": 0.1, "max_epochs": 3, "batch_size": 64, "seed": 0,
                 "momentum": 0.0})
    x = torch.randn(64, 784)
    y = torch.where(torch.rand(64) > 0.5, 1, -1).long()  # MIXED labels
    eps = 0.25
    w = m.linear.weight.detach().squeeze(0)
    b = m.linear.bias.detach()
    w1 = w.abs().sum()
    yf = y.float()
    x_adv = attack.fgsm(m, x, y, eps)
    fgsm_loss = float(m.loss(x_adv, y).item())
    # paper's tex:411 form (the slipped one)
    paper_margin = yf * (eps * w1 - (x @ w + b))
    paper_analytic = float(torch.nn.functional.softplus(paper_margin).mean().item())
    assert abs(fgsm_loss - paper_analytic) > 1e-2, \
        (f"paper tex:411 form matched the real FGSM loss on mixed labels "
         f"({fgsm_loss} vs {paper_analytic}) — the sign slip did not bite; "
         f"invariant lacks discriminating power")


def test_logreg_analytic_wrong_sign_differs():
    """negative (logreg_analytic_equivalence instrument): a WRONG-sign attack
    (perturbing x in the +y*sign(w) direction, which INCREASES the margin
    instead of decreasing it) does NOT match the correct closed form. Proves
    the equivalence check rejects a sign-flipped attack rather than rubber-
    stamping any sign."""
    torch.manual_seed(0)
    m = models.LogisticRegression3v7()
    x = torch.randn(64, 784)
    y = torch.where(torch.rand(64) > 0.5, 1, -1).long()
    eps = 0.25
    w = m.linear.weight.detach().squeeze(0)
    b = m.linear.bias.detach()
    w1 = w.abs().sum()
    yf = y.float()
    # WRONG-sign attack: x + eps*y*sign(w) (increases margin for the true label)
    x_adv_wrong = x + eps * (yf.unsqueeze(1) * torch.sign(w).unsqueeze(0))
    wrong_loss = float(m.loss(x_adv_wrong, y).item())
    # CORRECT closed form
    correct = float(torch.nn.functional.softplus(
        eps * w1 - yf * (x @ w + b)).mean().item())
    assert abs(wrong_loss - correct) > 1e-2, \
        f"wrong-sign attack matched the correct closed form ({wrong_loss} vs {correct}) — rubber stamp"


def test_loss_non_negative():
    """A cross-entropy / softplus loss is non-negative."""
    cases = [
        ("SoftmaxRegression", models.SoftmaxRegression(), 10),
        ("LogisticRegression3v7", models.LogisticRegression3v7(), None),
        ("MaxoutMLP", models.MaxoutMLP(32, 2, 3, {"input": 0.0, "hidden": 0.0}), 10),
        ("RBFNet", models.RBFNet(), 10),
    ]
    x = torch.randn(16, 784)
    for name, m, K in cases:
        if K is None:
            y = torch.where(torch.rand(16) > 0.5, torch.ones(16, dtype=torch.long),
                            -torch.ones(16, dtype=torch.long))
        else:
            y = torch.randint(0, K, (16,))
        loss = m.loss(x, y)
        assert loss.item() >= -1e-6, f"{name} loss negative: {loss.item()}"


def test_eps_trace_piecewise_linear_in_eps():
    """SPEC §4.15 / tex:762-770: with the FGSM direction fixed at eps=0, the
    logits are exactly (piecewise) linear in eps for a LINEAR model (softmax
    regression has no piecewise breaks, so fully linear)."""
    torch.manual_seed(0)
    m = models.SoftmaxRegression()
    x = torch.rand(1, 784)
    y = torch.tensor([4])
    eps_grid = torch.linspace(-10, 10, 21)
    logits = attack.fgsm_logits_trace(m, x[0], int(y[0]), eps_grid).detach().numpy()
    # for a linear model logits = W(x + eps*sign(g)) + b = (W x + b) + eps*(W sign(g))
    # => linear in eps. Check each class column is linear (3-point collinear).
    for k in range(10):
        col = logits[:, k]
        # slope between consecutive points must be constant
        diffs = np.diff(col)
        assert np.allclose(diffs, diffs[0], atol=1e-4), f"class {k} logit not linear in eps"


def test_adversarial_training_reduces_adv_err():
    """Ablation sanity (research-code skill): adversarial training with eps>0
    must REDUCE the adversarial validation error vs no adversarial training.
    Catches a mis-wired adversarial loop (wrong-sign perturbation, no
    perturbation, gradients not flowing through the adversarial half)."""
    d = data.load_mnist(0)
    t = {k: torch.from_numpy(v) for k, v in d.items()}
    t["x_train"] = t["x_train"][:1500]; t["y_train"] = t["y_train"][:1500]
    t["x_val"] = t["x_val"][:500]; t["y_val"] = t["y_val"][:500]
    # baseline
    mb = models.SoftmaxRegression()
    hb = train.train(mb, t, {"lr": 0.3, "max_epochs": 4, "batch_size": 128,
                            "seed": 0, "momentum": 0.9})
    base_adv = hb["adv_val_err"][-1]
    # adversarial
    ma = models.SoftmaxRegression()
    ha = train.train(ma, t, {"lr": 0.3, "max_epochs": 4, "batch_size": 128,
                            "seed": 0, "momentum": 0.9,
                            "adversarial": {"alpha": 0.5, "eps": 0.25}})
    adv_adv = ha["adv_val_err"][-1]
    assert adv_adv < base_adv - 1.0, \
        f"adversarial training did not reduce adv_err: {adv_adv} >= {base_adv}"


def test_empty_input_raises():
    """No success path returns OK on empty input (loud-failure contract)."""
    m = models.SoftmaxRegression()
    for fn in (lambda: m.logits(torch.empty(0, 784)),
               lambda: m.predict(torch.empty(0, 784)),
               lambda: ev.error(m, torch.empty(0, 784), torch.empty(0, dtype=torch.long)),
               lambda: attack.fgsm(m, torch.empty(0, 784), torch.empty(0, dtype=torch.long), 0.25)):
        try:
            fn()
            assert False, "empty input did not raise"
        except (ValueError, RuntimeError):
            pass


def test_fgsm_rejects_clipping_and_scaling():
    """negative (fgsm_inf_norm instrument): a clipping implementation (clamps
    x_adv to [0,1]) keeps x_adv in range and so would PASS a naive range check
    but FAIL test_fgsm_no_clipping; a scaled (non-sign) perturbation gives
    ||eta||_inf != eps and fails test_fgsm_inf_norm_equals_eps. Proves the two
    fgsm invariants catch the two named defect classes, not just the happy path."""
    m = models.SoftmaxRegression()
    torch.manual_seed(0)
    m.linear.weight.data = torch.sign(torch.randn_like(m.linear.weight))
    x = torch.ones(8, 784)  # at the boundary -> clipping would bite
    y = torch.zeros(8, dtype=torch.long)
    eps = 0.25
    x_in = x.clone().detach().requires_grad_(True)
    g = torch.autograd.grad(m.loss(x_in, y), x_in)[0]
    x_in.requires_grad_(False)

    # (a) clipping defect: x_adv clamped to [0,1] stays in range -- the no-clipping
    # assertion (that SOME pixel escapes [0,1] at the boundary) would reject it.
    x_adv_clip = (x + eps * torch.sign(g)).clamp(0.0, 1.0).detach()
    assert not ((x_adv_clip > 1.0).any() or (x_adv_clip < 0.0).any()), \
        "fixture: clipping should keep x_adv in range"
    # the no-clipping assertion, applied to the clipped output, MUST fail:
    try:
        assert (x_adv_clip > 1.0).any() or (x_adv_clip < 0.0).any()
        assert False, "no-clipping check accepted a clipped x_adv (rubber stamp)"
    except AssertionError:
        pass  # expected: the check rejects the clipped output

    # (b) scaling defect: a 0.5*eps*sign(g) perturbation has ||eta||_inf == 0.5*eps
    #     != eps, so the inf-norm-equals-eps assertion rejects it.
    x_adv_scaled = (x + 0.5 * eps * torch.sign(g)).detach()
    eta = (x_adv_scaled - x).abs()
    assert not torch.allclose(eta.max(), torch.tensor(eps)), \
        "inf-norm check accepted a scaled (0.5*eps) perturbation (rubber stamp)"


def test_rubbish_eval_softmax_in_range_and_shares_sum_to_100():
    """positive (rubbish_any_prob_threshold instrument): eval.rubbish_eval on a
    confident softmax returns rubbish_err in [0,100]; when mistakes exist the
    class_shares keys are exactly '0'..'9' and sum to 100."""
    torch.manual_seed(0)
    m = models.SoftmaxRegression()
    m.linear.weight.data = torch.randn_like(m.linear.weight) * 5.0  # confident on rubbish
    r = ev.rubbish_eval(m, 784, 256, seed=0)
    assert 0.0 <= r["rubbish_err"] <= 100.0, f"rubbish_err out of range: {r['rubbish_err']}"
    assert set(r["rubbish_class_shares"].keys()) == {str(k) for k in range(10)}
    if r["rubbish_err"] > 0.0:
        total = sum(r["rubbish_class_shares"].values())
        assert abs(total - 100.0) < 1e-3, f"class_shares sum {total} != 100"


def test_rubbish_eval_rbf_near_zero():
    """positive (rubbish_any_prob_threshold instrument): an RBF network far from
    the data assigns every class prob well below 0.5 on Gaussian rubbish, so
    rubbish_err ~ 0 (paper: 'RBF network ... error rate of 0%'). This is the
    oracle that proves the 0.5 'any class prob > 0.5' threshold is correct --
    a softmax would score ~100% here; the RBF scores ~0%. (Shift of 0.5 keeps
    the exp-quadratic probs positive but tiny -- no float underflow -- so the
    threshold, not arithmetic, is what's exercised.)"""
    torch.manual_seed(0)
    m = models.RBFNet()
    with torch.no_grad():
        m.mu.add_(0.5)  # means just far enough that max prob ~ 1e-5 (< 0.5, > 0)
    r = ev.rubbish_eval(m, 784, 256, seed=0)
    assert 0.0 <= r["rubbish_err"] <= 5.0, f"RBF rubbish_err not ~0: {r['rubbish_err']}"
    assert set(r["rubbish_class_shares"].keys()) == {str(k) for k in range(10)}


def test_rubbish_rejects_wrong_threshold():
    """negative (rubbish_any_prob_threshold instrument): a buggy threshold
    ('any prob > 0.0', which is always true for the exp-quadratic RBF whose
    probs are positive) would report ~100% rubbish_err for the robust RBF that
    the correct eval ('any prob > 0.5') scores ~0%. Proves the 0.5 threshold is
    load-bearing and the instrument rejects the always-true-threshold bug.
    (Shift of 0.5 keeps probs positive -- no underflow to exactly 0.0, which
    would make even the buggy >0.0 threshold report 0% and hide the defect.)"""
    torch.manual_seed(0)
    m = models.RBFNet()
    with torch.no_grad():
        m.mu.add_(0.5)
    x = torch.from_numpy(data.rubbish(784, 256, seed=0))
    with torch.no_grad():
        prob = m.prob(x)
        conf = prob.max(dim=-1).values
    correct_err = float((conf > 0.5).float().mean().item()) * 100.0
    buggy_err = float((conf > 0.0).float().mean().item()) * 100.0  # the bug
    assert correct_err <= 5.0, f"correct eval not ~0 on robust RBF: {correct_err}"
    assert buggy_err >= 95.0, f"buggy eval not ~100 on robust RBF: {buggy_err}"


def test_rbf_nu_trainable_receives_gradient():
    """positive (RBF autograd invariant): when nu_trainable=True the nu
    Parameter MUST receive a gradient from loss.backward(). An earlier
    implementation wrapped self.nu in float(...) inside _quad, which detached
    it from the autograd graph and silently zeroed the nu gradient. The default
    (nu_trainable=False, a buffer) is unaffected, but the trainable path must
    not be silently broken."""
    torch.manual_seed(0)
    m = models.RBFNet(nu=0.01, nu_trainable=True)
    assert m.nu.requires_grad, "nu_trainable=True must make nu a learnable Parameter"
    x = torch.from_numpy(data.rubbish(784, 32, seed=0))
    y = torch.zeros(32, dtype=torch.long)
    loss = m.loss(x, y)
    loss.backward()
    assert m.nu.grad is not None, "nu received no gradient (float(nu) detach bug?)"
    assert torch.isfinite(m.nu.grad), "nu gradient is not finite"


def test_rbf_nu_buffer_default_is_not_learnable():
    """negative: by default nu is a fixed buffer (not a Parameter), so it has
    no .grad attribute and requires_grad=False. This is the paper-silent default
    (SPEC 4.4); the trainable path above is opt-in."""
    torch.manual_seed(0)
    m = models.RBFNet(nu=0.01)  # nu_trainable defaults False
    assert not isinstance(m.nu, torch.nn.Parameter), "default nu must be a buffer"
    assert not m.nu.requires_grad, "default nu must not require grad"


def test_build_x_adv_uses_eval_mode_no_dropout():
    """positive (adversarial-training FGSM mode invariant): the in-training FGSM
    perturbation is computed with the model in EVAL mode (dropout OFF), so the
    perturbation is non-zero on essentially all pixels with a non-zero input
    gradient. Under the old bug (train mode, active dropout mask) the input
    gradient is exactly 0 on masked pixels. With input dropout p=0.99 the
    train-mode gradient is ~0 on ~99% of pixels while the eval-mode perturbation
    is non-zero on most — a >50pp gap. Guards the review finding that the
    adversarial-training attack ran through an active dropout mask."""
    torch.manual_seed(0)
    m = models.MaxoutMLP(32, 2, 3, {"input": 0.99, "hidden": 0.0})
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    m.train()
    x_adv = train._build_x_adv(m, x, y, 0.25)
    assert m.training, "_build_x_adv did not restore the model's train mode"
    frac_eval = ((x_adv - x).abs() > 0).float().mean().item()
    # contrast: a TRAIN-mode FGSM (the bug) zeros the input gradient on masked pixels
    m.train()
    x_req = x.detach().clone().requires_grad_(True)
    g_train = torch.autograd.grad(m.loss(x_req, y), x_req)[0]
    frac_train = (g_train.abs() > 0).float().mean().item()
    assert frac_eval > 0.5, f"eval-mode perturbation mostly zero ({frac_eval:.3f})"
    assert frac_eval > frac_train + 0.5, \
        f"eval-mode frac {frac_eval:.3f} not >> train-mode {frac_train:.3f}; dropout not off in _build_x_adv"


def test_rbf_has_no_log_temp_and_no_loss_clamp():
    """positive (RBF equation-faithfulness invariant, tex:595): the printed
    RBF equation has NO per-class temperature and NO loss clamp. An earlier
    version carried an inert `log_temp` Parameter and a `nll.clamp(min=-50)`
    (dead, since RBF logits <= 0 => NLL >= 0). Both are removed; this test
    guards against their reintroduction and asserts NLL is non-negative by
    construction (so no clamp is needed)."""
    m = models.RBFNet()
    assert not hasattr(m, "log_temp"), "RBFNet should not carry a log_temp Parameter"
    torch.manual_seed(0)
    x = torch.randn(16, 784)
    y = torch.randint(0, 10, (16,))
    with torch.no_grad():
        logits = m.logits(x)
    assert (logits <= 1e-6).all(), f"RBF logits should be <= 0, got max {logits.max().item()}"
    loss = m.loss(x, y).item()
    assert loss >= -1e-6, f"RBF NLL negative ({loss}); clamp would be dead but NLL should be >= 0"


def test_conv_maxout_has_no_post_relu():
    """positive (CIFAR arch-faithfulness invariant): maxout is itself the
    nonlinearity (Goodfellow et al. 2013c); NO ReLU CALL is applied after a
    conv-maxout stage (an earlier version inserted F.relu, an extra nonlinearity
    the paper never describes). Guards against reintroduction. (Checks for a
    relu() call, not the word 'relu' in comments.)"""
    import inspect
    import re
    src = inspect.getsource(models.ConvMaxoutCIFAR.forward_features)
    # strip comments so a docstring/comment mentioning 'relu' does not trigger
    code_lines = [ln.split("#")[0] for ln in src.splitlines()]
    code = "\n".join(code_lines)
    assert not re.search(r"\brelu\s*\(", code), \
        "ConvMaxoutCIFAR.forward_features calls relu() (should be pure maxout)"
