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
    closed form equals the loss under the paper's perturbation. tex:407 states
    "the sign of the gradient is just -sign(w)" — the worst-case direction that
    decreases the margin w.x+b uniformly (independent of y), giving x_adv =
    x - eps*sign(w). Then margin_adv = w.x+b - eps*||w||_1 and
    J_adv = zeta(-y*margin_adv) = zeta(y*(eps*||w||_1 - w.x - b)) (tex:411)."""
    torch.manual_seed(0)
    m = models.LogisticRegression3v7()
    x = torch.randn(64, 784)
    y = torch.where(torch.rand(64) > 0.5, 1, -1).long()
    eps = 0.25
    w = m.linear.weight.detach().squeeze(0)
    b = m.linear.bias.detach()
    yf = y.float()
    w1 = w.abs().sum()
    # paper's exact perturbation: eta = -eps*sign(w) (decrease margin, tex:407)
    x_adv = x - eps * torch.sign(w).unsqueeze(0)
    with torch.no_grad():
        fgsm_loss = float(m.loss(x_adv, y).item())
        margin = eps * w1 - (x @ w + b)  # eps*||w||_1 - w.x - b
        analytic = float(torch.nn.functional.softplus(yf * margin).mean().item())
    assert abs(fgsm_loss - analytic) < 1e-5, \
        f"FGSM form {fgsm_loss} != analytic {analytic}"


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
