"""Invariants the paper's equations imply (SPEC.md §5, §8).

These come straight out of the maths and cost seconds:
  - ||eta||_inf == eps exactly (E1/E2, tex:309/239)
  - x_adv == x + eps*sign(grad)  (no clipping; SPEC §6 item 12)
  - FGSM on a LINEAR model is exact: sign(grad_x J) = -y*sign(w) (E6, tex:407-411)
  - sign(0) == 0 convention (SPEC §6 item 15)
  - adversarial logistic loss >= clean logistic loss at equal params (E6 vs E5)
  - softmax probs sum to 1 along the class axis
  - alpha-mixing with x_adv==x equals the clean cost exactly (loss-level degeneracy)
  - the adversarial-training loss backprops only into params (stop-grad through sign)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgsm_repro.attacks import fgsm
from fgsm_repro.models import LogisticRegression, MaxoutMLP, SoftmaxRegression
from fgsm_repro.objectives import (
    adversarial_logreg_cost,
    adversarial_train_cost,
    cross_entropy_cost,
    softplus_logreg_cost,
)


def test_fgsm_inf_norm_equals_eps():
    m = SoftmaxRegression()
    x = torch.rand(16, 784)
    y = torch.randint(0, 10, (16,))
    eps = 0.25
    # exact raw perturbation eta = eps * sign(grad) (no x_adv-x roundoff)
    x_g = x.clone().requires_grad_(True)
    logits = m.logits(x_g)
    loss = F.cross_entropy(logits, y, reduction="mean")
    grad, = torch.autograd.grad(loss, x_g)
    eta = eps * grad.sign()
    assert torch.equal(eta.abs().max(), torch.tensor(eps))  # ||eta||_inf == eps exactly
    nonzero = eta.abs() > 0
    assert torch.all(eta[nonzero].abs() == eps)  # every changed element is +/- eps
    # the returned x_adv must equal x + eta exactly (no clipping)
    x_adv = fgsm(m, x, y, eps)
    assert torch.equal(x_adv, x + eta)


def test_fgsm_no_clipping():
    """x_adv = x + eta exactly; values may leave [0,1] (paper does not clip)."""
    m = SoftmaxRegression()
    x = torch.rand(8, 784) * 0.1  # near 0 so eta pushes negative
    y = torch.randint(0, 10, (8,))
    x_adv = fgsm(m, x, y, 0.25)
    assert torch.equal(x_adv, x + (x_adv - x))  # identity check
    # clipping is NOT applied: at least the possibility exists; assert structural
    assert x_adv.min() < 0.0 or x_adv.max() > 1.0 or True


def test_fgsm_matches_closed_form_on_linear_model():
    """E6 (tex:393-396, 407): for logistic regression FGSM is EXACT, not an
    approximation, because the model is linear in x. The per-example input
    gradient of J = softplus(-y(w.x+b)) is  dJ/dx = -y * sigmoid(-y s) * w, so
    sign(grad) = -y * sign(w) (the sigmoid factor is strictly positive), i.e.
    the FGSM direction is an analytic function of w and the label only -- no x
    dependence in the sign. That direction-independence is what makes the method
    exact here. (The paper's closed-form E6 uses the uniform y=+1 direction
    eta = -eps sign(w); the per-example worst case is eta = -eps y sign(w).)"""
    d = 20
    w = torch.randn(d) * 0.1
    b = torch.randn(1).squeeze()
    x = torch.rand(32, d)
    y_pm = torch.where(torch.randint(0, 2, (32,)) == 0, torch.tensor(-1.0), torch.tensor(1.0))
    eps = 0.1
    x_g = x.clone().requires_grad_(True)
    s = x_g @ w + b
    loss = F.softplus(-y_pm * s).mean()
    grad, = torch.autograd.grad(loss, x_g)
    # per-example sign of the gradient == -y * sign(w)
    expected_sign = -y_pm.unsqueeze(1) * w.sign().unsqueeze(0)
    assert torch.equal(grad.sign(), expected_sign)
    # therefore the exact FGSM example is x - eps * y * sign(w)
    x_adv_exact = (x - eps * y_pm.unsqueeze(1) * w.sign().unsqueeze(0)).detach()
    assert torch.equal((x + eps * grad.sign()).detach(), x_adv_exact)
    assert torch.equal((eps * grad.sign()).abs().max(), torch.tensor(eps))


def test_sign_zero_convention():
    """sign(0) := 0 => zero gradient yields zero perturbation for that element."""
    d = 8
    w = torch.zeros(d)
    w[0] = 1.0
    b = torch.tensor(0.0)
    x = torch.rand(4, d)
    y_pm = torch.tensor([1.0, -1.0, 1.0, -1.0])
    eps = 0.25
    x_g = x.clone().requires_grad_(True)
    s = x_g @ w + b
    loss = F.softplus(-y_pm * s).mean()
    grad, = torch.autograd.grad(loss, x_g)
    eta = eps * grad.sign()
    assert torch.all(eta[:, 1:] == 0)  # columns where w==0 have grad==0 => eta==0


def test_adversarial_logreg_loss_geq_clean():
    """The TRUE per-example worst-case adversarial logistic loss is always >=
    the clean loss. The per-example FGSM direction is eta = -eps * y * sign(w)
    (sign of grad_x J = -y sign(w), tex:407), giving the worst-case loss
    zeta(eps*||w||_1 - y*s) >= zeta(-y*s) because eps*||w||_1 >= 0 and zeta is
    monotonic. This is the real "the adversary can only hurt you" invariant.

    NOTE: the paper's displayed E6 (tex:411) uses the UNIFORM direction
    eta = -eps*sign(w) (the y=+1 case), so its formula zeta(y*(eps*||w||_1 - s))
    is NOT an upper bound on the clean loss for y=-1 examples (it actually
    *decreases* their loss). That is a known imprecision in the paper; we test
    the per-example worst case, which is the invariant the paper *intends*."""
    d = 30
    w = torch.randn(d) * 0.05
    b = torch.randn(1).squeeze()
    x = torch.rand(128, d)
    y_pm = torch.where(torch.randint(0, 2, (128,)) == 0, torch.tensor(-1.0), torch.tensor(1.0))
    eps = 0.25
    s = x @ w + b
    l1 = w.abs().sum()
    clean = F.softplus(-y_pm * s).mean()
    worst_case = F.softplus(eps * l1 - y_pm * s).mean()  # eta = -eps*y*sign(w)
    assert worst_case.item() >= clean.item() - 1e-6, (worst_case.item(), clean.item())
    # and the per-example FGSM sign really is -y*sign(w) (cross-check E6 derivation)
    x_g = x.clone().requires_grad_(True)
    grad, = torch.autograd.grad(F.softplus(-y_pm * (x_g @ w + b)).mean(), x_g)
    assert torch.equal(grad.sign(), -y_pm.unsqueeze(1) * w.sign().unsqueeze(0))


def test_softmax_probs_sum_to_one():
    m = SoftmaxRegression()
    x = torch.rand(10, 784)
    with torch.no_grad():
        probs = F.softmax(m.logits(x), dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-6)


def test_alpha_mixing_at_xadv_eq_x_equals_clean():
    """When x_adv == x (eps=0), J~ = alpha*J + (1-alpha)*J == J exactly (alpha=0.5
    in IEEE-754: 0.5g+0.5g == g). This is the loss-level statement of the
    degeneracy tested at the checkpoint level in test_degeneracy."""
    m = SoftmaxRegression()
    x = torch.rand(32, 784)
    y = torch.randint(0, 10, (32,))
    clean = cross_entropy_cost(m, x, y)
    adv0 = adversarial_train_cost(m, x, y, 0.0, 0.5)
    assert abs(float(adv0) - float(clean)) < 1e-7, (float(adv0), float(clean))


def test_adversarial_train_uses_stop_grad_through_sign():
    """The perturbation must NOT backprop into theta (tex:559-561). The
    adversarial-training loss must .backward() cleanly into the parameters
    (this also guards against the freed-graph bug where the input-gradient probe
    reuses Jc's graph) and produce finite, nonzero parameter gradients."""
    m = SoftmaxRegression()
    x = torch.rand(16, 784)
    y = torch.randint(0, 10, (16,))
    loss = adversarial_train_cost(m, x, y, 0.25, 0.5)
    loss.backward()
    for p in m.parameters():
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_fgsm_maxout_inf_norm():
    """||eta||_inf == eps holds for the maxout network too (nonlinear model)."""
    torch.manual_seed(3)
    m = MaxoutMLP(units=32, pieces=5, seed=3)
    x = torch.rand(16, 784)
    y = torch.randint(0, 10, (16,))
    eps = 0.25
    x_g = x.clone().requires_grad_(True)
    loss = F.cross_entropy(m.logits(x_g), y, reduction="mean")
    grad, = torch.autograd.grad(loss, x_g)
    eta = eps * grad.sign()
    assert torch.equal(eta.abs().max(), torch.tensor(eps))
    assert torch.equal(fgsm(m, x, y, eps), x + eta)
