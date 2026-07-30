"""Constructed-truth tests — oracles the implementation is checked against.

These encode the "constructed truth" of SPEC.md's `## Constructed truth` section:
cheap, exact references the paper hands us for free (or that follow from its
equations), so a wrong implementation is caught in seconds without a full
paper-scale run. Each test is an oracle a reader can verify without trusting the
measured numbers.

Categories exercised here:
  * DEGENERACY — the method at its no-op (eps=0) reproduces the baseline
    exactly (covered in test_degeneracy.py; referenced, not duplicated).
  * BRUTE FORCE at toy scale vs a closed form claiming a worst case — E6
    (tex:410-412) is the closed-form max-norm worst-case adversarial logistic
    loss; random perturbations within the box must NOT beat it.
  * SAME QUANTITY TWO WAYS — E6 closed form vs the empirical FGSM loss
    (also in test_instruments.py; here we add the brute-force envelope).
  * PLANTING A KNOWN STRUCTURE — plant a logistic regression with known w and
    require the FGSM perturbation to be the analytically-known direction
    eta = -eps * y * sign(w) (tex:407-411).
  * LIMITING CASES — eps=0 is the identity (test_fgsm_eps0_is_identity); large
    eps drives a fitted linear model to ~100% error.
  * NAIVE vs FAST — the per-example analytical FGSM direction equals the
    autograd-computed sign of the input gradient.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT / "src"))

from fgsm_repro.attacks import fgsm_logreg  # noqa: E402
from fgsm_repro.objectives import (  # noqa: E402
    adversarial_logreg_cost,
    softplus_logreg_cost,
)


def test_brute_force_e6_is_the_maxnorm_worst_case():
    """E6 (closed form) is the worst-case loss in the L-inf box of radius eps.

    Draw many random perturbations eta with ||eta||_inf <= eps; NONE may yield a
    higher softplus loss than E6 (the claimed worst case). This is a brute-force
    oracle for a closed form the paper asserts is the maximum (tex:407-412).
    """
    torch.manual_seed(0)
    w = 0.05 * torch.randn(784)
    b = torch.tensor(0.0)
    x = torch.rand(512, 784)
    y_pm = torch.ones(512)
    eps = 0.25
    e6 = adversarial_logreg_cost(w, b, x, y_pm, eps).item()
    worst = e6
    for _ in range(200):
        # random eta with ||eta||_inf <= eps (uniform in [-eps, eps] per coord)
        eta = (2 * torch.rand_like(x) - 1) * eps
        loss = softplus_logreg_cost(w, b, x + eta, y_pm).item()
        worst = max(worst, loss)
    # E6 is the worst case up to float slack from the random search not hitting
    # the exact sign(w) vertex; assert E6 is at least as large as the best random
    # point found (the random search cannot exceed the true worst case).
    assert e6 >= worst - 1e-3, f"E6={e6} < brute-force worst={worst}"


def test_plant_known_linear_structure_fgsm_direction():
    """Plant a logistic regression with known w; require the FGSM perturbation
    to be the analytically-known worst-case direction eta = -eps*y*sign(w)
    (tex:407-411: 'the sign of the gradient is just -sign(w)')."""
    torch.manual_seed(0)
    w = 0.05 * torch.randn(784)
    b = torch.tensor(0.0)
    x = torch.rand(16, 784)
    y_pm = torch.tensor([1.0, -1.0] * 8)
    eps = 0.25
    x_tilde = fgsm_logreg(w, b, x, y_pm, eps)
    eta = x_tilde - x
    expected = -eps * y_pm.unsqueeze(1) * torch.sign(w).unsqueeze(0)
    assert torch.allclose(eta, expected)
    # and ||eta||_inf == eps exactly (E2, tex:239); wherever w != 0 the step is eps
    assert float(eta.abs().max()) == pytest.approx(eps)


def test_limiting_case_large_eps_drives_linear_model_to_all_wrong():
    """Limiting case: as eps grows, the FGSM attack on a fitted linear model
    drives the error rate toward 1.0 (the paper's 'broad subspaces' claim,
    tex:631-639). A correct attack is monotone-ish and saturates near 1.0."""
    from fgsm_repro.models import SoftmaxRegression
    from fgsm_repro.eval import eval_fgsm
    torch.manual_seed(0)
    m = SoftmaxRegression(784, 10)
    x = torch.rand(512, 784)
    y = torch.randint(0, 10, (512,))
    e0 = eval_fgsm(m, x, y, 0.0).error_rate
    e_big = eval_fgsm(m, x, y, 5.0).error_rate
    assert e_big > e0
    assert e_big > 0.9  # large eps -> almost everything wrong


def test_naive_per_example_fgsm_equals_autograd_sign():
    """Naive (per-element analytic) sign of the input gradient for the logistic
    cost equals the autograd-computed sign used by the fast FGSM path. For
    J = mean softplus(-y(w.x+b)), dJ/dx = -y * sigmoid(-y s) * w, whose sign is
    -y * sign(w) (the sigmoid factor is strictly positive) — independent of x.
    """
    torch.manual_seed(0)
    w = 0.05 * torch.randn(784)
    b = torch.tensor(0.0)
    x = torch.rand(32, 784, requires_grad=True)
    y_pm = torch.ones(32)
    # autograd sign of the INPUT gradient (the FGSM direction)
    s = x @ w + b
    J = F.softplus(-y_pm * s).mean()
    g = torch.autograd.grad(J, x)[0]
    autograd_sign = torch.sign(g)
    # naive analytic dJ/dx = -y * sigmoid(-y s) * w; the sigmoid factor is
    # strictly positive, so sign(dJ/dx) = -y * sign(w) — independent of x.
    naive_sign = -y_pm.unsqueeze(1) * torch.sign(w).unsqueeze(0)
    assert torch.equal(autograd_sign, naive_sign)
