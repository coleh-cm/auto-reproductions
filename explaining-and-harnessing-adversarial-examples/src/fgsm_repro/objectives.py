"""Objective functions for FGSM reproduction.

Implements the cost functions referenced in Goodfellow, Shlens, Szegedy (2015),
"Explaining and Harnessing Adversarial Examples".  Equation citations below
point at `paper/source/iclr2015.tex`.

- ``cross_entropy_cost``        J for softmax-output models (mean NLL).
- ``softplus_logreg_cost``      E5 (tex:401-404).
- ``adversarial_logreg_cost``   E6 closed-form worst-case (tex:410-412).
- ``adversarial_train_cost``    E7 with stop-grad inside the sign (tex:486-488,
                                tex:559-561).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import Classifier


def cross_entropy_cost(model: Classifier, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood (cross-entropy) for softmax models.

    J = (1/B) * sum_b -log p(y_b | x_b), with p = softmax(logits).
    Returns a graph-attached float32 scalar.
    """
    logits = model.logits(x)
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, y.view(-1, 1))
    return nll.mean()


def softplus_logreg_cost(
    w: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    y_pm: torch.Tensor,
) -> torch.Tensor:
    """Logistic-regression softplus loss (E5, tex:401-404).

    J = E zeta(-y (w^T x + b)),  zeta(z) = log(1 + exp(z)) = softplus(z).

    ``y_pm`` is float32 with values in {-1, +1}.
    """
    s = x @ w + b
    z = -y_pm * s
    return F.softplus(z).mean()


def adversarial_logreg_cost(
    w: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    y_pm: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Closed-form adversarial logistic-regression loss (E6, tex:410-412).

    J_adv = E zeta(y (eps * ||w||_1 - w^T x - b)).

    Closed-form worst case under an L-infinity box of radius ``eps``: the
    adversary picks eta = eps * sign(-w) so that w^T eta = -eps * ||w||_1,
    which is *subtracted* from w^T x + b.  No autograd through eta.
    """
    s = x @ w + b
    val = y_pm * (eps * w.abs().sum() - s)
    return F.softplus(val).mean()


def adversarial_train_cost(
    model: Classifier,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Adversarial training cost (E7, tex:486-488) with stop-grad (tex:559-561).

    J_tilde = alpha * J(theta, x, y) + (1 - alpha) * J(theta, x_tilde, y),
    where x_tilde = x + eps * sign(grad_x J(theta, x, y)).

    STOP-GRAD: the input gradient used to form the sign is detached, and
    x_tilde is detached, so the adversary's reaction is not anticipated
    (derivative of sign is zero or undefined everywhere).

    Implementation note: the forward used to compute the input gradient is a
    SEPARATE graph from the outer loss. ``autograd.grad(Jc, x)`` (default
    ``retain_graph=False``) frees Jc's graph, so we do NOT reuse ``Jc`` in the
    returned loss -- we recompute a fresh ``loss_clean`` for the outer
    ``.backward()``. Reusing Jc would raise "Trying to backward through the
    graph a second time" on the train loop's ``loss.backward()``.

    Degeneracy: when eps == 0, x_tilde == x EXACTLY (x + 0*sign(g) == x), so
    loss_clean == loss_adv and the returned value equals the clean cost exactly
    (with alpha=0.5, 0.5*Jc + 0.5*Jc == Jc in IEEE-754). No clipping, no eps fudge.
    """
    x = x.detach().clone()
    x.requires_grad_(True)
    Jc = cross_entropy_cost(model, x, y)
    g = torch.autograd.grad(Jc, x)[0].detach()          # frees Jc's graph
    x_tilde = (x.detach() + eps * torch.sign(g)).detach()
    x.requires_grad_(False)
    # fresh graphs for the outer backward (not shared with the grad probe)
    loss_clean = cross_entropy_cost(model, x, y)
    loss_adv = cross_entropy_cost(model, x_tilde, y)
    return alpha * loss_clean + (1.0 - alpha) * loss_adv
