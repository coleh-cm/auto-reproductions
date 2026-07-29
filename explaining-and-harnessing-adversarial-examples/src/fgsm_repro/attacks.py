"""Adversarial-example attacks for FGSM reproduction.

Implements the attacks of Goodfellow, Shlens, Szegedy (2015),
"Explaining and Harnessing Adversarial Examples".  Equation citations below
point at `paper/source/iclr2015.tex`.

- ``fgsm``              Algorithm A  (E1/E2, tex:309, tex:235,239).
- ``fgsm_ensemble``     E1 ensemble variant (grad of mean CE over members).
- ``fooling_sign_step`` Algorithm D sign-step targeted fooling (E9, tex:954).
- ``sample_rubbish``    Rubbish-example sampling, x ~ N(0, I_dim) (tex:905).
"""

from __future__ import annotations

import torch

from .models import Classifier
from .objectives import cross_entropy_cost


def fgsm(model: Classifier, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Fast Gradient Sign Method (Algorithm A; E1, E2).

    eta   = eps * sign(grad_x J(theta, x, y))     # tex:309
    x_tilde = x + eta                            # tex:235; ||eta||_inf < eps (tex:239)

    Returns the DETACHED adversarial batch x_tilde.  No clipping to [0,1]
    (the paper's displayed method does not clip).

    When eps == 0, eta is the zero tensor, so x_tilde == x exactly.
    """
    x = x.detach().clone()
    x.requires_grad_(True)
    J = cross_entropy_cost(model, x, y)
    g = torch.autograd.grad(J, x)[0]
    eta = eps * torch.sign(g)
    x_tilde = (x + eta).detach()
    return x_tilde


def fgsm_ensemble(
    models: list,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Ensemble FGSM (E1): gradient of the mean cross-entropy over members.

    eta = eps * sign(grad_x (1/M) * sum_m J(theta_m, x, y)).
    Returns the detached adversarial batch.  No clipping.
    """
    x = x.detach().clone()
    x.requires_grad_(True)
    total = 0.0
    for m in models:
        total = total + cross_entropy_cost(m, x, y)
    J = total / len(models)
    g = torch.autograd.grad(J, x)[0]
    eta = eps * torch.sign(g)
    x_tilde = (x + eta).detach()
    return x_tilde


def fooling_sign_step(
    model: Classifier,
    x: torch.Tensor,
    cls: int,
    eps: float,
) -> torch.Tensor:
    """Targeted fooling sign step (Algorithm D; E9 sign variant, tex:954).

    Maximize logit-probability of class ``cls`` via a single gradient *sign*
    step on the input:
        x_tilde = x + eps * sign(grad_x p(y = cls | x))
    where p = softmax(model.logits(x)).  Returns detached x_tilde.
    """
    x = x.detach().clone()
    x.requires_grad_(True)
    p = torch.softmax(model.logits(x), dim=-1)
    obj = p[:, cls].sum()
    g = torch.autograd.grad(obj, x)[0]
    x_tilde = (x + eps * torch.sign(g)).detach()
    return x_tilde


def sample_rubbish(n: int, dim: int, gen: torch.Generator) -> torch.Tensor:
    """Draw ``n`` rubbish examples ~ N(0, I_dim) (tex:905).

    Returns a float32 tensor of shape [n, dim].
    """
    return torch.randn(n, dim, generator=gen, dtype=torch.float32)
