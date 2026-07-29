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

    SURROGATE-MODE CHOICE (SPEC.md section 6 item 23; addresses the
    dropout-state-during-surrogate-generation gap): the paper says only that
    the adversarial examples should "resist the current version of the model"
    (tex:490-491) and never states the dropout state used while generating
    them. The evaluation-time attacker (eval.py: eval_fgsm / fgsm) runs with
    the model in ``eval()`` mode (dropout OFF, deterministic). To make the
    training-time adversary MATCH the deterministic evaluation attacker, we
    compute the input-gradient probe with the model temporarily in ``eval()``
    mode (no dropout mask drawn, deterministic), then restore the caller's
    mode and compute the two loss terms in the original (typically ``train()``)
    mode so the regular training-time dropout still regularises the loss.
    This keeps the surrogate's direction identical to the eval-time attacker's
    and avoids consuming dropout RNG inside the probe (which previously caused
    the probe, loss_clean, and loss_adv to each use a *different* random mask).

    Implementation note: the forward used to compute the input gradient is a
    SEPARATE graph from the outer loss. ``autograd.grad(Jc, x)`` (default
    ``retain_graph=False``) frees Jc's graph, so we do NOT reuse ``Jc`` in the
    returned loss -- we recompute a fresh ``loss_clean`` for the outer
    ``.backward()``. Reusing Jc would raise "Trying to backward through the
    graph a second time" on the train loop's ``loss.backward()``.

    Degeneracy: when eps == 0, x_tilde == x EXACTLY (x + 0*sign(g) == x), so
    loss_clean == loss_adv and the returned value equals the clean cost exactly
    (with alpha=0.5, 0.5*Jc + 0.5*Jc == Jc in IEEE-754). No clipping, no eps fudge.
    With dropout OFF (the runner's default for the degeneracy gate), eval() and
    train() mode are identical, so the mode switch is a no-op there.
    """
    x = x.detach().clone()
    x.requires_grad_(True)
    # Compute the FGSM surrogate in eval() mode so it matches the deterministic
    # evaluation-time attacker (eval.py uses model.eval()). Restore the caller's
    # mode afterwards so the loss terms keep the original dropout behaviour.
    was_training = model.training
    model.eval()
    Jc = cross_entropy_cost(model, x, y)
    g = torch.autograd.grad(Jc, x)[0].detach()          # frees Jc's graph
    if was_training:
        model.train()
    x_tilde = (x.detach() + eps * torch.sign(g)).detach()
    x.requires_grad_(False)
    # fresh graphs for the outer backward (not shared with the grad probe),
    # computed in the caller's original mode.
    loss_clean = cross_entropy_cost(model, x, y)
    loss_adv = cross_entropy_cost(model, x_tilde, y)
    return alpha * loss_clean + (1.0 - alpha) * loss_adv


# --------------------------------------------------------------------------- #
# M7: noise-training controls (tex:555-557)
# --------------------------------------------------------------------------- #
def noise_train_cost(
    model: Classifier,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    noise_type: str,
    gen: "torch.Generator | None" = None,
) -> torch.Tensor:
    """Noise-training control cost (M7, tex:555-557) -- NOISE-ONLY batches.

    The paper's control: "we trained a maxout network with noise based on
    randomly adding $\\pm\\eps$ to each pixel, or adding noise in
    $U(-\\eps, \\eps)$" (tex:555-556). The prose reads as training on NOISY
    inputs (every example perturbed by random noise of max-norm <= eps); it does
    NOT state a clean/noisy mixture. We therefore train on noise-only batches:
        L = J(theta, x + eta, y)         # no clean term, no alpha
    with eta regenerated every batch (the standard per-batch default; the
    paper does not state the noise regeneration cadence -- tex:490-491 refers
    to FGSM adversarial-example regeneration, not additive noise) and
    detached (no gradient through the noise sampling).

    Noise forms:
      * ``"bernoulli"``: eta = eps * b, b ~ Bernoulli-sign{+1,-1} i.i.d.
        (tex:555 "randomly adding $\\pm\\eps$ to each pixel"); ||eta||_inf == eps.
      * ``"uniform"``:   eta ~ U(-eps, eps) per pixel (tex:556); ||eta||_inf <= eps.

    SPEC §6 item 27 records this choice (the prior implementation used an
    unpapered 0.5-clean/0.5-noisy alpha-mixture mirroring E7; the prose does
    not state a mixture, so noise-only is the faithful reading and is a HARDER
    control -- more noise exposure -- for the paper's "noise << FGSM" claim).

    This is the paper's CONTROL for FGSM adversarial training -- it is expected
    to be a WEAKER regularizer than FGSM (the paper reports it confers little
    benefit, tex:555-557). Comparing FGSM-vs-noise is the point of M7.
    """
    x = x.detach()
    if noise_type == "bernoulli":
        # +/- eps per pixel, i.i.d. sign. ||eta||_inf == eps exactly.
        signs = torch.randint(
            0, 2, x.shape, generator=gen, dtype=torch.float32, device=x.device
        ).mul_(2.0).sub_(1.0)
        eta = (eps * signs).detach()
    elif noise_type == "uniform":
        eta = (eps * (2.0 * torch.rand(
            x.shape, generator=gen, dtype=torch.float32, device=x.device
        ).sub_(1.0))).detach()  # U(-eps, eps)
    else:
        raise ValueError(f"unknown noise_type {noise_type!r}")
    x_noisy = (x + eta).detach()
    return cross_entropy_cost(model, x_noisy, y)


# --------------------------------------------------------------------------- #
# M9: independent-sigmoid top cost (tex:908-909 "independent sigmoids")
# --------------------------------------------------------------------------- #
def sigmoid_top_cost(model: Classifier, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-class independent-sigmoid training cost (M9 sigmoid-top, tex:908-909).

    The paper's rubbish appendix contrasts a maxout+softmax net with the same
    net whose top layer is "independent sigmoids" (tex:908-909). For independent
    per-class sigmoid outputs p(y=k|x) = sigmoid(logit_k), the faithful training
    cost is the sum over classes of per-class binary cross-entropy against the
    one-hot label, averaged over the batch (the standard multilabel cost):

        J = (1/B) sum_b sum_k BCE(sigmoid(logit_k(x_b)), 1[y_b == k]).

    This TRAINS the sigmoid-top net (SPEC §6 item 25), rather than copying the
    softmax-trained readout weights and applying sigmoids with no retraining
    (which mechanically forces rubbish error -> 1.0 regardless of training).
    ``model.logits`` returns the pre-sigmoid readout scores [B, K].

    Normalization: SUM over classes, MEAN over the batch
        J = (1/B) sum_b sum_k BCE(sigmoid(logit_k(x_b)), 1[y_b == k])
    (NOT PyTorch reduction='mean', which divides by B*K and scales the
    gradient by 1/K -- the paper does not specify the normalization, so we use
    the sum-over-classes / mean-over-batch form, the standard multilabel cost
    where each example's loss is the sum of its per-class BCEs.)
    """
    logits = model.logits(x)  # [B, K]
    k = logits.shape[-1]
    target = F.one_hot(y, num_classes=k).to(logits.dtype)  # [B, K]
    # reduction='none' -> per-element BCE [B, K]; sum over classes, mean over batch.
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return bce.sum(dim=1).mean()


# --------------------------------------------------------------------------- #
# M-L1: L1 weight-decay control (Section 5, tex:426-433)
# --------------------------------------------------------------------------- #
def l1_first_layer_penalty(model: Classifier, coeff: float) -> torch.Tensor:
    """L1 weight-decay penalty on the FIRST maxout layer (Section 5, tex:426-433).

    The paper's Section 5 contrast of adversarial training vs L1 weight decay
    applies L1 decay to "the first layer": "When applying L1 weight decay to the
    first layer, we found that even a coefficient of .0025 was too large, and
    caused the model to get stuck with over 5% error on the training set"
    (tex:429-432). We add ``coeff * ||W_0||_1`` (sum of absolute values of the
    first maxout layer's incoming weights) to the training cost -- the standard
    L1 weight-decay form (a penalty ADDED to the cost; the paper notes this is
    more pessimistic than adversarial training, which SUBTRACTS the penalty from
    the activation, tex:418-424).

    Returns a graph-attached zero for models without a first maxout layer so the
    same trainer can call it generically.
    """
    layer0 = getattr(model, "layer0", None)
    if layer0 is None and hasattr(model, "trunk"):
        layer0 = getattr(model.trunk, "layer0", None)
    if layer0 is None or coeff == 0.0:
        return torch.zeros((), dtype=torch.float32)
    W = layer0.W  # [in, out]
    return coeff * W.abs().sum()
