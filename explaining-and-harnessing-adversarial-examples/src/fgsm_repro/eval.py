"""Evaluation metrics. Definitions are FIXED here (SPEC.md §4 `eval.py`).

``AttackEval``:
    error_rate            = fraction of ALL evaluated examples misclassified
                          = mean(argmax(logits) != y)               (SPEC.md §6 item 13)
    mean_confidence_on_errors = softmax probability assigned to the PREDICTED class,
                          averaged over the MISCLASSIFIED subset only (§6 item 14).
                          0.0 when no example is misclassified.
    n                     = number of evaluated examples.

``AgreementStats`` (§8, tex:679-688): the paper reports two different agreement
fractions for the same (m1, m2) pair:
    * over m1's errors           -> "16.0% / 54.6% / 53.6%"  (tex:681-683, 687)
    * over the BOTH-wrong subset -> "84.6% / 54.3%"          (tex:684-686)
Both must be producible, so both are exposed as separate fields.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .attacks import fgsm, sample_rubbish
from .models import Classifier


@dataclass
class AttackEval:
    error_rate: float
    mean_confidence_on_errors: float
    n: int


@dataclass
class AgreementStats:
    # count of examples BOTH models misclassify on m1's FGSM adversarials
    n_both_wrong: int
    # fraction (over m1's-error subset) where m2 assigns m1's (predicted) class.
    # This is the paper's primary §8 number: 16.0% / 54.6% / 53.6% (tex:681-683,687).
    p_pred_match_over_m1_errors: float
    # fraction (over BOTH-wrong subset) where m2 assigns m1's (predicted) class.
    # The paper's 84.6% / 54.3% numbers (tex:684-686).
    p_pred_match_over_both_wrong: float


def _probs(model: Classifier, x: torch.Tensor) -> torch.Tensor:
    """softmax(model.logits(x), dim=-1) -> [B, K]."""
    return F.softmax(model.logits(x), dim=-1)


def _eval_from_probs_pred(
    probs: torch.Tensor, pred: torch.Tensor, y: torch.Tensor
) -> AttackEval:
    """Build an AttackEval from probs, predicted class, and true labels.

    `pred` is the argmax over classes (the PREDICTED class). The confidence is
    the softmax probability of that predicted class, averaged over the
    misclassified subset only; 0.0 when nothing is misclassified.
    Denominator for error_rate is ALL n evaluated examples.
    """
    n = int(pred.numel())
    wrong = pred != y
    error_rate = wrong.float().mean().item()
    if wrong.sum() > 0:
        # prob of the PREDICTED class for each example
        conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
        mean_conf = conf[wrong].mean().item()
    else:
        mean_conf = 0.0
    return AttackEval(error_rate=error_rate, mean_confidence_on_errors=mean_conf, n=n)


def eval_clean(model: Classifier, x: torch.Tensor, y: torch.Tensor) -> float:
    """Clean ACCURACY = mean(argmax(logits) == y) in [0, 1]."""
    model.eval()
    with torch.no_grad():
        pred = model.logits(x).argmax(dim=1)
    return (pred == y).float().mean().item()


def eval_fgsm(
    model: Classifier, x: torch.Tensor, y: torch.Tensor, eps: float
) -> AttackEval:
    """Evaluate the model on its OWN FGSM adversarial examples (no clipping)."""
    model.eval()
    x_adv = fgsm(model, x, y, eps)
    with torch.no_grad():
        logits = model.logits(x_adv)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=1)
    return _eval_from_probs_pred(probs, pred, y)


def eval_transfer(
    src: Classifier,
    tgt: Classifier,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> AttackEval:
    """M6: craft FGSM examples from `src`, score on `tgt`."""
    src.eval()
    tgt.eval()
    x_adv = fgsm(src, x, y, eps)
    with torch.no_grad():
        logits = tgt.logits(x_adv)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=1)
    return _eval_from_probs_pred(probs, pred, y)


def class_agreement(
    m1: Classifier,
    m2: Classifier,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> AgreementStats:
    """§8 cross-model class agreement (tex:679-688).

    Build FGSM adversarials from m1, predict on both m1 and m2, then report:
      * p_pred_match_over_m1_errors: over the subset where m1 misclassifies
        its own adversarial example, how often m2 picks the SAME class m1 did.
        (paper's 16.0% / 54.6% / 53.6%)
      * p_pred_match_over_both_wrong: over the subset where BOTH misclassify,
        how often m2 picks m1's (wrong) class. (paper's 84.6% / 54.3%)
    Both fractions are 0.0 when their denominator is empty.
    """
    m1.eval()
    m2.eval()
    x_adv = fgsm(m1, x, y, eps)
    with torch.no_grad():
        p1 = m1.logits(x_adv).argmax(dim=1)
        p2 = m2.logits(x_adv).argmax(dim=1)
    m1_err = p1 != y
    both_wrong = m1_err & (p2 != y)
    n_both_wrong = int(both_wrong.sum().item())

    if m1_err.sum() > 0:
        p_pred_match_over_m1_errors = (
            (p2[m1_err] == p1[m1_err]).float().mean().item()
        )
    else:
        p_pred_match_over_m1_errors = 0.0

    if both_wrong.sum() > 0:
        p_pred_match_over_both_wrong = (
            (p2[both_wrong] == p1[both_wrong]).float().mean().item()
        )
    else:
        p_pred_match_over_both_wrong = 0.0

    return AgreementStats(
        n_both_wrong=n_both_wrong,
        p_pred_match_over_m1_errors=p_pred_match_over_m1_errors,
        p_pred_match_over_both_wrong=p_pred_match_over_both_wrong,
    )


def eval_rubbish(
    model: Classifier, n: int, dim: int, seed: int
) -> AttackEval:
    """Appendix rubbish examples (tex:905-906).

    Draw n ~ N(0, I_dim) samples; "error" := max_k p(y=k|x) > 0.5.
    Confidence = mean over the erroring subset of that max prob; 0.0 if none.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    x = sample_rubbish(n, dim, gen)
    with torch.no_grad():
        probs = _probs(model, x)
        max_prob = probs.max(dim=1).values
        pred = probs.argmax(dim=1)
    wrong = max_prob > 0.5
    error_rate = wrong.float().mean().item()
    if wrong.sum() > 0:
        mean_conf = max_prob[wrong].mean().item()
    else:
        mean_conf = 0.0
    return AttackEval(error_rate=error_rate, mean_confidence_on_errors=mean_conf, n=n)
