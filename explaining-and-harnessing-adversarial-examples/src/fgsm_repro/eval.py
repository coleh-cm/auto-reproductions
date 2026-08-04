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
    # count of examples m1 misclassifies on its own FGSM adversarials
    n_m1_errors: int
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


def _rbf_unnorm_probs(model: Classifier, x: torch.Tensor) -> torch.Tensor:
    """RBF UNNORMALIZED per-class probabilities (M8/M9, SPEC §6 item 9).

    The paper prints only the binary RBF form ``p(y=1|x) = exp((x-mu)^T beta
    (x-mu))`` (E8, tex:595) — a per-class *unnormalized* probability in (0, 1]
    (bounded by 1 only when beta is negative-definite). For the multiclass
    extension we take the independent per-class reading ``p_k(x) = exp(q_k(x))``
    (NOT a softmax over the q_k). This preserves the paper's measured mechanism:
    confidence decays toward 0 away from the class centres mu_k, so the model is
    low-confidence on off-manifold inputs (FGSM-pushed, N(0,I) rubbish).

    A 10-way SOFTMAX over the q_k was the prior implementation; its max prob is
    bounded below by 1/K = 0.1, so it structurally CANNOT reproduce the paper's
    conf-on-mistakes 1.2% / clean-confidence 60.6% / rubbish-error 0%
    (tex:600-604, 923). The unnormalized exp(q) reading can. The argmax
    prediction is unchanged (softmax is monotonic, so argmax(q) == argmax(softmax
    q)); only the confidence/threshold metrics change.

    The exponent is clamped to <= 80 to avoid float32 overflow if a trained
    beta drifts positive (which would itself contradict the RBF property); this
    clamp only affects pathological drift, never a faithful neg-definite beta.
    """
    q = model.logits(x)  # [B, K] the quad forms
    return torch.exp(q.clamp(max=80.0))


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
    if n == 0:
        raise ValueError("_eval_from_probs_pred: empty input set; a grader must raise, not return a vacuous verdict")
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
    """Clean ACCURACY = mean(prediction == y) in [0, 1].

    Multiclass (K>1): prediction = argmax(logits).  Binary (K==1, e.g. M2
    logistic regression with y in {-1,+1}): prediction = +1 if score>0
    else -1, i.e. the logistic decision rule sign(score).  The argmax-over-one
    column would always yield 0 and never match {-1,+1} labels, so the binary
    case is special-cased here.
    """
    if int(x.shape[0]) == 0:
        raise ValueError("eval_clean: empty input set; a grader must raise, not return a vacuous verdict")
    model.eval()
    with torch.no_grad():
        logits = model.logits(x)
        if logits.shape[-1] == 1:
            # binary: predict +1 where score>0, else -1 (matches y in {-1,+1})
            pred = torch.where(
                logits.squeeze(-1) > 0,
                torch.ones_like(y, dtype=y.dtype),
                -torch.ones_like(y, dtype=y.dtype),
            )
        else:
            pred = logits.argmax(dim=1)
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


def eval_clean_confidence_rbf(model: Classifier, x: torch.Tensor) -> float:
    """M8 RBF clean confidence = mean over ALL examples of max_k exp(q_k)
    (paper 60.6%, tex:603). Uses the UNNORMALIZED per-class exp(q) reading
    (SPEC §6 item 9); a softmax reading is bounded below by 1/K and cannot
    reproduce the paper's number.

    No-success-on-empty: an empty input set is a degenerate batch, not a valid
    confidence of nan -- it must RAISE (the contract SPEC §11 names; a vacuous
    nan propagating silently into claims c22/c23 is exactly the failure the
    no-empty-success oracle exists to catch)."""
    if int(x.shape[0]) == 0:
        raise ValueError("eval_clean_confidence_rbf: empty input set; a grader must raise, not return a vacuous verdict")
    model.eval()
    with torch.no_grad():
        probs = _rbf_unnorm_probs(model, x)  # [B, K]
    return float(probs.max(dim=1).values.mean())


def eval_fgsm_rbf(
    model: Classifier, x: torch.Tensor, y: torch.Tensor, eps: float
) -> AttackEval:
    """Evaluate an RBF model on its OWN FGSM adversarial examples (M8, tex:600-604).

    Same construction as ``eval_fgsm`` (the FGSM attack uses the model's
    training cost J = cross-entropy/softmax over the quad forms, tex:309), but
    the CONFIDENCE metric uses the UNNORMALIZED per-class exp(q) reading
    (SPEC §6 item 9). The error rate uses argmax(q) (normalization-invariant,
    identical to the softmax argmax). No clipping of x_tilde.
    """
    model.eval()
    x_adv = fgsm(model, x, y, eps)
    with torch.no_grad():
        logits = model.logits(x_adv)            # [B, K] quad forms
        probs = torch.exp(logits.clamp(max=80.0))  # unnormalized exp(q)
        pred = logits.argmax(dim=1)             # == argmax(softmax(q))
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

    This is the special case ``agreement_on_adv(m1, m1, m2, ...)`` (the attack
    source and the reference whose class is predicted are the SAME model m1),
    which is the paper's construction for the first four §8 numbers: all five
    numbers use adversarial examples generated on the deep MAXOUT network
    (tex:679-680 "we generated adversarial examples on a deep maxout network
    and classified these examples using a shallow softmax network and a shallow
    RBF network"). See ``agreement_on_adv`` for the 53.6% case where the attack
    source (maxout) differs from the reference whose class is predicted
    (softmax).
    """
    return agreement_on_adv(m1, m1, m2, x, y, eps)


def agreement_on_adv(
    attacker: Classifier,
    ref: Classifier,
    pred: Classifier,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> AgreementStats:
    """§8 cross-model class agreement with the attack source separated from the
    reference model (tex:679-688).

    The paper fixes ONE adversarial-example set per paragraph: "we generated
    adversarial examples on a deep maxout network and classified these examples
    using a shallow softmax network and a shallow RBF network" (tex:679-680).
    All five §8 agreement numbers therefore use the MAXOUT-generated examples
    (``attacker``). For the first four the predicted class is the MAXOUT's class
    (``ref == attacker == maxout``); for "the RBF network can predict softmax
    regression's class 53.6% of the time" (tex:687) the predicted class is the
    SOFTMAX's class, so ``ref == softmax`` while ``attacker`` stays maxout.

    Build FGSM adversarials from ``attacker``; let ``p_ref`` be the reference
    model's prediction (the class being predicted) and ``p_pred`` the predictor
    model's prediction. Report:
      * p_pred_match_over_m1_errors: over the subset where ``ref`` misclassifies
        the attacker's adversarial, how often ``pred`` picks ``ref``'s class.
        (paper's 16.0% / 54.6% / 53.6% over the attacker/reference's errors)
      * p_pred_match_over_both_wrong: over the subset where BOTH ``ref`` and
        ``pred`` misclassify, how often ``pred`` picks ``ref``'s (wrong) class.
        (paper's 84.6% / 54.3%)
    Both fractions are 0.0 when their denominator is empty.

    The prior implementation called ``class_agreement(m_soft, rbf)`` for the
    53.6% number, which built NEW adversarials from the softmax model rather
    than the paragraph's fixed maxout-generated set; that reading is retained
    as a SECONDARY diagnostic in m8_rbf.py. SPEC §6 records the ambiguity.
    """
    attacker.eval()
    ref.eval()
    pred.eval()
    x_adv = fgsm(attacker, x, y, eps)
    with torch.no_grad():
        p_ref = ref.logits(x_adv).argmax(dim=1)
        p_pred = pred.logits(x_adv).argmax(dim=1)
    ref_err = p_ref != y
    both_wrong = ref_err & (p_pred != y)
    n_m1_errors = int(ref_err.sum().item())
    n_both_wrong = int(both_wrong.sum().item())

    if ref_err.sum() > 0:
        p_pred_match_over_m1_errors = (
            (p_pred[ref_err] == p_ref[ref_err]).float().mean().item()
        )
    else:
        p_pred_match_over_m1_errors = 0.0

    if both_wrong.sum() > 0:
        p_pred_match_over_both_wrong = (
            (p_pred[both_wrong] == p_ref[both_wrong]).float().mean().item()
        )
    else:
        p_pred_match_over_both_wrong = 0.0

    return AgreementStats(
        n_m1_errors=n_m1_errors,
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

    No-success-on-empty: n<=0 is a degenerate count (an empty synthetic batch),
    not a valid verdict -- it must RAISE (SPEC §11 grader contract); returning
    AttackEval(error_rate=nan, ...) would propagate a vacuous nan silently.
    """
    if n <= 0:
        raise ValueError("eval_rubbish: empty input set (n<=0); a grader must raise, not return a vacuous verdict")
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    x = sample_rubbish(n, dim, gen)
    with torch.no_grad():
        probs = _probs(model, x)
        max_prob = probs.max(dim=1).values
    wrong = max_prob > 0.5
    error_rate = wrong.float().mean().item()
    if wrong.sum() > 0:
        mean_conf = max_prob[wrong].mean().item()
    else:
        mean_conf = 0.0
    return AttackEval(error_rate=error_rate, mean_confidence_on_errors=mean_conf, n=n)


def eval_rubbish_rbf(
    model: Classifier, n: int, dim: int, seed: int
) -> AttackEval:
    """Appendix rubbish examples for the RBF model (M9, tex:905-906, 923).

    Draw n ~ N(0, I_dim); for the RBF the per-class probability is the
    UNNORMALIZED exp(q_k) (SPEC §6 item 9, E8 tex:595). A rubbish sample is an
    "error" iff ANY class's exp(q_k) > 0.5 (tex:906 "assigning a probability
    greater than 0.5 to any class"). Confidence = mean over the erroring subset
    of the max exp(q_k); 0.0 if none. With a negative-definite beta the RBF is
    far from every mu_k on N(0,I) noise, so exp(q_k) -> 0 and the error rate is
    ~0 (paper 0%, tex:923) — the structural property a softmax reading cannot
    reproduce (a 10-way softmax max-prob is bounded below by 0.1).

    No-success-on-empty: n<=0 is a degenerate count (an empty synthetic batch),
    not a valid verdict -- it must RAISE (SPEC §11 grader contract).
    """
    if n <= 0:
        raise ValueError("eval_rubbish_rbf: empty input set (n<=0); a grader must raise, not return a vacuous verdict")
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    x = sample_rubbish(n, dim, gen)
    with torch.no_grad():
        probs = _rbf_unnorm_probs(model, x)  # [B, K] unnormalized exp(q)
        max_prob = probs.max(dim=1).values
    wrong = max_prob > 0.5
    error_rate = wrong.float().mean().item()
    if wrong.sum() > 0:
        mean_conf = max_prob[wrong].mean().item()
    else:
        mean_conf = 0.0
    return AttackEval(error_rate=error_rate, mean_confidence_on_errors=mean_conf, n=n)


def eval_rubbish_sigmoid(
    model: Classifier, n: int, dim: int, seed: int
) -> AttackEval:
    """Appendix rubbish examples, INDEPENDENT-SIGMOID top (M9, tex:908-909).

    Same protocol as ``eval_rubbish`` but with per-class independent sigmoid
    outputs: p(y=k|x) = sigmoid(logit_k(x)). A rubbish sample is an "error"
    iff ANY class probability > 0.5 (tex:906 "assigning a probability greater
    than 0.5 to any class"). Confidence = mean over the erroring subset of
    the MAX per-class sigmoid probability; 0.0 if none.

    The subject model here is a TRAINED independent-sigmoid net (per-class BCE),
    not a frozen softmax-to-sigmoid weight swap (SPEC §6 item 25): the paper
    says "Changing the top layer to independent sigmoids" (tex:908-909), whose
    natural reading is the architecture trained with the sigmoid-appropriate
    cost, evaluated on the same N(0,I) rubbish.

    No-success-on-empty: n<=0 is a degenerate count (an empty synthetic batch),
    not a valid verdict -- it must RAISE (SPEC §11 grader contract).
    """
    if n <= 0:
        raise ValueError("eval_rubbish_sigmoid: empty input set (n<=0); a grader must raise, not return a vacuous verdict")
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    x = sample_rubbish(n, dim, gen)
    with torch.no_grad():
        probs = torch.sigmoid(model.logits(x))   # [B, K] independent sigmoids
        max_prob = probs.max(dim=1).values
    wrong = max_prob > 0.5
    error_rate = wrong.float().mean().item()
    if wrong.sum() > 0:
        mean_conf = max_prob[wrong].mean().item()
    else:
        mean_conf = 0.0
    return AttackEval(error_rate=error_rate, mean_confidence_on_errors=mean_conf, n=n)
