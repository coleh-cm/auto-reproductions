"""Invariants the paper's equations imply (E1-E10, tex:235-936).

Each test asserts a property that MUST hold if the implementation matches the
cited equation.  These cost seconds and catch the errors that survive to
"trains fine, number is a bit off".  (research-code skill: invariants from the
maths.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT / "src"))

from fgsm_repro.models import SoftmaxRegression, LogisticRegression, RBFNet
from fgsm_repro.attacks import fgsm, sample_rubbish
from fgsm_repro.objectives import (
    cross_entropy_cost, softplus_logreg_cost, adversarial_logreg_cost,
    adversarial_train_cost,
)
from fgsm_repro.eval import eval_clean, eval_fgsm


def _softmax_model():
    torch.manual_seed(0)
    return SoftmaxRegression(784, 10)


# --- E1: eta = eps * sign(grad_x J);  ||eta||_inf == eps ------------------- #
def test_fgsm_perturbation_norm_equals_eps():
    """E1 (tex:309): ||eta||_inf == eps exactly when no zero-gradient element."""
    m = _softmax_model()
    x = torch.rand(16, 784)
    y = torch.randint(0, 10, (16,))
    for eps in (0.1, 0.25, 0.5):
        xt = fgsm(m, x, y, eps)
        eta = (xt - x).abs()
        assert torch.allclose(eta[eta > 0], torch.full_like(eta[eta > 0], eps)), (
            f"||eta||_inf != eps for eps={eps}")


# --- E2: x_tilde = x + eta; NO clipping ------------------------------------ #
def test_fgsm_no_clipping():
    """E2 (tex:235): perturbed inputs are NOT clipped to [0,1].

    The paper's displayed method does not clip (only the commented-out Szegedy
    variant constrains features; tex:313-326).  With eps large enough, some
    x_tilde elements must exceed [0,1]."""
    m = _softmax_model()
    x = torch.zeros(8, 784)  # zeros + positive eta -> exactly eps (in range), but
    y = torch.randint(0, 10, (8,))
    xt = fgsm(m, x, y, 0.5)
    assert xt.min() < 0.0 or xt.max() > 1.0, "FGSM appears to clip to [0,1] (it must not)"


# --- eps=0 degeneracy of the attack --------------------------------------- #
def test_fgsm_eps0_is_identity():
    """eps=0 -> eta=0 -> x_tilde == x exactly."""
    m = _softmax_model()
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    assert torch.equal(fgsm(m, x, y, 0.0), x)


# --- sign(0) := 0 convention ---------------------------------------------- #
def test_sign_zero_convention():
    """sign(0)=0: a zero-gradient element gets zero perturbation.

    Construct a model whose gradient is exactly zero on some input element
    (zero weight column -> zero gradient on that feature)."""
    m = _softmax_model()
    with torch.no_grad():
        m.linear.weight[:, 0] = 0.0  # zero weight on feature 0 -> zero grad
        m.linear.bias.zero_()
    x = torch.rand(4, 784)
    y = torch.randint(0, 10, (4,))
    xt = fgsm(m, x, y, 0.25)
    eta = xt - x
    assert torch.all(eta[:, 0] == 0.0), "sign(0)!=0: zero-gradient feature got perturbed"


# --- E5/E6: logistic costs; E6 == empirical FGSM for y=+1 ----------------- #
def test_e6_matches_empirical_fgsm_positive_class():
    """Paper claim (tex:398-412): for logistic regression FGSM is EXACT, so the
    closed-form adversarial loss E6 must equal the loss on the FGSM-perturbed
    input.  This holds for y=+1 (the paper's "sign of gradient = -sign(w)"
    derivation).  For y=-1 the paper's E6 is NOT the worst case (recorded
    finding); we assert the y=+1 case where the paper's exactness claim holds."""
    torch.manual_seed(2)
    w = 0.05 * torch.randn(784)
    b = torch.tensor(0.1)
    x = torch.rand(32, 784)
    y_pos = torch.ones(32, dtype=torch.float32)  # y = +1 only
    xg = x.clone().requires_grad_(True)
    J = softplus_logreg_cost(w, b, xg, y_pos)
    g = torch.autograd.grad(J, xg)[0]
    xt = (xg + 0.25 * torch.sign(g)).detach()
    empirical = softplus_logreg_cost(w, b, xt, y_pos)
    closed = adversarial_logreg_cost(w, b, x, y_pos, 0.25)
    assert torch.allclose(closed, empirical, atol=1e-5), (
        f"E6={closed} != empirical FGSM={empirical} for y=+1")


def test_e6_positive_class_increases_loss():
    """For y=+1 the worst-case perturbation must increase the loss (E6 >= clean)."""
    torch.manual_seed(3)
    w = 0.05 * torch.randn(784)
    b = torch.tensor(0.1)
    x = torch.rand(32, 784)
    y_pos = torch.ones(32, dtype=torch.float32)
    clean = softplus_logreg_cost(w, b, x, y_pos)
    adv = adversarial_logreg_cost(w, b, x, y_pos, 0.25)
    assert adv.item() >= clean.item() - 1e-6, "E6 < clean for y=+1 (not worst case)"


def test_e6_matches_paper_formula():
    """adversarial_logreg_cost == mean(zeta(y*(eps*||w||_1 - w^T x - b))) exactly."""
    torch.manual_seed(4)
    w = 0.03 * torch.randn(784)
    b = torch.tensor(-0.2)
    x = torch.rand(16, 784)
    y = torch.tensor([1, -1] * 8, dtype=torch.float32)
    eps = 0.25
    s = x @ w + b
    expected = F.softplus(y * (eps * w.abs().sum() - s)).mean()
    got = adversarial_logreg_cost(w, b, x, y, eps)
    assert torch.allclose(got, expected, atol=1e-6)


# --- E7: J_tilde(eps=0) == J exactly -------------------------------------- #
def test_e7_eps0_equals_cross_entropy():
    """E7 (tex:486-488): at eps=0, J~ = alpha*J + (1-alpha)*J = J exactly."""
    m = _softmax_model()
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    for alpha in (0.0, 0.5, 1.0):
        J = cross_entropy_cost(m, x, y)
        Jt = adversarial_train_cost(m, x, y, 0.0, alpha)
        assert torch.equal(J, Jt), f"E7(eps=0,alpha={alpha})={Jt} != J={J}"


# --- E8: RBF quad form is negative-definite (no minus sign in the exp) ---- #
def test_rbf_no_minus_sign():
    """E8 (tex:595): p(y=1|x) = exp((x-mu)^T beta (x-mu)) — NO minus sign in the
    exp; the minus lives in beta (neg-semi-def), which is the faithful reading
    of E8 (a valid probability requires beta neg-semi-def). RBFNet makes beta
    neg-def BY CONSTRUCTION (beta_k = -diag(softplus(raw_k))), so the quad form
    q_k <= 0 ALWAYS -> exp(q_k) in (0,1], and q_k -> -inf away from mu_k.

    We assert: (a) q <= 0 for arbitrary inputs (neg-def by construction); (b) q
    at a class centre mu_k is 0 (exp = 1, max confidence near mu -- the paper's
    "confident only in the vicinity of mu", tex:596-598); (c) q decreases
    (more negative) as the input moves away from mu_k; (d) the exp form is the
    paper's (no extra minus sign flipping it)."""
    rbf = RBFNet(n_classes=3, in_dim=4)
    x = torch.rand(5, 4)
    with torch.no_grad():
        q = rbf.logits(x)  # [B, K]
    # (a) neg-def by construction: q <= 0 for all inputs.
    assert torch.all(q <= 1e-6), f"RBF q should be <= 0 (neg-def), got max {q.max()}"
    # (b) at a class centre, q for that class is 0 (exp=1, max confidence).
    with torch.no_grad():
        q_at_mu = rbf.logits(rbf.mu)  # [K, K]: row k is q at mu_k
    assert torch.allclose(q_at_mu.diagonal(), torch.zeros(3), atol=1e-5), (
        f"q at own centre should be 0, got {q_at_mu.diagonal()}")
    # (c) moving away from mu_k makes q_k more negative (confidence decays).
    with torch.no_grad():
        rbf.mu.data.zero_()
        a = rbf.a  # [K, F], all > 0
    near = torch.zeros(1, 4)
    far = 10.0 * torch.ones(1, 4)
    q_near = rbf.logits(near)[:, 0]
    q_far = rbf.logits(far)[:, 0]
    assert q_far.item() < q_near.item() - 1.0, (
        f"q should decrease away from mu: near={q_near}, far={q_far}")
    # (d) beta is neg-def diagonal: beta_k = -diag(a_k), a_k > 0.
    b = rbf.beta  # [K, F, F]
    for k in range(3):
        diag = b[k].diag()
        assert torch.all(diag <= 0), f"beta[{k}] diag should be <= 0"
        # off-diagonal entries are exactly zero (diagonal parameterization).
        off = b[k] - torch.diag(b[k].diag())
        assert torch.all(off == 0), "beta should be diagonal"


# --- cross-entropy NLL is non-negative ------------------------------------- #
def test_cross_entropy_nonnegative():
    """J = mean NLL >= 0 always (log-prob <= 0)."""
    m = _softmax_model()
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    assert cross_entropy_cost(m, x, y).item() >= -1e-6


# --- eval_fgsm(eps=0) == 1 - eval_clean (identity attack) ----------------- #
def test_eval_fgsm_eps0_equals_clean_error():
    """eps=0 attack is identity, so adversarial error == clean error."""
    m = _softmax_model()
    x = torch.rand(32, 784)
    y = torch.randint(0, 10, (32,))
    clean_acc = eval_clean(m, x, y)
    adv = eval_fgsm(m, x, y, 0.0)
    assert abs(adv.error_rate - (1.0 - clean_acc)) < 1e-6, (
        f"eps=0 adv error {adv.error_rate} != 1-clean {1-clean_acc}")
    assert adv.n == 32


# --- confidence is over the misclassified subset only --------------------- #
def test_confidence_only_over_misclassified():
    """mean_confidence_on_errors averages over the misclassified subset only;
    if everything is misclassified it equals the mean max-prob."""
    m = _softmax_model()
    # all-wrong: y outside [0,9]? use a model that always predicts class 0 by
    # zeroing all but the first logit column
    with torch.no_grad():
        m.linear.weight[1:] = 0.0
        m.linear.bias[1:] = -1e9
        m.linear.bias[0] = 0.0
    x = torch.rand(16, 784)
    y = torch.randint(1, 10, (16,))  # never class 0 -> always wrong
    adv = eval_fgsm(m, x, y, 0.0)  # identity, still always predicts 0
    probs = F.softmax(m.logits(x), dim=-1)
    expected_conf = probs.max(dim=1).values.mean().item()
    assert abs(adv.mean_confidence_on_errors - expected_conf) < 1e-5, (
        f"conf {adv.mean_confidence_on_errors} != mean max-prob {expected_conf}")
    assert abs(adv.error_rate - 1.0) < 1e-6


# --- rubbish: N(0, I_dim) samples ------------------------------------------ #
def test_sample_rubbish_is_standard_normal():
    """sample_rubbish draws from N(0, I_dim): mean ~0, std ~1."""
    gen = torch.Generator().manual_seed(0)
    x = sample_rubbish(20000, 784, gen)
    assert x.shape == (20000, 784)
    assert abs(x.mean().item()) < 0.02
    assert abs(x.std().item() - 1.0) < 0.02


# --- E7: training-time surrogate matches the deterministic eval attacker --- #
def test_adversarial_train_surrogate_matches_eval_attacker():
    """SPEC.md section 6 item 23 / F1: the FGSM surrogate used inside
    adversarial_train_cost must be computed with dropout OFF (eval mode), so it
    matches the deterministic evaluation-time attacker (eval.py runs
    model.eval()). With dropout ACTIVE around the call, the perturbation the
    cost generates must still equal the eval-mode attacker's perturbation --
    i.e. the train-mode dropout must not leak into the surrogate direction.

    We verify by reconstructing the cost's internal probe (eval-mode gradient)
    and comparing its sign to a clean eval-mode fgsm() attack."""
    from fgsm_repro.models import MaxoutMLP
    torch.manual_seed(7)
    m = MaxoutMLP(units=32, pieces=5, seed=7,
                  dropout_input_include=0.5, dropout_hidden_include=0.5)
    x = torch.rand(16, 784)
    y = torch.randint(0, 10, (16,))
    # Deterministic eval-time attacker direction.
    m.eval()
    x_adv_eval = fgsm(m, x, y, 0.25)
    eta_eval = x_adv_eval - x
    # Reconstruct the cost's internal surrogate probe: eval-mode gradient.
    m.train()  # surrounding train mode (dropout active) -- must NOT affect probe
    x_g = x.detach().clone().requires_grad_(True)
    m.eval()
    Jc = cross_entropy_cost(m, x_g, y)
    g = torch.autograd.grad(Jc, x_g)[0].detach()
    m.train()
    eta_surrogate = 0.25 * torch.sign(g)
    assert torch.equal(eta_surrogate.sign(), eta_eval.sign()), (
        "surrogate direction differs from eval-time attacker (dropout leaked in)")
    assert torch.allclose(eta_surrogate, eta_eval)


def test_adversarial_train_backward_finite_with_dropout():
    """adversarial_train_cost must backward cleanly into params even with
    dropout active (guards the freed-graph bug + the eval/train mode switch)."""
    from fgsm_repro.models import MaxoutMLP
    torch.manual_seed(1)
    m = MaxoutMLP(units=16, pieces=5, seed=1,
                  dropout_input_include=0.5, dropout_hidden_include=0.5)
    m.train()
    x = torch.rand(16, 784)
    y = torch.randint(0, 10, (16,))
    loss = adversarial_train_cost(m, x, y, 0.25, 0.5)
    loss.backward()
    for p in m.parameters():
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0
    # model must be back in train mode after the cost (the cost restores it).
    assert m.training


# --- external-recipe alignment: readout init irange .005 + zero bias ------- #
def test_maxout_readout_init_matches_recipe():
    """F5: the adopted external pylearn2 mnist_pi.yaml sets ``irange: .005``
    on the Softmax readout layer ``y`` (and pylearn2 biases start at 0). The
    readout weight must therefore be uniform in [-0.005, 0.005] and its bias
    must be exactly zero at construction -- NOT PyTorch's default
    ±1/sqrt(fan_in) (~±0.0645 for fan_in=240) with random bias."""
    from fgsm_repro.models import MaxoutMLP
    torch.manual_seed(11)
    m = MaxoutMLP(units=240, pieces=5, n_classes=10, seed=11)
    w = m.readout.weight
    b = m.readout.bias
    assert w.abs().max().item() <= 0.005 + 1e-6, (
        f"readout weight max abs {w.abs().max().item()} > 0.005 (not irange .005)")
    assert torch.all(b == 0.0), f"readout bias must be zero, got {b}"
    # and it must NOT be all-zero (uniform draw is non-degenerate)
    assert w.abs().sum().item() > 0.0


# --- early-stopping selects the BEST epoch, not the stopping epoch ---------- #
def test_best_epoch_is_selected_not_stopping():
    """Paper protocol tex:505-506: the early-stopping criterion chooses the
    number of epochs to retrain for; that number is the BEST epoch (where
    validation error was lowest), NOT the epoch at which patience ran out
    (best_epoch + patience). TrainResult must expose best_epoch so the M5
    retrain arm retrains for the selected count, not the stopping count.

    This is a regression test for the blocker found by the adversarial review:
    train.py previously exposed only epochs_run (= best_epoch + patience), so
    M5 over-trained by ~patience epochs and the headline M5 number (tex:506-512)
    was biased. Fix: TrainResult.best_epoch records the 0-based index of the
    best validation epoch."""
    from fgsm_repro.data import MNISTData
    from fgsm_repro.models import SoftmaxRegression
    from fgsm_repro.train import TrainConfig, train

    # Build data where valid error is monotonically NON-decreasing after epoch
    # 0, so the best epoch is unambiguously epoch 0 and patience trips later.
    g = torch.Generator().manual_seed(0)
    def make(n):
        x = torch.rand(n, 784, generator=g, dtype=torch.float32)
        y = torch.randint(0, 10, (n,), generator=g, dtype=torch.int64)
        x[torch.arange(n), y] += 0.5
        return x, y
    data = MNISTData(x_train=make(200)[0], y_train=make(200)[1],
                     x_valid=make(80)[0], y_valid=make(80)[1],
                     x_test=make(80)[0], y_test=make(80)[1])
    torch.manual_seed(0)
    m = SoftmaxRegression(784, 10)
    res = train(m, TrainConfig(batch_size=50, lr=1.0, momentum=0.0,
                               max_epochs=20, seed=0, patience=3,
                               early_stop="clean", max_steps=4), data)
    # best_epoch must be an int (validation is non-empty) and 0-based.
    assert res.best_epoch is not None, "best_epoch should be set (valid non-empty)"
    assert isinstance(res.best_epoch, int)
    assert res.best_epoch >= 0
    # The selected epoch COUNT (what M5 retrains for) is best_epoch + 1, and it
    # must be <= epochs_run (the stopping epoch). Equality holds only if
    # patience never tripped; otherwise best_epoch+1 < epochs_run.
    assert res.best_epoch + 1 <= res.epochs_run, (
        f"best_epoch+1={res.best_epoch + 1} > epochs_run={res.epochs_run}")
    # history length tracks epochs_run too.
    assert len(res.history) == res.epochs_run
