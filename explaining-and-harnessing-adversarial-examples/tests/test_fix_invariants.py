"""Invariants for the REJECT-feedback fixes (RBF metric, sigmoid-top training,
ensemble prediction, noise-only control, L1 weight decay).

Each test asserts a property the fix must satisfy, checked against the paper's
mechanism.  (research-code skill: invariants from the maths.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT / "src"))

from fgsm_repro.models import RBFNet, MaxoutMLP, SigmoidTopMLP
from fgsm_repro.eval import (
    eval_fgsm_rbf, eval_clean_confidence_rbf, eval_rubbish_rbf, _rbf_unnorm_probs,
)
from fgsm_repro.objectives import (
    sigmoid_top_cost, l1_first_layer_penalty, noise_train_cost,
)
from fgsm_repro.attacks import fgsm, sample_rubbish
from fgsm_repro.train import TrainConfig, train, _maxout_layers_and_readout
from fgsm_repro.data import MNISTData


# ---------------- Fix 1: RBF unnormalized exp(q) metric ----------------- #
def test_rbf_unnorm_probs_are_exp_of_logits():
    """The RBF 'probability' is the UNNORMALIZED per-class exp(q_k) (E8, tex:595),
    NOT a softmax. _rbf_unnorm_probs == exp(logits) elementwise (up to the
    overflow clamp)."""
    rbf = RBFNet(n_classes=3, in_dim=4)
    x = torch.rand(5, 4)
    with torch.no_grad():
        q = rbf.logits(x)
        probs = _rbf_unnorm_probs(rbf, x)
    assert torch.allclose(probs, torch.exp(q.clamp(max=80.0)))
    # NOT a softmax: rows do not sum to 1 in general.
    assert not torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-3)


def test_rbf_unnorm_confidence_decays_off_manifold():
    """Core paper mechanism (tex:596-598): a neg-def RBF is confident only near
    its centres mu_k; far from every mu_k (e.g. N(0,I) rubbish far from MNIST
    mu), max_k exp(q_k) -> 0. With beta=-I the quad form is -||x-mu||^2, so a
    point far from all mu has tiny max exp(q). A softmax reading could not do
    this (bounded below by 1/K)."""
    rbf = RBFNet(n_classes=3, in_dim=8)
    with torch.no_grad():
        # centres near 0; a point far away (large norm) -> tiny confidence.
        rbf.mu.data.zero_()
        rbf.beta.data = -torch.eye(8).unsqueeze(0).expand(3, 8, 8).contiguous()
    near = torch.zeros(1, 8)  # at a centre -> q=0 -> exp=1
    far = 10.0 * torch.ones(1, 8)  # far from all centres -> q very negative
    conf_near = _rbf_unnorm_probs(rbf, near).max(dim=1).values
    conf_far = _rbf_unnorm_probs(rbf, far).max(dim=1).values
    assert abs(conf_near.item() - 1.0) < 1e-4, conf_near
    assert conf_far.item() < 1e-10, conf_far  # decays toward 0 off-manifold


def test_rbf_fgsm_error_uses_argmax_invariant():
    """RBF FGSM error rate uses argmax(q) which is normalization-invariant:
    argmax(q) == argmax(softmax(q)). The error rate must equal the error of
    argmax(softmax(q)) on the same adversarial inputs."""
    rbf = RBFNet(n_classes=4, in_dim=16)
    x = torch.rand(32, 16)
    y = torch.randint(0, 4, (32,))
    ev = eval_fgsm_rbf(rbf, x, y, 0.25)
    # Recompute the adversarial prediction two ways and confirm they agree.
    x_adv = fgsm(rbf, x, y, 0.25)
    with torch.no_grad():
        q = rbf.logits(x_adv)
        pred_argmax_q = q.argmax(dim=1)
        pred_argmax_sm = F.softmax(q, dim=1).argmax(dim=1)
    assert torch.equal(pred_argmax_q, pred_argmax_sm)
    manual_err = float((pred_argmax_q != y).float().mean().item())
    assert abs(ev.error_rate - manual_err) < 1e-9


def test_rbf_rubbish_can_be_zero_error():
    """With a neg-def beta the RBF assigns exp(q) << 0.5 to N(0,I) rubbish far
    from MNIST centres, so the any-class-p>0.5 error can be 0 (paper 0%,
    tex:923) -- structurally impossible under a softmax reading (bounded below
    by 1/K > 0.5 for K<2, and for K=10 softmax max-prob >= 0.1 but the >0.5
    threshold can still trip). Here we force centres far from the rubbish and
    a strong neg-def beta to verify the 0% mechanism is reachable."""
    rbf = RBFNet(n_classes=10, in_dim=784)
    with torch.no_grad():
        rbf.mu.data = 5.0 * torch.ones_like(rbf.mu.data)  # centres far from N(0,I)
        rbf.beta.data = -0.5 * torch.eye(784).unsqueeze(0).expand(10, 784, 784).contiguous()
    ev = eval_rubbish_rbf(rbf, n=200, dim=784, seed=0)
    assert ev.error_rate == 0.0, f"neg-def RBF far from rubbish should give 0% error, got {ev.error_rate}"


def test_eval_clean_confidence_rbf_in_unit_range_for_neg_def():
    """For a neg-def beta, exp(q) <= 1, so clean confidence in [0,1]."""
    rbf = RBFNet(n_classes=10, in_dim=784)
    with torch.no_grad():
        rbf.beta.data = -0.01 * torch.eye(784).unsqueeze(0).expand(10, 784, 784).contiguous()
    x = torch.rand(16, 784)
    c = eval_clean_confidence_rbf(rbf, x)
    assert 0.0 <= c <= 1.0


# ---------------- Fix 2: sigmoid-top is TRAINED (per-class BCE) --------- #
def test_sigmoid_top_cost_is_per_class_bce():
    """sigmoid_top_cost = mean_b sum_k BCE(sigmoid(logit_k), onehot_k)."""
    torch.manual_seed(0)
    m = SigmoidTopMLP(units=8, pieces=2, in_dim=784, n_classes=10,
                     dropout_input_include=1.0, dropout_hidden_include=1.0, seed=0)
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    got = sigmoid_top_cost(m, x, y)
    with torch.no_grad():
        logits = m.logits(x)
        target = F.one_hot(y, 10).float()
        expected = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    assert torch.allclose(got, expected, atol=1e-6)


def test_sigmoid_top_trains_and_predicts():
    """SigmoidTopMLP trains via train(cost='sigmoid_top') and predicts with
    argmax (== argmax sigmoid). Confirms the trained sigmoid-top path runs and
    is not a frozen swap."""
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
    m = SigmoidTopMLP(units=16, pieces=2, in_dim=784, n_classes=10,
                      dropout_input_include=1.0, dropout_hidden_include=1.0, seed=0)
    res = train(m, TrainConfig(batch_size=32, lr=0.1, momentum=0.5, max_epochs=5,
                               seed=0, cost="sigmoid_top", patience=100,
                               early_stop="clean", max_steps=10), data)
    assert res.steps_run == 10 and len(res.best_state_dict) > 0
    m.eval()
    with torch.no_grad():
        logits = m.logits(data.x_test)
        # argmax of pre-sigmoid logits == argmax of sigmoid probs.
        assert torch.equal(logits.argmax(dim=1), torch.sigmoid(logits).argmax(dim=1))


def test_sigmoid_top_not_a_frozen_softmax_swap():
    """The trained sigmoid-top readout must DIFFER from a fresh softmax net's
    readout (it is trained, not copied). Two independent trainings under the
    same seed but different costs produce different readout weights."""
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
    sm = MaxoutMLP(units=16, pieces=2, dropout_input_include=1.0,
                   dropout_hidden_include=1.0, seed=0)
    train(sm, TrainConfig(batch_size=32, lr=0.1, momentum=0.5, max_epochs=5,
                          seed=0, cost="softmax", patience=100,
                          early_stop="clean", max_steps=10), data)
    torch.manual_seed(0)
    st = SigmoidTopMLP(units=16, pieces=2, in_dim=784, n_classes=10,
                      dropout_input_include=1.0, dropout_hidden_include=1.0, seed=0)
    train(st, TrainConfig(batch_size=32, lr=0.1, momentum=0.5, max_epochs=5,
                          seed=0, cost="sigmoid_top", patience=100,
                          early_stop="clean", max_steps=10), data)
    assert not torch.equal(sm.readout.weight, st.readout.weight), (
        "sigmoid-top readout == softmax readout (frozen swap not trained)")


def test_sigmoid_top_recipe_applied():
    """The external max_col_norm recipe applies to SigmoidTopMLP too (so the M9
    comparison trains identically to the maxout+softmax net). After training,
    every output unit's incoming-weight norm is <= 1.9365 on both maxout layers
    and the readout."""
    from fgsm_repro.train import _MAX_COL_NORM
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
    st = SigmoidTopMLP(units=16, pieces=2, in_dim=784, n_classes=10,
                      dropout_input_include=1.0, dropout_hidden_include=1.0, seed=0)
    train(st, TrainConfig(batch_size=32, lr=0.1, momentum=0.5, max_epochs=3,
                          seed=0, cost="sigmoid_top", patience=100,
                          early_stop="clean", max_steps=20), data)
    layers, rw = _maxout_layers_and_readout(st)
    for layer in layers:
        assert layer.W.norm(dim=0).max().item() <= _MAX_COL_NORM + 1e-4
    assert rw.norm(dim=1).max().item() <= _MAX_COL_NORM + 1e-4


# ---------------- Fix 3: ensemble prediction metric ------------------- #
def test_ensemble_prediction_is_mean_prob_argmax():
    """The E1 ensemble prediction = argmax(mean of members' softmax probs)."""
    from experiments.e1_ensemble import _ensemble_predict  # noqa: PLC0415
    models = [MaxoutMLP(units=8, pieces=2, dropout_input_include=1.0,
                        dropout_hidden_include=1.0, seed=i) for i in range(3)]
    x = torch.rand(10, 784)
    pred = _ensemble_predict(models, x)
    with torch.no_grad():
        probs = sum(F.softmax(m.logits(x), dim=1) for m in models) / 3
        expected = probs.argmax(dim=1)
    assert torch.equal(pred, expected)


# ---------------- Fix 4: M7 noise-only (no clean mixture) -------------- #
def test_noise_train_cost_is_noise_only():
    """noise_train_cost returns ONLY the noisy loss (no alpha*clean term).
    With a zero-noise eta the noisy loss == clean loss; with a non-zero eta the
    returned value equals J(x+eta) exactly (no 0.5*J(x) mixture)."""
    torch.manual_seed(0)
    m = MaxoutMLP(units=8, pieces=2, dropout_input_include=1.0,
                  dropout_hidden_include=1.0, seed=0)
    x = torch.rand(16, 784)
    y = torch.randint(0, 10, (16,))
    gen = torch.Generator().manual_seed(1)
    from fgsm_repro.objectives import cross_entropy_cost
    # uniform noise with eps=0 -> eta=0 -> noisy loss == clean loss.
    loss0 = noise_train_cost(m, x, y, 0.0, "uniform", gen=gen)
    clean = cross_entropy_cost(m, x, y)
    assert torch.allclose(loss0, clean, atol=1e-6), (loss0, clean)
    # bernoulli eps=0 -> eta=0 -> same.
    loss0b = noise_train_cost(m, x, y, 0.0, "bernoulli", gen=gen)
    assert torch.allclose(loss0b, clean, atol=1e-6)


def test_noise_train_cost_no_clean_mixture_term():
    """For non-zero eps the noise-only loss must equal J(x+eta) and NOT
    0.5*J(x)+0.5*J(x+eta). We use a hand-built CONFIDENT linear model
    (SoftmaxRegression with a per-class diagonal weight) on data where the
    label is the index of the single bright pixel, so the clean loss is ~0 and
    a sizeable perturbation makes the noisy loss large -- guaranteeing
    clean != noisy so the mixture is distinguishable from the noise-only loss.
    """
    from fgsm_repro.models import SoftmaxRegression
    from fgsm_repro.objectives import cross_entropy_cost
    # Data: one bright pixel whose index IS the label (label in 0..9).
    g = torch.Generator().manual_seed(3)
    n, Fdim, K = 8, 784, 10
    x = torch.zeros(n, Fdim, dtype=torch.float32)
    y = torch.randint(0, K, (n,), generator=g, dtype=torch.int64)
    x[torch.arange(n), y] = 1.0
    # Confident linear classifier: logit_k = w * x[k] (diagonal), so argmax = y.
    m = SoftmaxRegression(in_dim=Fdim, n_classes=K)
    with torch.no_grad():
        m.linear.weight.zero_()
        for k in range(K):
            m.linear.weight[k, k] = 10.0
        m.linear.bias.zero_()
    clean = cross_entropy_cost(m, x.detach(), y)
    assert clean.item() < 1e-3, clean.item()  # confident -> near-zero clean loss

    gen = torch.Generator().manual_seed(7)
    loss = noise_train_cost(m, x, y, 3.0, "uniform", gen=gen)
    gen2 = torch.Generator().manual_seed(7)
    eta = 3.0 * (2.0 * torch.rand(x.shape, generator=gen2, dtype=torch.float32) - 1.0)
    x_noisy = (x.detach() + eta).detach()
    expected = cross_entropy_cost(m, x_noisy, y)
    mixture = 0.5 * clean + 0.5 * expected
    # Noise-only: loss == J(x+eta) (within thread-nondeterminism tolerance).
    assert torch.allclose(loss, expected, atol=1e-5), (loss.item(), expected.item())
    # Large noise flips many predictions -> noisy loss >> near-zero clean loss.
    assert expected.item() > clean.item() + 0.5, (clean.item(), expected.item())
    # So the noise-only loss must NOT equal the 0.5 clean/noisy mixture.
    assert not torch.allclose(loss, mixture, atol=1e-3), "noise cost still a 0.5 mixture"


# ---------------- Fix 5: L1 weight-decay penalty ---------------------- #
def test_l1_penalty_is_coeff_times_l1_of_first_layer():
    """l1_first_layer_penalty == coeff * ||layer0.W||_1 (sum of abs)."""
    torch.manual_seed(0)
    m = MaxoutMLP(units=8, pieces=2, dropout_input_include=1.0,
                  dropout_hidden_include=1.0, seed=0)
    for coeff in (0.0, 1e-3, 0.0025):
        pen = l1_first_layer_penalty(m, coeff)
        if coeff == 0.0:
            assert pen.item() == 0.0
        else:
            expected = coeff * m.layer0.W.abs().sum()
            assert torch.allclose(pen, expected, atol=1e-7)
    # Only the FIRST layer is penalized (not layer1 or readout).
    pen = l1_first_layer_penalty(m, 1.0)
    assert torch.allclose(pen, m.layer0.W.abs().sum())


def test_l1_penalty_increases_loss_and_zero_for_non_maxout():
    """Adding the L1 penalty increases the total loss; for a model with no
    first maxout layer it returns 0 (graceful)."""
    torch.manual_seed(0)
    m = MaxoutMLP(units=8, pieces=2, dropout_input_include=1.0,
                  dropout_hidden_include=1.0, seed=0)
    from fgsm_repro.objectives import cross_entropy_cost
    x = torch.rand(16, 784)
    y = torch.randint(0, 10, (16,))
    base = cross_entropy_cost(m, x, y).item()
    pen = l1_first_layer_penalty(m, 0.0025).item()
    assert pen > 0 and base + pen > base
    # Non-maxout model (RBF) -> zero penalty.
    rbf = RBFNet(10, 784)
    assert l1_first_layer_penalty(rbf, 0.0025).item() == 0.0


def test_l1_first_layer_coeff_trains():
    """train() with l1_first_layer_coeff runs and applies the recipe (max_col_norm
    still enforced on the L1 arm)."""
    from fgsm_repro.train import _MAX_COL_NORM
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
    m = MaxoutMLP(units=16, pieces=2, dropout_input_include=1.0,
                  dropout_hidden_include=1.0, seed=0)
    res = train(m, TrainConfig(batch_size=32, lr=0.1, momentum=0.5, max_epochs=3,
                               seed=0, l1_first_layer_coeff=0.0025, patience=100,
                               early_stop="clean", max_steps=10), data)
    assert res.steps_run == 10
    assert m.layer0.W.norm(dim=0).max().item() <= _MAX_COL_NORM + 1e-4
