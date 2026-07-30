"""Invariant tests: facts that must hold because the paper's equations say so.

Each test cites the equation it checks (SPEC section 5).  These are seconds-cheap
checks that catch the errors that survive "it runs".
"""

import numpy as np
import pytest

from gdr.types import GroupProblem
from gdr import objectives as obj, lewis, metrics, solvers


def _problem(seed=0, m=6, d=4, ni=10):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m * ni, d))
    b = rng.standard_normal(m * ni)
    gid = np.repeat(np.arange(m, dtype=np.int32), ni)
    return GroupProblem(A, b, gid)


# ---- E10: smoothed gradient/Hessian match the chain-rule formulas ----------
def test_smoothed_grad_matches_numerical():
    p = _problem(seed=1)
    sm = obj.make_smoothed(p, beta=0.7, delta=0.4)
    x = np.random.default_rng(2).standard_normal(p.d)
    g = sm.grad(x)
    fd = np.zeros(p.d)
    eps = 1e-6
    for i in range(p.d):
        xp = x.copy(); xp[i] += eps; xm = x.copy(); xm[i] -= eps
        fd[i] = (sm.value(xp) - sm.value(xm)) / (2 * eps)
    assert np.max(np.abs(g - fd)) < 1e-6


def test_smoothed_hess_matches_numerical_and_symmetric():
    p = _problem(seed=1)
    sm = obj.make_smoothed(p, beta=0.7, delta=0.4)
    x = np.random.default_rng(3).standard_normal(p.d)
    H = sm.hess(x)
    assert np.allclose(H, H.T, atol=1e-10), "Hessian must be symmetric"
    eps = 1e-5
    fdH = np.zeros((p.d, p.d))
    for j in range(p.d):
        xp = x.copy(); xp[j] += eps; xm = x.copy(); xm[j] -= eps
        fdH[j] = (sm.grad(xp) - sm.grad(xm)) / (2 * eps)
    assert np.max(np.abs(H - fdH)) < 1e-4


# ---- E8: |f~ - sqrt(F)| <= beta log m + delta  (paper/body.tex:231) ----------
def test_smoothed_approximation_bound():
    p = _problem(seed=4, m=8)
    beta, delta = 0.3, 0.2
    sm = obj.make_smoothed(p, beta, delta)
    rng = np.random.default_rng(5)
    for _ in range(20):
        x = rng.standard_normal(p.d) * 3
        f_tilde = sm.value(x)
        sqF = np.sqrt(p.worst_loss(x))      # sqrt(F) -- the folded-norm the surrogate tracks
        assert abs(f_tilde - sqF) <= beta * np.log(p.m) + delta + 1e-9


# ---- E11: softmax weights sum to 1, are nonnegative -------------------------
def test_softmax_sums_to_one():
    from gdr.objectives import _softmax
    rng = np.random.default_rng(6)
    s = rng.standard_normal(20) * 5
    sigma = _softmax(s, beta=0.5)
    assert sigma.shape == (20,)
    assert np.all(sigma >= 0)
    assert abs(sigma.sum() - 1.0) < 1e-12


# ---- E12/E13: block Lewis ellipsoid sandwich  (paper/body.tex:170-172) -------
def test_lewis_ellipsoid_sandwich():
    p = _problem(seed=7, m=6, d=4, ni=10)
    w = lewis.block_lewis_weights(p, p=None)
    Wdiag = lewis.expand_weights(w, p)
    rng = np.random.default_rng(8)
    for _ in range(15):
        x = rng.standard_normal(p.d) * 2
        c = rng.standard_normal()
        r = p.A @ x - c * p.b
        g_inf = max(np.linalg.norm(r[s:e]) for s, e in p.slices)
        Wr = Wdiag * r
        lower = g_inf
        mid = np.linalg.norm(Wr)
        upper = np.sqrt(2 * (p.d + 1)) * g_inf
        assert lower <= mid + 1e-9, (lower, mid)
        assert mid <= upper + 1e-9, (mid, upper)


# ---- leverage scores: in [0,1]-ish, sum = rank  (paper/other_proofs.tex:11) --
def test_leverage_scores_sum_to_rank():
    p = _problem(seed=9, m=4, d=4, ni=8)
    Ahat = p.augmented()
    tau = lewis.leverage_scores(Ahat)
    assert np.all(tau >= -1e-9)
    rank = int(np.linalg.matrix_rank(Ahat))
    assert abs(tau.sum() - rank) < 1e-6, (tau.sum(), rank)


# ---- E14: weighted-LS init satisfies the normal equations -------------------
def test_weighted_ls_init_is_stationary():
    p = _problem(seed=10, m=5, d=4, ni=12)
    w = lewis.block_lewis_weights(p, p=None)
    w = lewis.reset_to_identity(w, p.m)
    x0 = lewis.weighted_ls_init(p, w, p=None)
    # gradient of || W^{1/2}(A x - b) ||^2 at x0 should be ~0
    Wdiag = lewis.expand_weights(w, p)
    g = 2 * (p.A * Wdiag[:, None]).T @ (p.A @ x0 - p.b)
    assert np.max(np.abs(g)) < 1e-6, np.max(np.abs(g))


# ---- E1/E2/E3: group losses / worst / ERM match definitions -----------------
def test_group_losses_and_erm_definitions():
    p = _problem(seed=11, m=5, d=3, ni=9)
    x = np.random.default_rng(12).standard_normal(p.d)
    losses = p.group_losses(x)
    manual = np.array([np.sum((p.A[s:e] @ x - p.b[s:e]) ** 2) / (e - s)
                       for s, e in p.slices])
    assert np.allclose(losses, manual)
    assert p.worst_loss(x) == pytest.approx(losses.max())
    # ERM is the group-averaged minimizer -> its (weighted) normal-equation residual ~0
    xe = p.erm()
    w = 1.0 / p.sizes.astype(float)
    Wdiag = np.repeat(w, p.sizes.astype(np.int64))
    g = (p.A * Wdiag[:, None]).T @ (p.A @ xe - p.b)
    assert np.max(np.abs(g)) < 1e-6


# ---- E5: gap curve is best-so-far (monotone non-increasing) -----------------
def test_gap_curve_best_so_far():
    worst = np.array([10.0, 5.0, 7.0, 3.0, 4.0])   # non-monotone
    opt = 1.0
    curve = metrics.gap_curve(worst, opt)
    assert np.allclose(curve, [9.0, 4.0, 4.0, 2.0, 2.0])  # running min of gaps


# ---- E6: cost_to_rel_gap known-correct + known-wrong ------------------------
def test_cost_to_rel_gap_known_correct():
    # gap0 = 9, opt = 1, 1% of gap0 = 0.09 -> reach when gap<=0.09 i.e. worst<=1.09
    worst = np.array([10.0, 5.0, 1.05, 1.0])
    curve = metrics.gap_curve(worst, 1.0)
    t = metrics.cost_to_rel_gap(curve, gap0=9.0, rel=0.01, base="init")
    assert t == 2   # gap first <=0.09 at index 2 (gap=0.05)


def test_cost_to_rel_gap_known_wrong_not_reached():
    worst = np.array([10.0, 9.5, 9.0])
    curve = metrics.gap_curve(worst, 1.0)
    t = metrics.cost_to_rel_gap(curve, gap0=9.0, rel=0.01, base="init")
    assert t is None   # never reaches 1%


def test_cost_to_rel_gap_refuses_empty():
    with pytest.raises(ValueError):
        metrics.cost_to_rel_gap(np.array([]), 1.0, 0.01, base="init")


# ---- no success path may report OK on an empty result -----------------------
def test_reference_optimum_returns_finite_on_valid_problem():
    # a valid problem must yield a finite OPT >= the minimum group-loss floor;
    # the reference arm never returns a silent placeholder.
    p = _problem(seed=20, m=5, d=3, ni=9)
    xstar, opt = solvers.reference_optimum(p)
    assert np.isfinite(opt)
    assert p.worst_loss(xstar) == pytest.approx(opt, rel=1e-4)


def test_arms_refuse_zero_budget():
    p = _problem()
    x0 = p.erm()
    with pytest.raises(ValueError):
        solvers.solve_subgradient(p, x0, {"step": 1e-3, "schedule": "fixed"}, 0)
    with pytest.raises(ValueError):
        solvers.solve_smooth(p, x0, {"method": "gd", "beta": 1.0, "delta": 1.0,
                                    "step": 1e-3}, 0)
    with pytest.raises(ValueError):
        solvers.solve_ball_oracle(p, x0, {"geometry": "naive", "radius0": 1.0,
                                         "beta": 0.2, "delta": 0.1}, 0)
