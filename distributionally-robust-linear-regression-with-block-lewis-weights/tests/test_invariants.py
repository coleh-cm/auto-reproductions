"""Invariant tests: facts the paper's equations imply (research-code skill).

Each test asserts something that MUST hold if the maths is implemented right;
they cost seconds and catch the errors that survive to "trains fine".
Citations are `file:line` into paper/arxiv-2607.00252-src/.
"""
from __future__ import annotations

import numpy as np
import pytest

from gdr.lewis import block_lewis_weights, geometry_M, leverage_scores, should_reset_W
from gdr.objectives import smoothed, smoothed_grad_hess, p_objective, p_grad_hess
from gdr.problem import group_losses, group_norms, max_loss, max_loss_unsquared


# ---------------------------------------------------------------------------
# Lemma 6.1 (body.tex:230-231): |f_tilde - f| <= beta log m + delta
# ---------------------------------------------------------------------------
def test_smoothed_approximation_bound(small_problem):
    prob = small_problem
    m = prob["m"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal(prob["d"])
        f = max_loss_unsquared(prob, x)              # ||Ax-b||_{G_inf}
        for beta, delta in [(0.5, 0.5), (0.2, 0.1), (1.0, 0.3), (0.05, 0.05)]:
            ft = smoothed(prob, x, beta, delta)
            assert abs(ft - f) <= beta * np.log(m) + delta + 1e-9, (ft, f, beta, delta)


# ---------------------------------------------------------------------------
# smoothed surrogate gradient & Hessian vs finite differences (E4)
# ---------------------------------------------------------------------------
def test_smoothed_grad_hess_finite_diff(small_problem):
    prob = small_problem
    rng = np.random.default_rng(1)
    x = rng.standard_normal(prob["d"])
    beta, delta = 0.3, 0.2
    v, g, H = smoothed_grad_hess(prob, x, beta, delta)
    assert abs(v - smoothed(prob, x, beta, delta)) < 1e-9
    eps = 1e-6
    gfd = np.zeros(prob["d"])
    for i in range(prob["d"]):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        gfd[i] = (smoothed(prob, xp, beta, delta) - smoothed(prob, xm, beta, delta)) / (2 * eps)
    assert np.max(np.abs(g - gfd)) < 1e-5, np.max(np.abs(g - gfd))
    Hfd = np.zeros((prob["d"], prob["d"]))
    for i in range(prob["d"]):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        _, gp, _ = smoothed_grad_hess(prob, xp, beta, delta)
        _, gm, _ = smoothed_grad_hess(prob, xm, beta, delta)
        Hfd[:, i] = (gp - gm) / (2 * eps)
    assert np.max(np.abs(H - Hfd)) < 1e-4, np.max(np.abs(H - Hfd))


def test_smoothed_hessian_is_psd(small_problem):
    """lse is convex => its Hessian is PSD (the -(1/beta) grad grad^T term is
    exactly compensated by the softmax-curvature term, E4)."""
    prob = small_problem
    rng = np.random.default_rng(2)
    for _ in range(10):
        x = rng.standard_normal(prob["d"])
        _v, _g, H = smoothed_grad_hess(prob, x, beta=0.2, delta=0.1)
        assert np.linalg.eigvalsh(H).min() > -1e-7


# ---------------------------------------------------------------------------
# E6 block-Lewis overestimate + ||w||_1 <= 2(d+1)  (SPEC E8; other_proofs.tex)
# ---------------------------------------------------------------------------
def test_block_lewis_overestimate(small_problem):
    prob = small_problem
    d = prob["d"]
    w = block_lewis_weights(prob, p=np.inf)
    # ||w||_1 <= 2(d+1)  (SPEC E8; MO25 Lemma 5.6 gives 1.5(d+1) <= 2(d+1))
    assert w.sum() <= 2 * (d + 1) + 1e-6, w.sum()
    # overestimate condition (E6) on the augmented matrix A_hat=[A|b]:
    #   (sum_{j in S_i} tau_j(W^{1/2} A_hat)) / w_i <= 1  for all i
    A = prob["A"]; b = prob["b"]; off = prob["offsets"]
    Ahat = np.column_stack([A, b])
    w_rows = np.repeat(w, np.diff(off))
    tau = leverage_scores(np.sqrt(w_rows)[:, None] * Ahat)  # W^{1/2} A_hat
    block_tau = np.array([tau[off[i]:off[i + 1]].sum() for i in range(prob["m"])])
    assert (block_tau / w).max() <= 1.0 + 1e-6, (block_tau / w).max()


# ---------------------------------------------------------------------------
# E7 residual sandwich (Thm 3.3 / Thm 2.3):
#   ||Ax-cb||_{G_inf} <= ||W^{1/2}(Ax-cb)||_2 <= sqrt(2(d+1)) ||Ax-cb||_{G_inf}
# ---------------------------------------------------------------------------
def test_e7_residual_sandwich(small_problem):
    prob = small_problem
    d = prob["d"]
    w = block_lewis_weights(prob, p=np.inf)
    A = prob["A"]; b = prob["b"]; off = prob["offsets"]
    w_rows = np.repeat(w, np.diff(off))
    rng = np.random.default_rng(5)
    for _ in range(100):
        x = rng.standard_normal(d)
        c = rng.standard_normal()
        r = A @ x - c * b
        blk = np.array([np.sqrt(np.sum(r[off[i]:off[i + 1]] ** 2)) for i in range(prob["m"])])
        gn = blk.max()                                   # ||Ax-cb||_{G_inf}
        lhs = np.sqrt(np.sum(w_rows * r ** 2))            # ||W^{1/2}(Ax-cb)||_2
        assert gn - 1e-9 <= lhs <= np.sqrt(2 * (d + 1)) * gn + 1e-9, (lhs, gn)


# ---------------------------------------------------------------------------
# p-objective (E5) gradient vs finite differences
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [2.0, 4.0, 8.0])
def test_p_grad_finite_diff(small_problem, p):
    prob = small_problem
    rng = np.random.default_rng(4)
    x = rng.standard_normal(prob["d"])
    _v, g, _H = p_grad_hess(prob, x, p)
    eps = 1e-6
    gfd = np.zeros(prob["d"])
    for i in range(prob["d"]):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        gfd[i] = (p_objective(prob, xp, p) - p_objective(prob, xm, p)) / (2 * eps)
    # relative tolerance: gradients can be large (adversarial curvature), so
    # compare against the gradient magnitude, not an absolute floor.
    scale = np.maximum(np.abs(g), np.abs(gfd)).max() + 1e-12
    assert np.max(np.abs(g - gfd)) / scale < 1e-4, (p, np.max(np.abs(g - gfd)), scale)


# ---------------------------------------------------------------------------
# subgradient is a valid subgradient of F at the argmax group (E22)
# ---------------------------------------------------------------------------
def test_subgradient_validity(small_problem):
    """For F(x)=max_i ||r_i||^2, the subgradient 2 A_{S_{i*}}^T r_{i*} at the
    argmax group i* must satisfy the supporting-hyperplane inequality
    F(y) >= F(x) + <g, y-x> for random y (E22, experiments.tex:57-58)."""
    prob = small_problem
    from gdr.solvers.subgradient import _subgradient
    rng = np.random.default_rng(6)
    for _ in range(20):
        x = rng.standard_normal(prob["d"])
        g = _subgradient(prob, x)
        Fx = max_loss(prob, x)
        for _ in range(5):
            y = x + 0.05 * rng.standard_normal(prob["d"])
            Fy = max_loss(prob, y)
            assert Fy >= Fx + g @ (y - x) - 1e-6, (Fy, Fx + g @ (y - x))


# ---------------------------------------------------------------------------
# T5: ball-oracle outer iterates are non-increasing in F (trust region cannot
# increase the smoothed objective, and the center update keeps the better point)
# ---------------------------------------------------------------------------
def test_ball_oracle_monotone(small_problem, small_problem_opt):
    prob = small_problem
    _xstar, opt = small_problem_opt
    x0 = np.linalg.lstsq(prob["A"], prob["b"], rcond=None)[0]
    cfg = {"geometry": "euclidean", "r0_grid": [10.0], "shrink_grid": [0.5],
           "beta_grid": [1e-1], "delta_grid": [1e-1], "tol_inner": 1e-8, "opt": opt}
    h = run_arm_help("ball_oracle", cfg, prob, x0, opt)
    gaps = h["gap"]
    # gaps (F - opt) must be non-increasing across outer iterations
    for k in range(1, len(gaps)):
        assert gaps[k] <= gaps[k - 1] + 1e-9, (k, gaps[:k + 1])


def run_arm_help(arm, cfg, prob, x0, opt):
    from gdr.runner import run_arm
    return run_arm(arm, cfg, prob, x0, opt, max_outer=8, time_budget=60.0)
