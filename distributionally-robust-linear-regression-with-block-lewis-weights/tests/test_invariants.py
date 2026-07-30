"""Invariant tests: assertions that follow directly from the paper's equations.

These come *out of* the paper's maths, cost seconds, and catch the errors that
survive to "runs fine, number is a bit off" (research-code skill).
"""

from __future__ import annotations

import numpy as np
import pytest

from gdr.types import GroupProblem
from gdr.objectives import make_smoothed, Smoothed, _Folded, _softmax
from gdr.lewis import block_lewis_weights
from gdr.solvers import reference_optimum
from gdr.metrics import gap_curve, cost_to_rel_gap


def _toy(seed=0, m=6, d=3, n_per=5):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m * n_per, d))
    b = rng.standard_normal(m * n_per)
    gid = np.repeat(np.arange(m, dtype=np.int32), n_per)
    return GroupProblem(A, b, gid)


# --- E1/E2/E3: group loss, worst loss, ERM --------------------------------
def test_erm_minimizes_average_group_loss():
    """E3: x_ERM = argmin (1/m) sum_i ell_i.  A perturbation must not decrease it."""
    p = _toy(seed=1)
    e = p.erm()
    base = float(p.group_losses(e).mean())
    rng = np.random.default_rng(5)
    for _ in range(20):
        d = rng.standard_normal(p.d) * 1e-6
        assert p.group_losses(e + d).mean() >= base - 1e-9


def test_worst_loss_is_max_of_group_losses():
    """E2: F(x) = max_i ell_i(x) == problem.worst_loss."""
    p = _toy(seed=2)
    rng = np.random.default_rng(8)
    for _ in range(10):
        x = rng.standard_normal(p.d)
        np.testing.assert_allclose(p.worst_loss(x), np.max(p.group_losses(x)), rtol=1e-12)


def test_group_loss_is_mean_squared_residual():
    """E1: ell_i = (1/n_i)||A_{S_i}x - b_{S_i}||^2."""
    p = _toy(seed=3)
    rng = np.random.default_rng(9)
    x = rng.standard_normal(p.d)
    gl = p.group_losses(x)
    for i, (s, e) in enumerate(p.slices):
        ri = p.A[s:e] @ x - p.b[s:e]
        np.testing.assert_allclose(gl[i], (ri @ ri) / (e - s), rtol=1e-12)


# --- E4: reference optimum is a lower bound --------------------------------
def test_reference_optimum_le_erm_worst_loss():
    """E4: OPT = min_x max_i ell_i <= max_i ell_i(x_ERM) = F(ERM)."""
    p = _toy(seed=4)
    xstar, opt = reference_optimum(p)
    f_erm = p.worst_loss(p.erm())
    assert opt <= f_erm + 1e-6, (opt, f_erm)
    # and the returned x* achieves (approx) OPT
    assert abs(p.worst_loss(xstar) - opt) <= 1e-4 * max(1.0, abs(opt))


# --- E7/E10: smoothed surrogate calculus invariants -----------------------
def test_softmax_weights_sum_to_one():
    """sigma = softmax(s/beta) is a probability distribution (body.tex:285)."""
    p = _toy(seed=5)
    folded = _Folded(p)
    sm = Smoothed(folded, beta=0.1, delta=0.05)
    rng = np.random.default_rng(12)
    for _ in range(5):
        x = rng.standard_normal(p.d)
        _, _, h, s, _, _ = sm._inner(x)
        sigma = _softmax(s, 0.1)
        assert sigma.shape == (p.m,)
        assert np.all(sigma >= -1e-12)
        np.testing.assert_allclose(sigma.sum(), 1.0, atol=1e-12)


def test_smoothed_grad_matches_finite_difference():
    """E10: grad f_tilde is the analytic gradient (body.tex:287)."""
    p = _toy(seed=6)
    sm = make_smoothed(p, beta=0.1, delta=0.05)
    rng = np.random.default_rng(13)
    x = rng.standard_normal(p.d)
    g = sm.grad(x)
    fd = np.zeros(p.d)
    eps = 1e-6
    for k in range(p.d):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        fd[k] = (sm.value(xp) - sm.value(xm)) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


def test_smoothed_hess_symmetric():
    """Hessian of a scalar function is symmetric."""
    p = _toy(seed=7)
    sm = make_smoothed(p, beta=0.1, delta=0.05)
    rng = np.random.default_rng(14)
    x = rng.standard_normal(p.d)
    H = sm.hess(x)
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_smoothed_approx_error_bound_e8():
    """E8: |f_tilde - max_i||r~_i||| <= beta*log(m) + delta (body.tex:231)."""
    p = _toy(seed=8, m=10)
    folded = _Folded(p)
    beta, delta = 0.1, 0.05
    sm = Smoothed(folded, beta, delta)
    rng = np.random.default_rng(15)
    bound = beta * np.log(p.m) + delta
    for _ in range(10):
        x = rng.standard_normal(p.d)
        ginf = float(np.max(np.sqrt(folded.group_norm_sq(x))))
        assert abs(sm.value(x) - ginf) <= bound + 1e-9


# --- E5: gap curve is best-so-far (non-increasing) -------------------------
def _fake_history(losses):
    class H:
        pass
    h = H()
    h.worst_losses = list(losses)
    return h


def test_gap_curve_nonincreasing():
    """E5: best-so-far gap is non-increasing (experiments.tex:52)."""
    losses = [10.0, 8.0, 9.0, 5.0, 6.0, 4.0]   # not monotone, but best-so-far is
    curve = gap_curve(losses, opt=3.0)
    assert len(curve) == len(losses)
    diffs = np.diff(curve)
    assert np.all(diffs <= 1e-12), f"gap curve not non-increasing: {curve}"


def test_gap_curve_raises_on_bad_opt():
    """A large negative gap signals OPT was computed wrong — must raise, not hide."""
    losses = [10.0, 8.0]
    with pytest.raises(ValueError):
        gap_curve(losses, opt=100.0)   # gap = 8 - 100 = -92 << -1e-3


# --- Lewis overestimate condition (Definition 3.2) ------------------------
def test_lewis_weights_are_overestimates():
    """Defn 3.2: sum_{j in S_i} tau_j(W^{1/2-1/p} A_hat) / w_i <= 1 (other_proofs.tex:20)."""
    from gdr.lewis import leverage_scores
    p = _toy(seed=9, m=6)
    w = block_lewis_weights(p, p=None, n_iters=6)          # p=None -> inf
    # leverage scores of W^{1/2} A_hat (p=inf exponent 1/2-1/p = 1/2).
    # leverage_scores(A, Wdiag) returns tau_j(W^{1/2} A) with Wdiag = per-row w.
    ahat = p.augmented()
    w_rows = np.repeat(w, p.sizes.astype(np.int64))        # broadcast w_i to rows
    tau = leverage_scores(ahat, Wdiag=w_rows)
    for i, (s, e) in enumerate(p.slices):
        ratio = tau[s:e].sum() / w[i]
        assert ratio <= 1.01 + 1e-9, (i, ratio)            # 1.01 tol: (3/2) overestimate


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-x"]))
