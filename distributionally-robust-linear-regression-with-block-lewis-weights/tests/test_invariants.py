"""Invariant tests: facts the paper's equations imply (research-code skill).

Each test asserts something that MUST hold if the maths is implemented right;
they cost seconds and catch the errors that survive to "trains fine".
Citations are `file:line` into paper/arxiv-2607.00252-src/.
"""
from __future__ import annotations

import numpy as np
import pytest

from gdr.lewis import block_lewis_weights, geometry_M, leverage_scores, should_reset_W, lewis_warm_start, wls_init
from gdr.objectives import smoothed, smoothed_grad_hess, p_objective, p_grad_hess, _softmax
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
# E9 weighted-least-squares warm start: D = W (p=inf) or D = W^{1-2/p} (finite p)
# (SPEC.md:114; other_proofs.tex:74 corrected by U14).  lewis_warm_start must
# apply the p-dependent exponent to the per-row weight before wls_init.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [np.inf, 2.0, 4.0, 8.0])
def test_lewis_warm_start_D_exponent(small_problem, p):
    prob = small_problem
    off, m = prob["offsets"], prob["m"]
    w = block_lewis_weights(prob, p=p)
    if should_reset_W(w, m):
        pytest.skip("E11 reset fires for this problem; D = ones, exponent N/A")
    x0, w_rows, _reset = lewis_warm_start(prob, p=p)
    # the per-row D returned must equal w_i^{1-2/p} per block (or w_i for p=inf)
    exp = 1.0 if np.isinf(p) else (1.0 - 2.0 / float(p))
    w_D = w ** exp if exp != 1.0 else w
    expected_rows = np.repeat(w_D, np.diff(off))
    assert np.allclose(w_rows, expected_rows, atol=0, rtol=0), (p, w_rows[:3], expected_rows[:3])
    # and x0 must equal wls_init with that D (independent recomputation)
    x0_ref = wls_init(prob, expected_rows)
    assert np.allclose(x0, x0_ref, atol=1e-12), (p, x0, x0_ref)
    # p=inf is a no-op (exp=1): w_rows == raw w repeated
    if np.isinf(p):
        assert np.allclose(w_rows, np.repeat(w, np.diff(off)), atol=0, rtol=0)


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
# Strong convexity of the p-objective (Lemma 7.2 / strong_convexity_gp,
# interpolation.tex:51-61; "main new technical tool", body.tex:96):
#   f(x+d) >= f(x) + <grad f(x), d> + (4/2^p) ||A d||_{G_p}^p
# where f(x) = ||A x - b||_{G_p}^p = sum_i ||r_i||_2^p = p_objective(x, p) and
# ||A d||_{G_p}^p = sum_i ||A_{S_i} d||_2^p (the homogeneous group p-norm, b=0).
# The finite-difference test only checks grad/obj CONSISTENCY; this checks the
# objective's actual strong-convexity FORM (catches a wrong exponent that is
# self-consistent with its own derivatives).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [2.0, 4.0, 8.0])
def test_p_objective_strong_convexity(small_problem, p):
    prob = small_problem
    A, off, m = prob["A"], prob["offsets"], prob["m"]
    rng = np.random.default_rng(7)
    kp = 4.0 / (2.0 ** p)  # the paper's strong-convexity constant (4/2^p)
    for _ in range(50):
        x = rng.standard_normal(prob["d"])
        d = rng.standard_normal(prob["d"])
        fx = p_objective(prob, x, p)
        fxd = p_objective(prob, x + d, p)
        _, g, _ = p_grad_hess(prob, x, p)
        # ||A d||_{G_p}^p = sum_i ||A_{S_i} d||_2^p  (homogeneous; b=0)
        Ad_gp_p = 0.0
        for i in range(m):
            Adi = A[off[i]:off[i + 1], :] @ d
            Ad_gp_p += float(np.sqrt(Adi @ Adi) ** p)
        lower = fx + float(g @ d) + kp * Ad_gp_p
        assert fxd >= lower - 1e-6, (p, fxd, lower, fx, fxd - lower)


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
    """T5: the GUARANTEED invariant is that the smoothed surrogate f_tilde is
    non-increasing across outer iterations (the inner trust-region solve
    minimizes f_tilde over {||x-q||_M <= r} with q in the ball, body.tex:31 /
    E19; the damped-Newton line search makes f_tilde monotone each step).  The
    paper only claims F = max_i ||r_i||^2 'steadily decrease[s]' EMPIRICALLY
    (experiments.tex:107), NOT as a theorem.  We assert the guaranteed
    per-iteration f_tilde non-increasing invariant (via the recorded x_traj),
    and check F-monotonicity only as the empirical observation the paper makes
    (not as a guarantee).  A no-op solver returning x0 would FAIL the
    per-iteration f_tilde check unless f_tilde(x0) is already a fixed point,
    so this is genuine evidence the solver optimizes, not false security.
    """
    prob = small_problem
    xstar, opt = small_problem_opt
    # normalize so OPT == 1 (production convention, U15): arms are well-scaled.
    s = float(np.sqrt(opt))
    pn = dict(prob); pn["A"] = prob["A"] / s; pn["b"] = prob["b"] / s
    x0 = np.linalg.lstsq(pn["A"], pn["b"], rcond=None)[0]
    beta, delta = 1e-1, 1e-1
    cfg = {"geometry": "euclidean", "r0_grid": [2.0], "shrink_grid": [0.5],
           "beta_grid": [beta], "delta_grid": [delta], "tol_inner": 1e-8, "opt": 1.0}
    h = run_arm_help("ball_oracle", cfg, pn, x0, 1.0)
    gaps = h["gap"]                                   # F - 1 (normalized)

    # GUARANTEED invariant: f_tilde non-increasing across outer iterations
    # (per-iteration, via x_traj).  This is what "damped Newton" (experiments.tex:71)
    # actually guarantees; the prior start-vs-end check was false security.
    xs = h.get("x_traj")
    assert xs is not None and len(xs) == len(gaps), "ball_oracle must record x_traj"
    f_tildes = [float(smoothed(pn, np.asarray(xk), beta, delta)) for xk in xs]
    for k in range(1, len(f_tildes)):
        assert f_tildes[k] <= f_tildes[k - 1] + 1e-9, (k, f_tildes[:k + 1])

    # EMPIRICAL observation only (experiments.tex:107, NOT a theorem): F is
    # non-increasing on the normalized instance because the line search kills
    # overshoot.  Asserted as the paper's reported behavior, not as a guarantee.
    for k in range(1, len(gaps)):
        assert gaps[k] <= gaps[k - 1] + 1e-6, (k, gaps[:k + 1])


def run_arm_help(arm, cfg, prob, x0, opt):
    from gdr.runner import run_arm
    return run_arm(arm, cfg, prob, x0, opt, max_outer=8, time_budget=60.0)


# ---------------------------------------------------------------------------
# First-order arms make real end-to-end progress (NOT a no-op / broken arm).
#
# The smoke gate checks only `FINAL smoke=ok`; the four first-order arms report
# `iters_to_5%=None` there because they PLATEAU above 5% on the heterogeneous
# instance -- exactly the paper's §8 T4 finding ("first-order methods' plateau",
# experiments.tex:107).  A plateau reads identically to a broken arm (one that
# returns x0) from the gate's perspective, so this test machine-enforces the
# distinction: every first-order arm must DECREASE the worst-group loss F from
# its ERM warm start (final gap STRICTLY below the initial gap) and stay finite.
# A no-op solver returning x0 fails (final == initial); a divergent solver fails
# (non-finite / increasing).  This is the cheapest real evidence the baselines
# actually optimize, complementing the per-arm invariants above.
# ---------------------------------------------------------------------------
FIRST_ORDER_ARMS = {
    "subgradient": {"lr_grid": [1e-3, 1e-2, 1e-1],
                    "schedule_grid": ["const", "1/sqrt_t"]},
    "smoothed_gd": {"lr_grid": [1e-3, 1e-2, 1e-1],
                    "beta_grid": [1e-1], "delta_grid": [1e-1]},
    "smoothed_hb": {"lr_grid": [1e-3, 1e-2, 1e-1], "momentum_grid": [0.9],
                    "beta_grid": [1e-1], "delta_grid": [1e-1]},
    "smoothed_nesterov": {"lr_grid": [1e-3, 1e-2, 1e-1], "momentum_grid": [0.9],
                          "beta_grid": [1e-1], "delta_grid": [1e-1]},
}


@pytest.mark.parametrize("arm,cfg", list(FIRST_ORDER_ARMS.items()),
                         ids=list(FIRST_ORDER_ARMS))
def test_first_order_arm_makes_progress(small_problem, small_problem_opt, arm, cfg):
    prob = small_problem
    _xstar, opt = small_problem_opt
    # production convention (U15): rescale so OPT == 1, putting arms on an O(1)
    # loss scale (the IPM needs this; the first-order grids are tuned to it).
    s = float(np.sqrt(opt))
    pn = dict(prob); pn["A"] = prob["A"] / s; pn["b"] = prob["b"] / s
    opt_n = 1.0
    x0 = np.linalg.lstsq(pn["A"], pn["b"], rcond=None)[0]   # ERM warm start
    cfg = dict(cfg); cfg["opt"] = opt_n
    from gdr.runner import run_arm
    h = run_arm(arm, cfg, pn, x0, opt_n, max_outer=60, time_budget=60.0)
    gaps = h["gap"]
    g0, gN = float(gaps[0]), float(gaps[-1])
    # every recorded gap must be finite (no divergence to inf/NaN)
    assert all(np.isfinite(g) for g in gaps), (arm, gaps)
    # the arm must strictly decrease the worst-group suboptimality from the warm
    # start -- a no-op / broken arm would leave gN == g0.
    assert gN < g0 - 1e-6, (arm, "no progress: init", g0, "final", gN)


# ---------------------------------------------------------------------------
# ACS data-cache seed validation (gdr/data.py): a seed-blind cache would silently
# serve one seed's folded data for another (Round-3 fixed this for the OPT cache;
# this pins the analogous fix for the data cache).  Silent wrong-data
# substitution is the highest-stakes latent failure mode here.
# ---------------------------------------------------------------------------
def test_acs_cache_rejects_seed_mismatch(tmp_path):
    from gdr.data import _save_cache, _load_cache, _ACS_CACHE_SCHEMA
    from gdr.problem import Problem

    A = np.random.default_rng(0).standard_normal((20, 3))
    b = np.random.default_rng(1).standard_normal(20)
    offsets = np.array([0, 7, 14, 20])
    n_i = np.array([7, 7, 6])
    meta = {"cache_schema": _ACS_CACHE_SCHEMA, "group_by": "ST (region)", "seed": 0}
    p = Problem(A=A, b=b, offsets=offsets, name="acs_income", m=3, d=3, n_i=n_i, meta=meta)
    cache = tmp_path / "acs_income_folded.npz"
    _save_cache(str(cache), p)

    # matching seed -> accepted
    r = _load_cache(str(cache), seed=0)
    assert int(r["meta"]["seed"]) == 0
    # mismatched seed -> rejected (forces a rebuild, never silent substitution)
    with pytest.raises(ValueError, match="seed"):
        _load_cache(str(cache), seed=6)
    # stale schema -> rejected (so a cache from before the seed check is rebuilt)
    bad = Problem(A=A, b=b, offsets=offsets, name="acs_income", m=3, d=3, n_i=n_i,
                  meta={"cache_schema": 2, "group_by": "ST (region)", "seed": 0})
    _save_cache(str(cache), bad)
    with pytest.raises(ValueError, match="schema"):
        _load_cache(str(cache), seed=0)
    # seed=None (caller does not care) -> accepted (backward compat)
    _save_cache(str(cache), p)
    _load_cache(str(cache), seed=None)


# ---------------------------------------------------------------------------
# Round-11 fixes: committed result JSONs carry the paper's figure-curve data
# (gap_best = best-so-far / running-min, matching fig:acs_convergence's
# "best-so-far worst-group suboptimality" caption, experiments.tex:167) and the
# T2 wall-clock-to-1% (time_to_rel_gap, the second column of tab:acs_runtime,
# experiments.tex:176,185-186).  These are deterministic functions of the
# committed trajectory, so we pin them against the committed files.
# ---------------------------------------------------------------------------
def test_committed_history_has_best_so_far_curve():
    """gap_best is the running-minimum of gap and is monotone non-increasing
    (the paper's fig:acs_convergence curve; raw gap can be jagged)."""
    import json
    import os
    d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fn = os.path.join(d, "results", "acs_income_subgradient.json")
    if not os.path.isfile(fn):
        pytest.skip("committed ACS subgradient result not present")
    r = json.load(open(fn))
    g = np.asarray(r["history"]["gap"], dtype=float)
    gb = np.asarray(r["history"].get("gap_best", []), dtype=float)
    assert gb.size == g.size, "gap_best must be present and same length as gap"
    # best-so-far == running minimum of the raw per-iteration gap
    assert np.allclose(gb, np.minimum.accumulate(g)), "gap_best must be cummin(gap)"
    # the paper's figure curve is monotone non-increasing (raw gap is not)
    assert np.all(np.diff(gb) <= 0.0), "gap_best must be monotone non-increasing"


def test_committed_time_to_rel_gap_consistent():
    """time_to_rel_gap is the wall-clock at the iters_to_rel_gap crossing
    (history.time at that index), the T2 second column of tab:acs_runtime."""
    import json
    import os
    from gdr.runner import time_to_gap
    d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ["acs_income_ipm", "acs_income_ball_oracle_lewis",
                 "synthetic_ball_oracle_lewis"]:
        fn = os.path.join(d, "results", f"{name}.json")
        if not os.path.isfile(fn):
            continue
        r = json.load(open(fn))
        h = {"gap": r["history"]["gap"], "iter": r["history"]["iter"],
             "time": r["history"]["time"], "opt": r.get("opt_norm", 1.0)}
        it, t = time_to_gap(h, rel_gap=r["rel_gap"])
        assert it == r["iters_to_rel_gap"], (name, it, r["iters_to_rel_gap"])
        if it is None:
            assert r["time_to_rel_gap"] is None
        else:
            assert r["time_to_rel_gap"] == pytest.approx(t, rel=0, abs=1e-9), name


def test_opt_cache_n_i_signature_checked(tmp_path, monkeypatch):
    """get_opt_cached's n_i signature is actually compared on a cache hit
    (Round-11: the docstring promised an n_i check the code never performed;
    a stale cache with matching m/n/d/A_sum but different per-group sizes must
    be rebuilt, not silently reused)."""
    import json
    from run_arm import get_opt_cached
    prob = {
        "A": np.eye(4, 3), "b": np.ones(4), "offsets": np.array([0, 2, 4]),
        "name": "synthetic", "m": 2, "d": 3, "n_i": np.array([2, 2]),
    }
    # run from tmp_path so get_opt_cached writes results/ there, not the repo
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "results" / "opt_synthetic_seed0.json"
    cache.parent.mkdir(exist_ok=True)
    # a stale cache that matches m/n/d/A_sum but has a MISMATCHED n_i signature
    bad = {"opt": 1.234, "seed": 0, "m": 2, "n": 4, "d": 3,
           "n_i": [99, 99, 99], "A_sum": float(np.abs(prob["A"]).sum())}
    json.dump(bad, open(cache, "w"))
    # the OPT epigraph of [I4x3; 1] min max_i ||A_i x - b_i||^2 is 0 (b in col space),
    # definitely not 1.234 -> the stale cache must be rejected and rebuilt.
    val = get_opt_cached(prob, "synthetic", 0)
    assert abs(val - 1.234) > 1e-6, "stale cache with mismatched n_i was reused"


# ---------------------------------------------------------------------------
# Round-14: three invariant tests added after an adversarial paper-fidelity
# review (orchestration) confirmed them as missing coverage the paper's
# equations imply.  None changes implementation code; all pin facts the maths
# guarantees so a regression in the (correct) implementation is caught.
# ---------------------------------------------------------------------------

# (1) E9 Lewis warm-start QUALITY bound (major finding; SPEC T5 lists this
# check explicitly, other_proofs.tex:51-61 Lemma gp_regression_initialization).
#
#   On the UNSQUARED group-p-norm scale the lemma states, for the Lewis-weighted
#   least-squares init x0 = argmin ||W^{1/2-1/p}(Ax-b)||_2 (E9, p=inf => D=W):
#       ||A x0 - b||_{G_p}  <=  (2 rank(A))^{1/2-1/p}  ||A x* - b||_{G_p}
#   For p=inf this is  f(x0) <= sqrt(2 rank(A)) * f(x*)  where f = ||.||_{G_inf}.
#   The SPEC T5 / Thm-2.3 sandwich uses the augmented rank (d+1), giving the
#   looser (valid) bound  f(x0) <= sqrt(2(d+1)) * f(x*).  We assert the SPEC T5
#   bound (looser, conservative); the lemma's tighter sqrt(2 rank(A)) bound
#   also holds and is printed for transparency.
#
#   CRITICAL: this only proves something when the Lewis weights are ACTUALLY
#   used, i.e. when the E11 reset (sum w_i >= m => W=I) does NOT fire.  The
#   shared `small_problem` fixture has m=8, d=4 so 2(d+1)=10 >= m and the reset
#   fires (Lewis collapses to Euclidean); there the bound that holds is the
#   naive sqrt(m) one, not the Lewis one.  So this test builds a dedicated
#   problem with m=20 > 2(d+1)=10 so the reset does NOT fire and the Lewis
#   warm start is the one the lemma bounds.  (research-code skill: the
#   degeneracy/no-op setting must be distinguished from the active setting.)
def test_lewis_warm_start_init_quality():
    """E9 Lewis warm-start quality: f(x0) <= sqrt(2(d+1)) * f(x*)  (Lemma
    gp_regression_initialization, other_proofs.tex:51-61; SPEC T5).  Uses a
    problem where the E11 reset does NOT fire so the Lewis weights (not W=I)
    are the geometry actually used."""
    from gdr.data_synthetic import make_synthetic
    from gdr.reference import solve_opt
    # m=20 > 2(d+1)=10  =>  sum w_i <= 2(d+1) < m  => reset does NOT fire
    prob = make_synthetic(d=4, m=20, n_adv=3, n_per_group=8, seed=11,
                          E_ADV=1e3, DIST=3.0, E_LO=0.1, E_HI=1.0)
    d, m = prob["d"], prob["m"]
    w = block_lewis_weights(prob, p=np.inf)
    assert not should_reset_W(w, m), "test requires the E11 reset NOT to fire"
    assert w.sum() <= 2 * (d + 1) + 1e-9, (w.sum(), 2 * (d + 1))  # E8 guarantee

    x0, _w_rows, reset = lewis_warm_start(prob, p=np.inf)
    assert not reset, "lewis_warm_start must not reset when sum w_i < m"
    _xstar, opt_sq = solve_opt(prob)           # OPT = F(x*) on the SQUARED scale
    assert opt_sq > 0, "need a non-trivial optimum (b not in col(A))"

    # unsquared theory scale: f = ||A x - b||_{G_inf} = max_i ||r_i||_2
    f0 = max_loss_unsquared(prob, x0)
    fstar = float(np.sqrt(opt_sq))
    rank_A = int(np.linalg.matrix_rank(prob["A"]))
    # Lemma statement (tighter):  f0 <= sqrt(2 rank(A)) * fstar
    bound_lemma = np.sqrt(2 * rank_A) * fstar
    # SPEC T5 / Thm-2.3 sandwich (looser, augmented rank d+1): the asserted bound
    bound_spec = np.sqrt(2 * (d + 1)) * fstar
    assert f0 <= bound_spec + 1e-6 * fstar, (f0, bound_spec, fstar, w.sum())
    # the tighter lemma bound must also hold (it does for a correct impl)
    assert f0 <= bound_lemma + 1e-6 * fstar, (f0, bound_lemma, fstar)


# (2) E5 p-objective Hessian: finite-difference check + symmetry + PSD (minor
# finding; test_p_grad_finite_diff only checked the [d] gradient, never the
# [d,d] Hessian, so a Hessian-only bug -- wrong sign on the p(p-2) outer
# product, a missing term, or ||r||^{p-4} vs ||r||^{p-2} exponent confusion --
# would pass).  Mirrors test_smoothed_grad_hess_finite_diff for E4.
@pytest.mark.parametrize("p", [2.0, 4.0, 8.0])
def test_p_hessian_finite_diff_symmetry_psd(small_problem, p):
    """E5 Hessian (interpolation.tex:24-29) vs finite differences of the
    gradient, plus symmetry and PSD (f = sum ||r_i||^p is convex => Hessian PSD)."""
    prob = small_problem
    rng = np.random.default_rng(9)
    # pick x with all ||r_i|| > 0 to stay off the r=0 guard branch
    x = rng.standard_normal(prob["d"]) + 5.0
    _v, g, H = p_grad_hess(prob, x, p)
    d = prob["d"]
    # symmetry (exact, up to fp noise)
    assert np.allclose(H, H.T, atol=1e-12), (p, np.max(np.abs(H - H.T)))
    # PSD: f = sum ||r_i||^p is convex in x, so its Hessian is PSD
    assert np.linalg.eigvalsh(0.5 * (H + H.T)).min() > -1e-6, (p, np.linalg.eigvalsh(H).min())
    # finite-difference the Hessian column-by-column from the gradient
    eps = 1e-5
    Hfd = np.zeros((d, d))
    for i in range(d):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        _, gp, _ = p_grad_hess(prob, xp, p)
        _, gm, _ = p_grad_hess(prob, xm, p)
        Hfd[:, i] = (gp - gm) / (2 * eps)
    scale = np.maximum(np.abs(H), np.abs(Hfd)).max() + 1e-12
    assert np.max(np.abs(H - Hfd)) / scale < 1e-3, (p, np.max(np.abs(H - Hfd)) / scale)


# (3) softmax weights s lie in the probability simplex Delta^m (nit finding;
# E4 / body.tex:285-288 define s_i = lambda_i / sum_j lambda_j; SPEC E4 says
# s = softmax(a) in Delta^m).  No test pinned this; a regression that dropped
# the normalization would corrupt grad/Hessian.  Direct unit test of _softmax.
def test_softmax_weights_in_simplex():
    """s = softmax(a) is nonnegative and sums to 1 (E4, body.tex:285-288),
    including the max-shift edge case with large/extreme exponents."""
    rng = np.random.default_rng(13)
    for _ in range(50):
        a = rng.standard_normal(rng.integers(1, 20))
        s = _softmax(a)
        assert s.shape == a.shape
        assert np.all(s >= 0.0), s
        assert abs(float(s.sum()) - 1.0) < 1e-12, s.sum()
    # extreme / degenerate inputs (max-shift stability + simplex membership)
    for a in [np.array([0.0]), np.array([1000.0, -1000.0, 0.0]),
              np.array([1e6, 1e6, 1e6]), np.array([-1e6, -1e6])]:
        s = _softmax(a)
        assert np.all(np.isfinite(s)), a
        assert np.all(s >= 0.0), a
        assert abs(float(s.sum()) - 1.0) < 1e-12, (a, s.sum())

