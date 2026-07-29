"""Degeneracy test (research-code skill): the method at its no-op setting must
reproduce the baseline EXACTLY.  Two no-op settings for this paper:

  D1. Lewis ball-oracle at the E11 reset (sum_i w_i >= m  =>  W <- I) must be
      bit-identical to the Euclidean ball-oracle: same geometry M = A^T A, same
      warm start, same cfg => same iterates.  This is the cheapest real
      correctness evidence (the Lewis machinery collapses to the baseline).

  D2. The interpolating p-objective at p=2 reduces to plain least squares:
      f(x) = sum_i ||r_i||^2 = ||A x - b||^2, whose minimizer is the ERM
      solution (intro objective 1.1; body.tex:27).  Minimizing p_objective at
      p=2 must reproduce the ERM solution exactly.
"""
from __future__ import annotations

import numpy as np

from gdr.objectives import p_objective
from gdr.runner import run_arm


def test_lewis_reset_equals_euclidean(reset_problem):
    """D1: at the E11 reset, ball_oracle_lewis == ball_oracle_euclidean exactly."""
    prob = reset_problem
    x0 = np.linalg.lstsq(prob["A"], prob["b"], rcond=None)[0]
    cfg = {
        "r0_grid": [10.0], "shrink_grid": [0.5],
        "beta_grid": [1e-1], "delta_grid": [1e-1],
        "tol_inner": 1e-8, "opt": None, "max_outer": 5,
    }
    # confirm the reset actually fires for this problem
    from gdr.lewis import block_lewis_weights, should_reset_W
    w = block_lewis_weights(prob, p=np.inf)
    assert should_reset_W(w, prob["m"]), "test problem must trigger the E11 reset"

    cfg_e = dict(cfg); cfg_e["geometry"] = "euclidean"
    cfg_l = dict(cfg); cfg_l["geometry"] = "lewis"
    h_e = run_arm("ball_oracle", cfg_e, prob, x0, opt=0.0,
                  max_outer=5, time_budget=60.0)
    h_l = run_arm("ball_oracle", cfg_l, prob, x0, opt=0.0,
                  max_outer=5, time_budget=60.0)
    # bit-identical gaps (deterministic; same M, same x0, same cfg)
    assert h_e["gap"] == h_l["gap"], (h_e["gap"], h_l["gap"])
    assert np.allclose(h_e["x"], h_l["x"], atol=0, rtol=0)


def test_p2_objective_is_least_squares(small_problem):
    """D2: p=2 objective minimizer == ERM (plain least squares)."""
    prob = small_problem
    A, b = prob["A"], prob["b"]
    x_erm = np.linalg.lstsq(A, b, rcond=None)[0]
    # f(x) = sum_i ||r_i||^2 = ||Ax-b||^2  (body.tex:27, p=2)
    f_erm = p_objective(prob, x_erm, p=2.0)
    # the ERM solution is the global minimizer of ||Ax-b||^2; check a random
    # perturbation cannot beat it
    rng = np.random.default_rng(123)
    for _ in range(50):
        x = x_erm + 0.1 * rng.standard_normal(prob["d"])
        assert p_objective(prob, x, p=2.0) >= f_erm - 1e-9
    # gradient of the p=2 objective at ERM is (numerically) zero
    from gdr.objectives import p_grad_hess
    _v, g, _H = p_grad_hess(prob, x_erm, p=2.0)
    assert np.max(np.abs(g)) < 1e-8, np.max(np.abs(g))
