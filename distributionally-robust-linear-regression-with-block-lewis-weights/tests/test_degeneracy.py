"""Degeneracy test: the method at its no-op setting must reproduce the baseline EXACTLY.

The paper's own Algorithm 1 (``paper/body.tex:173-174``) carries a reset rule:
when the block-Lewis weight sum ``sum_i w_i >= m`` the geometry collapses to
``W = I_n``, i.e. ``M = A^T I A = A^T A`` -- the *naive Euclidean* geometry (E15,
``paper/body.tex:599-602``).  So the Lewis arm at the identity-weight setting is,
by construction, the Euclidean arm.  This is the cheapest real correctness check:
if the Lewis arm with forced identity weights does not bit-match the Euclidean
arm, the wiring is wrong and no full run can be trusted.

We assert bit-identical iterates (same M, same deterministic damped-Newton solver,
same surrogate -> identical trajectory).
"""

import numpy as np

from gdr.types import GroupProblem
from gdr import solvers, lewis


def _problem():
    rng = np.random.default_rng(123)
    m, d, ni = 5, 4, 12
    A = rng.standard_normal((m * ni, d))
    b = rng.standard_normal(m * ni)
    gid = np.repeat(np.arange(m, dtype=np.int32), ni)
    return GroupProblem(A, b, gid)


def test_geometry_identity_equals_naive():
    """geometry_matrix with w=ones (p=inf) == naive A^T A, exactly."""
    p = _problem()
    m = p.m
    M_naive = lewis.geometry_matrix(p, w=None)
    M_lewis_id = lewis.geometry_matrix(p, w=np.ones(m), p=None)
    assert np.array_equal(M_naive, M_lewis_id), "identity Lewis geometry must equal naive"


def test_reset_to_identity_when_sum_exceeds_m():
    """Algorithm 1 line 3: sum w >= m -> uniform ones."""
    p = _problem()
    w_big = np.full(p.m, 2.0)        # sum = 2m >= m
    assert np.array_equal(lewis.reset_to_identity(w_big, p.m), np.ones(p.m))
    w_small = np.full(p.m, 0.1)      # sum = 0.1m < m -> unchanged
    assert np.array_equal(lewis.reset_to_identity(w_small, p.m), w_small)


def test_ball_oracle_lewis_identity_matches_euclidean():
    """Flagship degeneracy: Lewis arm at W=I reproduces the Euclidean arm exactly."""
    p = _problem()
    x0 = p.erm()
    common = dict(radius0=10.0, decay=1.0, beta=0.2, delta=0.1,
                  reg_on=False, inner_iters=15)
    h_eu = solvers.solve_ball_oracle(
        p, x0, {**common, "geometry": "naive"}, 4)
    h_lw = solvers.solve_ball_oracle(
        p, x0, {**common, "geometry": "lewis",
                "lewis_p": None, "lewis_iters": None,
                "weights_override": np.ones(p.m)}, 4)
    assert len(h_eu.x) == len(h_lw.x), (len(h_eu.x), len(h_lw.x))
    for a, b in zip(h_eu.x, h_lw.x):
        assert np.allclose(a, b, atol=0, rtol=0), "Lewis@I must bit-match Euclidean"


def test_smoothed_at_warm_start_is_baseline():
    """The smoothed surrogate at a no-op smoothing is just a constant shift of the
    robust norm; its gradient direction is a convex combination of group gradients,
    and with a single dominating group it coincides with that group's gradient
    (E11, ``paper/body.tex:294`` ``grad f~ <= max_i grad s_i`` sanity bound)."""
    p = _problem()
    x = p.erm()
    sm = solvers.make_smoothed(p, beta=1e6, delta=1e6)   # near-uniform softmax
    g = sm.grad(x)
    # each grad_s_i = A~_i^T r~_i / h_i ; with uniform sigma, g = mean of grad_s_i
    from gdr.objectives import _Folded
    f = _Folded(p)
    r = f.residuals(x)
    gs = []
    for s, e in f.slices:
        ri = r[s:e]
        h = np.sqrt(1e12 + ri @ ri)
        gs.append(f.A[s:e].T @ ri / h)
    g_expect = np.mean(gs, axis=0)
    assert np.allclose(g, g_expect, atol=1e-4), (g[:3], g_expect[:3])
