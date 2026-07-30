"""Degeneracy tests: the method at its no-op setting must reproduce the baseline
EXACTLY.

Per the research-code skill: "Almost every method has a setting where it reduces
to something you already have... In that setting the method must reproduce the
baseline exactly."  These are the cheapest real correctness evidence — they let
a reader verify without trusting us.

Three degeneracies for this paper:

1. **Lewis geometry at p=2 == naive Euclidean geometry.**  The block-Lewis
   geometry matrix is ``M = A^T W^{1-2/p} A`` (paper/body.tex:511).  At the
   no-op setting p=2 the exponent ``1-2/p = 0`` so ``W^0 = I`` and ``M = A^T A``
   — EXACTLY the naive Euclidean geometry (E15, paper/body.tex:599).  The
   paper's data-dependent geometry reduces to the baseline geometry bit-for-bit.

2. **Regularized objective at coef=0 == unregularized smoothed objective.**  The
   T2 regularized surrogate is ``f_hat = f_tilde + coef * ||sqrt_w A(x-x0)||^2``
   (paper/body.tex:182).  At the no-op setting coef=0 the regularizer vanishes
   and ``f_hat == f_tilde`` in value, gradient, AND Hessian.  The flagship
   ball-oracle's regularized variant reduces to the plain smoothed variant.

3. **Smoothed surrogate -> max-norm as (beta, delta) -> 0.**  The smoothed
   surrogate f_tilde approximates the group-infinity norm max_i ||r_i||_2
   (theory convention, paper/body.tex:27) within ``beta*log(m) + delta`` (E8,
   paper/body.tex:231).  As the smoothing vanishes, f_tilde -> max_i ||r_i||
   exactly; the method's surrogate reduces to the nonsmooth objective it
   smooths.
"""

from __future__ import annotations

import numpy as np
import pytest

from gdr.types import GroupProblem
from gdr.lewis import block_lewis_weights, geometry_matrix
from gdr.objectives import make_smoothed, make_regularized


def _toy(seed: int = 0, m: int = 5, d: int = 3, n_per: int = 7) -> GroupProblem:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m * n_per, d))
    b = rng.standard_normal(m * n_per)
    gid = np.repeat(np.arange(m, dtype=np.int32), n_per)
    return GroupProblem(A, b, gid)


# --- Degeneracy 1: Lewis p=2 == naive geometry -------------------------------
def test_lewis_geometry_p2_equals_naive_identity():
    """At p=2, W^{1-2/p}=W^0=I, so the Lewis geometry M = A^T A == naive (E15).

    This is the degeneracy: the paper's data-dependent block-Lewis geometry at
    its no-op setting (p=2, where the group norm is plain l2) reduces EXACTLY to
    the naive Euclidean geometry the baselines use.
    """
    p = _toy(seed=1)
    w = block_lewis_weights(p, p=2, n_iters=4)            # any weights
    M_lewis = geometry_matrix(p, w, p=2)                  # A^T W^{0} A = A^T A
    M_naive = geometry_matrix(p, w=None, p=None)          # A^T A (E15)
    assert M_lewis.shape == (p.d, p.d)
    np.testing.assert_allclose(M_lewis, M_naive, atol=1e-12, rtol=0,
                               err_msg="Lewis p=2 must equal naive A^T A exactly")


def test_lewis_geometry_p2_independent_of_weights():
    """At p=2 the weights drop out entirely — different w give the same M."""
    p = _toy(seed=2)
    w1 = block_lewis_weights(p, p=2, n_iters=3)
    w2 = np.full(p.m, 0.5)                                 # arbitrary positive weights
    M1 = geometry_matrix(p, w1, p=2)
    M2 = geometry_matrix(p, w2, p=2)
    np.testing.assert_allclose(M1, M2, atol=1e-12, rtol=0)


# --- Degeneracy 2: regularized coef=0 == smoothed --------------------------
def test_regularized_coef0_equals_smoothed():
    """f_hat with coef=0 must equal f_tilde in value, grad, hess (T2, body.tex:182).

    The flagship ball-oracle's regularized objective at its no-op setting
    (regularizer switched off) reproduces the plain smoothed surrogate EXACTLY.
    """
    p = _toy(seed=3)
    beta, delta = 0.1, 0.05
    sm = make_smoothed(p, beta, delta)
    x0 = p.erm()
    sqrt_w = np.ones(p.n)                                   # naive geometry weights
    reg0 = make_regularized(p, beta, delta, sqrt_w, x0, coef=0.0)
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = x0 + rng.standard_normal(p.d) * 0.3
        np.testing.assert_allclose(reg0.value(x), sm.value(x), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(reg0.grad(x), sm.grad(x), rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(reg0.hess(x), sm.hess(x), rtol=1e-10, atol=1e-12)


# --- Degeneracy 3: smoothed -> max-norm as (beta,delta) -> 0 ----------------
def test_smoothed_converges_to_group_infinity_norm():
    """f_tilde -> max_i ||r~_i||_2 as (beta, delta) -> 0 (E8, body.tex:231).

    The smoothed surrogate reduces to the nonsmooth max-norm objective it
    smooths.  We check the approximation error is bounded by beta*log(m)+delta
    and shrinks to 0 with the smoothing.
    """
    from gdr.objectives import Smoothed, _Folded
    p = _toy(seed=4, m=8)
    folded = _Folded(p)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(p.d)
    # group-infinity norm in the folded (theory) convention: max_i ||r~_i||
    norms = np.sqrt(folded.group_norm_sq(x))               # [m] = ||r~_i||
    ginf = float(np.max(norms))
    errs = []
    for beta, delta in [(1.0, 1.0), (0.1, 0.1), (0.01, 0.01), (1e-3, 1e-3)]:
        sm = Smoothed(folded, beta, delta)
        ft = sm.value(x)
        bound = beta * np.log(p.m) + delta                  # E8
        assert abs(ft - ginf) <= bound + 1e-9, (beta, delta, abs(ft - ginf), bound)
        errs.append(abs(ft - ginf))
    # error must be (non-strictly) decreasing as smoothing shrinks
    assert errs[-1] < errs[0] + 1e-12, errs
    assert errs[-1] < 1e-3, f"smoothing did not converge to max-norm: {errs}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-x"]))
