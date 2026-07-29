"""Shared pytest fixtures for the gdr reproduction tests."""
from __future__ import annotations

import numpy as np
import pytest

from gdr.data_synthetic import make_synthetic
from gdr.reference import solve_opt


@pytest.fixture(scope="session")
def small_problem():
    """A tiny folded GDR problem (d=4, m=8, 2 adversarial) for fast tests."""
    return make_synthetic(d=4, m=8, n_adv=2, n_per_group=12, seed=3,
                          E_ADV=1e3, DIST=3.0, E_LO=0.1, E_HI=1.0)


@pytest.fixture(scope="session")
def small_problem_opt(small_problem):
    x_star, opt = solve_opt(small_problem)
    return x_star, float(opt)


@pytest.fixture(scope="session")
def reset_problem():
    """A problem with m small enough that the E11 reset fires (sum_i w_i >= m).

    ||w||_1 <= 2(d+1); with d=4, m=4 we have 2(d+1)=10 >= 4 so the reset fires,
    making the Lewis geometry collapse to the Euclidean one (degeneracy test).
    """
    return make_synthetic(d=4, m=4, n_adv=1, n_per_group=10, seed=7,
                          E_ADV=1e2, DIST=2.0, E_LO=0.1, E_HI=1.0)
