"""Reference robust optimum via the CVXPY epigraph QCQP (SPEC E20, experiments.tex:40-50).

    min_{x in R^d, t in R}  t   s.t.  ell_i(x) = ||A_{S_i} x - b_{S_i}||^2 <= t   for all i

(on the *folded* data, so ell_i == (1/n_i)||orig residual||^2).  The returned
epigraph value t* is OPT; all plots report F(x) - OPT  (experiments.tex:52).
"""
from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
    _HAVE_CVXPY = True
except Exception:  # pragma: no cover
    _HAVE_CVXPY = False

from .problem import Problem


def solve_opt(problem: Problem, solver: str = "CLARABEL") -> tuple[np.ndarray, float]:
    """Solve the epigraph QCQP; return (x_star [d], OPT).  Tries CLARABEL then SCS."""
    if not _HAVE_CVXPY:
        raise RuntimeError("cvxpy not installed")
    A = problem["A"]
    b = problem["b"]
    off = problem["offsets"]
    d = problem["d"]
    x = cp.Variable(d)
    t = cp.Variable()
    cons = []
    for i in range(problem["m"]):
        Ai = A[off[i]:off[i + 1], :]
        bi = b[off[i]:off[i + 1]]
        cons.append(cp.sum_squares(Ai @ x - bi) <= t)
    prob = cp.Problem(cp.Minimize(t), cons)
    last_err = None
    for sv in [solver, "CLARABEL", "SCS", "ECOS", "OSQP", "SCIPY"]:
        try:
            val = prob.solve(solver=getattr(cp, sv) if hasattr(cp, sv) else sv)
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
        if x.value is None or val is None or not np.isfinite(val):
            last_err = f"solver {sv} returned no solution"
            continue
        return np.asarray(x.value, dtype=np.float64), float(val)
    raise RuntimeError(f"all solvers failed; last error: {last_err}")
