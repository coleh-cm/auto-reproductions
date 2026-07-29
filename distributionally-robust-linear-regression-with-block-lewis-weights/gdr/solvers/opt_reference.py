"""OPT reference arm — the CVXPY epigraph QP optimum (E20).

This arm is not an optimizer; it returns the exact robust optimum from
``gdr.reference.solve_opt`` (E20, experiments.tex:40-50) so the OPT curve is a
flat line at gap 0.  ``cfg['opt']`` may be absent; in that case the optimum is
computed here and used as the gap reference (the returned gap is exactly 0.0).
"""

from __future__ import annotations

import time
import numpy as np

from gdr.solvers import get_opt


def run(problem, cfg, x0, max_outer, time_budget):
    """Return the exact optimum as a one-point history at iter 0, gap 0.0."""
    cfg = cfg or {}
    t0 = time.perf_counter()
    from gdr.reference import solve_opt  # sibling unit (frozen interface)

    x_star, opt = solve_opt(problem, solver=cfg.get("solver", "clarabel"))
    x_star = np.asarray(x_star, dtype=np.float64)  # [d]
    opt = float(opt)
    # Make sure cfg carries opt for any downstream consumer that reads it.
    cfg["opt"] = opt
    t = time.perf_counter() - t0
    return {
        "iter": [0],
        "gap": [0.0],          # F(x_star) - opt == 0 by definition
        "time": [float(t)],
        "x": x_star,
    }


# Expose get_opt for convenience (some runners import it from the arm module).
__all__ = ["run", "get_opt"]
