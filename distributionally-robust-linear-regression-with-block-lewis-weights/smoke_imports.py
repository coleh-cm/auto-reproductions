"""Environment smoke test for the reproduction of
"Distributionally Robust Linear Regression With Block Lewis Weights"
(Manoj & Patel, arXiv:2607.00252).

This script proves the pinned dependencies in ``requirements.txt`` resolve and
that the toolchain the paper's experiments rely on actually *works*, not just
imports:

  * numpy / scipy / cvxpy / clarabel / scs / pandas / matplotlib / folktables /
    scikit-learn / pytest all import;
  * CVXPY sees the convex solvers needed for the robust-optimum reference (E20,
    experiments.tex:40-50);
  * the folktables ACSIncome task exposes exactly the 10 features the paper uses
    (experiments.tex:148; SPEC D2);
  * the epigraph QCQP of E20 solves correctly with both clarabel and scs and
    matches an independent scipy minimizer on the same problem.

Run with the repro venv: ``.venv/bin/python smoke_imports.py``.
Exits nonzero on any failure; prints ``SMOKE OK`` on success.
"""

import sys

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import cvxpy as cp
import clarabel
import scs
import pandas
import matplotlib
import folktables
from folktables import ACSIncome, ACSDataSource
import sklearn
import pytest


def _check_solvers():
    solvers = cp.installed_solvers()
    for name in ("CLARABEL", "SCS"):
        if name not in solvers:
            raise SystemExit(f"missing CVXPY solver {name}; have {solvers}")
    return solvers


def _check_folktables_features():
    expected = [
        "AGEP", "COW", "SCHL", "MAR", "OCCP",
        "POBP", "RELP", "WKHP", "SEX", "RAC1P",
    ]
    got = list(ACSIncome.features)
    if got != expected:
        raise SystemExit(f"ACSIncome.features mismatch: {got} != {expected}")
    return got


def _check_epigraph_qcqp():
    """Solve the E20 epigraph QCQP with clarabel + scs, compare to scipy brute force."""
    rng = np.random.default_rng(0)
    m, d = 5, 3
    sizes = [4, 5, 6, 3, 5]
    As = [rng.standard_normal((n, d)) for n in sizes]
    bs = [
        As[i] @ rng.standard_normal(d) + 0.1 * rng.standard_normal(sizes[i])
        for i in range(m)
    ]
    # fold the 1/sqrt(n_i) factors into the data (paper convention, body.tex:27)
    As = [As[i] / np.sqrt(sizes[i]) for i in range(m)]
    bs = [bs[i] / np.sqrt(sizes[i]) for i in range(m)]

    x = cp.Variable(d)
    t = cp.Variable()
    cons = [cp.sum_squares(As[i] @ x - bs[i]) <= t for i in range(m)]
    prob = cp.Problem(cp.Minimize(t), cons)
    val_clarabel = prob.solve(solver=cp.CLARABEL)
    val_scs = prob.solve(solver=cp.SCS)

    from scipy.optimize import minimize

    def F(xx):
        return max(float(np.sum((As[i] @ xx - bs[i]) ** 2)) for i in range(m))

    r = minimize(
        F, np.zeros(d), method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000},
    )
    if not np.isclose(val_clarabel, r.fun, atol=1e-5):
        raise SystemExit(
            f"clarabel OPT {val_clarabel} != brute {r.fun}"
        )
    if not np.isclose(val_scs, r.fun, atol=1e-3):
        raise SystemExit(f"scs OPT {val_scs} != brute {r.fun}")
    return val_clarabel, val_scs, r.fun


def main():
    print(f"python {sys.version.split()[0]}")
    print(f"numpy {np.__version__}")
    print(f"scipy {scipy.__version__}")
    print(f"cvxpy {cp.__version__}")
    print(f"clarabel {clarabel.__version__}")
    print(f"scs {scs.__version__}")
    print(f"pandas {pandas.__version__}")
    print(f"matplotlib {matplotlib.__version__}")
    print(f"folktables {folktables.__version__}")
    print(f"scikit-learn {sklearn.__version__}")
    print(f"pytest {pytest.__version__}")

    solvers = _check_solvers()
    print(f"cvxpy installed_solvers: {solvers}")

    feats = _check_folktables_features()
    print(f"ACSIncome.features ({len(feats)}): {feats}")

    vc, vs, vb = _check_epigraph_qcqp()
    print(f"epigraph QCQP (E20): clarabel={vc:.8g} scs={vs:.8g} brute={vb:.8g}")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
