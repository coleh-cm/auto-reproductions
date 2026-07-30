# Distributionally Robust Linear Regression With Block Lewis Weights — reproduction

Reproduction of *Distributionally Robust Linear Regression With Block Lewis
Weights* (Naren Sarayu Manoj, Kumar Kshitij Patel, 2026; arXiv 2607.00252).

- **paper_ref:** `0a8cf406-bd9b-4ebe-8a3f-c9e3d62a2a94`
- **project_id:** `d7735ece-02c4-4228-985c-00834c92b8f3`

No upstream code exists for this paper (the arXiv bundle ships only LaTeX; the
experiments section references an "included Jupyter notebook" that is absent
from the bundle), so the method is implemented from scratch. The authoritative
LaTeX source for all maths lives in `paper/`; the method spec, equation
citations, and the unstated-parameters register live in `SPEC.md`; the
numbers-gate arm contract lives in `arms.json`.

## Scope

The checkable numbers are the paper's Section 8 empirical evaluation
(`paper/experiments.tex`): an adversarial synthetic regression instance
(d=10, m=100, 5 adversarial, κ(stacked Gram)≈10⁵) and the ACS Income task
(m=51 regions, d=10, 200/region → n=10,200), comparing seven optimizer arms
plus a CVXPY reference optimum. See `SPEC.md` for the full breakdown.

## Environment

Python 3.13 with pinned dependencies. The method needs:
`numpy`, `scipy`, `cvxpy` (with conic solvers), `pandas`, `folktables`,
`matplotlib`, plus `pytest` for the gate. Everything is pinned in
`requirements.txt` (a full lock, not just direct deps).

## Quickstart

### Option A — local venv with `uv` (what CI runs)

```bash
# from the repo root
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt

# prove the environment imports resolve and a conic solver loads
.venv/bin/python -c "import numpy, scipy, cvxpy, pandas, matplotlib, folktables, sklearn, requests; import cvxpy as cp; x=cp.Variable(2); p=cp.Problem(cp.Minimize(cp.sum_squares(x)),[x>=1]); assert abs(p.solve(solver=cp.CLARABEL)-2.0)<1e-6; print('env OK')"

# run the test gate
.venv/bin/pytest -q
```

`uv` is not required to be preinstalled only for the install: if it is missing,
install it first:

```bash
pip install uv==0.11.31
```

### Option B — Docker (reproduces from a clean base image)

```bash
docker build -t gdr-blw .
docker run --rm -it gdr-blw          # drops you in /repo with the venv active
# inside the container:
.venv/bin/python -c "import cvxpy, folktables; print('OK')"
.venv/bin/pytest -q
```

The Docker image is CPU-only (the problems are tiny: d=10), so no GPU is
needed. ACS data is downloaded on demand by `folktables` from census.gov and
cached under `data/`.

## Layout

```
requirements.txt   full pinned dependency lock (Python 3.13)
Dockerfile         clean-base reproducible build of the same environment
README.md          this file
SPEC.md            method spec, equation citations, unstated-parameters register
arms.json          numbers-gate arm contract (8 arms)
REPRODUCTION.md    reproduction log
paper/             authoritative arXiv LaTeX source (maths) + companion [MO25]
gdr/               implementation (see SPEC section 7 for the file contract)
```
