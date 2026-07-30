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

## What runs

`run_all_arms.sh` runs the paper's eight arms at full configuration on the
**ACS Income** instance (m=51, d=10, 200/region → n=10,200; real data from
census.gov via `folktables`) and prints one `FINAL <arm>=<value>` line per arm.
The value is the arm's iterations to 1% relative worst-group gap
(base = initial ERM gap, the scale-invariant reading of the unstated 1%-reference,
SPEC §6 item 4); the CVXPY reference arm prints OPT.

```bash
.venv/bin/bash run_all_arms.sh     # ACS, all 8 arms (downloads data on first use)
.venv/bin/bash smoke.sh           # tiny self-contained problem, one FINAL line
.venv/bin/pytest -q               # 20 tests: degeneracy, invariants, grader
```

`smoke.sh` proves the code path runs end to end; its number is **not** evidence
about the paper (never report it as a result).

### Reproduced results (this run, committed under `results/`)

Headline gate — **both ball-oracle arms reach 1% in a single outer iteration on
ACS, matching the paper's flagship claim** (`paper/experiments.tex:181-182`):

| arm | iters to 1% (run / paper) |
|---|---|
| ball_oracle_euclidean | 1 / 1 ✓ |
| ball_oracle_lewis | 1 / 1 ✓ |
| smoothed_heavy_ball | 34 / 47 (same order) |
| ipm | 22 / 8 (qual. ✓, exact count not matched) |
| subgradient | 58 / "not reached" (discrepancy — see REPRODUCTION.md) |

Statistical context (`paper/experiments.tex:189`): ERM avg 104.9 / σ 11.6 /
worst 135.1 (California); Max/Mean 1.29→1.02 (paper 1.28→1.02); California
loss decrease −26.2 (paper −24.3). The synthetic instance (κ(A^T A)=9.7e4,
5 adversarial groups) reproduces the qualitative picture: first-order methods
stall far above OPT, IPM converges rapidly to the best final loss, both ball
oracles decrease steadily to near-OPT. See `REPRODUCTION.md` for the full
numbers, discrepancies, and the unstated-parameters register.

## Layout

```
requirements.txt   full pinned dependency lock (Python 3.13)
Dockerfile         clean-base reproducible build of the same environment
README.md          this file
SPEC.md            method spec, equation citations, unstated-parameters register
arms.json          numbers-gate arm contract (8 arms)
REPRODUCTION.md    reproduction log (decisions, blockers, results, discrepancies)
run_all_arms.sh    gate entrypoint: all 8 arms on ACS, prints FINAL lines
smoke.sh           self-contained smoke (proves the path runs; not a result)
results/           committed run outputs (acs_all.json, synthetic_all.json, ...)
tests/             degeneracy + invariant + grader tests (20 tests)
paper/             authoritative arXiv LaTeX source (maths) + companion [MO25]
gdr/               implementation (see SPEC section 7 for the file contract)
  types.py         GroupProblem dataclass (E1/E2/E3)
  objectives.py    smoothed surrogate f~_{β,δ} (E7/E10), regularised fhat (T2)
  lewis.py         block Lewis weights (E12), geometry (E13/E15), W=I reset, E14
  solvers.py       8 arms: cvxpy ref, subgradient, smooth×3, ipm, ball_oracle×2
  metrics.py       gap curve (E5), cost-to-1%-gap (E6), statistical context
  data_synth.py    reconstructed synthetic instance (κ≈1e5, 5 adversarial)
  data_acs.py      folktables ACS Income (m=51, 200/region, target-scale match)
  harness.py       grid-search tuning + FINAL-line emission
```
