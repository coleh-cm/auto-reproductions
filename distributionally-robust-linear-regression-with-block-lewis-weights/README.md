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
(`paper/experiments.tex`): the ACS Income task (m=51 regions, d=10, 200/region
→ n=10,200) comparing seven optimizer arms plus a CVXPY reference optimum
(Table `tab:acs_runtime`), plus a reconstructed adversarial synthetic instance
(d=10, m=100, 5 adversarial, κ(stacked Gram)≈10⁵) for qualitative checks.

## What runs (and only what runs)

| command | what it does | time |
|---|---|---|
| `bash smoke.sh` | tiny self-contained problem, one `FINAL` line (plumbing check only) | seconds |
| `bash run_all_arms.sh acs` | the full ACS gate: 8 arms, one `FINAL <arm>=<value>` line each | ~40 s |
| `bash run_all_arms.sh synthetic` | the synthetic gate | ~12 s |
| `.venv/bin/python -m pytest -q` | 20 tests: degeneracy, equation invariants, grader | ~1 s |
| `.venv/bin/python -m gdr.harness context --instance acs` | the statistical-context vector | ~30 s |

Each arm prints exactly one line `FINAL <arm>=<value>` where `value` is
iterations to 1% relative worst-group gap (base=init), `not_reached`, or the
OPT value for the reference arm. Arm names match `arms.json`. Results JSON is
written to `results/` (committed, not gitignored).

## Headline results (ACS, seed 0)

| arm | measured | paper |
|---|---|---|
| ball_oracle_euclidean | 1 | 1 ✓ |
| ball_oracle_lewis | 1 | 1 ✓ (flagship) |
| smoothed_heavy_ball | 41 | 47 |
| ipm | 16 | 8 |
| subgradient | 58 (reached) | not reached |
| reference OPT | 110.3 | band 107-114 ✓ |

Statistical context: ERM avg 104.9 / std 11.6 / worst 135.1 (California);
robust Max/Mean 1.287→1.022 (paper 1.28→1.02 ✓); California loss decrease 26.2
(paper 24.3 ✓). See `REPRODUCTION.md` for the full discrepancy record.

## Environment

Python 3.13 with pinned dependencies in `requirements.txt` (a full lock).
`numpy`, `scipy`, `cvxpy` (CLARABEL/SCS conic solvers), `pandas`, `folktables`,
`matplotlib`, `pytest`. CPU-only (d=10 problems are tiny). ACS data is
downloaded on demand by `folktables` from census.gov and cached under `data/`
(gitignored; regenerable).

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv -r requirements.txt
.venv/bin/pytest -q                      # tests
bash run_all_arms.sh acs                  # the gate
```

## Layout

```
gdr/                implementation (see SPEC section 7 for the file contract)
  types.py           frozen GroupProblem data contract
  data_synth.py      reconstructed synthetic generator
  data_acs.py        folktables ACS Income loader
  objectives.py      E7/E10 smoothed surrogate + T2 regularizer
  lewis.py           E12 block Lewis weights + E13 geometry
  solvers.py         8 arms + CVXPY reference optimum
  metrics.py         E5/E6 gap curves + statistical context
  harness.py         grid tuning + FINAL-line emission
tests/               degeneracy, invariant, and grader tests (20 total)
run_all_arms.sh      full ACS gate
smoke.sh             plumbing check
arms.json            arm -> shell-command mapping
results/             committed result JSON (NOT gitignored)
SPEC.md              method spec, equation citations, unstated-parameters register
REPRODUCTION.md      reproduction log + measured-vs-claimed + discrepancy record
paper/               authoritative arXiv LaTeX source (maths) + companion [MO25]
```
