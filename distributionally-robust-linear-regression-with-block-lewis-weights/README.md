# Reproduction: Distributionally Robust Linear Regression With Block Lewis Weights

Reproduction of **"Distributionally Robust Linear Regression With Block Lewis
Weights"** (Naren Sarayu Manoj, Kumar Kshitij Patel, arXiv:2607.00252, ICLR 2026).

The paper gives an algorithm for the **group distributionally robust (GDR) least
squares** problem — minimizing the worst-group mean-squared error

```
min_x  max_{i in [m]}  (1/n_i) * ||A_{S_i} x - b_{S_i}||_2^2
```

using block Lewis weights to choose a data-dependent Euclidean geometry, then a
damped-Newton trust-region ("ball-oracle") solver on a log-sum-exp smoothing of
the max. Its headline claim is an Õ(min{rank(A), m}^{1/3} ε^{−2/3})
linear-system-solve complexity (Theorem 1, `intro.tex:27-37`). The empirical
evaluation (§8, `experiments.tex`) benchmarks the *unaccelerated* ball-oracle
variant against subgradient / smoothed-first-order / IPM baselines on a
synthetic heterogeneous regression instance and on the ACS Income task.

## What this repo contains

- `paper/arxiv-2607.00252-src/` — the authoritative LaTeX source (maths in the
  PDF extraction are not trusted; all equation citations resolve here).
- `SPEC.md` — the full reproduction specification: problem notation with
  shapes, every equation implemented (E1–E22) with `file:line` citations, the
  frozen component interfaces, the 7 arms × 2 datasets plan, the numbers gate
  targets (T1–T5), and the 18 unstated-items list (U1–U18).
- `REPRODUCTION.md` — running log of the read/spec step.
- `requirements.txt` — every dependency pinned (direct + transitive).
- `Dockerfile` — builds the environment from scratch.
- `smoke_imports.py` — proves the pinned toolchain resolves *and works*
  (imports, CVXPY solvers, folktables features, and a correct epigraph-QCQP
  solve vs an independent scipy minimizer).
- Implementation package (`gdr/`), tests (`tests/`), `conftest.py`, and
  `arms.json` are added by the subsequent implementation step per SPEC §4–§5.

> **Upstream code: none.** The arXiv page has no code link; both authors'
> GitHub accounts have zero public repos; and the "included Jupyter notebook"
> referenced at `experiments.tex:38` is absent from the arXiv tarball. The
> method is re-implemented from scratch from the LaTeX source. See SPEC §6.

## Method in one paragraph

Fold the per-group factors `1/√n_i` into the data (`body.tex:27`), so the
worst-group objective is `F(x) = max_i ||A_{S_i}x − b_{S_i}||_2^2`. Smooth the max
with the log-sum-exp surrogate `f̃_{β,δ}(x) = β·log Σ_i exp((√(δ²+‖r_i‖²)−δ)/β)`
(E3, `body.tex:54-56`) with `β = ε/(4 log m)`, `δ = ε/4` (Lemma 6.1). Compute
**block Lewis weights** `w` via [MO25, Algorithm 2] (re-derived concretely as
E8 from arXiv:2311.10013) on the augmented matrix `[A|b]`, giving a
data-dependent geometry `M = AᵀWA` (E7/E11; fall back to `M = AᵀA` when
`Σ_i w_i ≥ m`). Warm-start at the weighted least-squares solution `x₀`
(E9, `body.tex:176`), then run the ball-oracle: repeatedly minimize `f̃_{β,δ}`
over `{x : ‖x − q‖_M ≤ r}` by a damped trust-region Newton (Moré–Sorensen), update
`q ← x`, optionally shrink `r` (E19, `experiments.tex:71-78`). The robust
optimum `OPT` for evaluation is the CVXPY epigraph QCQP (E20,
`experiments.tex:40-50`); all plots report `F(x) − OPT`.

## Environment

Python 3.13. Direct dependencies: numpy, scipy, cvxpy, clarabel, scs, pandas,
matplotlib, folktables, pytest (transitive deps pinned too — see
`requirements.txt`). No GPU/CUDA. All native extensions ship wheels for both
`linux/amd64` and `linux/arm64`, so `python:3.13-slim` is a sufficient base.

> `ecos` is intentionally **not** installed: ecos 2.0.14 has no cp313/aarch64
> wheel and its source build needs `Python.h`. The paper only requires "a
> standard convex solver" via CVXPY (`experiments.tex:50`); **clarabel**
> (CVXPY's default) and **scs** cover the epigraph QCQP and are verified in
> `smoke_imports.py` to solve it correctly (matching an independent scipy
> minimizer to 1e-5).

## Quickstart

### With `uv` (recommended; this is the verified environment)

```bash
# from this reproduction folder
uv venv --python 3.13 --clear .venv
uv pip install --python .venv -r requirements.txt

# prove the toolchain resolves and the E20 epigraph QCQP solves correctly
.venv/bin/python smoke_imports.py
#   -> prints versions, cvxpy installed_solvers, the 10 ACSIncome features,
#      and "SMOKE OK"
```

### With plain pip + a system Python 3.13

```bash
python3.13 -m venv --clear .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python smoke_imports.py
```

### With Docker

```bash
docker build -t gdr-block-lewis .
docker run --rm gdr-block-lewis          # runs smoke_imports.py
```

## Running the experiments

The experiment entrypoint (`gdr/runner.py` per SPEC §4) and the arm
configurations (`arms.json` per SPEC §5) are produced by the implementation
step. Once present, the intended interface is:

```bash
# run one arm on one dataset (SPEC §4 frozen interface)
.venv/bin/python -m gdr.runner --arm ball_oracle_lewis --dataset synthetic
.venv/bin/python -m gdr.runner --arm ball_oracle_lewis --dataset acs_income

# the 7 arms x 2 datasets comparison of §8 (subgradient, smoothed_gd/hb/nesterov,
# ipm, ball_oracle_euclidean, ball_oracle_lewis) + the CVXPY opt_reference
.venv/bin/python -m gdr.runner --all
```

Until the implementation lands, `smoke_imports.py` is the runnable proof that
the environment is correct.

## Running the tests

```bash
.venv/bin/python -m pytest -q     # equation-invariant + degeneracy gates (SPEC §5 T5)
```

## Target numbers (§8)

**ACS Income, cost to reach 1% relative worst-group suboptimality**
(`tab:acs_runtime`, `experiments.tex:171-186`):

| Method                  | Iterations to 1% gap | Wall-clock (s) |
|-------------------------|----------------------|----------------|
| Subgradient             | — (not reached)      | —              |
| Smoothed Heavy-Ball     | 47                   | 0.062          |
| IPM                     | 8                    | 0.066          |
| Ball-Oracle (Euclidean) | 1                    | 0.019          |
| Ball-Oracle (Lewis)     | 1                    | 0.019          |

Gate semantics (SPEC §5 T1–T5): both ball-oracle arms reach the target in ≤ 2
outer iterations with strict ordering `iters(BO) < iters(IPM) ≤ iters(HB)` and
the subgradient arm does not reach 1% within budget; the synthetic instance
shows IPM lowest, both BO arms strictly decreasing and beating the first-order
plateau, Lewis ≤ Euclidean finally; the E6/E7/E9/Lemma-6.1 unit checks hold.
Exact counts depend on undisclosed hyperparameter grids (U2), so the honest
gate is ordering + boundedness, not exact-equality.

## Notes / open risks (carried from SPEC §7)

- The benchmarked ball-oracle arms are **unaccelerated** (`experiments.tex:78`),
  so §8's numbers do not exercise Theorem 1's Algorithm 1 as written (U3).
- The synthetic data recipe is not in the paper (U1); the "included Jupyter
  notebook" (`experiments.tex:38`) is absent from the arXiv tarball — the
  concrete generator is our declared choice in `arms.json`.
- ACS data download via `folktables` needs network at run time (D2); the
  synthetic instance (D1) is fully self-contained.
- ACS preprocessing (year/horizon, standardization, subsample scheme/seed, log
  transform) is unspecified (U4); declared choices live in `arms.json`.
