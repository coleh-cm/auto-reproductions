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

```bash
# prove the toolchain resolves and the E20 epigraph QCQP solves (env smoke)
.venv/bin/python smoke_imports.py

# one arm on one dataset (prints FINAL <dataset>_<arm>=<iters-to-1%>)
.venv/bin/python run_arm.py --arm ball_oracle --geometry lewis --dataset synthetic --seed 0
.venv/bin/python run_arm.py --arm ipm                  --dataset acs_income --seed 6

# all 7 method arms x 2 datasets + opt_reference (the §8 comparison)
bash run_all_arms.sh                 # prints one FINAL line per (dataset, arm)

# a tiny end-to-end smoke (all 7 arms, same code path, finishes in <1 s; NOT a result)
bash smoke.sh
```

Each arm prints exactly `FINAL <dataset>_<arm>=<value>` where `<value>` is the
arm's own outer iterations to reach 1% relative worst-group suboptimality
`(F−OPT)/OPT ≤ 0.01` (T1, `experiments.tex:176`), or `NR` if not reached within
budget. Full per-arm histories are written to `results/<dataset>_<arm>.json`
(committed, not gitignored).

## Running the tests

```bash
.venv/bin/python -m pytest -q     # 19 tests: degeneracy + equation invariants (SPEC T5)
```

`tests/test_degeneracy.py` — the method at its no-op setting must reproduce the
baseline exactly: (D1) the Lewis ball-oracle at the E11 reset (Σwᵢ≥m ⇒ W←I) is
bit-identical to the Euclidean ball-oracle; (D2) the p=2 interpolating objective
reduces to plain least squares (ERM). `tests/test_invariants.py` — Lemma 6.1
(`|f̃−f| ≤ β log m + δ`), the E6 block-Lewis overestimate & `‖w‖₁ ≤ 2(d+1)`,
the E7 residual sandwich, smoothed grad/Hessian finite-difference + PSD,
p-objective gradient finite-difference, **Lemma 7.2 strong-convexity of ‖·‖ₚ²**,
**lewis_warm_start D-exponent** (p=∞/2/4/8), subgradient validity, ball-oracle
**per-iteration f̃-monotonicity** (via the recorded `x_traj`).

## Results (this reproduction)

**Synthetic (D1, seed=0, DIST=5.0; cond(AᵀA)=1.40e5; ERM/robust worst-group ratio 1.47)**
— T4 is qualitative (`experiments.tex:106-109`): subgradient=NR (plateau ≈10.0%),
all smoothed first-order=NR (plateau ≈9.1%), ipm=5, ball_oracle_euclidean=9,
ball_oracle_lewis=5 (both final 0.34%). Matches the paper: IPM reaches the lowest
final loss; both BO arms strictly decrease the gap over outer iterations and
beat the first-order plateau; **Lewis ≤ Euclidean** — both BO arms reach the
0.34% smoothing floor, Lewis in 5 outer iterations vs Euclidean's 9 (the
paper's "very slight benefit from Lewis", `experiments.tex:109`). Both BO
curves are monotone non-increasing in the smoothed objective (damped Newton).

**ACS Income (D2, seed=6; California is the worst ERM group; ERM mean 107.3)**
— T1 gate (`experiments.tex:171-186`):

| Method                  | Paper (iters to 1%) | This repro |
|-------------------------|---------------------|------------|
| Subgradient             | — (not reached)     | 3 (reaches) |
| Smoothed Heavy-Ball     | 47                  | 10         |
| IPM                     | 8                   | 10         |
| Ball-Oracle (Euclidean) | 1                   | 1          |
| Ball-Oracle (Lewis)     | 1                   | 1          |

Gate: BO arms ≤ 2 ✓; ordering `iters(BO)=1 < iters(IPM)=10 ≤ iters(HB)=10` ✓
(paper `1 < 8 < 47`); subgradient reaches 1% ✗ (paper: never). The ordering and
BO=1 reproduce; the subgradient-NR condition and the IPM/HB magnitudes do not,
because the reproduced ACS ERM-robust gap is ~1.8% vs the paper's ~25% (Blocker
B1 below). T3 (report-only): ERM mean 107.3 (paper 108.2 ±5 ✓), worst state
California ✓, robust Max/Mean 1.030 (paper 1.02 ✓); ERM worst 112.7 (paper
138.1 ✗), ERM Max/Mean 1.051 (paper 1.28 ✗).

## Blocker B1 (ACS heterogeneity, U4/U7)

The paper's per-state heterogeneity (ERM Max/Mean 1.28, worst 138.1) is **not
reproducible** from the disclosed preprocessing with `d=10` / no-intercept /
`log1p(PINCP)` / 200-per-state: a 40-seed scan caps the ERM worst at 116.8
(Max/Mean ≤ 1.09). With the faithful setup the reproduced ERM-robust gap is
~1.8% (vs the paper's ~25%), so the subgradient method reaches the 1% target
(the paper reports it never does) and the IPM/HB counts are compressed. The
**structure** reproduces — California is the worst ERM group, the robust
optimum narrows the band to Max/Mean ≈ 1.03 (paper 1.02), ERM mean ≈ 107.3
(paper 108.2) — but the **magnitude** does not, and is attributed to the
undisclosed ACS preprocessing/seed (U4/U7). This is a partial-reproduction
blocker on the T1 subgradient-NR condition and the T3 magnitudes; the
`iters(BO) < iters(IPM) ≤ iters(HB)` ordering and `BO = 1` hold.

## Notes / open risks (carried from SPEC §7)

- The benchmarked ball-oracle arms are **unaccelerated** (`experiments.tex:78`),
  so §8's numbers do not exercise Theorem 1's Algorithm 1 as written (U3); the
  accelerated MS-oracle arms (E16/E17/E18) are a theory-fidelity stretch not
  exercised by any §8 number and are not implemented here.
- The synthetic data recipe is not in the paper (U1); the "included Jupyter
  notebook" (`experiments.tex:38`) is absent from the arXiv tarball — the
  concrete generator (`gdr/data_synthetic.py`) is our declared choice.
- ACS data (51 state PUMS CSVs, ~575 MB) are gitignored; download with
  `scripts/download_acs.py` (or `folktables` `get_data(download=True)`).
- All open choices are recorded in SPEC.md §8A; the OPT=1 normalization (U15)
  is required numerically (the IPM's damped-Newton centering stalls on O(1e5)
  losses at cond 1e4+ but converges at the same cond with O(1) losses).
