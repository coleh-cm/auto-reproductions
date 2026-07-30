# Reproduction log: Distributionally Robust Linear Regression With Block Lewis Weights

- **paper_ref:** 0a8cf406-bd9b-4ebe-8a3f-c9e3d62a2a94 · **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **arXiv:** 2607.00252 (Manoj & Patel, 2026)

## Status

### 2026-07-30 — COMPREHENSION
- Fetched arXiv e-print LaTeX source (authoritative for maths) into `paper/`; fetched the
  companion paper MO25 (arXiv 2311.10013) whose Algorithm 2 the paper's Lewis-weight step
  invokes, stored grep-ably as `paper/mo25_main.tex`.
- No upstream code exists (no links in paper/arXiv page; GitHub search empty; the
  "included Jupyter notebook" cited at `paper/experiments.tex:38` is absent from the bundle).
- Wrote `SPEC.md` (method as algorithms, symbol shapes, equation citations, 18-item
  unstated-parameters register, frozen component interfaces, arm list) and `arms.json`.

### 2026-07-30 — IMPLEMENTATION (this commit)
- Implemented the full `gdr/` package from scratch against the frozen `gdr/types.py`
  contract, built via a two-wave parallel orchestration with adversarial review of each
  module against `paper/*.tex`. Modules: `types` (frozen `GroupProblem`), `data_synth`
  (reconstructed adversarial instance), `data_acs` (folktables loader), `objectives`
  (E7/E10 smoothed surrogate + T2 regularizer), `lewis` (E12 block Lewis weights + E13
  geometry), `metrics` (E5/E6 gap curves + statistical context), `solvers` (8 arms),
  `harness` (grid tuning + FINAL-line emission).
- Tests: `tests/test_degeneracy.py` (3 no-op=baseline degeneracies), `tests/test_invariants.py`
  (E1-E8 equation invariants), `tests/test_grader.py` (1%-gap grader on known-correct/wrong
  + subprocess via `sys.executable`). 20 tests, all passing.
- Scripts: `run_all_arms.sh` (full ACS gate), `smoke.sh` (tiny plumbing check).

## Resolved open items (SPEC section 6)

1. **Synthetic recipe** — `[reconstructed]`. Shared Haar eigenbasis; normal groups
   log-uniform curvature [1e-1,1e1] near x_pop=0 with σ_n=0.5; 5 adversarial groups with
   one extreme-curvature direction (distinct each) and far optimum (offset ~5-10), σ_adv=0.05;
   n_per_group=50. Tuned so cond(A^T A) ≈ 1e5 (achieved 1.000000e+05 across 30 seeds). The
   ERM-vs-OPT gap is visible (~14% relative). See `gdr/data_synth.py`.
2. **n_i synthetic** — `[reconstructed]` 50 rows/group (n=5000).
3. **Hyperparameter grids** — `[reconstructed]` small grids in `gdr/harness.py:GRIDS`.
   Subgradient step {1e-9..1e-2}×{fixed,inv_sqrt}; smoothed β∈{0.003-0.02}, δ=0.01,
   step∈{0.01-0.2}, momentum∈{0.85-0.93}; IPM barrier0∈{1,2,5,10}, growth∈{1,2,5};
   ball-oracle radius∈{50,500}, β∈{0.005,0.02}, δ=0.01, reg_on=False.
4. **1%-gap reference** — unstated; we compute BOTH base='init' (gap/gap0) and base='opt'
   (gap/OPT). The FINAL line uses base='init' (the most natural reading of "relative
   suboptimality"); both are stored in the results JSON. The paper's table values are
   reproducible under base='init' for the flagship ball-oracle arms.
6. **CVXPY solver** — `[reconstructed]` CLARABEL (installed); falls back to default if
   unavailable. Raises on non-optimal status (HARD RULE 3).
9. **Lewis routine** — `[reconstructed]` exact leverage solves, T=⌈2 ln m⌉; p=∞ (matches
   Algorithm 1, the max objective).
12. **ACS target scale** — `[reconstructed]`. The SPEC assumed the target would be
    z-scored then rescaled to lift MSE to ~108. Empirically, the RAW log1p(PINCP) target
    (no standardization, no intercept) already reproduces the paper: ERM avg 104.9 (paper
    108.2), worst 135.1 California (paper 138.1), std 11.6 (paper 11.8), Max/Mean 1.287
    (paper 1.28). Mechanism: with no intercept and centered features the global mean of
    log-income (~10.6) cannot be fit, so ERM MSE is dominated by that squared offset and
    high-income states are worst. target_scale=1.0 (auto-selected). See `gdr/data_acs.py`.

## Measured results vs paper claims (ACS Income, m=51, the headline instance)

### Within-noise caveat — read this first

The two flagship arms, `ball_oracle_euclidean` and `ball_oracle_lewis`, come out
**identical within floating-point noise**: both reach the 1% target in exactly 1
outer iteration, and their best-so-far gap curves coincide to 12 decimal places
(`final_gap` = 0.04593018017129**68** vs 0.04593018017133**52** — a difference at
the 13th digit). A measurement that cannot tell the two geometries apart has
**not tested** the paper's claim that "the Lewis-weight geometry is marginally
faster than the Euclidean ball" (`experiments.tex:139,172`). We reproduce the
weaker claim — *each* ball oracle reaches 1% in a single iteration — but the
Euclidean-vs-Lewis comparison itself is not resolved by these numbers; it is
within noise, not decided. (The single-iteration result is not a warm-start
artifact: the initial gap is `gap0 = F(x0) − OPT = 24.759`, the 1%-of-init
threshold is 0.2476, and one trust-region Newton step drops the gap to 0.0459,
i.e. it closes 99.8% of the gap in one step.)

### Iteration-count comparison (the metric in Table `tab:acs_runtime`)

The exact command per arm is `arms.json`; each is
`.venv/bin/python -m gdr.harness arm --arm <ARM> --instance acs` (no `--budget`
flag, so the harness default applies: 100 for the first-order/IPM arms, 15 for
the ball-oracle arms). The whole gate is reproduced by `bash run_all_arms.sh acs`,
which prints one `FINAL <arm>=<value>` line per arm and writes `results/acs_all.json`.

| arm | exact command | measured (iters to 1% gap, base=init) | paper claim | match |
|---|---|---|---|---|
| reference_cvxpy | `.venv/bin/python -m gdr.harness arm --arm reference_cvxpy --instance acs` | OPT = 110.316 | band 107-114 (`experiments.tex:189`) | ✓ |
| ball_oracle_euclidean | `.venv/bin/python -m gdr.harness arm --arm ball_oracle_euclidean --instance acs` | 1 | 1 (`experiments.tex:181`) | ✓ exact |
| ball_oracle_lewis | `.venv/bin/python -m gdr.harness arm --arm ball_oracle_lewis --instance acs` | 1 | 1 (`experiments.tex:182`) | ✓ exact (flagship) |
| smoothed_heavy_ball | `.venv/bin/python -m gdr.harness arm --arm smoothed_heavy_ball --instance acs` | 41 | 47 (`experiments.tex:179`) | ~ close |
| ipm | `.venv/bin/python -m gdr.harness arm --arm ipm --instance acs` | 16 | 8 (`experiments.tex:180`) | 2× off |
| subgradient | `.venv/bin/python -m gdr.harness arm --arm subgradient --instance acs` | 58 (reached) | not reached (`experiments.tex:178`) | ✗ discrepancy |
| smoothed_gd | `.venv/bin/python -m gdr.harness arm --arm smoothed_gd --instance acs` | 70 | (not in table) | — |
| smoothed_nesterov | `.venv/bin/python -m gdr.harness arm --arm smoothed_nesterov --instance acs` | 33 | (not in table) | — |

Re-running `bash run_all_arms.sh acs` reproduces these `FINAL` lines
**byte-for-byte** (deterministic, seed 0): the re-run was diffed against
`/tmp/arms.log` and is identical. `OPT = 110.316` sits inside the paper's
107-114 robust band; `F(x0) = 135.075` (paper ERM worst 138.1, California).

### Wall-clock comparison (the other column of Table `tab:acs_runtime`)

The paper also reports wall-clock and claims the ball-oracle methods are fastest
on **both** axes (0.019 s, "roughly 3× faster than IPM and Heavy-Ball"). Our
wall-clock does **not** reproduce that: each ball-oracle outer step solves a
damped-Newton trust-region subproblem, so one iteration is expensive.

| arm | measured wall (s) | paper wall (s) |
|---|---|---|
| ball_oracle_euclidean | 0.813 | 0.019 |
| ball_oracle_lewis | 0.801 | 0.019 |
| ipm | 0.084 | 0.066 |
| smoothed_heavy_ball | 0.048 | 0.062 |
| smoothed_nesterov | 0.048 | — |
| smoothed_gd | 0.047 | — |
| subgradient | 0.007 | — |

So in our run the ball-oracle arms are fastest **in iteration count** (1) but
**slowest in wall-clock** (≈0.8 s). The paper's iteration-count claim is
reproduced; the paper's wall-clock-speed claim is not. We did not optimise the
Newton sub-solve (the paper notes it did not either, `experiments.tex:130`), but
the gap is large enough that this is a genuine divergence, not noise. (Wall-clock
varies a few % run-to-run — only the `wall_final` fields of `results/acs_all.json`
differ between the committed run and a re-run; every iteration count, `final_gap`,
and `OPT` is bit-identical. The headline metric is deterministic; wall-clock is
not, and is reported as measured, not as a reproducible constant.)

**Statistical context** (ACS, seed 0): ERM avg 104.9 / std 11.6 / worst 135.1 (California);
robust Max/Mean 1.287→1.022 (paper 1.28→1.02 ✓); worst-ERM-group (California) loss
decrease 26.2 (paper 24.3 ✓).

## Discrepancies and their attribution (honest record)

1. **subgradient reaches 1% on ACS (58 iters); paper says "not reached".** Our subgradient
   uses the paper's MSE subgradient `g=(2/n_{i*})A_{S_i*}^T r` (`experiments.tex:57-58`) and
   the paper's "best of fixed/diminishing" protocol. On the well-conditioned ACS instance
   (d=10, standardized features) the active group stays fixed (California) so subgradient
   descent is effectively gradient descent on one smooth quadratic and converges. The
   paper's "pinned at the ERM gap" likely reflects their specific (unstated, SPEC item 3)
   step-size grid; with a reasonable reconstructed grid it converges. We report the measured
   58 honestly rather than forcing "not reached".

2. **IPM = 16 vs paper 8.** Our IPM is a basic log-barrier method with reconstructed
   barrier/growth/damping (SPEC item 10). It reaches 1% in 16 centring phases. The paper's
   IPM (which "CVXPY natively uses" per `experiments.tex:107`) is a production-grade
   interior-point solver. 16 is within 2× and within the unstated-tuning latitude.

3. **Synthetic: first-order methods make ~zero progress (paper: "stall far above optimum").**
   The synthetic loss scale (~1e4) is large; the reconstructed step grids bracket either
   no-movement (too small) or divergence (too large) for the smoothed arms, so the best
   config stays at the warm start. The paper's "stall far above optimum" implies some
   progress; ours makes none. Attributable to the unstated grid (SPEC item 3). Ball oracles
   and IPM DO make progress on synthetic (ball-oracle reaches 1% in 3-6 iters ✓; IPM gap
   1279→92.7), matching the paper's qualitative "ball oracles steadily decrease" and "IPM
   converges rapidly". Ball-oracle beats IPM on synthetic in our run; the paper says IPM is
   best — our basic IPM stops early (a known limitation of a hand-rolled barrier method vs
   CVXPY's production solver).

## Research-readiness gates

Verdict recorded 2026-07-30 by re-running the checks in this checkout.

| # | gate | verdict | evidence |
|---|---|---|---|
| 1 | Builds from scratch | **partial** | `Dockerfile` is a clean `python:3.13-slim` build from the full lock `requirements.txt` with a build-time import + CVXPY smoke test. No Docker daemon in this sandbox, so `docker build` was not executed here; the env was built with `uv pip install -r requirements.txt` instead and works. |
| 2 | README is accurate | **pass** | The README quickstart (`uv venv --python 3.13 .venv && uv pip install --python .venv -r requirements.txt` then `bash run_all_arms.sh acs`) was followed; `.venv` builds, the gate runs, and the table matches `results/acs_all.json`. |
| 3 | Packages are clear | **pass** | `requirements.txt` is a 71-line full lock (every dep pinned, transitive conic solvers included). Install resolves and every method import + CVXPY solve loads. |
| 4 | Entrypoint is obvious | **pass** | One command, flag-driven: `bash run_all_arms.sh [acs\|synthetic]`; per-arm via `arms.json`. No source edits needed. |
| 5 | Fast path | **pass** | `bash smoke.sh` runs the full code path (surrogate → Lewis → trust-region Newton → gap) in seconds, printing `FINAL smoke=0.0845631`. |
| 6 | Deterministic / noise quantified | **pass** | `bash run_all_arms.sh acs` re-run diffed byte-for-byte identical to the recorded `/tmp/arms.log` for the headline iteration-count metric (seed 0). Wall-clock is non-deterministic (±a few % run-to-run) and is reported as measured, not as a constant. |
| 7 | Degeneracy test in repo | **pass** | `tests/test_degeneracy.py` (3 no-op=baseline cases) + `tests/test_invariants.py` + `tests/test_grader.py`; `pytest -q` → 20 passed. |
| 8 | Data provenance stated | **pass** | ACS PUMS downloaded on demand by `folktables` from census.gov (2018 1-Year), cached under `data/` (gitignored, regenerable). README + `gdr/data_acs.py` state it. |
| 9 | Recorded number reproducible | **pass** | The exact command (`bash run_all_arms.sh acs`) re-run produces the recorded `FINAL` lines and `results/acs_all.json` identically. |
| 10 | Nothing depends on hidden local state | **pass** | Fresh-clone-runnable: `.venv/` and `data/` are gitignored and regenerable; source, `SPEC.md`, `arms.json`, `results/`, `paper/` are committed. |

## Budget / gate-status markers

- `$HOME/.build_attempts = 2` — the environment was built (2 attempts) and works:
  `pytest -q` → 20 passed, `smoke.sh` runs, the 8-arm ACS gate runs and reproduces.
  The build gate is not failing; the only unbuilt artifact is the Docker image
  (gate 1, partial — no daemon in sandbox), which does not affect the numbers.
- `$HOME/.env_attempts` — not present; no environment-assembly budget was spent
  on a still-failing gate.
- `$HOME/.review_rounds = 1` — review budget was used. The commit log records
  that round-1 and round-2 adversarial-review findings were fixed (`700ecc1`,
  `68296e9`, `ea1ef8c`); the final tree is clean, 20 tests pass, and the gate is
  deterministic. No outstanding objection is recorded in the persisted
  artifacts, so we report the review as resolved rather than budget-exhausted —
  but we cannot certify from the artifacts alone that the reviewers went quiet,
  only that every finding we can see was addressed.

## What runs

- `bash smoke.sh` — tiny self-contained problem, one FINAL line, ~seconds. Plumbing only.
- `bash run_all_arms.sh acs` — the full ACS gate (8 arms), ~40 s. Prints one `FINAL <arm>=<value>`
  per arm; writes `results/acs_all.json`.
- `bash run_all_arms.sh synthetic` — the synthetic gate, ~12 s.
- `.venv/bin/python -m pytest -q` — 20 tests (degeneracy, invariants, grader).
- `.venv/bin/python -m gdr.harness context --instance acs` — the statistical-context vector.

Results JSON lives in `results/` (NOT gitignored — see `.gitignore`: "RESULTS are committed").
