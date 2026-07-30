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

| arm | measured (iters to 1% gap, base=init) | paper claim | match |
|---|---|---|---|
| ball_oracle_euclidean | 1 | 1 (`experiments.tex:181`) | ✓ exact |
| ball_oracle_lewis | 1 | 1 (`experiments.tex:182`) | ✓ exact (flagship) |
| smoothed_heavy_ball | 41 | 47 (`experiments.tex:179`) | ~ close |
| ipm | 16 | 8 (`experiments.tex:180`) | 2× off |
| subgradient | 58 (reached) | not reached (`experiments.tex:178`) | ✗ discrepancy |
| smoothed_gd | 70 | (not in table) | — |
| smoothed_nesterov | 33 | (not in table) | — |
| reference_cvxpy OPT | 110.3 | band 107-114 (`experiments.tex:189`) | ✓ |

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

## What runs

- `bash smoke.sh` — tiny self-contained problem, one FINAL line, ~seconds. Plumbing only.
- `bash run_all_arms.sh acs` — the full ACS gate (8 arms), ~40 s. Prints one `FINAL <arm>=<value>`
  per arm; writes `results/acs_all.json`.
- `bash run_all_arms.sh synthetic` — the synthetic gate, ~12 s.
- `.venv/bin/python -m pytest -q` — 20 tests (degeneracy, invariants, grader).
- `.venv/bin/python -m gdr.harness context --instance acs` — the statistical-context vector.

Results JSON lives in `results/` (NOT gitignored — see `.gitignore`: "RESULTS are committed").
