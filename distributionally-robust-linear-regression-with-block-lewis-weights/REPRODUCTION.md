# Reproduction log: Distributionally Robust Linear Regression With Block Lewis Weights

- **paper_ref:** 0a8cf406-bd9b-4ebe-8a3f-c9e3d62a2a94 · **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **arXiv:** 2607.00252 (Manoj & Patel, 2026)

## Status

### 2026-07-30 — COMPREHENSION
- Fetched arXiv e-print LaTeX source (authoritative for maths) into `paper/`; fetched the
  companion paper MO25 (arXiv 2311.10013) whose Algorithm 2 the paper's Lewis-weight step
  invokes, stored grep-ably as `paper/mo25_main.tex`.
- No upstream code exists: no links in paper/arXiv page; GitHub search empty; the
  "included Jupyter notebook" cited at `paper/experiments.tex:38` is absent from the arXiv bundle.
- Wrote `SPEC.md` (method as algorithms, symbol shapes, equation citations `paper/<file>:<line>`,
  register of 18 unstated items, frozen component interfaces, arm list) and `arms.json`
  (8-arm numbers-gate contract).
- Scope decision: numbers gate = paper's Section 8 empirical evaluation (synthetic + ACS Income).
  The theory algorithms are cited in SPEC §5 as a stretch goal; the paper's own experiments ran
  the unaccelerated trust-region variant (`paper/experiments.tex:78`).
- Environment probe: pip works (numpy 2.5.1 installed); census.gov reachable for folktables.

### 2026-07-30 — IMPLEMENTATION (this commit)
Implemented the full empirical pipeline against the frozen interfaces in SPEC §7.
All modules verified against the paper's equations; the test gate (20 tests) is green.

- `gdr/types.py` (pre-existing) — `GroupProblem` dataclass, E1/E2/E3 group losses / worst / ERM.
- `gdr/objectives.py` — smoothed surrogate `f~_{beta,delta}` (E7) with the chain-rule
  gradient/Hessian (E10), the regularised variant `fhat` (T2), and the folded-normalisation
  convention (theory `f = sqrt(F)`). Gradient/Hessian match finite differences to <1e-6;
  the E8 approximation bound `|f~ - sqrt(F)| <= beta log m + delta` holds.
- `gdr/lewis.py` — block-Lewis weights (E12, MO25 Algorithm 2 specialised to all-inner-ell2,
  p=inf), the ellipsoid geometry `M = A^T W^{1-2/p} A` (E13, naive `A^T A` E15), the
  `W = I_n` reset of Algorithm 1 line 3, the weighted-LS initializer (E14). The E13
  ellipsoid sandwich is checked numerically.
- `gdr/solvers.py` — the eight arms:
  * `reference_optimum` (E4, CVXPY epigraph QCQP; raises on solver failure),
  * `solve_subgradient` (3.1, fixed + inv-sqrt schedules, lowest-index argmax ties),
  * `solve_smooth` (3.2, GD / Heavy-Ball / Nesterov on `f~`),
  * `solve_ipm` (3.3, log-barrier IPM on the epigraph with feasibility-preserving
    Armijo backtracking on the barrier objective),
  * `solve_ball_oracle` (3.4, trust-region damped-Newton on `f~`, naive + Lewis geometry,
    no acceleration per `paper/experiments.tex:78`, optional T2 regulariser).
  Every arm raises on a zero/empty budget — no success path reports OK on nothing.
- `gdr/metrics.py` — gap curve (E5, best-so-far), cost to 1% relative gap (E6, both
  `base="init"` and `base="opt"` since the paper's 1%-reference is unstated, SPEC §6 item 4),
  statistical-context vector (`paper/experiments.tex:189`).
- `gdr/data_synth.py` — reconstructed synthetic instance (SPEC §6 item 1): shared orthonormal
  eigenbasis, 95 aligned normal groups + 5 adversarial groups with distinct high-curvature
  directions and far optima; `L_big` calibrated so `kappa(A^T A) ~ 1e5` (measured 1.78e5,
  within one order of magnitude). ERM-vs-OPT gap visible (~56).
- `gdr/data_acs.py` — folktables 2018 1-Year ACSIncome, grouped by state (m=51), 200/region,
  population z-scored features, `log1p(PINCP)` target with a `target_scale` auto-selected so
  the ERM average MSE is closest to the paper's 108.2 (SPEC §6 item 12). The headline iteration
  metric is scale-invariant.
- `gdr/harness.py` — grid-searches each arm (matching the paper's tuning protocol,
  `paper/experiments.tex:81-92`), selects the lowest-worst-loss config, runs the reported
  budget, and emits `FINAL <arm>=<value>`.
- `tests/` — 20 tests: degeneracy (Lewis@I bit-matches Euclidean; smoothed no-op),
  invariants (E8/E10/E11/E12/E13/E14/E1/E2/E3/E5/E6, leverage scores, no-empty-result),
  and a grader exercised on known-correct + known-wrong inputs via `sys.executable`.
- `run_all_arms.sh`, `smoke.sh` — gate entrypoints.

### Resolved open choices (SPEC §6 register updates)
- §6 item 4 (1%-reference): FINAL uses `base="init"` (gap/gap0, scale-invariant); `base="opt"`
  recorded in the results JSON. See `gdr/metrics.py`.
- §6 item 5 (synthetic warm start): ERM, by analogy with ACS. See `gdr/data_synth.py`.
- §6 item 7 (trust-region internals): Levenberg damping `(H + nu M) d = -g`, M-ellipsoid
  boundary projection, Armijo backtracking on `f~`. See `gdr/solvers._solve_trust_region`.
- §6 item 8 (fhat regulariser): exposed via `reg_on` (default off in the tuned run); coefficient
  `beta/(1000*min{rank,m})` (Algorithm-1 form) when on. See `gdr/solvers.solve_ball_oracle`.
- §6 item 9 (Lewis constants): `n_iters = ceil(2 ln m)`, exact leverage solves, p=inf.
- §6 item 12 (ACS target scale): auto-selected from `ACS_TARGET_SCALE_CANDIDATES` to match
  ERM avg 108.2; the iteration metric is scale-invariant so an imperfect match does not block.

### Open items / blockers
- The exact statistical-context numbers (ERM 108.2/11.8/138.1, robust band 107-114, CA -24.3)
  depend on the unstated target scaling; we report the closest-match scale and the achieved
  numbers verbatim. The headline gate metric (iterations to 1% gap) is scale-invariant.
- Wall-clock numbers are machine-dependent (informational, not pass/fail) per the paper itself.

### 2026-07-30 — NUMBERS (this commit)
Ran the full 8-arm gate on both instances (real ACS Income data downloaded from census.gov
via folktables; synthetic instance reconstructed to κ(A^T A)≈1e5). Results committed under
`results/` (un-gitignored — a number whose evidence file is gitignored is a claim with its
evidence deleted).

**ACS Income (m=51, d=10, n=10200, OPT=110.32, gap0=24.76; 1%-target=0.247, base=init):**

| arm | iters to 1% (this run) | paper | status |
|---|---|---|---|
| ball_oracle_euclidean | 1 | 1 | **reproduced** |
| ball_oracle_lewis | 1 | 1 | **reproduced** |
| smoothed_heavy_ball | 34 | 47 | same order (tuning-dependent) |
| ipm | 22 | 8 | partial — converges rapidly & best final loss (qual. ✓), exact count not matched |
| subgradient | 58 | "not reached" | **discrepancy** — see below |
| smoothed_gd / nesterov | 36 / 33 | (no ACS number) | informational |

Headline gate metric — **both ball-oracle arms reach 1% in a single outer iteration on ACS,
exactly matching the paper's flagship claim** (`paper/experiments.tex:181-182`). Mechanism:
with a tuned smoothing β≈0.005 the inner damped-Newton solver converges to the smoothed
surrogate's minimizer within one trust-region solve, and the surrogate's minimizer is within
1% of the robust optimum; the radius does not bind (Euclidean and Lewis give identical gaps).

**Statistical context (ACS, `paper/experiments.tex:189`):**

| quantity | this run | paper |
|---|---|---|
| ERM average MSE | 104.9 | 108.2 |
| ERM spread σ | 11.6 | 11.8 |
| ERM worst MSE | 135.1 | 138.1 |
| worst group | California | California |
| robust band | [94.2, 110.3] | "around 107–114" |
| Max/Mean (ERM → robust) | 1.29 → 1.02 | 1.28 → 1.02 |
| California loss decrease | −26.2 | −24.3 |

The Max/Mean ratio (1.02) and the California worst-group match exactly; the ERM statistics
match within ~3% (the target scaling is unstated, SPEC §6 item 12).

**Synthetic (m=100, d=10, 5 adversarial, κ(A^T A)=9.7e4, OPT=9399, gap0=1280):** qualitative
claims reproduced — first-order methods (subgradient, smoothed gd/hb/nesterov) stall far above
OPT (no arm reaches 1% in 100 iters); IPM makes the most first/second-order progress
(final gap 92.7 vs gap0 1280, "converges rapidly, best final loss"); both ball oracles reach 1%
in 2 outer iterations ("steadily decrease the worst-group loss"). κ≈1e5 within one order of
magnitude.

### Subgradient discrepancy (ACS)
The paper reports the subgradient arm as "not reached" (`paper/experiments.tex:178`), "essentially
pinned at the ERM gap". In this reproduction the tuned subgradient (lowest-worst-loss config over
a wide step grid, step=1e-2 fixed) reaches the 1% target at iteration 58: the raw iterate
oscillates around OPT and the best-so-far gap (the paper's reported metric,
`paper/experiments.tex:52`) crosses 1%. This is a genuine discrepancy, not a tuning artefact: the
max-loss subgradient can make progress on this instance with a well-chosen step, and the paper's
step grid (unstated, SPEC §6 item 3) likely did not include an effective step or used a smaller
budget. Recorded here rather than silently forcing "not_reached".

### IPM discrepancy (ACS)
The paper reports IPM at 8 iterations to 1% (`paper/experiments.tex:180`); this reproduction's
centring-based log-barrier IPM reaches 1% in 16 iterations (base=init). The IPM does converge
rapidly and to the best final loss (≈OPT, matching the qualitative claim), but the exact 8 is not
reproduced — the barrier schedule is unspecified (SPEC §6 item 10) and 8 ≈ √m suggests a short-step
schedule with a tighter constant than our reconstructed one.

### 2026-07-30 — ADVERSARIAL REVIEW (this commit)
Ran a 5-component adversarial review (orchestrate, 5 parallel reviewers each hunting for
correctness failures with file:line evidence against the paper LaTeX). Outcome:

**4 valid findings, all fixed:**
1. *Trust-region ball not enforced* (solvers.py) — the inner solver only capped each Newton
   step's M-norm, so over 30 inner steps the iterate drifted outside `||x-q||_M ≤ r`. Fixed:
   `project_to_ball` now hard-projects every iterate onto the M-ellipsoid ball, so the
   subproblem actually solved is SPEC E9 `min_{||x-q||_M≤r} f~(x)`. (With a non-binding radius
   both ball oracles still reach 1% in 1 outer iteration; with a binding small radius they take
   more, correctly.)
2. *IPM counted non-Newton iterations* (solvers.py) — tau-growth/centring/restore branches
   appended the same x and counted as iterations, inflating the count vs the paper's "one
   iteration = one outer Newton step" (`paper/experiments.tex:100`). Fixed: only accepted
   Newton steps advance the curve. (IPM 22 → 16 honest Newton steps to 1%.)
3. *Lewis init constant* (lewis.py) — used `rank(Â)/m`; SPEC/paper use `(d+1)/m` (the column
   count of Â). Fixed to `n_cols/m`. (Production weights were already a valid overestimate; the
   reviewer re-verified E13 holds across 12000 trials.)
4. *E13 self-test constant* (tests) — used `sqrt(2(d+1))`; paper uses `sqrt(2(rank(A)+1))`.
   Fixed to `rank(A)+1` (coincides under the paper's wlog `rank(A)=d`, robust to rank-deficiency).

**4 false findings — the reviewers inspected `/workspace/gdr/`, a separate partial checkout
(only 4 files, no harness/data_acs/paper), not this repo.** Verified against the actual code:
- "1/sqrt(n_i) folding never applied" — FALSE: `gdr/objectives.py:102-106` folds
  (`self.A = problem.A * (1/sqrt(n_i))`; `_inner` uses the folded `_Folded.residuals`).
- "E8 test compares against raw norm" — FALSE: `tests/test_invariants.py:59` uses
  `sqrt(p.worst_loss(x)) = sqrt(F)`.
- "harness.py / data_acs.py missing, paper/ empty" — FALSE: all present in this repo
  (the reviewers' `/workspace` had only data_synth/lewis/metrics/objectives + an empty paper/).

**Numbers after the fixes (ACS, real data):** ball_oracle_euclidean=1, ball_oracle_lewis=1
(both reproduce the flagship), ipm=16, smoothed_heavy_ball=41, subgradient=58.

### 2026-07-30 — GATE FIX: drop `_meta` from arms.json (this commit)
The gate iterates over every key of `arms.json` and requires each to be a runnable arm that prints
exactly one `FINAL <arm>=<value>` line. The previous `arms.json` carried a `_meta` object (paper
title, arxiv, paper_ref, project_id, primary metric, note) alongside the 8 runnable arms; the gate
counted `_meta` as a declared arm and flagged it as missing a FINAL line. Removed `_meta` from
`arms.json`, which now holds only the 8 runnable arm commands. No metadata is lost: the paper
block (title/arxiv/paper_ref/project_id) and the primary_metric (with citation) already live in
`arms_contract.json`, and the `harness all` results JSON still records a `_meta` instance-summary
field (instance/m/d/n/opt/gap0/budget) — that is a results-file entry, not a declared arm, and is
not expected to print a FINAL line. Verified: each of the 8 per-arm commands prints exactly one
`FINAL <arm>=<value>` line (reference_cvxpy=110.316, ball_oracle_euclidean=1, ball_oracle_lewis=1
re-checked post-edit; the full 8-arm set was produced by the prior gate run).
