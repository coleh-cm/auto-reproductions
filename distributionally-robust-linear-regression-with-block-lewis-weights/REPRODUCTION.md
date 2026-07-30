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
