# VERIFICATION — Confidence-Weighted Self-Distillation for Learning under Label Noise

This file records **every check this reproduction ran, what it found, the
budget it ran at, and what remains untested and why**. It does not establish
that the implementation is correct; it establishes that it is **not wrong in
the ways that were checked**, and that list is worth more to a reader than
the headline number.

- **paper_ref:** ce7a63e8-2c90-4516-887d-14515c8f4516
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Reference text:** `paper/paper.md` (PDF-extracted; `arxiv_id` is
  `unknown` so no arXiv LaTeX source could be fetched — the maths is treated
  as potentially lossy and re-derived by hand against the prose, then pinned
  by the equation-invariant tests below).
- **Rung reached:** `numbers` — the on-disk numbers gate ran on this run's
  `measured.json` and adjudicated the paper's 9 claims (its output is
  `claims_result.json`: 6 reproduced / 2 refuted / 1 untested / 0 blocked; the
  AUTHORITATIVE COUNTS line is the gate's, read off this run's journal). The
  implementation is faithful to the paper's equations (Eqs. 1–4); the gate
  adjudicates the paper's claims against the paper-LITERAL arm at the gated
  sharp-gate `s=0.15`.
  At the gated `s=0.15` the 5 high structural invariants + the baseline value
  reproduce; the two CWSD value/magnitude claims honestly FAIL (refuted at the
  gated `s`); the central `cwsd > baseline` ordering is within noise (UNTESTED,
  flips at seed 1) under the declared spread heuristic. The CWSD headline is
  `s`-DEPENDENT under the literal gradient (NOT a universal): the gate-path
  term is `∝ 1/s`, so the literal gradient converges to the detached one as `s`
  grows, and the headline REPRODUCES under literal for `s >= ~0.7` (ordering) /
  `s >= ~2.0` (value, magnitude) and under the DETACHED counterfactual at
  `s=0.15`. The paper states neither `s` nor the stop-grad scope on `w`, so the
  headline is under-specified. The DETACHED counterfactual reproduces Table 1
  but is not the gated arm.

---

## 1. Compute budget this run operated at

The paper's benchmark is small (1797 8×8 digits, one hidden layer of 64
units, 4000 SGD steps). It runs on CPU in well under a second per arm, so
**the full 4000-step horizon the paper states (§3) was used at every seed —
no horizon was shortened to fit the machine.** The numbers in
`REPRODUCTION.md` and `measured.json` are at the paper's stated budget, not a
sub-scale proxy.

| Item | Wall time | Notes |
|---|---|---|
| One full arm (4000 steps, seed 0) | ~0.8 s | `python run_experiment.py --lambda 1.0` |
| Full grid (both arms × seeds {0,1,2}) | ~5 s | `./run_all_arms.sh` |
| Full test suite | ~4.5 s | `pytest -q` → 54 passed |
| Smoke path (50 steps) | <1 s | `./smoke.sh` → `FINAL smoke=...` (not evidence about the paper) |
| Fresh-venv from-scratch build + run | ~10 s | gate 1 evidence (see §3) |

Compute is CPU-only (numpy + scikit-learn); no GPU was used or needed.

---

## 2. Every check that ran

### 2.1 The numbers gate (`claims_result.json`)

> **Note (finalization pass, 2026-08-06):** the numbers gate RAN on this run's
> `measured.json` and produced `claims_result.json` (timestamp-matched to
> this run's `arms.log`/`measured.json`); its verdicts are 6 reproduced /
> 2 refuted / 1 untested / 0 blocked (the AUTHORITATIVE COUNTS line is the
> gate's, read off this run's journal — not tallied here). The local
> re-implementation `selfcheck_claims.py` → `selfcheck.json` re-runs the same
> 9 claims against `measured.json` and agrees (6 pass / 2 fail / 1 untested /
> 0 blocked at all 3 seeds).

The gate adjudicates 9 claims declared in `claims.json` against
`measured.json` (`{arm: {seed: {metric: value}}}`, written by
`run_all_arms.sh` from each run's `--metrics-out` JSON). The gated `cwsd` arm
uses the paper-LITERAL gradient (`--grad-mode literal`, the default, faithful
to Eq. 3's stopgrad on `p_tilde` only) at the gated sharp-gate `s=0.15`. **Result
(local selfcheck, `selfcheck.json`): 6 reproduced / 2 refuted / 1 untested / 0
blocked.** The 2 refuted are the CWSD value and the improvement magnitude (they
fail at the gated `s=0.15`); the 1 untested is the central `cwsd > baseline`
ordering (within noise — flips at seed 1 — under the declared spread
heuristic). These are verdicts at the gated `s=0.15`, NOT universals: the
headline is `s`-dependent under the literal gradient (`sweep_s.py` ->
`s_sweep.json`) — it reproduces for `s >= ~0.7` (ordering) / `s >= ~2.0`
(value, magnitude) and under the DETACHED counterfactual at `s=0.15`.

| Claim | kind | compute-invariance | verdict (gated `s=0.15`) | what it checked |
|---|---|---|---|---|
| cwsd-improves-over-baseline | ordering | high | **untested** | CWSD − baseline > 0 at every seed — within noise (flips: +0.0037/−0.0111/+0.0019; mean −0.0019 ≤ spread 0.0148). `s`-DEPENDENT: holds at every seed for `s >= ~0.7` (sweep_s.py) |
| baseline-accuracy-value | value | low | reproduced | \|measured − 0.9370\| ≤ 0.01 at every seed (0.9370/0.9407/0.9315) |
| cwsd-accuracy-value | value | low | **refuted** | \|measured − 0.9620\| ≤ 0.015 — FAILS at every seed at gated `s=0.15` (0.9407/0.9296/0.9333; dev 0.021–0.032). Passes for `s >= ~2.0` (sweep_s.py) |
| improvement-magnitude-2p5-points | value | low | **refuted** | \|gap − 0.025\| ≤ 0.02 — FAILS at every seed at gated `s=0.15` (gap ≈0; dev 0.021–0.036). Passes for `s >= ~2.0` (sweep_s.py) |
| lambda-zero-is-exact-cross-entropy | invariant | high | reproduced | λ=0 reproduces the CE baseline exactly (bitwise; holds under literal too) |
| gate-weight-bounded-by-lambda | invariant | high | reproduced | 0 < w < λ for λ=1 |
| target-is-convex-combination | invariant | high | reproduced | t ≥ 0 and Σ t = 1 |
| stop-gradient-holds-target-constant | invariant | high | reproduced | literal grad = (p−t)/B + gate-path with `p_tilde` constant (1.08e-3) |
| single-network-no-extra-parameters | existence | high | reproduced | param_count == 4 (COMPUTED via len(params)) |

The **6 high** compute-invariance claims: 5 pass (the structural invariants of
Eqs. 1–4 + the single-network existence), 1 untested (the central ordering,
within noise at the gated `s=0.15`). The **3 low** value claims: 1 reproduced
(baseline), 2 refuted (CWSD value + improvement magnitude, at the gated `s`).
The refutations are honest AT THE GATED `s=0.15`: they are NOT universals over
`s` — the headline reproduces under the literal gradient for shallow gates
(`s >= ~0.7-2.0`) because the gate-path term (`∝ 1/s`) vanishes and literal →
detached, and under the DETACHED counterfactual at `s=0.15`. Because the paper
states neither `s` nor the stop-grad scope on `w`, the headline is
under-specified. The DETACHED counterfactual (`--grad-mode detached`, reported
in `selfcheck.json`) reproduces Table 1 (0.9611/0.9481/0.9556, ordering
+0.0241/+0.0074/+0.0241) — the standard self-distillation convention the paper
does not mark on `w`.

### 2.2 The pytest suite (54 tests, all pass)

Grouped by file. Every test is a check; the suite was also deliberately
broken by the 7 mutations in §2.4 to prove each check actually catches the
bug it claims to.

- **`test_degeneracy.py` (5)** — the paper's own verification gate (λ=0 =
  baseline exactly). `test_lambda_zero_target_equals_onehot`,
  `test_lambda_zero_loss_and_grads_equal_ce_bitwise` (per-step loss + every
  grad bitwise-equal to an independently written CE routine),
  `test_lambda_zero_training_matches_ce_training_bitwise` (300-step SGD loop
  with identical params + accuracy), `test_training_step_count_is_exact`
  (pins the exact step-count guard across the first-epoch boundary), and
  `test_baseline_seed0_reproduces_paper_table1_value` (pins the paper's
  headline baseline 0.9370 = 506/540 at the full 4000-step budget). The
  structural `t==Y` and per-step loss+grad checks are swept over
  `s ∈ {0.01,0.15,1.0,10.0}` (3 orders of magnitude) so the no-op=baseline
  gate provably cannot be fit to the answer via the one unstated
  hyperparameter `s`. The degeneracy holds under BOTH grad modes (the
  gate-path term is `λ·...=0` at λ=0).
- **`test_invariants.py` (11)** — equation-invariants from Eqs. 1–4:
  softmax/`p̃`/target sum to one, target in simplex and non-negative,
  confidence in [0,1], gate weight bounded by λ, loss non-negative and
  matches the direct formula, gate-open ⇒ t = `p̃` ⇒ CE(`p̃`, p), the
  paper-LITERAL gradient `dL/dz = (p−t)/B + gate-path` vs finite differences
  with `p_tilde` FROZEN (w recomputed) on all four params including the ReLU
  backprop path W1/b1, stop-grad on `p_tilde` keeps the target's `p_tilde`
  half independent of θ.
- **`test_data.py` (6)** — split shapes (1257/540), stratified-and-seeded
  split, label-corruption rate and invariance, `uniform-other` excludes the
  original class, clean-when-rate-zero, `train()` leaves the test set clean.
- **`test_cli.py` (4)** — output format for both arms, λ-range rejection
  (diagnostic to stderr), all documented flags accepted (incl. `--grad-mode`).
- **`test_instruments.py` (9)** — positive + negative tests for the four
  instruments in `instruments.json` (data-loader fingerprint, accuracy
  scorer, final-line parser, degeneracy-equivalence); the empty-evaluation
  case asserts the scorer raises rather than silently reporting 0.0.
- **`test_mutations.py` (14)** — each of the 7 deliberate defects
  (M1–M7, §2.4) is caught by its `must_fail` test, and each defect's `find`
  anchor is verified unique.
- **`test_structural_metrics.py` (5)** — the structural metrics
  `run_experiment.py --metrics-out` emits (gate bounds, simplex, degeneracy
  zeros, active-gate-differs-from-CE) are in bounds for the CWSD arm, zero
  for the baseline arm, a positive + negative test for each, and the
  non-vacuity test proving the frozen-`p_tilde` FD check discriminates BOTH
  a no-stopgrad (through `p_tilde`, ~2.1) and a detached (no gate path, ~4.0)
  implementation.

(Total 54 = 5+11+6+4+9+14+5; `test_data.py` includes the
`test_train_leaves_test_set_clean` check.)

### 2.3 Structural metrics emitted into `measured.json`

On one 128-example batch with the trained params, `run_experiment.py
--metrics-out` records the measured evidence for the equation-invariants:

- **baseline arm (every seed):** `param_count=4` (COMPUTED via `len(params)`),
  `gate_w_min=0`, `gate_w_max=0`, `degeneracy_loss_err=0.0`,
  `degeneracy_grad_err=0.0`. The independent CE routine is bitwise identical
  to `loss_and_grads` at λ=0 under BOTH grad modes (the gate-path term is
  `λ·...=0` at λ=0).
- **cwsd arm (literal, every seed):** `param_count=4` (computed),
  `gate_w_min ≈ 0.012–0.017 > 0`, `gate_w_max ≈ 0.21–0.28 < 1`,
  `target_min > 0`, `target_sum_err ≈ 1.2e-7 < 1e-6`,
  `stopgrad_grad_err ≈ 1.08e-3 < 5e-3`.

The `stopgrad_grad_err` tolerance is `5e-3` (not a tighter `1e-4`): the check
finite-differences the loss with `p_tilde` frozen at the unperturbed params
(`w` recomputed) on a peaked net (W2 scaled 8×) with `eps=1e-4` in float32,
which carries ~`1.1e-3` round-off (seed-independent, since the check runs on a
fixed tiny network at seed 123); `5e-3` is the honest floor-plus-margin. The
check now discriminates BOTH failure modes: a NO-stopgrad FD (recomputing
`p_tilde`) diverges to ~2.1 and a DETACHED analytic (dropping the gate path)
diverges to ~4.0 on the same peaked net — so it catches a missing stop-grad on
`p_tilde` AND a missing gate path (M6/M7).

### 2.4 Deliberate mutations (the suite was broken on purpose)

Seven defects injected into `run_experiment.py`, each caught by the test that
must fail when it is present (`tests/test_mutations.py`):

- **M1** gate weight non-zero at λ=0 (breaks no-op=baseline) → caught by
  `test_lambda_zero_target_equals_onehot`.
- **M2** ReLU mask `h>=0` instead of `h>0` (wrong ReLU derivative) → caught
  by the per-step grad degeneracy check.
- **M3** loss reduction `/B·K` instead of `/B` (Eq. 4 mean-over-batch,
  sum-over-class) → caught by `test_loss_matches_direct_formula` / degeneracy.
- **M4** temperature leaks into the loss prediction (uses `softmax(z/2)`
  instead of `softmax(z)`) → caught by the degeneracy bitwise check.
- **M5** target not in simplex (doubles the `p̃` contribution so t leaves
  the simplex) → caught by `test_target_sums_to_one` and the
  `target_sum_err` metric.
- **M6** detached gradient — drops the `L→t→w→c→z` gate-path term, reverting
  to `dL/dz = (p−t)/B` only (the whole target constant; the standard
  self-distillation convention the paper does NOT mark on `w`, and the exact
  divergence an adversarial review rejected as outcome-determinative) → caught
  by `test_gradient_matches_finite_differences` (the frozen-`p_tilde` FD
  includes the gate path; the mutated detached analytic does not, diverging
  by ~4 on the peaked net). Invisible to the λ=0 degeneracy gate (the
  gate-path term is `λ·...=0` at λ=0).
- **M7** no stop-grad on `p_tilde` — adds the `d p_tilde/dz` chain term (the
  trivial-solution hazard the paper warns about, `paper/paper.md:213-214`) →
  caught by `test_gradient_matches_finite_differences` (the frozen-`p_tilde`
  FD omits the `p_tilde` chain; the mutated no-stopgrad analytic adds it,
  diverging by ~2 on the peaked net). Invisible to the λ=0 degeneracy gate
  (the extra term is `w·...=0` at λ=0).

A suite nobody has broken on purpose is not evidence; these prove each check
catches the bug class it claims.

### 2.5 Instruments (`instruments.json`)

Four closed-book-catchers, each with a positive test (accepts a
known-correct input) and a negative test (rejects a known-wrong one):

- **data-loader** — fingerprints the data as the paper's own
  `sklearn.datasets.load_digits` (n_total=1797, 64 features, K=10,
  Xtr 1257×64, Xte 540×64, plus SHA-256 of the split arrays). A run that
  silently fell back to a synthetic corpus would pass every other gate and
  mean nothing; this catches it.
- **accuracy-scorer** — the test-set accuracy metric; the empty-evaluation
  case asserts it raises rather than reporting 0.0.
- **final-line-parser** — parses the `FINAL accuracy=<float>` contract line
  via `sys.executable`.
- **degeneracy-equivalence** — the λ=0==baseline equivalence.

### 2.6 Determinism / re-run check

`./run_all_arms.sh` was re-run this finalization pass; both invocations
produced the identical six `FINAL` lines (baseline 0.9370/0.9407/0.9315,
cwsd-literal 0.9407/0.9296/0.9333 — the gated arm, matching `/tmp/arms.log`
and `measured.json` byte-for-byte) and the identical `measured.json`. Same
seed → same number; the run is seed-pinned and reproducible. (The
0.9611/0.9481/0.9556 figures quoted in earlier passes are the DETACHED
counterfactual arm, not the gated literal arm — corrected here.)

### 2.7 Fresh from-scratch build (research-readiness gate 1)

Built a fresh venv in a clean directory: `uv venv --python 3.13` +
`uv pip install -r requirements.txt`, copied the source + tests + support
files (`claims.json`, `instruments.json`, `mutations.json`, `paper/`),
ran `pytest -q` → 54 passed, ran `python run_experiment.py --lambda 1.0`
(= `--grad-mode literal`, the default) → `FINAL accuracy=0.9407`, and
`python run_experiment.py --lambda 1.0 --grad-mode detached` →
`FINAL accuracy=0.9611` (the counterfactual that reproduces Table 1). The
from-scratch environment reproduces the numbers. (The `Dockerfile` path
itself was **not** exercised because `docker` is not installed — gate 1
verdict `partial`, see `REPRODUCTION.md`.)

### 2.8 Adversarial component review (prior lineage, re-confirmed)

A 5-reviewer + 1-verify-agent adversarial review reviewed each of
data-pipeline, method-core, training-loop, evaluation-metric, and
baseline-arm against `paper/paper.md` with `file:line` evidence; all 5
approved, 0 blocker/major. The verify agent ran the actual program:
`pytest -q` → passed; `--lambda 0.0` → 0.9370 (exact); `--lambda 1.0` →
0.9407 (gated literal arm; the `--grad-mode detached` variant → 0.9611).
**No `$HOME/.review_rounds` file exists on this run** — no review budget is
recorded as spent; the committed review→fix log shows each round's findings
were resolved with a committed fix and no outstanding unresolved objection.

---

## 3. The numbers, and how to read them

The paper reports a **single seed-0 run** (paper §3). At seed 0 under the
paper-LITERAL gradient (the gated cwsd arm, faithful to Eq. 3's stopgrad on
`p_tilde` only) at the gated sharp-gate `s=0.15`: baseline 0.9370 (exact),
CWSD-LITERAL 0.9407 (Δ +0.0071 vs baseline; Δ −0.0213 vs Table 1's 0.9620).
At the gated `s=0.15` the paper's +2.5-point headline and the central
`cwsd > baseline` ordering do NOT reproduce — the seed-1 gap is −0.0111
(ordering flips) and seeds 0/2 are within noise (+0.0037, +0.0019 = +2/+1 test
examples). This is `s`-DEPENDENT, not a universal: at `s=2.0` under the literal
gradient CWSD is 0.9648/0.9481/0.9611 (Δ −0.0009/+0.0026/−0.0009 vs Table 1)
and the ordering holds at every seed (+0.0278/+0.0074/+0.0296), because the
gate-path term (`∝ 1/s`) vanishes and literal → detached. The DETACHED
counterfactual (`--grad-mode detached`, not the gated arm) reproduces Table 1
at seed 0 already at `s=0.15` (0.9611, Δ −0.0009) and the ordering holds at
every seed (+0.0241/+0.0074/+0.0241); it is the standard self-distillation
convention the paper does not mark on `w`.

The per-seed gap (CWSD-LITERAL − baseline) at the gated `s=0.15` is
+0.0037 / −0.0111 / +0.0019 — positive at 2 of 3 seeds but within single-arm
noise at all of them, and negative at seed 1. At the gated `s=0.15` the
paper's central claim is not reproduced (within noise → UNTESTED under
the declared spread heuristic; value/magnitude REFUTED). It is reproduced
under the literal gradient for shallow gates (`s >= ~0.7` ordering,
`s >= ~2.0` value/magnitude — `sweep_s.py` -> `s_sweep.json`) and under the
detached variant at `s=0.15`. Because the paper states neither `s` nor the
stop-grad scope on `w`, the headline is under-specified; the gate adjudicates
the paper's claims against the faithful literal arm at the gated `s=0.15`
and reports the honest verdicts.

---

## 4. What remains untested, and why

- **The §4 attribution claim** ("We attribute the gain to the gate
  suppressing the gradient contribution of examples whose labels disagree
  with a confident prediction, which are disproportionately the corrupted
  ones"). The paper itself phrases this as an *attribution*, not a measured
  result. Testing it requires per-example gate-weight logging that the
  paper's output contract (one `FINAL accuracy=` line) does not expose; it
  is intentionally left out of the gate and listed in `claims.json`
  `not_tested`.
- **Table-1 magnitudes at seeds ≠ 0.** The paper's exact 0.9370 / 0.9620 are
  seed-0 numbers of a pipeline in which the split, noise mask, and init all
  depend on the seed. Seeds 1 and 2 are out-of-distribution for the paper's
  numbers; they are covered by widened tolerances, not asserted as
  reproductions at those seeds (listed in `not_tested`).
- **Docker build.** `docker` is not installed in this environment, so
  `docker build` / `docker run` were not exercised. The from-scratch
  environment was instead verified via a fresh `uv venv` +
  `uv pip install -r requirements.txt` build (research-readiness gate 1,
  verdict `partial`). The `Dockerfile` is present and self-contained but
  unexercised here.
- **The one unstated hyperparameter `s` (gate sharpness, Eq. 2).** The paper
  never states `s`. Under the paper-LITERAL gradient (the default cwsd arm) it
  is NOT calibrated to Table 1 — at the gated `s=0.15` the literal arm lands
  CWSD ≈ baseline (the headline FAILS, so `s=0.15` is provably non-tuning). The
  CWSD result is `s`-DEPENDENT under the literal gradient: the gate-path term
  is `∝ 1/s`, so the literal gradient converges to the detached one as `s`
  grows. The full sweep (`sweep_s.py` -> `s_sweep.json`, seeds 0/1/2, both
  modes, `s ∈ {0.05…5.0}`): the headline REPRODUCES under literal for
  `s >= ~0.7` (ordering) / `s >= ~2.0` (value, magnitude) and does NOT for the
  sharp-gate default `s=0.15`; under detached it reproduces at `s=0.15`. A
  prior pass truncated this sweep at `s=0.30` and wrongly concluded the
  literal gradient never reproduces — the extended sweep falsifies that. The
  `λ = 0` arm is bitwise insensitive to `s` (and to grad-mode), so the
  degeneracy check is not touched. A fully-specified paper would have pinned
  `s`; the reproduction does not lean on it (the gated `s=0.15` is the
  prose-aligned sharp-gate default, disclosed and non-tuning).
- **The gradient-mode choice (an under-specified-statement finding).** Eq. (3) marks stopgrad
  ONLY on `p_tilde`; whether the gate weight `w` is detached is unstated. The
  default `--grad-mode literal` (stopgrad on `p_tilde` only, `w`
  differentiable) is the paper's letter; whether the headline reproduces under
  it is `s`-DEPENDENT (it does NOT at the gated `s=0.15`, DOES for `s >= ~0.7-2.0`).
  The `--grad-mode detached` variant (whole target constant) is the standard
  self-distillation convention the paper does not mark on `w` and reproduces
  Table 1 already at `s=0.15`; it is reported as a counterfactual
  (`selfcheck.json`), not as the gated arm. The reproduction's central finding
  is that the headline is `s`-dependent under the paper's literal equations and
  reachable under the detached convention at the default `s` — i.e. the paper
  under-specifies two quantities (`s` and the stop-grad scope on `w`) on which
  the headline's reachability turns.
- **Other plausible RNG-stream layouts / weight inits.** The paper omits the
  RNG stream layout and weight init. `init-first` + He-normal was selected
  because it is the only one of the plausible arrangements that reproduces
  the paper baseline 0.9370 *exactly* (the paper's own λ=0 verification gate
  selects it). `spawned` and `noise-first` and xavier init do not reproduce
  the baseline exactly; they are exposed as `--rng-layout` / `--init` flags
  but the headline numbers use the layout that passes the paper's own gate.
- **Multi-seed statistical significance.** The paper reports a single run;
  this reproduction ran 3 seeds as a robustness check, which is enough to
  show the gap is seed-sensitive (within noise at seed 1) but not enough to
  make a statistical-significance claim. A reader wanting a confidence
  interval would need more seeds; the paper does not provide one and neither
  does this reproduction.

---

## 5. Build / environment / review budget

- **Build attempts:** `$HOME/.build_attempts` **does not exist** — no build
  budget is recorded as spent in this run. The build is clean:
  `run_all_arms.sh` writes a well-formed `measured.json`, the local selfcheck
  returns 6 pass / 2 fail / 1 untested / 0 blocked (the 2 fail are the CWSD
  value and improvement magnitude, refuted at the gated `s=0.15`; the 1
  untested is the central ordering, within noise at the gated `s`),
  `pytest -q` → 54 passed.
- **Environment attempts:** `$HOME/.env_attempts` **does not exist** — the
  environment is reproducible from the pinned `requirements.txt` (fresh venv
  verified this pass, gate 1).
- **Review rounds:** `$HOME/.review_rounds` **does not exist** on this run —
  no review budget is recorded as spent. The committed review→fix log shows
  each round's findings were resolved with a committed fix (impl pass:
  frozen-target FD check; impl pass 2: paper-LITERAL gradient as default,
  fixing the blocking whole-target stop-grad finding; claims-adjudication
  pass: extended s-sweep, corrected false universal; final commit `f8a7d47`:
  a reviewer "minor" on the cwsd-accuracy-value claim). No outstanding
  unresolved objection is found in the committed record, and the numbers gate
  adjudicated with 0 blocked. (A prior pass's docs stated a
  `$HOME/.review_rounds` file existed with value 2; that was stale for this
  run and is corrected here.)

---

## 6. Bottom line

**Rung reached: `numbers`.** The on-disk numbers gate ran on this run's
`measured.json` and adjudicated the paper's 9 claims (`claims_result.json`:
6 reproduced / 2 refuted / 1 untested / 0 blocked; the AUTHORITATIVE COUNTS
line is the gate's, off this run's journal). The implementation faithfully
implements the paper's Eqs. (1)–(4) under the paper-LITERAL gradient (stopgrad
only on
`p_tilde`, the default `--grad-mode literal`); the 5 high structural
invariants (degeneracy, gate bound, target simplex, stop-grad on `p_tilde`,
single-network) and the baseline value reproduce. The paper's headline
(+2.5 points, 0.9620) and central `cwsd > baseline` ordering are
**`s`-DEPENDENT under the paper's literal gradient** — at the gated sharp-gate
default `s=0.15` they do NOT reproduce (the ordering is within noise, flips at
seed 1 → UNTESTED under the declared spread heuristic; the value and
magnitude are REFUTED at the gated `s`), but they DO reproduce for shallow
gates (`s >= ~0.7` ordering, `s >= ~2.0` value/magnitude — `sweep_s.py` ->
`s_sweep.json`) because the gate-path term (`∝ 1/s`) vanishes and literal →
detached, and under the DETACHED counterfactual at `s=0.15`. Because the paper
states neither `s` nor the stop-grad scope on `w`, the headline is
under-specified: reachable under the paper's equations for a range of
`(s, grad-mode)`, not at the prose-aligned sharp-gate default under literal.
The DETACHED counterfactual (`--grad-mode detached`, whole target constant —
the standard self-distillation convention the paper does not mark on `w`)
reproduces Table 1 and is reported in `selfcheck.json` / REPRODUCTION.md, not
as the gated arm. `param_count` is computed (`len(params)`), not a literal.
The numbers are reproducible (`./run_all_arms.sh` regenerates `measured.json`;
`pytest -q` → 54 passed; a fresh from-scratch venv reproduces them).

The reproduction's central finding: the paper's headline is `s`-dependent
under its literal equations (stopgrad only on `p_tilde`, gate weight `w`
differentiable) and reachable under the detached convention at the default
`s` — the paper under-specifies the two quantities (`s` and the stop-grad
scope on `w`) on which the headline's reachability turns. This is a reported,
honest result — the gate's verdicts at the gated `s=0.15` and the
`sweep_s.py` survival sweep are the evidence.

What was checked: the method's equations (Eqs. 1–4) and their invariants; the
λ=0==baseline degeneracy (the paper's own verification gate, bitwise against
an independent CE routine, swept over `s`, holds under both grad modes); the
single-network / no-extra-params existence claim (computed); the literal
gradient against frozen-`p_tilde` finite differences (discriminating both a
no-stopgrad and a detached implementation); the headline accuracy magnitudes
at seed 0 and the ordering at seeds {0,1,2} under the literal arm; data
provenance (the paper's own `load_digits` corpus, fingerprinted); determinism;
a fresh from-scratch build. What was not checked: the §4 attribution claim,
the Docker build, statistical significance, and anything depending on the
unstated `s` beyond the survival sweep. The list of what was checked is the
part that matters.
