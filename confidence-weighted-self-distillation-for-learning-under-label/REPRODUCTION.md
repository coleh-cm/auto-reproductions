# Reproduction: Confidence-Weighted Self-Distillation for Learning under Label Noise

- **Paper:** Confidence-Weighted Self-Distillation for Learning under Label Noise
- **Authors:** A. Bergstrom, M. Oyelaran, K. Vasquez (Institute for Applied Learning Systems)
- **Year:** unknown
- **arxiv_id:** unknown
- **Date started:** 2026-08-04
- **paper_ref:** ce7a63e8-2c90-4516-887d-14515c8f4516
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Branch:** `repro/confidence-weighted-self-distillation-for-learning-under-label`

## Status

**Ingest complete.** Work is in `setup` stage; no code has been (re)validated by this run yet.

- [x] Reproduction workspace set up from `main`, branch
      `repro/confidence-weighted-self-distillation-for-learning-under-label` pushed
- [x] Paper text saved to `paper/paper.md`
- [x] Comprehension (SPEC) — 2026-08-04: SPEC.md and claims.json re-verified
      independently against `paper/paper.md` (all citations re-grepped, all 11
      quotes verbatim-checked, `s`-has-no-value re-confirmed, both arms and all
      three seeds re-executed: seed 0 baseline 0.9370 exactly / CWSD 0.9611;
      `48 passed`; no upstream code found on GitHub). See SPEC.md's second
      re-verification note for the command-level evidence.
- [x] Implementation / verification of existing code against the paper
- [x] Adversarial review rounds clean
- [ ] Readiness gates
- [ ] Publish

## Source notes

- The objective's `arxiv_id` is `unknown`, so **no arXiv LaTeX source could be
  fetched** (`https://arxiv.org/e-print/<id>` is not applicable). Per the
  ingest instructions, this is recorded here and work proceeds with the
  PDF-extracted text at `paper/paper.md`, which is reliable for prose but
  **not** for maths — some symbols may be silently missing (e.g. the gate
  sharpness `s` in Eq. (2) has no stated value in the extracted text).
- This slug's folder already exists on `main`: it holds the complete output of
  a prior reproduction run of the same paper (its `REPRODUCTION.md` reported
  rung `numbers`). Those artifacts (`run_experiment.py`, tests, `SPEC.md`,
  `VERIFICATION.md`, `measured.json`, etc.) are left in place as starting
  material; this run re-verifies them against the paper rather than trusting
  them. The prior run's stale remote branch tip (`9a07636`) was fast-forwarded
  to current `main` — the slug folder content was identical (`git diff` empty).

## Claimed results (paper Table 1, §4)

| Method | λ | Test accuracy |
|---|---|---|
| Cross-entropy (baseline) | 0 | 0.9370 |
| CWSD (ours) | 1 | 0.9620 |

Setup (§3): scikit-learn `load_digits` (1797 8×8 digits, K=10, pixels ÷16),
30% stratified test split at seed 0, 20% symmetric label noise, 64-hidden-unit
ReLU MLP, SGD lr 0.1, batch 64, 4000 steps; λ=1, τ=0.9, T=2; single seed-0 run.

## Implementation / verification pass (2026-08-04)

The prior run's implementation was re-verified end-to-end against the paper
rather than trusted. `run_all_arms.sh` was re-run on the pinned env (numpy
2.5.1, scikit-learn 1.9.0, Python 3.13): seeds 0/1/2 → baseline
0.9370/0.9407/0.9315, CWSD 0.9611/0.9481/0.9556; `measured.json` reproduced
byte-identically; `pytest -q tests` → 48 passed; smoke `FINAL smoke=0.8370`
(50-step, not a result). The numbers gate re-ran on the regenerated
`measured.json` and reported 9/9 reproduced (gate_pass), with
`claims_result.json` carrying the `produced_by: reproduce-paper numbers gate`
stamp — i.e. the gate wrote it, not this step.

## Adversarial review (orchestrate, 5 components × 2 reviewers)

An `orchestrate` workflow reviewed the five independent components (data
pipeline, method core, training loop, evaluation metric, baseline/degeneracy
arm) against `paper/paper.md` and `SPEC.md`, each with two reviewers under
distinct lenses (correctness-against-the-paper; silent-failure-modes).
Result: 0 blockers, 0 majors; 3 minors/nits fixed, 2 deferred with rationale.

### Fixes applied from review

1. **`evaluate()` now raises on an empty test set** (`run_experiment.py:346`).
   The prior code returned `nan` (a `RuntimeWarning`), which the
   `instruments.json` contract describes as "a grader that cannot run must
   raise, never return a negative verdict". `nan` is not a fabricated 0.0, but
   it is not a raise either; the contract is now honoured literally.
   `test_accuracy_scorer_must_raise_on_empty` asserts `pytest.raises(ValueError)`
   instead of accepting `nan`. (minor, evaluation-metric.)

2. **`parse_final_line` regex tightened to reject negatives**
   (`tests/test_instruments.py:183`). The prior pattern
   `^FINAL accuracy=(-?\d+\.?\d*)$` would parse a fabricated `FINAL accuracy=-0.5`
   as a float; accuracy is in `[0,1]`, so the optional leading minus is dropped
   and a negative line is rejected to BLOCKED. A new negative-case assertion is
   added to `test_final_line_parser_negative`. (nit, evaluation-metric.)

3. **`mutations.json` `_comment` made honest about M5**. The blanket claim
   "Each defect covers `core` and `degeneracy`" was false for M5
   (`M5-target-not-in-simplex`): at `λ=0` the gate weight `w=0` makes the doubled
   `p_tilde` term inert, so M5 is invisible to the λ=0 degeneracy gate and is
   instead caught by the `test_target_sums_to_one` simplex invariant. The
   `_comment` now states that M1-M4 cover `core`+`degeneracy` and M5 covers
   `core`+`simplex`. No defect escapes the suite; the metadata no longer
   overstates the degeneracy coverage. (minor, baseline-degeneracy-arm.)

### Deferred (with rationale)

- **`--steps 0` misuse path** (training-loop, minor): `main()` does not guard
  against a zero-step budget, so a hypothetical `--steps 0` prints a
  `FINAL accuracy=<untrained>` line. This is never reachable in the actual
  pipeline (`run_all_arms.sh` and `smoke.sh` both use a positive budget; the
  paper fixes 4000), and `test_training_step_count_is_exact` already asserts
  the exact-step-count invariant. Not fixed to avoid changing the step-count
  contract the test pins; flagged here for honesty.
- **`run_all_arms.sh` relies on `run_experiment.py` defaults**
  (training-loop, nit): the script passes only `--lambda --seed --metrics-out`
  and relies on argparse defaults for the other hyperparameters. The defaults
  provably match `claims.json` config exactly (verified: steps 4000, lr 0.1,
  batch 64, s 0.15, tau 0.9, T 2.0, noise 0.2 uniform-all, init he,
  batch-mode epoch-permutation, rng-layout init-first), so the full paper
  config IS run. Pinning every field explicitly would guard against a future
  default drift but is not a current correctness issue.
- **`data-pipeline` and `method-core` reviewer stalls**: two of the ten
  reviewers returned no verdict. The covered components (training-loop,
  evaluation-metric, baseline-degeneracy-arm) all passed; the two stalled
  components were re-verified by direct inspection during the fixes above
  (data loader fingerprinted by `test_data_loader_*`; Eq.2/3/4 operator
  precedence, axis reductions, temperature scope, and stop-grad confirmed
  against `run_experiment.py:144-198` and `paper/paper.md:108-252`).
