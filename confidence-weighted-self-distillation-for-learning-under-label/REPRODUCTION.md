# Reproduction: Confidence-Weighted Self-Distillation for Learning under Label Noise

- **Paper:** Confidence-Weighted Self-Distillation for Learning under Label Noise
- **Authors:** A. Bergstrom, M. Oyelaran, K. Vasquez (Institute for Applied Learning Systems)
- **Year:** unknown
- **arxiv_id:** unknown
- **Date:** 2026-08-06
- **paper_ref:** ce7a63e8-2c90-4516-887d-14515c8f4516
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Branch:** `repro/confidence-weighted-self-distillation-for-learning-under-label`

## Status

Current rung: **numbers** — the on-disk numbers gate ran on this run's
`measured.json` and adjudicated the paper's 9 claims (`claims_result.json`:
6 reproduced / 2 refuted / 1 untested / 0 blocked; the AUTHORITATIVE COUNTS
line is the gate's, read off this run's journal). The implementation is
faithful to the paper's equations (Eqs. 1–4); the gate adjudicates against the
paper-LITERAL gated arm at the sharp-gate default `s=0.15`.

- [x] Reproductions repo cloned (shallow `--depth 1 --filter=blob:none`) into
      `$HOME`; workspace folder present at the slug name
- [x] `$HOME/.repro_dir` and `$HOME/.repro_branch` written (no trailing newlines)
- [x] Branch `repro/confidence-weighted-self-distillation-for-learning-under-label`
      checked out and pushed (the ref already existed on the remote from the
      prior run; work continues on top of it, no force-push, default branch
      untouched)
- [x] Paper text saved to `paper/paper.md` (re-saved verbatim from the
      objective; byte-identical to the prior run's copy)
- [x] Comprehension (SPEC) — 2026-08-06: SPEC.md rewritten for this run; 24
      grep anchors re-resolved, 12/12 claims.json quotes verbatim, s-token
      inventory confirms Eq. (2)'s `s` has no value, Eq. (2) token layout fixes
      the `(c−τ)/s` reading; no figures; GitHub upstream search re-run (none);
      pinned env rebuilt (Python 3.13.5 + numpy 2.5.1 + scikit-learn 1.9.0,
      pytest 9.1.1) and prior results spot-re-executed — 54 tests pass,
      seed-0 baseline → 0.9370 (Table 1 exact), CWSD literal `s=0.15` → 0.9407,
      literal `s=2.0` → 0.9648, detached `s=0.15` → 0.9611, full `s_sweep.json`
      consistent. New this pass: explicit per-arm Bennett restriction check
      (§6). Unstated-gap list with weakest-reading choices: SPEC §4 (11 items).
- [x] Implementation / verification — 2026-08-06: adversarial review of all five
      components (data pipeline, method core, training loop, evaluation metric,
      baseline arm) against `paper/paper.md` via 5 independent subagents
      (`orchestrate` run `cwsd-verify-against-paper`). Verdict: **5/5 components
      match the paper**, 0 P0, 0 P1, 4 P3 (3 actionable polish). P3 fixes
      applied this pass: (1) corrected a stale line citation in
      `tests/test_degeneracy.py::test_training_step_count_is_exact` (284-298 →
      556-559); (2) parametrized
      `test_lambda_zero_loss_and_grads_equal_ce_bitwise` over `grad_mode ∈
      {literal, detached}` so the degeneracy-under-both-modes claim is
      regression-guarded (the structural reason it holds: the gate-path term is
      `lam·…` and vanishes at `lam=0`, `run_experiment.py:213-214`); (3) added
      `test_baseline_seed0_reproduces_paper_table1_value` pinning the paper's
      headline baseline value `0.9370` (=506/540, `paper/paper.md:400-406`) at
      the full 4000-step budget, so a regression that perturbs the baseline
      accuracy without breaking the bitwise-CE degeneracy test is still caught.
      Suite now **54 passed**. `run_all_arms.sh` re-run → `measured.json`
      byte-identical to the committed copy; `selfcheck_claims.py` → 6 pass / 2
      fail / 1 untested (unchanged: the CWSD value/magnitude claims are REFUTED
      at the gated sharp-gate default `s=0.15` under the paper-LITERAL gradient,
      the ordering UNTESTED within noise — the central finding, unchanged). The
      P3 `s=0.15` default note belongs to the results/numbers dimension and is
      already documented in SPEC §4 item 1 / §10 and `claims.json`; no code
      change there.
- [x] Adversarial review rounds — no `$HOME/.review_rounds` file exists on this
      run; the committed review→fix log shows each round's findings were
      resolved with a committed fix and no review budget was spent without the
      reviewers going quiet.
- [x] Readiness gates — walked (table below).
- [x] Publish — `publish_reproduction` called at rung `numbers`.

## Measured vs claimed

**The arms came out within noise of each other.** The baseline and CWSD
ranges overlap entirely (baseline 0.9315–0.9407; CWSD 0.9296–0.9407), and the
CWSD–baseline gap is negative at seed 1 (−0.0111) and within single-arm noise
at seeds 0 and 2 (+0.0037, +0.0019, i.e. +2 and +1 test examples of 540). The
gate's declared-spread heuristic therefore marks the central
`cwsd > baseline` ordering claim **UNTESTED** (mean gap −0.0019 ≤ spread
0.0148). **A measurement that cannot tell the arms apart has not tested the
paper's comparison**, whatever else it shows: the headline +2.5-point
*ordering* is not reproduced as a separable effect at the gated `s=0.15`. The
*value* claims (CWSD = 0.9620; gap = +0.025) are REFUTED at the gated `s` at
every seed. These verdicts are at the gated sharp-gate default `s=0.15` under
the paper-LITERAL gradient; they are `s`-dependent, not universal — see
"Under-specification" below and `sweep_s.py` / `s_sweep.json`.

Numbers below are at the **paper's full stated budget** (1797 8×8 digits, 64-unit
hidden layer, SGD lr 0.1, batch 64, **4000 steps** — no horizon was shortened
to fit the machine) on the **paper's own `sklearn.datasets.load_digits`
corpus**, not a synthetic stand-in. Data provenance: the data-loader
instrument (`instruments.json`) fingerprints the corpus as
`sklearn.datasets.load_digits` (n=1797, 64 features, K=10, Xtr 1257×64, Xte
540×64, plus the SHA-256 of the split arrays), so a silent fallback to a
synthetic dataset would be caught.

Exact commands (run from this reproduction folder with `.venv/bin/python`;
`./run_all_arms.sh` runs both arms × seeds {0,1,2} and emits the `FINAL
<arm>=<value>` lines plus `measured.json`):

```
.venv/bin/python run_experiment.py --lambda 0.0 --grad-mode literal --seed 0   # baseline, seed 0
.venv/bin/python run_experiment.py --lambda 1.0 --grad-mode literal --seed 0   # CWSD (literal), seed 0
# (repeat --seed 1 and --seed 2 for each arm; the gated cwsd arm uses grad-mode literal, s=0.15, tau=0.9, T=2)
```

| Arm (paper claim)            | seed | measured (command above) | paper claim | Δ measured − claim |
|------------------------------|------|--------------------------|-------------|--------------------|
| Cross-entropy baseline (λ=0) | 0    | 0.9370                   | 0.9370      | 0.0000 (exact)     |
|                              | 1    | 0.9407                   | — (seed-0 only) | —               |
|                              | 2    | 0.9315                   | — (seed-0 only) | —               |
|                              | mean | 0.9364                   | 0.9370      | −0.0006 (within tol) |
| CWSD (λ=1, literal, s=0.15)  | 0    | 0.9407                   | 0.9620      | −0.0213            |
|                              | 1    | 0.9296                   | —           | —                  |
|                              | 2    | 0.9333                   | —           | —                  |
|                              | mean | 0.9345                   | 0.9620      | −0.0275            |
| Improvement CWSD − baseline  | 0    | +0.0037                  | +0.0250     | −0.0213            |
|                              | 1    | −0.0111                  | +0.0250     | −0.0361            |
|                              | 2    | +0.0019                  | +0.0250     | −0.0231            |
|                              | mean | −0.0019                  | +0.0250     | −0.0269            |

The paper reports a single seed-0 run (paper §3); seeds 1 and 2 are this
reproduction's robustness check, out-of-distribution for the paper's exact
numbers, and are covered by widened tolerances, not asserted as reproductions
at those seeds. At seed 0 the baseline reproduces the paper's 0.9370 exactly
(506/540); CWSD does not (0.9407 vs 0.9620).

### Under-specification (why the headline is reachable but not at the gated default)

The paper never states the gate sharpness `s` (Eq. 2 defines it; §3 lists only
`λ=1, τ=0.9, T=2`) and never states the stop-grad scope on the gate weight `w`
(Eq. 3 marks stopgrad only on `p̃`). Under the paper-LITERAL gradient (the
default `--grad-mode literal`, stopgrad on `p̃` only, `w` differentiable) the
CWSD headline is `s`-dependent: the gate-path term is `∝ 1/s`, so the literal
gradient converges to the detached one as `s` grows. The full sweep
(`sweep_s.py` → `s_sweep.json`, seeds 0/1/2, both modes, `s ∈ {0.05…5.0}`):
the headline **reproduces** under literal for `s ≥ ~0.7` (ordering) / `s ≥ ~2.0`
(value, magnitude) — e.g. `s=2.0`: 0.9648/0.9481/0.9611, gaps
+0.0278/+0.0074/+0.0296 — and **does not** at the prose-aligned sharp-gate
default `s=0.15`. The **DETACHED** variant (`--grad-mode detached`, whole
target constant — the standard self-distillation convention the paper does not
mark on `w`) reproduces Table 1 already at `s=0.15`
(0.9611/0.9481/0.9556, gaps +0.0241/+0.0074/+0.0241); it is reported as a
counterfactual in `selfcheck.json`, not as the gated arm. Because the paper
states neither `s` nor the stop-grad scope on `w`, the headline is
under-specified: reachable under the paper's equations for a range of
`(s, grad-mode)`, not at the prose-aligned sharp-gate default under the literal
gradient. The gate adjudicates the paper's claims against the faithful literal
arm at the gated `s=0.15` and reports the honest verdicts.

## Readiness gates

Walked on 2026-08-06. `partial` is a gate honestly short of `pass`.

| # | Gate | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Builds from scratch | **partial** | Fresh `uv venv --python 3.13 --clear .venv` + `uv pip install -r requirements.txt` in a clean dir reproduces the numbers (`pytest -q` → 54 passed; `--lambda 0.0` → 0.9370; `--lambda 1.0` → 0.9407; `--lambda 1.0 --grad-mode detached` → 0.9611). A `Dockerfile` is present and self-contained, but `docker` is not installed in this environment so `docker build`/`docker run` were **not** exercised — the gate is `partial`, not `pass`, on that account. |
| 2 | README is accurate | pass | Quickstart in `README.md` run verbatim in a clean checkout: `uv venv … + uv pip install … + ./run_all_arms.sh` produces the six `FINAL` lines and `measured.json`; `.venv/bin/python run_experiment.py --lambda 0.0` / `--lambda 1.0` print the documented `FINAL accuracy=<float>` line. |
| 3 | Packages are clear | pass | `requirements.txt` pins every dependency with a version (numpy 2.5.1, scikit-learn 1.9.0, scipy 1.18.0, joblib 1.5.3, threadpoolctl 3.6.0, narwhals 2.24.0, pytest 9.1.1 + transitive). Fresh install from clean succeeded and the code did not die on a missing import. |
| 4 | Entrypoint is obvious | pass | One documented command, `python run_experiment.py --lambda FLOAT …`, flags for every knob; no source edits required. `./run_all_arms.sh` runs the full grid. |
| 5 | Fast path exists | pass | `./smoke.sh` exercises the whole code path (data→corrupt→init→train→eval→print) at 50 steps in ~1s, emitting `FINAL smoke=…` (deliberately not the contract line; not evidence about the paper). |
| 6 | Deterministic / noise quantified | pass | Same seed → same number: `./run_all_arms.sh` re-run produces the identical six `FINAL` lines and `measured.json`. Run-to-run spread is measured across seeds {0,1,2} and recorded (baseline 0.0092, CWSD 0.0111, gap spread 0.0148). |
| 7 | Degeneracy test in the repo | pass | `tests/test_degeneracy.py`: λ=0 reproduces an independent CE routine bitwise (per-step loss + every grad), swept over `s ∈ {0.01,0.15,1.0,10.0}` and both grad modes; `test_baseline_seed0_reproduces_paper_table1_value` pins 0.9370 at the full 4000-step budget. |
| 8 | Data provenance stated | pass | `sklearn.datasets.load_digits` (n=1797, 8×8, K=10), 30% stratified split at seed 0, 20% symmetric noise; the data-loader instrument fingerprints the corpus + split SHA-256. A pinned library call, not a description. |
| 9 | Recorded number reproducible | pass | The exact command beside each number (table above), re-run, reproduces it within the quantified noise (deterministic per seed). |
| 10 | Nothing depends on hidden local state | pass | Runs in a fresh clone in a fresh venv (gate 1 evidence); no home-directory or manually-fetched-wheel dependency. |

## Source notes

- The objective's `arxiv_id` is `unknown`, so **no arXiv LaTeX source exists to
  fetch** (`https://arxiv.org/e-print/<id>` is not applicable). Per the ingest
  instructions this is recorded here and work proceeds with the PDF-extracted
  text in `paper/paper.md`.
- Consequence: **maths in the extracted text is not authoritative** —
  extraction drops some glyphs (e.g. in Eq. (2) as extracted, the gate
  sharpness `s` appears only as a prose symbol with no numeric value; §3's
  hyperparameter sentence lists `λ=1, τ=0.9, T=2` and stops). Every equation
  must be treated as provisional and cross-checked against the surrounding
  prose before use; the SPEC step must re-derive any symbol it cannot confirm.

## Provenance

- This branch — and the current default branch — already contain a **complete
  earlier reproduction of this exact paper** (same paper_ref/project_id),
  taken to `rung=numbers` on 2026-08-04/05 (branch tip `c174cdd` at ingest
  time). Its artifacts (code, tests, SPEC, measured results, prior
  REPRODUCTION.log) remain in this folder as the **starting state**.
- Nothing from that prior run is trusted without re-verification: later steps
  re-derive the SPEC from `paper/paper.md`, re-run the experiments, and
  adversarially re-review the code against the paper before any publish call.
  The prior run's claims are unverified prior work product, not ground truth.
  The prior run's full REPRODUCTION.md content is recoverable from git history
  (`git show HEAD~1:<folder>/REPRODUCTION.md` after this commit, or the blob at
  `c174cdd`).
- Setup is committed on this run's branch, not on `main`; only the publish
  step touches the default branch.

## Log

- 2026-08-06 — Ingest: workspace set up on top of the existing remote branch;
  branch tip pushed; paper text re-saved verbatim (byte-identical);
  `arxiv_id` unknown recorded; fresh REPRODUCTION.md started for this run.
- 2026-08-06 — Comprehension pass: `SPEC.md` re-derived from `paper/paper.md`
  for this run (algorithm + shapes + cited equations; 11 unstated gaps each
  with permitted readings and the weakest choice; frozen interfaces; arms with
  per-arm restriction checks; claims.json re-verified 12/12 quotes verbatim;
  figures and upstream-code searches re-executed). Central finding retained
  and spot-re-verified: the CWSD headline is **under-specified** — the paper
  states neither the gate sharpness `s` nor the stop-grad scope on the gate
  weight `w`; under the paper-LITERAL gradient it reproduces only for shallow
  gates (`s ≥ ~0.7` ordering, `s ≥ ~2.0` value/magnitude) or under the
   detached convention, not at the prose-aligned sharp-gate default `s=0.15`.
- 2026-08-06 — Comprehension fix pass (review feedback): `claims.json` gained
  the required top-level **`restrictions` map** keyed by arm
  (`{"<arm>": {"kind", "detail"}}`): `baseline` → `none` (unstated knobs inert
  at `λ=0`; seeds {0,1,2} expand situations without touching correctness),
  `cwsd` → `narrows_situations` (one gated point of the paper's permitted
  `(s, grad-mode)` region; criterion untouched; sweep reports the
  neighbourhood) — mirroring SPEC §6, which already carried the prose
  restriction check; SPEC §8 now documents the machine-readable map.
  Re-verified after the edit: claims.json parses and the restrictions keys
  equal the declared arms, 12/12 quotes verbatim, `selfcheck_claims.py`
   re-run on the pinned env (numpy 2.5.1 / sklearn 1.9.0) → 6 pass / 2 fail /
   1 untested, `selfcheck.json` regenerated byte-identical to the committed
   copy, pytest 53 passed.
- 2026-08-06 — Implementation / verification pass: `orchestrate` run
  `cwsd-verify-against-paper` fanned out 5 independent adversarial reviewers
  (one per component: data pipeline, method core, training loop, evaluation
  metric, baseline arm), each reading `paper/paper.md` and the implementation
  with its tools and citing `file:line` for every finding. Verdict: **5/5
  components match the paper**, 0 P0, 0 P1, 4 P3. Applied 3 actionable P3
  fixes to `tests/test_degeneracy.py`: corrected a stale line citation
  (284-298 → 556-559); parametrized the bitwise-CE degeneracy test over
  `grad_mode ∈ {literal, detached}` so the degeneracy-under-both-modes claim
  is regression-guarded; added
  `test_baseline_seed0_reproduces_paper_table1_value` pinning the paper's
  headline baseline `0.9370` (506/540) at the full 4000-step budget. Suite
  53 → **54 passed**. Re-ran `run_all_arms.sh` → `measured.json` byte-identical
  to the committed copy; `selfcheck_claims.py` → 6 pass / 2 fail / 1 untested
  (unchanged). The numbers are unchanged because the test-only edits do not
  touch `run_experiment.py`; the fixes harden the regression suite that
  guards the (already-faithful) implementation.
- 2026-08-06 — Finalization pass: re-ran the arms (`./run_all_arms.sh`) →
  `measured.json` byte-identical to the committed copy and to `/tmp/arms.log`
  (baseline 0.9370/0.9407/0.9315, cwsd-literal 0.9407/0.9296/0.9333); the
  on-disk numbers gate ran on this run's `measured.json` → `claims_result.json`
  (6 reproduced / 2 refuted / 1 untested / 0 blocked; the AUTHORITATIVE COUNTS
  line is the gate's, off this run's journal). Re-walked the 10
  research-readiness gates (table above; gate 1 `partial` — fresh venv
  reproduces, `docker` not installed so the Dockerfile path was not exercised).
  Fresh from-scratch `uv venv` build in a clean dir reproduces (54 passed;
  0.9370 / 0.9407 / 0.9611). `$HOME/.build_attempts`, `$HOME/.env_attempts`,
  and `$HOME/.review_rounds` do **not** exist on this run — no build/env/review
  budget is recorded as spent; the review→fix log shows no outstanding
  unresolved objection. Recorded the measured-vs-claimed table with the exact
  commands, the within-noise caveat (arms ranges overlap, ordering UNTESTED),
  and data provenance (paper's own `load_digits`, fingerprinted). Rung reached:
  `numbers`.
