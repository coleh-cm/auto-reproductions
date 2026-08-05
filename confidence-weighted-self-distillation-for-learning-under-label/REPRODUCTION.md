# Reproduction: Confidence-Weighted Self-Distillation for Learning under Label Noise

- **Paper:** Confidence-Weighted Self-Distillation for Learning under Label Noise
- **Authors:** A. Bergstrom, M. Oyelaran, K. Vasquez (Institute for Applied Learning Systems)
- **Year:** unknown
- **arxiv_id:** unknown
- **Date:** 2026-08-04
- **paper_ref:** ce7a63e8-2c90-4516-887d-14515c8f4516
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Branch:** `repro/confidence-weighted-self-distillation-for-learning-under-label`

## Status

Current rung: **numbers** — the on-disk numbers gate ran on this run's
`measured.json` and adjudicated all 9 claims (its output is `claims_result.json`,
verdicts 6 reproduced / 2 refuted / 1 untested / 0 blocked; the AUTHORITATIVE
COUNTS line is the gate's, read off this run's journal). The paper's Table-1 headline
(+2.5 points, 0.9620) and the central `cwsd > baseline` ordering are
**`s`-DEPENDENT under the paper's LITERAL gradient** (stopgrad only on `p_tilde`,
as Eq. 3 marks it) — NOT a universal. The paper never states the gate sharpness
`s` (Eq. 2 defines it; §3 lists only `lambda=1, tau=0.9, T=2`). At the gated
sharp-gate default `s=0.15` (prose-aligned — "s controls how sharply the gate
opens", paper/paper.md:164 — and provably non-tuning, since the headline FAILS
there under literal) the headline does NOT reproduce (CWSD ≈ baseline, ordering
within noise / flips at seed 1). But the gate-path gradient term is `∝ 1/s`, so
the literal gradient CONVERGES to the DETACHED one as `s` grows: the full s-sweep
(`sweep_s.py` -> `s_sweep.json`) across seeds 0/1/2 shows the headline
REPRODUCES under the literal gradient for shallow gates — ordering holds at
every seed for `s >= ~0.7`, value and magnitude for `s >= ~2.0` (e.g. `s=2.0`:
0.9648/0.9481/0.9611) — and under the DETACHED variant (whole target constant,
which the paper does not state on `w`) already at `s=0.15` (0.9611/0.9481/0.9556).
Because the paper states neither `s` nor the stop-grad scope on `w`, the
headline is **under-specified**: reachable under the paper's equations for a
range of `(s, grad-mode)`, but not at the prose-aligned sharp-gate default under
the literal gradient. The reproduction implements the literal gradient as the
default (faithful) and pins `s=0.15` for the gated arm with a disclosed
non-headline rationale; the detached variant is reported as a counterfactual.
The 5 high structural invariants + the baseline value reproduce; the two CWSD
value/magnitude claims are REFUTED at the gated `s=0.15`; the central ordering
is UNTESTED (within noise, flips at seed 1) under the declared spread
heuristic. `selfcheck_claims.py` → `selfcheck.json` reports 6 pass / 2 fail /
1 untested / 0 blocked at all 3 seeds; `measured.json` regenerated.

- [x] Reproductions repo cloned (shallow, blobless) into `$HOME`; workspace folder
      present at the slug name
- [x] `$HOME/.repro_dir` and `$HOME/.repro_branch` written (no trailing newlines)
- [x] Branch `repro/confidence-weighted-self-distillation-for-learning-under-label`
      created from `origin/main` and pushed
- [x] Paper text saved to `paper/paper.md`
- [x] Comprehension (SPEC)
- [x] Implementation / verification (this pass: paper-LITERAL gradient made the
      default; DETACHED kept as counterfactual; `param_count` computed;
      stop-grad FD check rewritten to freeze `p_tilde` (w recomputed) and
      discriminate both no-stopgrad and detached; M6 detached + M7 no-stopgrad
      mutations added; 53 tests pass; arms re-run; measured.json regenerated)
- [x] Adversarial review rounds: the prior pass was REJECTED on one blocking,
      outcome-determinative finding (whole-target stop-grad contradicts Eq. 3);
      this pass fixes it
- [x] Readiness gates (see `## Research-readiness gates` below)
- [ ] Publish (this finalization pass: numbers recorded, gates walked,
      VERIFICATION.md updated, branch pushed and default branch brought up;
      `publish_reproduction` called)

## Measured vs paper-claimed (this run's arms)

**The two arms came out within noise of each other.** At the gated arm
(paper-LITERAL gradient, `--grad-mode literal`, sharp-gate default `s=0.15`)
the baseline and CWSD accuracy ranges overlap completely across seeds 0/1/2
(baseline 0.9315–0.9407, CWSD 0.9296–0.9407); the per-seed CWSD−baseline gap is
+0.0037 / −0.0111 / +0.0019 (mean −0.0019), whose spread across seeds (0.0148)
is ~8× the magnitude of the mean gap and straddles zero (the ordering flips at
seed 1). A measurement that cannot tell the arms apart has not tested the
paper's comparison, whatever else it shows, and the gated-arm numbers below are
reported on that basis. (The headline IS reachable under the paper's equations
for other `(s, grad-mode)` — see `## Literal vs detached, and the unstated s`
— but not by this gated arm.)

**Where the data came from.** scikit-learn `load_digits` — the paper's own
stated benchmark (1797 8×8 digits, K=10, §3), fingerprinted by the
`data-loader` instrument in `instruments.json` (n_total=1797, 64 features,
Xtr 1257×64, Xte 540×64, plus a SHA-256 of the split arrays). This is the
paper's dataset, not a synthetic stand-in; a number measured on a synthetic
substitute would say nothing about the paper and is not used here.

**Horizon.** Every arm ran the paper's full stated 4000-step SGD budget (§3);
no horizon was shortened to fit the machine. (A number produced at a horizon
too short to separate the arms would not be evidence about the paper's claim;
this run did not do that.)

**Exact commands.** The paper's own two commands (§5), run with all defaults
(`--seed 0`, `--grad-mode literal`, `s=0.15`, `τ=0.9`, `T=2`, 4000 steps,
lr 0.1, batch 64, 20% symmetric noise):

```
python run_experiment.py --lambda 0.0    # baseline  -> FINAL accuracy=0.9370
python run_experiment.py --lambda 1.0    # CWSD       -> FINAL accuracy=0.9407
```

The full grid (both arms × seeds {0,1,2}) used by the gate is produced by
`./run_all_arms.sh`, which invokes, per arm-seed:

```
.venv/bin/python run_experiment.py --lambda <0.0|1.0> --grad-mode literal --seed <0|1|2> --metrics-out <tmp>
```

and writes `measured.json` plus one `FINAL <arm>=<value>` line per arm-seed to
`/tmp/arms.log`.

| Arm | λ | Paper claim (Table 1) | Measured (seed 0 / 1 / 2) | Measured mean | Diff vs claim (seed 0) | Exact command (seed 0) |
|---|---|---|---|---|---|---|
| Cross-entropy (baseline) | 0 | 0.9370 | 0.9370 / 0.9407 / 0.9315 | 0.9364 | 0.0000 | `python run_experiment.py --lambda 0.0` |
| CWSD (gated, literal, `s=0.15`) | 1 | 0.9620 | 0.9407 / 0.9296 / 0.9333 | 0.9345 | −0.0213 | `python run_experiment.py --lambda 1.0` |

The paper's headline improvement is **+0.025** (0.9620 − 0.9370). This run's
gated arm measures **−0.0019** on average (CWSD *below* baseline), with the
per-seed gap +0.0037 / −0.0111 / +0.0019 — i.e. the gated-arm measurement does
not reproduce the +2.5-point claim and, within its own seed spread, cannot
distinguish the two arms. Tolerance is not this reproduction's to decide; the
numbers and the difference are stated and the reader is left to judge. The
numbers gate's own verdicts on these claims (read off this run's journal, not
tallied here) are reported in `claims_result.json` and summarised in
`VERIFICATION.md`.

**Counterfactual arms (NOT the gated arm — reported for completeness, since
the headline is reachable under them).** Under `--grad-mode detached` (whole
target constant, the standard self-distillation convention the paper does not
mark on `w`) at the same `s=0.15`: CWSD 0.9611 / 0.9481 / 0.9556 (gap
+0.0241 / +0.0074 / +0.0241 vs baseline) — Table 1 reachable. Under the literal
gradient at `s=2.0`: CWSD 0.9648 / 0.9481 / 0.9611 (gap +0.0278 / +0.0074 /
+0.0296) — Table 1 also reachable. These are recorded in `selfcheck.json`
(`counterfactual_detached`) and `s_sweep.json` (`sweep_s.py`).

## Research-readiness gates

| # | Gate | Verdict | Evidence |
|---|---|---|---|
| 1 | Builds from scratch | **partial** | Fresh `uv venv --python 3.13` + `uv pip install -r requirements.txt` reproduces the numbers (`pytest -q` → 53 passed; `--lambda 0.0` → 0.9370; `--lambda 1.0` → 0.9407). The `Dockerfile` path was NOT exercised — `docker` is not installed in this environment. The Dockerfile is present and self-contained but unexercised here. |
| 2 | Accurate README | **pass** | `README.md` describes the method, both grad modes, the unstated-`s` finding, the exact commands, and the measured-vs-claimed table; quickstart (uv / pip / Docker) reproduces. |
| 3 | Clear packages / deps | **pass** | `requirements.txt` pins numpy/scikit-learn/pytest; `.venv` excluded by `.gitignore`; no stray unpinned imports (`run_experiment.py` is numpy + sklearn only). |
| 4 | Obvious entrypoint | **pass** | `run_experiment.py` (single arm, `--lambda` required) and `run_all_arms.sh` (full grid → `measured.json`) are the documented entrypoints; each prints exactly one `FINAL accuracy=<float>` line as the paper's §5 contract specifies. |
| 5 | Reproducible number | **partial** | Seed-pinned: `./run_all_arms.sh` re-run produces byte-identical `measured.json` and the same six `FINAL` lines (determinism check, VERIFICATION.md §2.6). The baseline reproduces the paper's 0.9370 exactly at seed 0. The CWSD headline (0.9620) is NOT reproduced by the gated arm at `s=0.15` (it is `s`-dependent and reachable only under other `(s, grad-mode)`); the central ordering is within noise. The number is reproducible; whether it reproduces the paper is the open question the table above states. |
| 6 | Data provenance | **pass** | `load_digits` (the paper's own dataset) fingerprinted by the `data-loader` instrument; no synthetic stand-in. |
| 7 | Tests exist & pass | **pass** | `pytest -q tests` → 53 passed; 7 deliberate mutations each caught by a `must_fail` test (suite broken on purpose to prove the checks bite). |
| 8 | Adversarial review | **partial** | Review rounds ran (see `## Review-rounds note`); each surfaced finding was resolved with a committed fix (the blocking whole-target stop-grad finding → `--grad-mode literal` default; the false "never reproduces at any `s`" universal → extended s-sweep). No standing unresolved objection is recorded in the committed log; the numbers gate returned 0 blocked. The review-rounds file exists (`$HOME/.review_rounds` = 2), so this pass is recorded as `partial` rather than a clean pass — see the note. |

## Review-rounds note

`$HOME/.review_rounds` exists with value **2** on this run (the prior pass's
docs, written before this finalization, state the file did not exist — that
claim is now stale and corrected here). Two review rounds are recorded as
spent. The committed log shows each review round's findings were resolved
with a fix (impl pass: frozen-target FD check; impl pass 2: paper-LITERAL
gradient as default, fixing the blocking whole-target stop-grad finding;
claims-adjudication pass: extended s-sweep, corrected false universal; final
commit `f8a7d47`: a reviewer "minor" on the cwsd-accuracy-value claim noted).
No outstanding unresolved objection is found in the committed record, and the
numbers gate adjudicated with 0 blocked. Whether the review budget was fully
exhausted is not determinable from the file alone; this reproduction does not
claim an unqualified clean review pass on that account, and a reader should
weigh the committed review→fix log rather than the round count.

## Source notes

- The objective's `arxiv_id` is `unknown`, so **no arXiv LaTeX source could be
  fetched** (`https://arxiv.org/e-print/<id>` is not applicable). Per the ingest
  instructions this is recorded here and work proceeds with the PDF-extracted
  text in `paper/paper.md`.
- Consequence: **maths in the extracted text is not authoritative** — extraction
  drops some glyphs (e.g. in Eq. (2) as extracted, the gate sharpness `s` appears
  only as the prose symbol with no numeric value). Every equation must be
  treated as provisional and cross-checked against the prose before use; the
  SPEC step must re-derive any symbol it cannot confirm.

## Provenance

- This repository's default branch already contains a **complete earlier
  reproduction of this exact paper** (same paper_ref/project_id), merged via
  commit `53d7bb1` ("Merge repro/explaining-and-harnessing-adversarial-examples
  ... CWSD and other papers preserved from origin/main"). Its branch, now fully
  merged and redundant, still existed on the remote.
- This run branches from current `origin/main`, so that prior reproduction's
  artifacts (code, tests, SPEC, measured results) remain in this folder as the
  **starting state**. Nothing from the prior run is trusted without
  re-verification: later steps re-derive the SPEC from `paper/paper.md`, re-run
  the experiments, and adversarially re-review the code against the paper
  before any publish call. The prior run's claims are treated as unverified
  prior work product, not ground truth.
- Setup is committed on this run's branch, not on `main`; only the publish step
  touches the default branch.

## Log

- 2026-08-04 — Ingest: workspace set up; branch pushed; paper text saved;
  `arxiv_id` unknown recorded; fresh REPRODUCTION.md started for this run.
- 2026-08-04 — Comprehension (SPEC): re-derived from `paper/paper.md`, not from
  the prior run's notes. Independently re-verified: all 11 claims.json quotes
  verbatim (11/11), all 24 grep anchors, Eq. (2)'s `s` valueless (tokens only at
  paper.md:139/:162; hyperparameter sentence lists λ=1, τ=0.9, T=2 and stops);
  no figures (nothing for `read-figure`); no URLs in paper; GitHub
  repo+user searches all `total_count: 0` (no upstream code). Arms re-run on a
  fresh pinned env (numpy 2.5.1, sklearn 1.9.0, Python 3.12.13): baseline
  0.9370/0.9407/0.9315, CWSD 0.9611/0.9481/0.9556 (seeds 0/1/2) — byte-identical
  to committed `measured.json`; baseline seed 0 matches Table 1 exactly; ordering
  holds at every seed. Structural predicates all pass (degeneracy errors 0.0
  bitwise; gate/target/stop-grad bounds as claimed). Calibration evidence
  re-measured: rng-layout spawned 0.9315 / noise-first 0.9426; s sensitivity
  0.12→0.9593 … 0.18→0.9648; λ=0 bitwise s-independent. `pytest -q tests` →
  48 passed. SPEC.md rewritten with the single-pass verification note;
  claims.json unchanged (still correct per this pass).
- 2026-08-04 — Implementation pass (adversarial review → fix). An orchestration
  of 5 component reviewers (data pipeline, method core, training loop,
  evaluation metric, baseline/degeneracy arm) against `paper/paper.md`, each
  finding verified by a second refuter agent, surfaced ONE confirmed real
  (minor, validation-only) defect: `_stopgrad_grad_err` and
  `test_gradient_matches_finite_differences` finite-differenced the scalar loss
  value, but `loss_and_grads` recomputes the target `t` from the perturbed `z`
  on every call, so the value-FD returned the FULL no-stopgrad gradient — the
  check passed only by coincidence on a near-uniform tiny net (p≈uniform ⇒ the
  dt/dz chain term vanished). The core method (lines 130-198) was always
  correct (stop-grad structural via the hand-derived `dz=(p-t)/B` with `t` a
  plain detached array); only the validation check's evidentiary claim was
  wrong. FIX: added `_loss_with_frozen_target` (freezes `t` at unperturbed
  params, FDs only the log-p term); both checks now use it on a PEAKED net
  (W2×8, p non-uniform) where the no-stopgrad gradient diverges by ~6.1 — so
  the check now actually distinguishes a correct stop-grad from a no-stopgrad
  implementation. Added `test_stopgrad_grad_err_is_nonvacuous` proving the
  frozen-target FD matches the stopgrad analytic while the recomputing-t FD
  diverges; added mutation M6 (a no-stopgrad gradient `dz=(p-t)/B +
  lam*(p_tilde-p)/B` that is INVISIBLE to the λ=0 degeneracy gate by
  construction, since the extra term is `0` at λ=0) caught by
  `test_gradient_matches_finite_differences`. Suite 51 passed (was 48).
  `measured.json` regenerated: accuracy arms byte-identical; `stopgrad_grad_err`
  now 1.16e-3 (was 8.93e-4, both < 5e-3). Added `selfcheck_claims.py` →
  `selfcheck.json` (local evaluator; deliberately NOT `claims_result.json`,
  which the gate owns): 9/9 claims pass at all 3 seeds. Removed stale
  `claims_result.json` committed by a prior run (the gate refuses any copy it
  did not produce). All four other components cleared review with no confirmed
  issues.
- 2026-08-04 — Implementation pass 2 (adversarial REJECTION → blocking fix). A
  follow-up adversarial review REJECTED the above on one blocking,
  outcome-determinative finding (verified by independent re-execution, not
  from docs): the implementation applied the stop-gradient to the WHOLE target
  `t`, but Eq. (3) marks stopgrad ONLY on `p_tilde` ("the latter [= p_tilde]
  treated as a constant", paper/paper.md:171-174, :198). Under the paper-LITERAL
  gradient (stopgrad on `p_tilde` only; the gate weight `w = λσ((c−τ)/s)` is
  differentiable in `z` through `c = max_k p_k`, so the `L → t → w → c → z`
  path is included) the Table-1 headline +2.5 points and the central
   `cwsd > baseline` ordering do NOT reproduce at the gated `s=0.15` — re-measured at seeds 0/1/2:
   CWSD-LITERAL 0.9407 / 0.9296 / 0.9333 vs baseline 0.9370 / 0.9407 / 0.9315,
   i.e. the ordering flips at seed 1 and is within noise at seeds 0/2. (This
   pass stated the non-reproduction as a universal over `s`; the 2026-08-05
   entry below falsifies that — the literal gradient DOES reproduce for
   `s >= ~0.7-2.0`.) The
   detached-t variant (the prior primary arm) reproduces Table 1 (0.9611 /
   0.9481 / 0.9556), but it is the standard self-distillation convention the
   paper does NOT mark on `w`. The prior pass therefore reached the paper's
   number through a mechanism the paper forbids by omission — exactly what the
   gate exists to prevent.

  FIX (this pass):
  (1) Added `--grad-mode literal|detached` (default **literal**, paper-faithful):
  stopgrad ONLY on `p_tilde`, gate weight `w` differentiable; the full literal
  gradient `dL/dz = (p − t)/B + gate-path term` is hand-derived
  (`_gate_path_grad`) and verified against a frozen-`p_tilde` finite-difference
  to ~1e-3 on a peaked net. The DETACHED variant (whole target constant) is kept
  as a documented COUNTERFACTUAL — a variant under which Table 1 is reachable
  (the literal gradient also reaches it for shallow gates; see 2026-08-05)
  — reported in `selfcheck.json` / REPRODUCTION.md, NOT as the gated arm.
  (2) Rewrote the stop-grad FD check to freeze `p_tilde` (not the whole `t`),
  with `w` recomputed; it now discriminates BOTH failure modes — a no-stopgrad
  (through `p_tilde`) FD diverges (~2.1) and a detached (no gate path) analytic
  diverges (~4.0) on the peaked net.
  (3) Replaced the prior M6 mutation with two: M6 (detached, drop the gate-path
  term — the exact divergence the review rejected) and M7 (no stop-grad on
  `p_tilde`, the trivial-solution hazard the paper warns about); both caught by
  the rewritten FD check.
  (4) `param_count` is now COMPUTED (`len(params)`) rather than a hardcoded
  literal `4` (a prior review noted the literal would report 4 even if
  parameters were added).
   (5) `claims.json` cwsd arm uses `--grad-mode literal`; the value/ordering
   claims are retained as the paper's claims so the gate adjudicates them
   honestly against the literal arm (they fail at the gated `s=0.15`); the
   detached counterfactual is in `not_tested`. `--s` is no longer described as
   "calibrated to Table 1" (the literal arm does not reproduce Table 1 at the
   gated `s=0.15`; the s-sweep shows it DOES reproduce at `s >= ~0.7-2.0`).

   Result (re-measured this pass, seeds 0/1/2): baseline 0.9370/0.9407/0.9315
   (grad-mode-independent at λ=0, degeneracy holds under literal too — the
   gate-path term is `λ·...=0`); CWSD-LITERAL 0.9407/0.9296/0.9333; CWSD-DETACHED
   (counterfactual) 0.9611/0.9481/0.9556. `selfcheck.json`: 6 pass (5 high
   structural invariants + baseline value) / 2 fail (cwsd value, improvement
   magnitude, refuted at the gated `s=0.15`) / 1 untested (central ordering:
   within noise, flips at seed 1, under the declared spread heuristic) / 0
   blocked. `pytest -q tests` → 53 passed. The reproduction's honest conclusion:
   the paper's headline is `s`-dependent under the literal gradient — it does
   NOT reproduce at the prose-aligned sharp-gate default `s=0.15`, but DOES for
   shallow gates `s >= ~0.7-2.0` (literal → detached) and under the detached
   variant at `s=0.15`; because the paper states neither `s` nor the stop-grad
   scope on `w`, the headline is under-specified, not uniformly refuted.

- 2026-08-05 — Adversarial review → claims-adjudication fix. A follow-up
  adversarial review REJECTED the pass above on one blocking,
  outcome-determinative finding (verified by independent re-execution in the
  repo venv, not from docs): the repo's headline conclusion — "the paper-LITERAL
  gradient does NOT reproduce Table 1 at any `s`; the headline is reachable
  only under the DETACHED variant" — was empirically FALSE. The reviewer
  re-ran CWSD under `--grad-mode literal` at the unstated `s=2.0` and got
  0.9648/0.9481/0.9611 at seeds 0/1/2 vs baseline 0.9370/0.9407/0.9315, so the
  ordering, value and magnitude claims ALL pass at every seed. The mechanism is
  structural: the gate-path gradient term is `∝ 1/s`, so the literal gradient
  converges to the detached one as `s` grows; the prior pass's own sensitivity
  sweep stopped at `s=0.30`, exactly before the regime that falsifies the
  universal. The implementation of Eqs. (1)-(4) was faithful in both grad modes
  (all three reviews concur; 53 tests pass; degeneracy bitwise exact); the
  failure was in the claims-adjudication layer.

  FIX (this pass):
  (1) Extended the s-sweep to `s=5.0` across seeds 0/1/2 in BOTH grad modes
  (`sweep_s.py` -> `s_sweep.json`); documented crossover (ordering at `s~0.7`,
  value/magnitude at `s~2.0`) in SPEC §4 item 1.
  (2) Corrected every "at any `s`" / "only under detached" universal at
  claims.json, SPEC.md, VERIFICATION.md, run_experiment.py help text and this
  file to the `s`-dependent truth.
  (3) Re-adjudicated the cwsd arm under a STATED `s` policy: pin `s=0.15`
  (sharp-gate default, prose-aligned, provably non-tuning — the headline fails
  there). The value/magnitude claims are REFUTED at the gated `s`; the ordering
  is UNTESTED under the declared spread heuristic.
  (4) Declared ONE verdict rule for the ordering claim (a spread heuristic:
  within-noise → `untested`, not `refuted`), now consistent between
  `claims.json`, the numbers gate, and `selfcheck_claims.py`.
  (5) Removed the stale committed `claims_result.json` (the gate owns it).
   `measured.json` regenerated (byte-identical: the gated `s=0.15` is
   unchanged); `pytest -q tests` → 53 passed; `selfcheck.json` → 6 pass / 2 fail
   / 1 untested / 0 blocked.

- 2026-08-05 — Finalization / publish. Arms re-run by `./run_all_arms.sh`
  (`/tmp/arms.log`): baseline 0.9370/0.9407/0.9315, cwsd-literal 0.9407/0.9296/
  0.9333 (seeds 0/1/2) — byte-identical to `measured.json`; baseline seed 0 =
  Table 1 exactly. The numbers gate ran on this run's `measured.json` and
  produced `claims_result.json` (6 reproduced / 2 refuted / 1 untested / 0
  blocked — the AUTHORITATIVE COUNTS line is the gate's, off this run's
  journal). Rung set to **numbers** (the gate ran and adjudicated). Added the
  `## Measured vs paper-claimed (this run's arms)` section leading with the
  within-noise finding (the gated arms' ranges overlap fully; the gap spread
  0.0148 ≫ |mean gap| 0.0019 and straddles zero), the exact commands, the data
  provenance (`load_digits`, the paper's own dataset, fingerprinted — not a
  synthetic stand-in), and the full-horizon note (4000 steps, not shortened).
  Added the `## Research-readiness gates` table (5 pass / 3 partial: build
  without Docker, reproducible number that does not reproduce the headline at
  the gated `s`, adversarial review with `$HOME/.review_rounds` = 2). Added the
  `## Review-rounds note` correcting the prior docs' stale "no review_rounds
  file" claim. Updated `VERIFICATION.md` rung → `numbers` and the same
  review-rounds correction. `pytest -q tests` → 53 passed. Branch pushed and
  the default branch brought up to it (merge, not force).

## Literal vs detached, and the unstated `s` — the central finding

The paper's Eq. (3) annotates `stopgrad` ONLY on `p_tilde`
(`paper/paper.md:198`, "the latter treated as a constant", `:171-174`). The
gate weight `w = λσ((c−τ)/s)` is unmarked and is a function of `θ` through
`c = max_k p_k`. Two faithful readings of the under-specified stop-grad, AND one
unstated hyperparameter `s` (Eq. 2 defines it; §3 never assigns it), combine:

| reading | stop-grad on | gate path `w` | `s` | CWSD seed 0/1/2 | ordering | Table 1 (0.9620) |
|---|---|---|---|---|---|---|
| **literal** (default, paper's letter) | `p_tilde` only | differentiable (included) | 0.15 (gated default) | 0.9407 / 0.9296 / 0.9333 | within noise (flips s1) | NOT reachable at this `s` |
| **literal** | `p_tilde` only | differentiable (included) | 2.0 | 0.9648 / 0.9481 / 0.9611 | holds all seeds | reachable |
| **detached** (standard convention, counterfactual) | `p_tilde` AND `w` | constant (dropped) | 0.15 | 0.9611 / 0.9481 / 0.9556 | holds all seeds | reachable (±0.001) |

The literal gradient is the default because it is what Eqs. (2)–(4) literally
state. Its gate-path term is `∝ 1/s`, so it CONVERGES to the detached gradient
as `s` grows: the headline that does NOT reproduce at the sharp-gate default
`s=0.15` DOES reproduce under the literal gradient for shallow gates (ordering
from `s ≈ 0.7`, value/magnitude from `s ≈ 2.0` across seeds 0/1/2 —
`sweep_s.py` -> `s_sweep.json`), and the detached variant reproduces it already
at `s=0.15`. Because the paper states NEITHER `s` NOR the stop-grad scope on
`w`, the headline is **under-specified**: reachable under the paper's equations
for a range of `(s, grad-mode)`, but not at the prose-aligned sharp-gate default
under the literal gradient. The gated cwsd arm therefore pins `s=0.15` (a
disclosed, provably non-tuning choice — the headline fails there) and reports
the value/magnitude claims as REFUTED at the gated `s` and the ordering as
UNTESTED (within noise) under the declared spread heuristic; the detached
variant is implemented (`--grad-mode detached`) and reported as a
counterfactual (`selfcheck.json`), not as the gated arm. The `s`-dependence of
the headline under the paper's literal equations IS the reproduction's central
finding about the paper: its headline rests on two quantities it does not state
(`s` and the stop-grad scope on `w`). (Both readings agree at `λ = 0` — the
gate-path term is `λ·...=0` — so the paper's degeneracy gate holds under either,
at every `s`.)
