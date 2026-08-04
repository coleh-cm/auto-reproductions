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

Current rung: **correctness** (implementation faithful to the paper's equations;
adversarial review's blocking finding fixed). The paper's Table-1 headline
(+2.5 points, 0.9620) and the central `cwsd > baseline` ordering do NOT
reproduce under the paper's LITERAL gradient (stopgrad only on `p_tilde`, as
Eq. 3 marks it); they reproduce only under the DETACHED variant (whole target
constant), which the paper does not state on the gate weight `w`. The
reproduction implements the literal gradient as the default (faithful) and
reports the detached variant as a counterfactual. The 5 high structural
invariants + the baseline value reproduce; the central ordering + the two CWSD
value claims honestly fail. `selfcheck_claims.py` → `selfcheck.json` reports
6 pass / 3 fail / 0 blocked at all 3 seeds; `measured.json` regenerated.

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
- [ ] Readiness gates
- [ ] Publish

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
  `cwsd > baseline` ordering do NOT reproduce — re-measured at seeds 0/1/2:
  CWSD-LITERAL 0.9407 / 0.9296 / 0.9333 vs baseline 0.9370 / 0.9407 / 0.9315,
  i.e. the ordering flips at seed 1 and is within noise at seeds 0/2. The
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
  as a documented COUNTERFACTUAL — the variant under which Table 1 is reachable
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
  honestly against the literal arm (they fail); the detached counterfactual is
  in `not_tested`. `--s` is no longer described as "calibrated to Table 1"
  (the literal arm does not reproduce Table 1 at any `s`).

  Result (re-measured this pass, seeds 0/1/2): baseline 0.9370/0.9407/0.9315
  (grad-mode-independent at λ=0, degeneracy holds under literal too — the
  gate-path term is `λ·...=0`); CWSD-LITERAL 0.9407/0.9296/0.9333; CWSD-DETACHED
  (counterfactual) 0.9611/0.9481/0.9556. `selfcheck.json`: 6 pass (5 high
  structural invariants + baseline value) / 3 fail (central ordering at seed 1,
  cwsd value, improvement magnitude) / 0 blocked. `pytest -q tests` → 53
  passed (was 51; +2 from M6/M7). The reproduction's honest conclusion: the
  paper's headline and central ordering do NOT reproduce under the paper's
  stated equations; they reproduce only under the detached variant the paper
  does not state — a genuine, reported finding, not a fabricated "reproduced".

## Literal vs detached — the central finding

The paper's Eq. (3) annotates `stopgrad` ONLY on `p_tilde`
(`paper/paper.md:198`, "the latter treated as a constant", `:171-174`). The
gate weight `w = λσ((c−τ)/s)` is unmarked and is a function of `θ` through
`c = max_k p_k`. Two faithful readings of the under-specified stop-grad:

| reading | stop-grad on | gate path `w` | CWSD seed 0/1/2 | ordering | Table 1 (0.9620) |
|---|---|---|---|---|---|
| **literal** (default, paper's letter) | `p_tilde` only | differentiable (included) | 0.9407 / 0.9296 / 0.9333 | flips at seed 1 | NOT reachable |
| **detached** (standard convention, counterfactual) | `p_tilde` AND `w` | constant (dropped) | 0.9611 / 0.9481 / 0.9556 | holds all seeds | reachable (±0.001) |

The literal gradient is the default because it is what Eqs. (2)–(4) literally
state; under it the headline does not reproduce. The detached variant is the
standard self-distillation convention (treat the whole target as a constant),
which the paper does not state on `w`; it is the only reading under which
Table 1 is reachable, so it is implemented (`--grad-mode detached`) and
reported as a counterfactual (`selfcheck.json`), not as the gated arm. The
divergence between the two readings IS the reproduction's central finding
about the paper: its headline rests on a gradient computation it does not
explicitly state. (Both readings agree at `λ = 0` — the gate-path term is
`λ·...=0` — so the paper's degeneracy gate holds under either.)
