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

Current rung: **environment → comprehension → implementation** — workspace
set up, paper on disk, branch pushed; SPEC.md re-derived from
`paper/paper.md` and committed; all five components adversarially verified
against the paper (5/5 match, 0 P0/P1) and the full numbers regenerate
byte-identically (`measured.json`, `claims_result.json`).

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
      pinned env rebuilt (Python 3.12.13 + numpy 2.5.1 + scikit-learn 1.9.0,
      pytest 9.1.1) and prior results spot-re-executed — 53 tests pass,
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
- [ ] Adversarial review rounds
- [ ] Readiness gates
- [ ] Publish

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
