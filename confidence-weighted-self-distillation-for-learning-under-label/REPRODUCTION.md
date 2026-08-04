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

Current rung: **comprehension** (SPEC re-derived and re-verified this run).

- [x] Reproductions repo cloned (shallow, blobless) into `$HOME`; workspace folder
      present at the slug name
- [x] `$HOME/.repro_dir` and `$HOME/.repro_branch` written (no trailing newlines)
- [x] Branch `repro/confidence-weighted-self-distillation-for-learning-under-label`
      created from `origin/main` and pushed
- [x] Paper text saved to `paper/paper.md`
- [x] Comprehension (SPEC)
- [ ] Implementation / verification
- [ ] Adversarial review rounds clean
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
