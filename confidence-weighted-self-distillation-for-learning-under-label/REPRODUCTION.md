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

Current rung: **environment** (workspace setup for this run).

- [x] Reproductions repo cloned (shallow, blobless) into `$HOME`; workspace folder
      present at the slug name
- [x] `$HOME/.repro_dir` and `$HOME/.repro_branch` written (no trailing newlines)
- [x] Branch `repro/confidence-weighted-self-distillation-for-learning-under-label`
      created from `origin/main` and pushed
- [x] Paper text saved to `paper/paper.md`
- [ ] Comprehension (SPEC)
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
