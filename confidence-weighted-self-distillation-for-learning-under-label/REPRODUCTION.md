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

Current rung: **environment** — workspace set up, paper on disk, branch pushed.
No code has been written or re-verified by this run yet.

- [x] Reproductions repo cloned (shallow `--depth 1 --filter=blob:none`) into
      `$HOME`; workspace folder present at the slug name
- [x] `$HOME/.repro_dir` and `$HOME/.repro_branch` written (no trailing newlines)
- [x] Branch `repro/confidence-weighted-self-distillation-for-learning-under-label`
      checked out and pushed (the ref already existed on the remote from the
      prior run; work continues on top of it, no force-push, default branch
      untouched)
- [x] Paper text saved to `paper/paper.md` (re-saved verbatim from the
      objective; byte-identical to the prior run's copy)
- [ ] Comprehension (SPEC)
- [ ] Implementation / verification
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
