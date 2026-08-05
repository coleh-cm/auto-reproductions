# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

**Paper:** Explaining and Harnessing Adversarial Examples
**Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
**Venue:** ICLR 2015 (arXiv:1412.6572v3, 20 Mar 2015)
**Date:** 2026-08-05 (this run)

## Status

**Setup / ingest.** Workspace initialized; reproduction not yet started on this run.

- [x] Reproduction folder + branch `repro/explaining-and-harnessing-adversarial-examples`
- [x] Paper text saved to `paper/paper.md` (PDF extraction; prose reliable, maths not)
- [x] arXiv LaTeX source (e-print 1412.6572) unpacked to `paper/source/`; only
      `.tex`/`.bbl` tracked, figures and style files gitignored. Verified byte-identical
      to a fresh download this run. Source contains no `.bib` (references are in the
      `.bbl`) and no `\def`/`\newcommand` macros for equations — symbols are literal.
- [ ] SPEC.md (method spec for this run)
- [ ] Implementation runs end-to-end on the smallest case
- [ ] Adversarial review loop clean
- [ ] Readiness gates
- [ ] Published

## Notes

- **Prior run preserved.** This repository's `main` already contains a completed
  reproduction of this paper (merged commit `11f0a3b`, prior run dated 2026-08-04).
  Its files remain in this folder. Its final report is preserved as
  `REPRODUCTION_prior_run.md`; this file starts fresh for the current run. Later
  steps decide what of the prior work to reuse, redo, or replace.
- The authoritative reference for every equation, table, and reported number is
  `paper/source/iclr2015.tex` / `paper/source/iclr2015.bbl`, not `paper/paper.md`.
