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
- [x] SPEC.md (method spec for this run)
- [ ] Implementation runs end-to-end on the smallest case
- [ ] Adversarial review loop clean
- [ ] Readiness gates
- [ ] Published

## Log

### 2026-08-05 — SPEC step (this run)

- **Reviewed the paper from the LaTeX source** (`paper/source/iclr2015.tex`, 975 lines)
  rather than the PDF extraction; all equations and numbers below are cited by
  `<file>:<line>` into it.
- **Kept the prior run's SPEC.md + claims.json as the base** (merge commit `11f0a3b`
  reached rung=numbers with this structure) and re-verified everything against the
  paper on disk instead of trusting it:
  - all 70 claim quotes resolve verbatim at their cited lines (script, 0 failures);
  - `claims.json` is byte-identical to the SPEC §8 block and parses;
  - schema check: 18 arms, 70 claims spanning all six kinds (value/ordering/invariant/
    existence/curve), ≥3 seeds per claim (the large-model claims use the paper's 5),
    14 claims rated `high` compute-invariance (the gating set);
  - claims block format validated: ordering → quantity+direction; value →
    claimed+tolerance; invariant/existence → predicate; curve → quantity+x+comparison.
- **Restored the paper figures** to `paper/source/` from a fresh arXiv e-print download
  (1412.6572); they are gitignored by design, `read-figure` needs them on disk.
  `eps_curve.pdf` rasterized 4x with pymupdf for the reader.
- **Re-read the figures** with `read-figure` (23 new transcript exchanges in
  `figures/read-figure.jsonl`). Clean reads: x ticks -15/15, y ticks -2000/1000,
  "piecewise-linear", another curve ends above class 4 on x in [0,15] -> yes,
  class-4 logit at eps=+10 -> **-400**. Flagged (deliberation) reads for the remaining
  figure values land inside the prior reads' tolerances. Montage grids confirmed
  **programmatically** by autocorrelation (eps_curve_inputs and airplane are 10x10).
  Figure-3 naive-filter reads were phrasing-unstable this run -> recorded; nothing
  gated rests on them. Claims c65-c70 stand unchanged.
- **Upstream code re-checked live**: GitHub title search (13 repos, attack-only
  third-party FGSM reimplementations), `goodfeli/adversarial` = GAN code, pylearn2
  maxout scripts = the maxout-paper configs (reference-only). Conclusion unchanged:
  no usable upstream code exists for this paper's experiments. See SPEC §0.

## Notes

- **Prior run preserved.** This repository's `main` already contains a completed
  reproduction of this paper (merged commit `11f0a3b`, prior run dated 2026-08-04).
  Its files remain in this folder. Its final report is preserved as
  `REPRODUCTION_prior_run.md`; this file starts fresh for the current run. Later
  steps decide what of the prior work to reuse, redo, or replace.
- The authoritative reference for every equation, table, and reported number is
  `paper/source/iclr2015.tex` / `paper/source/iclr2015.bbl`, not `paper/paper.md`.
