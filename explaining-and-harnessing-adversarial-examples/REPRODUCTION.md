# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

- **Paper:** EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
- **Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
- **Year:** 2015 (ICLR 2015 conference paper)
- **arXiv:** [1412.6572](https://arxiv.org/abs/1412.6572) (v3, 20 Mar 2015)
- **Date reproduction started:** 2026-08-04

## Status

- [x] Reproduction workspace set up; repo cloned at `~/auto-reproductions` (shallow,
  blobless), working on branch `repro/explaining-and-harnessing-adversarial-examples`
- [x] Paper text saved to `paper/paper.md` (PDF-extracted prose)
- [x] arXiv LaTeX source fetched from `https://arxiv.org/e-print/1412.6572` and unpacked
  to `paper/source/`; `iclr2015.tex` and `iclr2015.bbl` committed (identical to the
  previously committed copies), figures/style files gitignored per folder `.gitignore`
- [ ] Paper read properly; `SPEC.md` written (algorithm, symbol shapes, equation citations, unstated details)
- [ ] Upstream code search recorded
- [ ] Implementation runs smallest end-to-end case
- [ ] Adversarial review rounds clean
- [ ] Readiness gates walked and recorded
- [ ] Numbers compared to paper and published

## Notes

- The LaTeX source under `paper/source/` (`iclr2015.tex`) is authoritative for every
  equation, table and reported number; preamble `\def`/`\newcommand` macros must be
  resolved before quoting any equation from it. The PDF-extracted text in
  `paper/paper.md` is reliable for prose only.
- **Prior run:** this slug already has a completed reproduction in this branch's history
  (branch tip `40237e7` at ingest time, landed on `main` as `4f8171c`): full `SPEC.md`,
  implementation under `src/`/`experiments/`, `arms.json`, `claims.json`,
  `measured.json`, `VERIFICATION.md`, `numbers_gate.py`, and the prior run's 54 KB
  `REPRODUCTION.md` log (see git history of this branch — it was replaced by this file
  at this run's ingest commit). This run re-uses and re-verifies that work rather than
  redoing it from scratch.
