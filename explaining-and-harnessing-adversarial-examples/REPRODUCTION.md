# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

- **Paper:** EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
- **Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
- **Year:** 2015 (ICLR 2015 conference paper)
- **arXiv:** [1412.6572](https://arxiv.org/abs/1412.6572) (v3, 20 Mar 2015)
- **Date reproduction started:** 2026-07-29

## Status

- [x] Reproduction workspace set up; repo cloned at `~/auto-reproductions`
- [x] Paper text saved to `paper/paper.md` (PDF-extracted prose)
- [x] arXiv LaTeX source fetched and unpacked to `paper/source/` (verified byte-identical to a fresh download of `https://arxiv.org/e-print/1412.6572`)
- [ ] Paper read properly; `SPEC.md` written (algorithm, symbol shapes, equation citations, unstated details)
- [ ] Upstream code search recorded
- [ ] Implementation runs smallest end-to-end case
- [ ] Adversarial review rounds clean
- [ ] Readiness gates walked and recorded
- [ ] Numbers compared to paper and published

## Notes

- The LaTeX source under `paper/source/` (`iclr2015.tex`, 975 lines) is authoritative for every
  equation, table and reported number; preamble `\def`/`\newcommand` macros must be resolved
  before quoting any equation from it. The PDF-extracted text in `paper/paper.md` is reliable
  for prose only.

## Log

- 2026-07-29: Workspace initialized. `paper/source/` was already present from a prior attempt;
  verified byte-identical against a fresh fetch of the arXiv e-print tarball. No fetch failure to
  record.
