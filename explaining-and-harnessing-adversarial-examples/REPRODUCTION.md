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
- [x] Paper read properly; `SPEC.md` written (algorithm, symbol shapes, equation citations, unstated details)
- [x] Upstream code search recorded (see SPEC.md §7: no author code; paper's only link is the 2013 maxout-paper pylearn2 configs)
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
- 2026-07-29: Read the full LaTeX source (975 lines, macros resolved) and wrote `SPEC.md`:
  four algorithms (FGSM attack, FGSM adversarial training, closed-form adversarial logistic
  regression, rubbish/targeted-fooling), full symbol table with shapes, 10 equations with
  grep-able file:line citations, 20-item unstated-details list, frozen component interfaces,
  and the upstream-search record. Key decisions fixed in the spec: no clipping of adversarial
  inputs; error rate over all evaluated examples; "average confidence" = predicted-class
  probability over misclassified examples; stop-gradient through the sign in adversarial
  training; maxout defaults (2 layers, 5 pieces, dropout include .8/.5, init irange .005,
  max_col_norm 1.9365, SGD batch 100 LR .1 momentum .5→.7, patience 100) adopted from the
  still-live external `lisa-lab/pylearn2` `mnist_pi.yaml` and explicitly marked as external,
  not paper-stated. Environment: Python 3.12, no torch yet, CPU-only, 16 cores.
