# Reproduction: LEACE: Perfect linear concept erasure in closed form

**Paper:** Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman. *LEACE: Perfect linear concept erasure in closed form.* NeurIPS 2023. arXiv:2306.03819
**Date reproduction started:** 2026-08-17

## Status

- [x] Workspace set up (`repro/leace-perfect-linear-concept-erasure-in-closed-form` branch)
- [x] Paper saved under `paper/` (arXiv 2306.03819 LaTeX source unpacked + rendered HTML, both fetches HTTP 200)
- [ ] Method read through; SPEC.md written with shapes + cited equations
- [ ] Upstream code located (or implementation from scratch)
- [ ] Smallest end-to-end case runs and produces a parsed number
- [ ] Adversarial review rounds clean
- [ ] Readiness gates passed
- [ ] Published

## Paper sources on disk

- `paper/main.tex`, `paper/rebuttal.tex` — LaTeX source (unpacked from e-print tarball; authoritative for equation/number quotes, cite as `<file>:<line>`)
- `paper/main.bbl`, `paper/citations.bib` — bibliography sources
- `paper/paper.html` — arXiv rendered HTML (MathML; reading surface)
- Not committed (gitignored): `eprint.tar.gz`, `figures/`, `oleace.pdf`, `neurips_2023.sty`

## Notes

- Both arXiv copies (e-print tarball and HTML) fetched successfully; no missing sources, so no equation has to be treated as unverified on extraction grounds. Macro resolution still required when quoting equations from the `.tex` preamble.

## Log

- 2026-08-17 — Initial setup: cloned repo, created branch, fetched and unpacked the paper, started this file.
