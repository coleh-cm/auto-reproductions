# Reproduction: CASteer: Cross-Attention Steering for Controllable Concept Erasure

- **Paper:** CASteer: Cross-Attention Steering for Controllable Concept Erasure
- **Authors:** Tatiana Gaintseva, Andreea-Maria Oncescu, Chengcheng Ma, Ziquan Liu, Martin Benning, Gregory Slabaugh, Jiankang Deng, Ismail Elezi
- **Year:** 2026
- **arXiv:** 2503.09630
- **Date started:** 2026-08-17
- **paper_ref:** d0ab97b5-e9bf-41af-9846-d19f846eef80
- **project_id:** 06910d54-1d99-4864-8bff-ab3007a0c70e

## Status

**In progress — setup complete.**

- [x] Reproduction folder created; branch `repro/casteer-cross-attention-steering-for-controllable-concept-erasure` pushed.
- [x] arXiv e-print 2503.09630 fetched and unpacked under `paper/` — `.tex`/`.bib` committed, figure images and style files gitignored (no `.bbl` present in the source tarball).
- [x] Rendered HTML saved as `paper/paper.html` and committed.
- [ ] SPEC.md — method written up as an algorithm with symbol shapes and cited equations.
- [ ] Upstream code search.
- [ ] Implementation / environment setup.
- [ ] Smallest end-to-end run producing a parsed number.
- [ ] Adversarial review loop clean.
- [ ] Readiness gates.
- [ ] Publish.

## Source layout

- `paper/paper.html` — reading surface (MathML rendering).
- `paper/iclr2026_conference.tex` — main file; `paper/math_commands.tex` — preamble macros (resolve these before quoting equations).
- `paper/content/*.tex` — abstract, intro, method (`method_2.tex`), experiments, related work, conclusion, supplementary.
- `paper/content/<model>_tables/*.tex` — results tables per model (SD 1.4, SD 1.4 step-0, SD 1.5, SDXL, Sana, multi-concept, strength, UNet-padding, constant-strength ablations).
- Quotes use `<file on disk>:<line>` citations against this `.tex` source.

## Log

- 2026-08-17: Repo cloned shallow/blobless. Branch `repro/casteer-cross-attention-steering-for-controllable-concept-erasure` created and pushed. Both arXiv copies fetched successfully (e-print tarball + HTML); no fetch failures to record.
