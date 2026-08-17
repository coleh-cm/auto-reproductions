# Reproduction: LEACE: Perfect linear concept erasure in closed form

**Paper:** Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman. *LEACE: Perfect linear concept erasure in closed form.* NeurIPS 2023. arXiv:2306.03819
**Date reproduction started:** 2026-08-17

## Status

- [x] Workspace set up (`repro/leace-perfect-linear-concept-erasure-in-closed-form` branch)
- [x] Paper saved under `paper/` (arXiv 2306.03819 LaTeX source unpacked + rendered HTML, both fetches HTTP 200)
- [x] Method read through; SPEC.md written with shapes + cited equations; claims.json (20 claims, 12 high-invariance, seeds [0,1,2]); all claim quotes verified byte-verbatim against paper/main.tex
- [x] Upstream code located: EleutherAI/concept-erasure (linked from paper abstract, main.tex:283) pinned @9f51753 + EleutherAI/tagged-pile @5719699; divergences from paper text recorded in SPEC.md §2-§3
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
- Environment: CPU-only (16 cores, 63 GB RAM, no GPU). All arms scoped accordingly; per-arm restrictions in SPEC.md §5.
- Figure readings (read-figure, transcript `paper/figure-transcript.md`): random-accuracy line 0.52; LEACE point ≈ (2.2, 0.52); leftmost INLP MSE ≈ 1.6; RLACE min acc ≈ 0.54; amnesic 'Ours' max damage at layer 11, INLP at layer 6 (matches main.tex:601); on-disk figure is the uncorrected No-Intervention=0.89 version (rebuttal.tex:23 documents 0.892→0.928).
- Key upstream-vs-paper divergences found: covariance shrinkage default ON (paper silent), svd_tol=0.01 truncation (paper silent), trace-constraint Q built from *whitened truncated* Σ_XZ vectors (paper's feasibility argument requires Q Σ_XZ = 0, i.e. SAL's Q), 2^25 vs 2^22 tokens, fp16, seed 42.

## Log

- 2026-08-17 — Initial setup: cloned repo, created branch, fetched and unpacked the paper, started this file.
- 2026-08-17 — Read full main.tex + rebuttal.tex; rendered and read all quantitative figures via read-figure (transcript committed); located and read upstream code (EleutherAI/concept-erasure, tagged-pile); wrote SPEC.md (method/shapes/cited equations/16 unstated items/frozen interfaces/restriction analysis/upstream findings) and claims.json (20 claims: 12 high, 6 low, 2 medium; 20 arms; seeds [0,1,2]); verified every claim quote byte-verbatim against paper/main.tex and every citation line-resolvable.
- 2026-08-17 — Review feedback round 1 addressed: added top-level `restrictions` map to claims.json keyed by all 20 arms (kind none/narrows_situations/changes_correctness + per-arm detail; 6 `none` synthetic arms, 14 `narrows_situations`, zero `changes_correctness`), SPEC.md §7 embedded copy kept byte-identical; fixed C15 curve claim to carry `quantity: measured.amnesic_leace.mlm_drop` with explicit `against: measured.amnesic_random.mlm_drop` for its `above` comparison (previously a difference quantity with `above`, which left the comparator curve implicit). Restrictions JSON-validated; embedded copy re-checked byte-identical.
