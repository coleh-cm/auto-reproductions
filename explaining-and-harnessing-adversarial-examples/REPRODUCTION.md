# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

- **Paper:** EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
- **Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
- **Year:** 2015 (ICLR 2015 conference paper)
- **arXiv:** [1412.6572](https://arxiv.org/abs/1412.6572) (v3, 20 Mar 2015)
- **paper_ref:** 43e841f0-05f8-4249-ba14-14c1a586f6ee
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Date reproduction started:** 2026-08-04
- **Reproduction dir:** `/root/auto-reproductions/explaining-and-harnessing-adversarial-examples`
- **Branch:** `repro/explaining-and-harnessing-adversarial-examples`

## Status

- [x] Reproduction workspace set up: repo `coleh-cm/auto-reproductions` cloned to
  `~/auto-reproductions` (shallow, blobless: `--depth 1 --filter=blob:none`), folder
  `explaining-and-harnessing-adversarial-examples/` in use.
- [x] Folder path written to `$HOME/.repro_dir` (no trailing newline); branch name
  written to `$HOME/.repro_branch`.
- [x] Branch `repro/explaining-and-harnessing-adversarial-examples` created locally from
  current `main` and pushed to origin.
- [x] Paper PDF-extracted text at `paper/paper.md` (verified to match this run's
  objective text; already on disk from prior merged run).
- [x] arXiv LaTeX source re-fetched fresh from `https://arxiv.org/e-print/1412.6572`
  (gzipped tarball, 704,379 bytes) and unpacked to `paper/source/`. `iclr2015.tex`
  (61,082 B) and `iclr2015.bbl` (6,795 B) are tracked; no `.bib` exists in the source
  (references live in the `.bbl`). Figures (`.png`/`.pdf`), style files
  (`.sty`/`.bst`) and the tarball are gitignored per folder `.gitignore`; they remain
  unpacked on disk for reference only. The freshly fetched `.tex`/`.bbl` are
  byte-identical to the previously committed copies (v3 source, unchanged).
- [ ] Paper read (method sections, equations from `paper/source/iclr2015.tex` with
  preamble macros `\eps` = ε, `\sign` = sign resolved); `SPEC.md` written.
- [ ] Upstream code search done.
- [ ] Implementation runs end to end on the smallest case.
- [ ] Adversarial review loop clean.
- [ ] Readiness gates walked.
- [ ] Numbers measured and published.

## Notes for later steps

**Workspace history.** This folder already contained a complete, merged reproduction
from an earlier run of this workflow (merge commit `536f58f` on `main`: "final EAE
reproduction"), including `SPEC.md`, `README.md`, `src/`, `experiments/`, `tests/`,
`results/`, claims/gate tooling, and a 308-line `REPRODUCTION.md` log. This run's log
starts fresh above; the prior log is recoverable from git history
(`git log main -- REPRODUCTION.md`). The prior implementation is treated as
pre-existing content of this workspace, not as output of this run; every claim about
"this run reproduced X" in later steps must come from work re-executed and
re-verified during this run.

**Paper math reference.** The LaTeX in `paper/source/iclr2015.tex` is authoritative.
Key macros from the preamble: `\eps` = ε, `\sign` = sign, `\veta`/`\vtheta`/`\vx` =
bold vectors. The headline FGSM equation (tex line 309) is
\(\eta = \epsilon\,\mathrm{sign}(\nabla_x J(\theta, x, y))\) — the PDF-extracted text
renders the epsilon as a small square that can be misread.

**arXiv fetch.** Succeeded this run via plain `curl`; no GnuTLS/HTTP-2 failure.
The repo clone (step 1) likewise succeeded shallow+blobless, so no tarball fallback
was needed.
