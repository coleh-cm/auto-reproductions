# MidSteer: Optimal Affine Framework for Steering Generative Models

Reproduction of "MidSteer: Optimal Affine Framework for Steering Generative Models"
(Gaintseva, Stepanov, Liu, Benning, Slabaugh, Deng, Elezi; 2026; arXiv:2605.05220).

- paper_ref: 0e6756c9-d827-4749-a8d7-3140a8c98a25
- project_id: 06910d54-1d99-4864-8bff-ab3007a0c70e
- Date: 2026-08-16
- Upstream code (linked in paper): https://github.com/Atmyre/MidSteer

## Status

**Phase: spec complete.** Paper read against the arXiv LaTeX source (authoritative); `SPEC.md`
and `claims.json` written and committed. Upstream code confirmed to exist and cloned for
inspection (`https://github.com/Atmyre/MidSteer`, HEAD `0f3b31e`); decision: **adopt upstream
code** as the base implementation. No running code yet.

### Setup log

- [x] Repository cloned (`--depth 1 --filter=blob:none`) and branch `repro/midsteer-optimal-affine-framework-for-steering-generative-models` created and pushed.
- [x] `paper/` contains the PDF-extracted paper text (`paper.txt`, convenience copy; maths NOT reliable from it) and the arXiv 2605.05220 v3 LaTeX source (authoritative for equations, tables, numbers).
- [x] LaTeX source: `main.tex`, `flipping_main.tex`, `content/*.tex`, `artefacts/**/*.tex`, `example_paper.bib`. Build system per `00README.json` (ignored): pdflatex + texlive 2025, `icml2026` style (`.sty`/`.bst` ignored, compile-time only).
- [x] `paper/.gitignore` keeps only `.tex`/`.bbl`/`.bib` + `paper.txt` + `paper.html`; figures (`img/*.png`, `artefacts/{main,pareto,phase}/*.pdf`), `eprint.tar.gz`, styles and metadata are ignored.
- [x] Rendered HTML copy saved as `paper/paper.html` (https://arxiv.org/html/2605.05220, LaTeXML, maths as MathML) for one-fetch reading; quotes/citations still come from the `.tex` source.
- [x] Read paper properly; wrote `SPEC.md` (method as algorithm, symbols+shapes, every
  implemented equation cited to `paper/*.tex:<line>`, 19-item gap list G1–G19 with weakest
  readings, frozen component interfaces, 4 arms, per-arm restriction analysis).
- [x] Wrote `claims.json`: 22 claims (16 high / 3 medium / 3 low), 4 arms, seeds {0,1,2};
  every claim has verbatim quote + on-disk citation + settling arithmetic; high claims carry
  sensitivity or `fixed_by_paper`. Figure-derived curve claims backed by
  `figure_reads/transcript.md` (5 reads, all constrained one/two-word answers, no deliberation
  flags).
- [x] Evaluated upstream code (https://github.com/Atmyre/MidSteer): exists, public. Mapped its
  math to the paper Eqs. 6/13/19/22/23 (matches, incl. mean-centering identity
  x' = x − β·Q(x−μ) ≡ Âx + b̂). Recorded 6 candidate divergences in SPEC.md §1 (projection-score
  clipping ON by default and not in the paper; mean-diff vs literal Cov(X,Z); pinv tolerance;
  per-head stats; hooks vs weight-folding; diffusion-step indexing).
- [ ] Vendor/clone upstream into the repo; get the smallest end-to-end case running
  (E1 synthetic closed-form checks first — CPU-only, then E2 smallest model arm) and produce a
  parsed number.
- [ ] Adversarial review rounds against the paper until clean.
- [ ] Readiness gates; publish.

### Key macro definitions to resolve when quoting equations

From `paper/main.tex` lines 65–71:

- `\sxx` → $\mathbf{\Sigma}_{\mathbf{X}\mathbf{X}}$; `\sxz` → $\mathbf{\Sigma}_{\mathbf{X}\mathbf{Z}}$
- `\swxz` → $\mathbf{\Sigma}_{\mathbf{W}\mathbf{X},\mathbf{Z}}$; `\swxwx` → $\mathbf{\Sigma}_{\mathbf{W}\mathbf{X},\mathbf{W}\mathbf{X}}$
- `\szz` → $\mathbf{\Sigma}_{\mathbf{Z}\mathbf{Z}}$; `\pwxz` → $\mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{\mathbf{X}\mathbf{Z}}}$
- `\cov` → $\mathrm{Cov}$; `\rv` (random-vec shorthand used throughout)

### Method core (from paper, to be verified against LaTeX)

Whitening $W = (\Sigma_{XX}^{1/2})^+$, cross-covariances $\Sigma_{XZ_i} = \mathrm{Cov}(X, Z_i)$ estimated on $N=1000$ concept prompts; $\Sigma_{XX}$ on $M=50000$ broad prompts via Welford's algorithm (Appendix A). Per layer (every self-attention layer for LLMs, every cross-attention layer for diffusion), affine map folded into the out-projection weights:

- LEACE (erasure): $\hat{A} = I - W^+(W\Sigma_{XZ})(W\Sigma_{XZ})^+ W$ (Eq. 6)
- LEACE-Switch: $\hat{A} = I - 2W^+(W\Sigma_{XZ})(W\Sigma_{XZ})^+ W$ (Eq. 13)
- MidSteer: $\hat{A} = I + W^+(\Sigma_{WX,Z_2} - \Sigma_{WX,Z_1})\Sigma_{WX,Z_1}^+ W$ (Eq. 19), with $\hat{b} = \mathbb{E}[X] - \hat{A}\mathbb{E}[X]$ and steering strength $\beta$ scaling the non-identity term (Eqs. 22–23).
- Vanilla steering baselines: erasure $(I - ss^T)$ and Householder switching $(I - 2ss^T)$ with unit-norm steering vector $s$ (Eqs. 24–25).
