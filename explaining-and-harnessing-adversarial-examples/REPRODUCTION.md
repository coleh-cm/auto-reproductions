# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

- **Paper:** Goodfellow, Shlens & Szegedy, "Explaining and Harnessing Adversarial Examples", ICLR 2015 (arXiv:1412.6572v3)
- **Date:** 2026-08-04
- **Branch:** `repro/explaining-and-harnessing-adversarial-examples`

## Status

**SPEC written.** No upstream code for this paper exists; implementing fresh from `SPEC.md`.

- [x] Paper text saved to `paper/paper.md` (PDF extraction; prose reliable, maths not)
- [x] arXiv LaTeX source fetched (https://arxiv.org/e-print/1412.6572) and unpacked to `paper/source/`
- [x] `paper/source/iclr2015.tex` and `paper/source/iclr2015.bbl` committed (no separate `.bib` in the e-print; the `.bbl` is the compiled bibliography). Figures, `.sty`/`.bst` files and the tarball are gitignored — see `.gitignore`.
- [x] SPEC.md (method, symbols/shapes, equations with source-line citations) + `claims.json` (70 claims, all quotes grep-verified against the tex; 19 high-invariance claims form the numbers gate) + `figures/read-figure.jsonl` (vision-read transcript for Fig. 4)
- [ ] Environment
- [ ] Implementation
- [ ] Verification against paper numbers
- [ ] Readiness gates
- [ ] Publish

## Reference notes for later steps

The LaTeX preamble defines math macros that must be resolved when quoting equations:
`\eps` → `\epsilon`, `\sign` → `\text{sign}`, `\veta` → `\bm{\eta}`, `\vtheta` → `\bm{\theta}`,
`\vx` → `\bm{x}`, plus the rest of the `\v<letter>` bold-vector family and `\g<letter>` greek shortcuts.

Key equations (verified in `paper/source/iclr2015.tex`):

- FGSM perturbation (§4, iclr2015.tex line 309):
  `η = ε sign(∇_x J(θ, x, y))`
- Adversarial training objective (§6, line 486–487):
  `J̃(θ, x, y) = α J(θ, x, y) + (1 − α) J(θ, x + ε sign(∇_x J(θ, x, y)))` with `α = 0.5`
- Adversarial logistic regression (§5): minimize `E ζ(y(ε||w||₁ − w⊤x − b))`, `ζ(z) = log(1 + e^z)`

Headline numbers to verify (from the tex): softmax regression FGSM error 99.9% @ ε=.25 (MNIST);
maxout FGSM error 89.4% (conf 97.6%) without adversarial training → 17.9% with; clean test error
0.94% → 0.84% → 0.782% avg (1600-unit maxout + adversarial training, 5 seeds: 4× 0.77%, 1× 0.83%);
transfer: 19.6% / 40.9%; RBF: 55.4% error but 1.2% confidence on mistakes; ensemble of 12: 91.1%/87.9%;
MNIST rubbish-class: maxout softmax 98.35% (conf 92.8%), sigmoid top 68% (87.9%), softmax regression
59.8% (70.8%), RBF 0%; logistic regression 3-vs-7: 1.6% clean → 99% FGSM @ ε=.25.

## Log

- 2026-08-04 — Ingest: cloned repo (shallow, blobless), branch created, prior merged run's folder
  reset (its final state remains in history on `main`, merge commit 858edac), paper text + LaTeX
  source committed.
- 2026-08-04 — SPEC pass: full method read from the LaTeX source; upstream-code check
  (`goodfeli/adversarial` is the GAN paper's repo, not this one; paper's only code link is
  pylearn2 CIFAR preprocessing; nothing runnable — implement fresh); SPEC.md + claims.json
  written; Figure 4 read via `read-figure` (eps_curve.pdf rasterized first); all 70 claim
  quotes audited as substrings of their cited tex lines (0 mismatches).
