# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

**Paper:** Explaining and Harnessing Adversarial Examples
**Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
**Venue:** ICLR 2015 (arXiv:1412.6572v3, 20 Mar 2015)
**Date started:** 2026-08-06 (this run)

## Status

**Setup complete; reproduction not yet started in this run.** Branch
`repro/explaining-and-harnessing-adversarial-examples` created and pushed.
Reference material is on disk and verified:

- [x] Reproduction folder + branch `repro/explaining-and-harnessing-adversarial-examples`
- [x] Paper text in `paper/paper.md` (full PDF-extracted text; prose reliable,
      maths not) — verified complete against the ingest payload
- [x] arXiv LaTeX source (1412.6572) in `paper/source/`: `iclr2015.tex` and
      `iclr2015.bbl` tracked and committed; everything else from the unpack is
      gitignored (`.gitignore` tracks only `.tex`/`.bbl`/`.bib` under
      `paper/source/`, ignores tarballs/figures)
- [x] SPEC.md (this run) — rewritten 2026-08-06, all 70 claim quotes re-verified verbatim
      against `paper/source/iclr2015.tex`; claims.json spliced byte-content-identical;
      this run's figure reads in `figures/read-figure_2026-08-06.jsonl`
- [x] Implementation runs end-to-end
- [x] Adversarial review loop clean
- [x] Readiness gates
- [ ] Numbers measured and compared against the paper
- [ ] Published

## Prior runs in this folder

This repository holds earlier reproductions of this same paper, and their notes
are preserved here:

- `REPRODUCTION_prior_run.md` — run started 2026-08-04 (implementation +
  self-check; reviewed).
- `REPRODUCTION_prior_run_2026-08-05.md` — run that reached rung `numbers`
  (published): measured-vs-paper table for all 70 claims, readiness gates, and
  the caveat that its 6-epoch horizon did **not** reproduce the headline
  clean-error regularization claim (0.94% → 0.84% → 0.782%) while the
  adversarial-robustness claims (89.4% → 17.9% error under FGSM after
  adversarial training) did reproduce. Existing code in this folder
  (`data.py`, `models.py`, `attack.py`, `train.py`, `eval.py`,
  `run_all_arms.py`, `tests/`, `SPEC.md`, `claims.json`) is that run's output
  and is the starting point this run evaluates before building on or replacing.

## Authoritative sources

For every equation, table, and reported number prefer
`paper/source/iclr2015.tex` over `paper/paper.md` — PDF extraction silently
drops some maths glyphs. The `.tex` preamble defines macros that hide symbols
used in the equations; resolve them before quoting:

- `\def\eps{{\epsilon}}` (line 14)
- `\def\sign{{\text{sign}}}` (line 16)
- `\def\veta{{\bm{\eta}}}`, `\def\vw{{\bm{w}}}`, `\def\vx{{\bm{x}}}`, etc.
  (lines 20–27 and surroundings)

Key equations/numbers live in the `.tex` around: linear explanation
(`w^T x̃ = w^T x + w^T η`, `η = ε·sign(w)`, ~l. 230–250); FGSM
(`η = ε·sign(∇_x J(θ, x, y))`, §4); the MNIST softmax claim (ε=.25 → 99.9%
err / 79.3% conf, l. 333); maxout 89.4% / 97.6% (l. 338); CIFAR-10 conv maxout
(ε=.1 → 87.15% err / 96.6% prob, l. 340); adversarial-training objective
(`J̃ = αJ(θ,x,y) + (1−α)J(θ, x + ε·sign(∇_x J))`, α=0.5, §6); and the trained
resistance numbers (89.4% → 17.9%, l. 515–517).

## Log

- 2026-08-06 — Setup: repo cloned (shallow, blobless), branch
  `repro/explaining-and-harnessing-adversarial-examples` created and pushed,
  paper + LaTeX source verified on disk, this file started.
- 2026-08-06 — SPEC step. Re-downloaded the arXiv e-print `1412.6572`; the on-disk
  `.tex`/`.bbl` are byte-identical to the fresh download, and the figure files are now
  preserved at `figures/paper/` (`eps_curve.pdf` rasterized to
  `figures/paper/eps_curve_raster.png`, 220 dpi). Upstream code re-checked live (GitHub
  title search: 13 attack-only third-party repos; `goodfeli/adversarial` = GAN code;
  pylearn2 maxout dir = maxout-paper configs, no FGSM-training code; `mnist_pi.yaml`
  fetched as fill-in provenance for §4). Wrote `SPEC.md` fresh: method algorithm, symbols
  with shapes, 9 cited equations, gap analysis G1–G24 with permitted/weakest readings,
  fixed interfaces, per-arm restriction log, figure readings. Figure reads for this run in
  `figures/read-figure_2026-08-06.jsonl` (12 exchanges; 4 value reads flagged as
  deliberation → weak evidence, consistent with prior clean reads; two cross-run-unstable
  reads recorded as unfit for gating). `claims.json` carried forward because every one of
  its 70 quotes re-verified verbatim at its cited line on 2026-08-06 (script: 0 failures;
  14 high-invariance claims gate; seeds [0,1,2], maxout_large_adv [0..4] per the paper's
  five trials). SPEC §8 embeds claims.json content-identically (script-enforced).
- 2026-08-06 — Implementation/review step. Ran an `orchestrate` workflow (run
  `90c34428-b218-4be8-8ab7-31e9787b71ae`): five parallel reviewers (data pipeline,
  method core, training loop, eval metric, baseline arms) each checked its
  component against `paper/source/iclr2015.tex`, with an adversarial verification
  pass that refutes each finding so only genuine paper-deviations survive.
  Result: 0 findings in data/method/training/eval; **2 confirmed** in the
  baseline-arms component, both fixed and covered by new tests + mutations:
  - **F1 (major)** `agreement_mnist` c42 metric `agree_rbf_on_softmax` (paper
    53.6%, tex:688) was computed on softmax-FGSM examples instead of the
    maxout-FGSM examples the preceding agreement metrics (54.3%, tex:686) use.
    The paper's 53.6 is a direct comparison to 54.3 (RBF predicts softmax's
    class vs RBF predicts maxout's class); only on the shared example set is
    the 53.6≈54.3 closeness the evidence of the RBF's "strong linear component".
    The bug measured ~1.5% and **inverted the paper's conclusion** (implying
    the RBF has no linear component). Fixed: `ag_rbf_on_sm =
    ev.agreement(softmax_model, rbf_model, x_adv, y_test)` reusing the
    maxout-FGSM `x_adv`. Guarded by
    `test_agreement_rbf_on_softmax_uses_maxout_fgsm_examples` / mutation
    `mut_agreement_rbf_on_softmax_uses_softmax_fgsm`.
  - **F2 (minor)** `transfer_mnist` in `main()` trained `large_adv` with
    `full60k=False`, contradicting the paper's from-scratch 60k retrain
    (tex:505-506) AND the arm's own restriction doc in claims.json (which
    states `full60k=True`). The standalone `arm_transfer_mnist` default was
    already `full60k=True`; only the `main()` entrypoint overrode it. Fixed:
    `full60k=True` in `main()`. Guarded by
    `test_transfer_mnist_main_runs_full60k_retrain` / mutation
    `mut_transfer_mnist_skips_60k_retrain`.
  All 39 tests pass (37 prior + 2 new); all 12 mutations are caught (10 prior
  + 2 new). CIFAR-10 confirmed network-reachable (HTTP 200, 170 MB) so the
  real-data run can obtain the paper's dataset here. Regenerating
  `measured.json` for the two affected arms (`agreement_mnist`,
  `transfer_mnist`) from real data with the fixes applied; the prior
  `measured.json` was the published run's output (these two arms' values
  were produced by the buggy code and are being replaced).
