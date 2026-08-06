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

- 2026-08-06 — Review-findings fix pass. The adversarial review of the
  F1/F2-fixed state (HEAD `ce02584`) raised three BLOCKING items and several
  recommended ones; this pass addresses them:
  - **B1 (blocking, review_divergence) — CIFAR preprocessing was not the
    referenced pipeline.** `data.py:_gcn_preprocess` subtracted the
    training-set *per-pixel* mean and applied one global scale; the paper's
    footnote 2 (tex:343-345) defers to the pylearn2 maxout scripts whose
    `GlobalContrastNormalization` centers and scales *each image
    independently*. Fixed: per-image GCN (subtract each image's own mean over
    its 3072 pixels, then one global scale `s = 0.5/std(centered)` so the
    train std is ~0.5, the paper's only stated property). The fingerprint no
    longer asserts the std~0.5 the recipe forces by construction (circular);
    it now asserts a RAW uint8 pixel-sum checksum in `_load_cifar_raw` (a
    property the GCN recipe cannot force — the CIFAR analogue of the MNIST
    checksum) plus the existing per-pixel-variance structural check. SPEC
    G20 updated. **Fingerprint correction this pass:** the raw-checksum
    constants committed in the earlier fix pass were placeholders that did
    NOT match the canonical cs.toronto.edu tar (train 10.2B / test 2.0B vs
    the real 18.54B / 3.73B sum of uint8 pixels); they would have *rejected*
    the real corpus on any rerun. Recomputed from the real tar this pass
    (`_CIFAR_RAW_TRAIN_SUM = 18540682003.0`, `_CIFAR_RAW_TEST_SUM =
    3733375634.0`; verified by loading all 50k/10k images). The CIFAR arm is
    rerun under the new preprocessing AND the corrected fingerprint at all 3
    seeds (real data on disk, ~170 MB): per-image GCN gives
    clean_err 27.6/27.0/27.0, adv_err 97.2/95.7/98.8 (vs the prior per-pixel
    GCN values 27.3/26.8/…, adv_err 98.7/99.1/99.7) — the preprocessing change
    is material and the numbers now pass the corrected corpus fingerprint.
  - **B2 (blocking, review_divergence) — `maxout_large_adv` skipped the
    paper's Phase-2 60k retrain.** `arm_maxout_large_adv` ran `full60k=False`
    while the claims c14/c18/c19/c20 quote the *post-retrain* model's numbers
    (17.9% / 0.782% / 81.4%, tex:510/506/523). Fixed: `full60k=True`
    (the from-scratch 60k retrain, tex:505-506). Rerun at all 5 seeds
    [0..4] this pass: the post-retrain model gives clean_err
    1.49/1.55/2.74/1.57/1.33 (mean 1.74), adv_err 5.78/8.83/24.96/10.89/5.27
    (mean 11.1), adv_conf_mistakes 70.4/74.6/70.4/72.2/71.0 (mean 71.7) — the
    retrain is materially more robust than the pre-retrain values it replaces
    (adv_err mean 19.2 → 11.1; clean_err mean 2.33 → 1.74), and c14 (adv_err
    17.9, tol 8) and c20 (adv_conf_mistakes 81.4, tol 12) both pass; c19
    (clean 0.782) still fails honestly at the 6-epoch sub-scale horizon.
    (`transfer_mnist` already used `full60k=True`; its measured values are
    unchanged-code and remain valid.)
  - **B3 (blocking, review_faithful/review_divergence) — `claims_result.json`
    stale vs `measured.json`/`selfcheck.json`.** `claims_result.json` is the
    workflow's numbers-gate verdict table (`produced_by: reproduce-paper
    numbers gate`); it is NOT writable from this repo (a script here writing
    that filename would collide with the gate and be refused). It was last
    regenerated at the 2026-08-05 publish and predates the F1/F2 fixes and
    this pass's measured.json. The in-repo current-verdict evidence is
    `selfcheck.json` (written by `selfcheck_claims.py`, the reproduction's
    own grader), which is regenerated from the current `measured.json` in
    this pass. The numbers gate / publish step regenerates
    `claims_result.json` from the then-current `measured.json`; it must not
    be hand-edited here.
  - **R1 (recommended) — Ensemble attack objective mismatched the evaluated
    decision rule.** `Ensemble.loss` was cross-entropy of the MEAN LOGITS
    while `predict` uses mean-PROBABILITY argmax — the loss of a different
    classifier than the one evaluated on c34–c37. Fixed: `Ensemble.loss` is
    now the NLL of the mean-probability classifier (`-log mean_k p_k[y]`),
    so the FGSM ensemble-attack target matches the evaluated decision rule
    (tex:822-823). Guarded by `test_ensemble_loss_is_mean_prob_nll` /
    mutation `mut_ensemble_loss_mean_logits`. (c36 is HIGH; the ensemble12
    arm is rerun at all 3 seeds this pass: the mean-prob NLL attack is
    materially stronger than the prior CE-of-mean-logits attack —
    adv_err_ensemble_crafted 98.5/97.8/97.9 vs 93.3/92.8/93.5 — and the
    whole-vs-single ordering c37 now resolves clearly, gap 5.2–6.7pp at
    every seed vs the prior ~1.3–3.3pp within-noise gap; the single-member
    attack is unchanged as expected.)
  - **R2 (recommended) — `LogisticRegression3v7.confidence` returned
    `sigmoid(margin)` = P(y=+1) regardless of the predicted sign** (~0 for
    confident y=−1 predictions; a latent wrong-metric bug no claim consumed
    today). Fixed: `sigmoid(|margin|)` = confidence in the PREDICTED class.
    Guarded by `test_logreg_confidence_on_predicted_class` / mutation
    `mut_logreg_confidence_p_y_plus_one`.
  - **R3 (recommended) — SPEC G1/G2 proclaimed pylearn2 `mnist_pi.yaml`
    fill-ins the code does not implement** (lr .1 / batch 100 / momentum
    .5→.7 / `irange .005` / `max_col_norm 1.9365` vs the code's lr 0.05 /
    batch 128 / constant momentum 0.9 / `N(0, 0.5/√fan_in)` / no max-col-norm).
    Reconciled: SPEC G1/G2 "Resolution" prose now describes what the code
    actually does (code-as-run), with the earlier pylearn2 claim recorded as
    superseded.
  - **R4 (recommended) — c65 satisfied by construction.** c65 (correct-class
    logit > max-wrong at ε=0) holds by the eps_trace example's *selection*
    (the example is chosen correctly classified, as the paper's Fig 4
    caption assumes "The correct class is 4"), so it is not an independent
    test. Moved c65 to `not_tested`; the tail/shape claims c66–c70 remain
    genuinely tested on the same example (c66 fails honestly on it).
  Test suite: 42 passed with the real CIFAR tar on disk (41 pass + 1 skip in a
  tar-less fresh clone — the skip is the CIFAR positive fingerprint test) after
  the fixes; 14 mutations all caught (12 prior + 2 new). `measured.json` is
  regenerated for the three changed-code arms (`cifar_conv_maxout`,
  `maxout_large_adv`, `ensemble12`) from real data at every seed this pass
  (EAE_ONLY + EAE_FORCE so the resume cache does not keep the pre-fix values);
  unchanged-code arms keep their prior real-data values. `selfcheck.json` is
  regenerated from the new `measured.json` (45 pass / 24 fail, 14/14 HIGH,
  gate=PASS). `claims_result.json` is left for the numbers-gate/publish step:
  the committed copy still predates this pass's `measured.json` and is stale
  (it carries pre-F1/F2 verdicts and the 3-seed c14); the gate overwrites it
  from the current `measured.json` at publish, at which point it must match
  `selfcheck.json` verdict-for-verdict.
