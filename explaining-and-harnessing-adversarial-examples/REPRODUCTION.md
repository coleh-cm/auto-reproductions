# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

**Paper:** Explaining and Harnessing Adversarial Examples
**Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
**Venue:** ICLR 2015 (arXiv:1412.6572v3, 20 Mar 2015)
**Date started:** 2026-08-06 (this run)

## Status

**Reproduction complete; published this run.** Branch
`repro/explaining-and-harnessing-adversarial-examples` pushed and `main` brought
up to it. Reference material verified on disk:

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
- [x] Readiness gates (see § Research-readiness gates below)
- [x] Numbers measured and compared against the paper (see § Numbers below)
- [x] Published

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

- 2026-08-06 — Numbers / publish step. All 16 arms ran (2 blocked per SPEC §9:
  `mp_dbm`, `googlenet_imagenet`); the numbers gate
  (`claims_result.json`, `produced_by: reproduce-paper numbers gate`) adjudicated
  69 claims → **46 reproduced / 21 refuted / 2 untested / 0 blocked**; the gate's
  load-bearing verdict over `compute_invariance == high` is **14 reproduced / 0
  refuted / 0 blocked → gate PASS**. The in-repo grader `selfcheck.json`
  (`selfcheck_claims.py` over the claim's full seed list) agrees verdict-for-
  verdict (45 pass / 24 fail, 14/14 HIGH). REPRODUCTION.md status flipped,
  measured-vs-paper table + readiness-gate table written, VERIFICATION.md
  refreshed, scratch files cleaned, branch pushed and `main` fast-forwarded.

## Numbers — measured vs paper

**Data source:** the real paper datasets, not synthetic stand-ins. MNIST is
loaded by `data.load_mnist` (50k/10k/10k×784, pixel-sum fingerprint matched to the
canonical corpus); CIFAR-10 by `data.load_cifar10` (45k/5k/10k×3072, per-image
GCN + a raw-uint8 pixel-sum fingerprint matched to the canonical cs.toronto.edu
tar; the 170 MB tar was downloaded in this environment). Both fingerprints are
asserted at load time (`tests/test_data_loader.py`). No arm substituted a
synthetic corpus for the paper's data.

**Exact command that produced every number below:**

```
.venv/bin/python run_all_arms.py     # trains every arm × seed -> measured.json
.venv/bin/python selfcheck_claims.py # grades claims.json against measured.json -> selfcheck.json
# the workflow's numbers gate then writes claims_result.json (produced_by: reproduce-paper numbers gate)
```

Each arm prints one `FINAL <arm>=<value>` line; those lines and the per-seed
`measured.json` are the raw measurements. `claims_result.json` pairs each claim's
measured mean with the paper's claimed value and a verdict. The table below uses
the gate's `claims_result.json` values (mean over the seeds the gate evaluated);
where the in-repo `selfcheck.json` (mean over the claim's full seed list) differs
materially it is footnoted. Verdicts: **reproduced** = within tolerance / ordering
holds; **refuted** = outside tolerance / ordering fails; **untested** = gap inside
the cross-seed spread (not separable from noise).

### Arms-within-noise caveat (read this first)

The paper's **headline clean-error regularization claim** is 0.94% (naive) →
0.84% (adversarial training) → 0.782% (large + Phase-2 60k retrain), a *decrease*
at each step. At this run's CPU sub-scale horizon the four clean-error arms came
out **within noise of each other** and **not** in the paper's order:

| arm | clean_err (mean over seeds) | paper |
|---|---|---|
| `maxout_naive` (c08) | 1.47% | 0.94% |
| `maxout_adv` (c11) | 1.95% | 0.84% |
| `maxout_large_naive` (c17) | 1.80% | 1.14% |
| `maxout_large_adv` (c19) | 1.93% (spread 1.25pp) | 0.782% |

The arms span ~0.5pp and the per-seed spread on `maxout_large_adv` (1.25pp) is
larger than the gaps between arms; the ordering claim c12 (adversarial training
*reduces* clean error, `maxout_naive.clean − maxout_adv.clean > 0`) is **refuted**
(−0.48pp). **A measurement that cannot separate the clean-error arms has not
tested the paper's regularization comparison**, whatever else it shows, and these
numbers are not presented as evidence about that claim. They are reported because
they are what the machine produced at the 6-epoch + 60k-retrain horizon, honestly
short of the paper's full budget.

The **adversarial-robustness arms separate clearly and were tested**: a naive
maxout is fooled ~89–94% by FGSM (c09 89.1%, c15 93.9%) while the adversarially-
trained large maxout falls to **c14 13.2%** (gate; selfcheck full-5-seed mean
11.1%) — far outside the cross-seed spread, reproducing the paper's 89.4% → 17.9%
effect in direction and magnitude.

### Horizon deviation

The 1600-unit maxout is trained **6 epochs** (paper: until validation-levels-off
+ early-stop on adversarial validation, typically dozens) plus the paper's
from-scratch 60k Phase-2 retrain (`tex:505-506`, implemented and guarded by
`test_degeneracy.py`). The conv maxout is trained **25 epochs** (paper: full
budget). This is shorter than the paper's. A number produced at a horizon too
short to separate the clean-error arms is not evidence about the clean-error
claim (stated above); the adversarial-robustness and FGSM-effect orderings, which
separate at this horizon, are evidence about those claims.

### Measured-vs-paper table (all 69 adjudicated claims)

| id | kind | paper claim (short) | claimed | measured | spread | verdict |
|---|---|---|---|---|---|---|
| c01 | value | softmax MNIST FGSM ε=.25 error | 99.9% | 100.0% | 0.0 | reproduced |
| c02 | value | softmax MNIST FGSM avg confidence | 79.3% | 99.2% | 0.33 | refuted |
| c03 | ordering | FGSM reliably misclassifies (softmax) | — | 91.1 | 0.31 | reproduced |
| c04 | value | logreg 3v7 clean error | 1.6% | 1.54% | 0.15 | reproduced |
| c05 | value | logreg 3v7 FGSM error | 99% | 100.0% | 0.0 | reproduced |
| c06 | ordering | logreg FGSM error > 0 | — | 98.5 | 0.15 | reproduced |
| c07 | invariant | sign(grad)=−y·sign(w); w·sign(w)=‖w‖₁ | — | — | — | reproduced |
| c08 | value | maxout naive clean error | 0.94% | 1.47% | 0.06 | refuted |
| c09 | value | maxout FGSM error | 89.4% | 89.1% | 4.72 | reproduced |
| c10 | value | maxout FGSM avg confidence | 97.6% | 92.8% | 2.14 | reproduced |
| c11 | value | maxout+adv clean error | 0.84% | 1.95% | 0.24 | refuted |
| c12 | ordering | adv training reduces clean error | — | −0.48 | 0.30 | refuted |
| c13 | ordering | adv training cuts adv error (maxout) | — | 80.6 | 3.95 | reproduced |
| c14 | value | large+adv FGSM error | 17.9% | 13.2%¹ | 19.2 | reproduced |
| c15 | value | large naive FGSM error | 89.4% | 93.9% | 1.78 | reproduced |
| c16 | ordering | adv training cuts adv error (large) | — | 80.8 | 20.8 | reproduced |
| c17 | value | large naive clean error | 1.14% | 1.80% | 0.28 | refuted |
| c18 | existence | 4/5 seeds ≤ 0.83% clean | — | — | — | refuted |
| c19 | value | large+adv clean error (best MNIST) | 0.782% | 1.93% | 1.25 | untested |
| c20 | value | large+adv misclass confidence | 81.4% | 71.8% | 4.21 | reproduced |
| c21 | value | orig-on-advfromnew transfer error | 40.9% | 71.5% | 0.74 | refuted |
| c22 | value | new-on-advfromorig transfer error | 19.6% | 39.8% | 4.61 | refuted |
| c23 | ordering | transfer asymmetry > 0 | — | 31.8 | 3.99 | reproduced |
| c24 | value | Rademacher-noise control FGSM error | 86.2% | 88.7% | 3.87 | reproduced |
| c25 | value | Rademacher-noise control confidence | 97.3% | 89.1% | 0.59 | refuted |
| c26 | value | uniform-noise control FGSM error | 90.4% | 88.1% | 1.98 | reproduced |
| c27 | value | uniform-noise control confidence | 97.8% | 91.3% | 1.90 | reproduced |
| c28 | ordering | noise controls still fooled > 0 | — | 74.8 | 21.5 | reproduced |
| c29 | value | RBF FGSM error | 55.4% | 98.5% | 0.46 | refuted |
| c30 | value | RBF misclass confidence | 1.2% | 22.4% | 0.10 | refuted |
| c31 | value | RBF clean confidence | 60.6% | 67.2% | 0.08 | reproduced |
| c32 | ordering | RBF low confidence when fooled | — | 70.1 | 2.15 | reproduced |
| c33 | ordering | RBF confidence drops on adv | — | 44.7 | 0.07 | reproduced |
| c34 | value | ensemble (whole-crafted) FGSM error | 91.1% | 98.1% | 0.63 | reproduced |
| c35 | value | single-member-crafted FGSM error | 87.9% | 92.0% | 1.55 | reproduced |
| c36 | ordering | whole-crafted > 0 | — | 96.5 | 0.67 | reproduced |
| c37 | ordering | whole > single (ensemble resistance) | — | 6.0 | 1.51 | reproduced |
| c38 | value | softmax predicts maxout's class (all) | 54.6% | 71.3% | 1.75 | refuted |
| c39 | value | RBF predicts maxout's class (all) | 16.0% | 38.0% | 4.96 | refuted |
| c40 | value | softmax predicts maxout's class (cond) | 84.6% | 74.2% | 1.58 | refuted |
| c41 | value | RBF predicts maxout's class (cond) | 54.3% | 59.2% | 4.58 | reproduced |
| c42 | value | RBF predicts softmax's class | 53.6% | 60.9% | 1.37 | reproduced |
| c43 | ordering | softmax > RBF agreement (cond) | — | 33.3 | 3.21 | reproduced |
| c44 | ordering | RBF linear-component > 0 | — | 15.0 | 3.38 | reproduced |
| c45 | existence | maxout FGSM never class 8 | — | — | — | reproduced |
| c46 | value | maxout Gaussian-rubbish error | 98.35% | 97.2% | 1.86 | reproduced |
| c47 | value | maxout rubbish confidence | 92.8% | 90.2% | 2.62 | reproduced |
| c48 | value | sigmoid-top rubbish error | 68.0% | 11.2% | 7.17 | refuted |
| c49 | value | sigmoid-top rubbish confidence | 87.9% | 81.2% | 2.78 | reproduced |
| c50 | value | softmax-reg rubbish error | 59.8% | 98.3% | 0.70 | refuted |
| c51 | value | softmax-reg rubbish confidence | 70.8% | 92.7% | 1.51 | refuted |
| c52 | value | RBF rubbish error | 0.0% | 0.0% | 0.0 | reproduced |
| c53 | ordering | RBF resists rubbish > 0 | — | 97.2 | 1.86 | reproduced |
| c54 | existence | CIFAR maxout never class airplane/auto/horse/ship/truck | — | — | — | reproduced |
| c55 | value | MNIST maxout rubbish classified as 5 | 45.3% | 29.7% | 14.5 | reproduced |
| c56 | value | CIFAR conv maxout FGSM ε=.1 error | 87.15% | 97.2% | 3.15 | refuted |
| c57 | value | CIFAR conv FGSM confidence | 96.6% | 93.9% | 4.39 | reproduced |
| c58 | value | CIFAR conv Gaussian-rubbish error | 93.4% | 99.4% | 1.90 | reproduced |
| c59 | value | CIFAR conv rubbish confidence | 84.4% | 97.1% | 8.04 | refuted |
| c60 | value | CIFAR fooling avg per-step success | 75.3% | 49.4% | 43.7 | untested |
| c61 | value | CIFAR fooling airplane success | 24.7% | 11.6% | 34.7 | reproduced |
| c62 | existence | frog&truck fooling 100% (3 seeds) | — | — | — | refuted |
| c63 | existence | airplane hardest (3 seeds) | — | — | — | refuted |
| c64 | existence | L1 coef .0025 stuck > 5% train err | — | — | — | reproduced |
| c66 | curve | Fig.4 correct-class thin manifold | — | — | — | reproduced |
| c67 | curve | Fig.4 wrong classes stable wide region | — | — | — | reproduced |
| c68 | curve | Fig.4 logits piecewise-linear in ε | — | — | — | reproduced |
| c69 | curve | Fig.4 logits within range | — | — | — | reproduced |
| c70 | curve | Fig.4 correct class is 4 | — | — | — | reproduced |

¹ c14: the numbers gate (`claims_result.json`) evaluated the claim over its
top-3 seeds → 13.2%; the in-repo grader `selfcheck.json` over the claim's full
5-seed list → 11.1% (per-seed 5.78/8.83/24.96/10.89/5.27). Both verdicts are
`reproduced` (paper 17.9%, tol 8). The two graders disagree only on seed
selection for this one claim, not on the verdict.

**Blocked arms (SPEC §9, not built):** `mp_dbm` (multi-prediction deep Boltzmann
machine — outside this run's compute scope) and `googlenet_imagenet` (Fig. 1 demo
needs a pretrained GoogLeNet + ImageNet). Their claims are absent from the table
(the gate reports 0 blocked — they are not adjudicated, not failed).

### What the numbers say, plainly

- **Reproduced and load-bearing (HIGH invariance, 14/14):** the FGSM effect on
  linear and maxout models (c03/c06/c13/c16), the analytic-logistic equivalence
  c07, the FGSM ‖η‖∞=ε invariant, the degeneracy no-op, and the orderings that
  are the paper's actual claims (adversarial training cuts FGSM error c13/c16;
  RBF low-confidence-when-fooled c32/c33; ensemble whole>single c37; transfer
  asymmetry c23; agreement c43/c44; noise-control still-fooled c28).
- **Reproduced as orderings, refuted as tight values:** several `low`/`medium`
  value claims fail in absolute terms (RBF 55.4%→98.5%, softmax-rubbish
  79.3%→99.2%, CIFAR adv 87.15%→97.2%) but the *ordering* the paper draws from
  them holds. The value gaps trace to training hyperparameters / model
  parametrizations the paper never states (RBF β/μ, exact softmax training) and
  to the sub-scale horizon.
- **Refuted, honestly:** the clean-error regularization values (c08/c11/c17/c19)
  and the c12 ordering — within noise at this horizon, not evidence about the
  paper's clean-error claim (see caveat above).
- **Untested (gap inside seed spread):** c19 (0.782% clean) and c60 (CIFAR
  fooling avg 75.3%) — not separable from run-to-run noise at this scale.

## Research-readiness gates

| # | gate | verdict | evidence |
|---|---|---|---|
| 1 | Builds from scratch | **partial** | `Dockerfile` is present and well-formed; `uv venv --python 3.13 .venv && uv pip install -r requirements.txt` builds the pinned closure cleanly and the imports resolve (torch 2.7.1+cpu, numpy 2.3.2, matplotlib 3.11.1, pytest 8.4.2). Docker itself is unavailable in this sandbox, so `docker build` was NOT run — a reader with docker should run it. |
| 2 | README is accurate | **pass** | README quickstart followed verbatim: `uv venv` + `uv pip install -r requirements.txt`, the import check prints the expected versions, `.venv/bin/pytest -q` → 42 passed, `python -m run_all_arms` + `selfcheck_claims.py` produce `measured.json`/`selfcheck.json`. |
| 3 | Packages are clear | **pass** | `requirements.txt` pins every direct + transitive dep to a version with a comment block; install succeeds and the code then imports cleanly (no missing-import death). |
| 4 | Entrypoint is obvious | **pass** | one documented command: `run_all_arms.py` (via `run_all_arms.sh`) takes flags (`EAE_ONLY`, `EAE_FORCE`, `EAE_SMOKE`, seeds) — no source edits needed. |
| 5 | Fast path | **pass** | `bash smoke.sh` trains softmax on 2k MNIST × 5 epochs and prints `FINAL softmax_reg=...` in ~1.5 s; exercises the whole train→FGSM→eval path. |
| 6 | Deterministic / noise quantified | **pass** | same seed → same number (softmax_reg seed 0 reproduced `adv_err=100.0` on re-run); every claim's cross-seed spread is recorded in `claims_result.json`/`selfcheck.json` and reported in the table above. |
| 7 | Degeneracy test in repo | **pass** | `tests/test_degeneracy.py`: ε=0 / noise-ε=0 / L1-coef=0 are bit-identical to baseline; ε>0 bit-differs; Phase-2 retrain `phase2_start_state == init_state` guard + negative test. |
| 8 | Data provenance stated | **pass** | `data.load_mnist` / `data.load_cifar10` download the canonical corpora into `./data/` (gitignored) and assert pixel-sum fingerprints at load; README states the source and version; positive + negative fingerprint tests in `tests/test_data_loader.py`. |
| 9 | Recorded number reproducible | **pass** | the command beside the numbers (`run_all_arms.py` + `selfcheck_claims.py`) reproduces `measured.json`/`selfcheck.json` within the recorded seed spread (softmax_reg re-run confirmed). |
| 10 | Nothing depends on hidden local state | **partial** | a fresh clone builds, installs, and passes tests (41 pass + 1 skip — the CIFAR positive fingerprint test skips until the 170 MB tar is re-downloaded, which `load_cifar10` does automatically). The tar is not checked in (gitignored); a reader needs network on first run. No home-directory or manual-wheel dependency. |
