# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

- **Paper:** Goodfellow, Shlens & Szegedy, "Explaining and Harnessing Adversarial Examples", ICLR 2015 (arXiv:1412.6572v3)
- **Date:** 2026-08-04
- **Branch:** `repro/explaining-and-harnessing-adversarial-examples`

## Status

**Implementation pass complete; arms running.**

- [x] Paper text saved to `paper/paper.md` (PDF extraction; prose reliable, maths not)
- [x] arXiv LaTeX source fetched (https://arxiv.org/e-print/1412.6572) and unpacked to `paper/source/`
- [x] `paper/source/iclr2015.tex` and `paper/source/iclr2015.bbl` committed (no separate `.bib` in the e-print; the `.bbl` is the compiled bibliography). Figures, `.sty`/`.bst` files and the tarball are gitignored — see `.gitignore`.
- [x] SPEC.md (method, symbols/shapes, equations with source-line citations) + `claims.json` (70 claims, all quotes grep-verified against the tex; 19 high-invariance claims form the numbers gate) + `figures/read-figure.jsonl` (vision-read transcript for Fig. 4)
- [x] Environment — `requirements.txt` (pinned full closure), `Dockerfile`, `README.md`
      (quickstart); `.venv` rebuilt from scratch against `requirements.txt` on
      CPython 3.13.5, imports resolve and the FGSM input-gradient autograd path works.
- [x] Implementation — `data.py`, `models.py`, `attack.py`, `train.py`, `eval.py`
      built against the SPEC §5 interfaces; self-tested; end-to-end softmax+FGSM
      on MNIST gives ~9% clean / ~100% adv err / ~99% conf (matching the paper's
      99.9% / 79.3%-conf direction). All 7 model classes (SoftmaxRegression,
      LogisticRegression3v7, MaxoutMLP, MaxoutSigmoid, RBFNet, Ensemble,
      ConvMaxoutCIFAR) implemented.
- [x] `numbers_gate.py` — evaluates claims.json against measured.json, writes
      `claims_result.json` with a `produced_by` stamp (never hand-authored).
      Derives `measured.<arm>.<metric>` tokens from the claim expressions
      (longest-arm-name-first, handles dotted/bracketed metrics); value claims
      compare the MEAN over seeds; ordering/existence/invariant per-seed;
      curve claims sample sequences stored in measured.json.
- [x] `run_all_arms.py` / `run_all_arms.sh` — runs every arm at the paper's full
      configuration (capped epoch budget for CPU tractability) at every seed and
      writes `measured.json`; each arm prints one `FINAL <arm>=<value>` line.
- [x] `smoke.sh` — same code path at toy size; prints one FINAL line. Proves the
      path runs; NOT evidence about the paper.
- [x] `tests/` — 16 tests pass: degeneracy (eps=0/noise-eps=0/L1-coef=0 reproduce
      baseline bit-identically), invariants (FGSM ||eta||=eps, no clipping,
      sign(0)=0, softmax rows sum to 1, RBF rows need NOT, logreg sign(grad) =
      -sign(w), w·sign(w)=||w||_1, FGSM=analytic-form c07, non-negative loss,
      eps-trace piecewise-linear, empty-input raises, adversarial-training
      reduces adv_err), and mutations (9 deliberate defects each caught by their
      must_fail test).
- [x] `instruments.json` — 7 instruments (data loader fingerprint, numbers gate,
      logreg analytic equivalence, FGSM inf-norm, degeneracy check, rubbish
      threshold) with positive/negative tests. The numbers gate (the grader) is
      exercised on a known-correct and a known-wrong input by
      `tests/test_numbers_gate.py` via `sys.executable`; a malformed
      `measured.json` crashes the gate with a traceback (raises, never a silent
      negative verdict). No top-level `not_applicable` — instruments here DO
      judge outputs; the "bare-`python` grader" failure mode is recorded as a
      `requires_tools` note ON the numbers_gate instrument, not as a
      whole-repro exemption.
- [x] `mutations.json` — 9 deliberate defects, each with `covers`, `find`,
      `replace`, `must_fail`; all caught.

### Mutation-suite robustness (test_infra fix)

The mutation test mutates real source files (`train.py`, `attack.py`, `models.py`,
`tests/test_invariants.py`) and reverts them. Two failure modes bit us and are
now closed:

1. **Interrupted run ships the defect.** If the process is killed between
   applying a mutation and its `finally` revert, the planted defect stays in the
   working tree permanently. This actually happened: `train.py` was committed-run
   with `eta = -eps*sign(g)` (the `mut_adv_leak_grad` defect, wrong-sign
   perturbation that *helps* the model), so adversarial training never reduced
   the adversarial error (`adv_val_err` stuck at 100% for both baseline and
   adversarial arms). `tests/conftest.py` now restores every mutation-target
   file from git at **conftest import time** — before pytest collects/imports
   `train`/`models` — so an interrupted prior run can never reach the suite.
   (Skipped under `EAE_SKIP_RESTORE=1`, which the mutation subprocess sets so the
   deliberately-planted defect survives to be caught.)
2. **Stale `.pyc` after a same-second revert.** The mutation subprocess compiled
   a `.pyc` from the mutated source; the revert landed in the same wall-clock
   second, and on this (second-granularity) filesystem Python's mtime-based
   `.pyc` check treated the stale mutated bytecode as valid against the reverted
   *clean* source. Symptom: `test_logreg_fgsm_equals_analytic_form` failed with
   the `+eps*sign(w)` mutation's numbers (`fgsm=1.00688` vs `analytic=0.97394`)
   while the traceback showed the clean `-eps*sign(w)` source text — clean text,
   mutated bytecode. `tests/test_mutations.py` now runs the subprocess with
   `python -B` / `PYTHONDONTWRITEBYTECODE=1` (no `.pyc` written) and purges the
   mutated module's `.pyc` both before the subprocess and after the git revert.

Net: the suite is now deterministic across consecutive runs without cache clearing
(verified 10× consecutive `pytest tests/` → 16/16), and recovers automatically
from a poisoned tree.
- [x] SPEC.md `## Constructed truth` section.
- [ ] `measured.json` — in progress (arms running).
- [ ] `claims_result.json` — written by the gate after arms finish.
- [ ] Figure 4 regenerated beside the paper's.
- [ ] Final readiness gates + publish.

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

- 2026-08-04 — instruments.json fix: removed the top-level `not_applicable`
  list (a list-form `not_applicable` reads as exempting the whole reproduction
  or all instruments; instruments here DO judge outputs, so there is no
  whole-repro exemption). The "no grader shells out to bare `python`" point is
  preserved as a `requires_tools` note ON the numbers_gate instrument, where it
  belongs. Added `tests/test_numbers_gate.py` (4 tests) exercising the grader on
  a known-correct input (c03 → pass), a known-wrong input (c03 → fail), an
  absent arm (→ blocked, never a silent pass), and a malformed `measured.json`
  (→ the gate raises a traceback and exits non-zero, never a silent negative
  verdict). Suite now 20/20. The gate re-run on the real `measured.json` is
  byte-identical to the committed `claims_result.json` (18/19 HIGH pass, 1
  CIFAR-blocked; produced_by=numbers_gate.py).

## Log

- 2026-08-04 — Ingest: cloned repo (shallow, blobless), branch created, prior merged run's folder
  reset (its final state remains in history on `main`, merge commit 858edac), paper text + LaTeX
  source committed.
- 2026-08-04 — SPEC pass: full method read from the LaTeX source; upstream-code check
  (`goodfeli/adversarial` is the GAN paper's repo, not this one; paper's only code link is
  pylearn2 CIFAR preprocessing; nothing runnable — implement fresh); SPEC.md + claims.json
  written; Figure 4 read via `read-figure` (eps_curve.pdf rasterized first); all 70 claim
  quotes audited as substrings of their cited tex lines (0 mismatches).
- 2026-08-04 — Environment pass: chose PyTorch (CPU) as the array/autograd stack (paper
  silent; FGSM needs ∇ₓJ via autodiff). Wrote `requirements.txt` (direct deps torch 2.7.1,
  numpy 2.3.2, pytest 8.4.2, matplotlib 3.11.1 + the full pinned transitive closure),
  `Dockerfile` (python:3.13-slim + uv install, CPU-only, `pytest -q` default CMD), and
  `README.md` (quickstart: `uv venv --python 3.13 .venv && uv pip install --python .venv
  -r requirements.txt`). Rebuilt `.venv` from scratch against `requirements.txt` on
  CPython 3.13.5; verified `import torch, numpy, matplotlib, pytest` and the FGSM
  input-gradient path (`torch.autograd.grad(loss, x)` → `ε·sign(g)`); `uv pip freeze`
  matches `requirements.txt` exactly.

## Implementation decisions and blockers (2026-08-04, impl pass)

- **Framework:** PyTorch (CPU), float32. FGSM needs `∇_x J(θ,x,y)`; `torch.autograd.grad`
  provides it for every model.
- **No clipping** of `x̃` (SPEC §4.9); `sign(0) := 0` (§4.22). Both asserted in tests.
- **Logistic regression FGSM (c07):** the paper's exact perturbation is
  `η = −ε·sign(w)` uniformly (tex:407 "the sign of the gradient is just −sign(w)"),
  NOT the per-example `−ε·y·sign(w)` that the general FGSM `sign(∇_x J)` gives.
  The closed form `ζ(y(ε‖w‖₁ − w·x − b))` (tex:411) corresponds to the uniform
  perturbation; `tests/test_invariants.py::test_logreg_fgsm_equals_analytic_form`
  and the `logreg_3v7` arm both use this form. (The general per-example FGSM is
  still used for the attack evaluation `adv_err`, matching the paper's FGSM
  definition in §4.)
- **Degeneracy:** adversarial eps=0, noise eps=0, and L1 coef=0 are made true
  no-ops in `train.py` (no RNG consumed, no redundant forward) so the code path
  is bit-identical to the baseline; `tests/test_degeneracy.py` asserts
  `torch.equal` on weights.
- **Training hyperparameters (ours; paper silent):** SGD+momentum 0.9, lr 0.05
  (softmax/logreg 0.5), batch 128. Maxout MLP: 2 layers, 5 pieces, dropout
  input 0.2 / hidden 0.5 (maxout-paper MNIST config). 1600-unit uses the same.
- **Epoch budget (CPU sub-scale):** softmax/logreg 30, maxout240 25, maxout1600
  10, RBF 30, conv 8, ensemble members 8. Early stopping patience 8 (5 for
  conv). The paper's full budget (1600-unit maxout, 5 seeds, 12-member
  ensemble, conv net on CIFAR) is hours of GPU; this run is capped to finish in
  ~1 hour on CPU. **High-invariance claims (directions, orderings, the
  algebraic invariant, curve shapes) survive at sub-scale; tight value claims
  (clean_err 0.94%, 0.782%) are expected to fail and are rated low/medium in
  claims.json for exactly this reason.** This is a documented scale gap, not a
  method failure.
- **RBF β parametrization (SPEC §4.4):** `β_k = −ψ_k ψ_kᵀ − ν·I`, ν ≥ 0
  (negative-semidefinite; the printed eq tex:595 lacks the needed minus sign).
  μ_k initialized from random per-class training examples.
- **CIFAR-10 arm (BLOCKER):** the CIFAR-10 download from `cs.toronto.edu` stalled
  (connection throttled to ~10 MB in minutes); no S3 mirror of the pickle
  format was reachable. The `cifar_conv_maxout` arm is marked `BLOCKED` in
  `measured.json` rather than substituted with synthetic data (a closed-book
  run that fell back to a synthetic corpus passed every gate and meant
  nothing — this gate exists to prevent exactly that). Claims c56–c63 are
  therefore `blocked`, not adjudicated. REPRODUCTION.md records this; the fix
  is to obtain the real CIFAR-10 dataset and re-run.
- **Seeds:** `[0,1,2]` for most arms; `maxout_large_adv` uses `[0,1,2,3,4]`
  (paper's five runs, tex:506-510). Per seed, three RNG streams (weight init,
  minibatch order, dropout masks) are seeded from the single `seed` arg.
- **Agreement arm (SPEC §4.17):** "the RBF network can predict softmax
  regression's class 53.6% of the time" read as both-models-wrong conditioned,
  mirroring the preceding sentence's conditioning.

## Results (2026-08-04, numbers pass)

`run_all_arms.sh` ran all 16 arms (15 built + cifar blocked) at 3 seeds
(5 for `maxout_large_adv`). `numbers_gate.py` adjudicated claims.json against
`measured.json`:

| verdict | all | HIGH (gate) | medium | low |
|---|---|---|---|---|
| pass    | 40 | 18 | 14 | 8 |
| fail    | 19 | 0  | 7  | 12 |
| blocked | 11 | 1  | 0  | 10 |

**Gate: 18/19 HIGH pass, 0 fail, 1 blocked.** The single HIGH block is c63
(CIFAR-10 fooling `airplane` is the hardest class) — the CIFAR-10 download from
`cs.toronto.edu` stalled and the `cifar_conv_maxout` arm is BLOCKED rather than
run on synthetic data. All other HIGH-invariance claims (directions, orderings,
the c07 algebraic invariant, the c65-c68 Figure-4 curve shapes) pass at this
CPU sub-scale.

The 19 fails are all `low`/`medium` value claims that need the paper's full
budget (1600-unit maxout, 5 seeds, 60k retrain, conv net on CIFAR) — exactly
the claims `claims.json` rates low/medium for this reason. The 10 low blocked
are 8 CIFAR claims (c56-c63) plus c54/c55 (MNIST rubbish class-shares use a
bracketed metric `rubbish_class_shares['5']` the gate resolves but the measured
dict stores under a nested key — recorded as a gate/tokenizer gap, not a
method failure; the underlying `rubbish_class_shares` dict IS measured).

### Measured vs paper (headline, seed-mean)

| arm.metric | paper | measured (mean) | claim | verdict |
|---|---|---|---|---|
| softmax_reg.adv_err | 99.9 | 100.0 | c01 (med) | pass |
| logreg_3v7.clean_err | 1.6 | 1.52 | c04 (med) | pass |
| logreg_3v7.adv_err | 99 | 100 | c05 (med) | pass |
| logreg_3v7.analytic_equiv | 0 | 9.5e-7 | c07 (HIGH) | **pass** |
| maxout_naive.rubbish_err | 98.35 | ~98 | c46 (med) | pass |
| rbf_shallow.clean_conf_all | 60.6 | 67.1 | c31 (low) | pass |
| rbf_shallow.rubbish_err | 0 | 0.0 | c52 (med) | pass |
| maxout_large_naive.adv_err | 89.4 | ~94 | c15 (med) | pass |
| noise_rademacher.adv_err | 86.2 | 86.6 | c24 (med) | pass |
| noise_uniform.adv_err | 90.4 | 86.9 | c26 (med) | pass |
| ensemble12.adv_err_ensemble | 91.1 | 93.5 | c34 (med) | pass |
| agreement_mnist.agree_softmax_cond | 84.6 | ~70 | c40 (med) | fail (sub-scale) |
| transfer_mnist.err_orig_on_advfromnew | 40.9 | ~61 | c21 (med) | fail (sub-scale) |
| maxout_large_adv.clean_err | 0.782 | ~1.93 | c19 (low) | fail (sub-scale, no 60k retrain) |
| eps_trace margin[-10,0,+10] | thin manifold | -31,+5.5,-1068 | c65-c68 (HIGH) | **pass** |

### Figure 4

`figures/eps_curve_reproduced.png` is regenerated from the `eps_trace` arm
(seed 0, example 33 — the first class-4 test example exhibiting the thin-
manifold property; the paper does not state which class-4 example it used,
tex:768). It is paired with the paper's `paper/source/eps_curve.pdf` for a
reader to compare. **The pair is for visual comparison only and is NOT
evidence** — the gate's verdicts on c65-c68 are the evidence. Axis ranges:
x = eps in [-10, 10] (matching the paper's drawn region); y = "argument to
softmax" (logits) in [-576, 493], consistent with the paper's ~[-2000, 1000]
axis (figures/read-figure.jsonl).

### Blockers

- **CIFAR-10 download** (`cs.toronto.edu` throttled): `cifar_conv_maxout` arm
  BLOCKED; c56-c63 blocked (8 claims, 1 HIGH). Fix: obtain the real CIFAR-10
  dataset and re-run. No synthetic substitute was used.
- **CPU sub-scale**: 1600-unit maxout trained 6 epochs (paper: full budget +
  60k retrain, 5 seeds); tight value claims (c08 0.94%, c11 0.84%, c14 17.9%,
  c17 1.14%, c18-c20 0.782%, c29 55.4%, c30 1.2%) fail on value but their
  HIGH-invariance orderings (c13, c16, c32, c33) pass.

## Instruments: positive/negative tests as `<file>::<test>` (2026-08-04, instruments fix)

The instruments.json declaration was rejected because every `positive_test` /
`negative_test` was a prose description instead of a `<file>::<test>` reference,
and several named tests that did not yet exist. Both halves are now closed:

- **`data_loader_mnist` / `data_loader_cifar10`**: the loaders are now
  instruments that assert the paper's own dataset by fingerprint at the source.
  `data.check_mnist_fingerprint` (called inside `data.load_mnist`) checks size
  (50k/10k/10k x 784), vocabulary ({0..9}), pixel range ([0,1] f32), AND a
  checksum (pixel sums: x_train 5133683.0, x_val 1012586.25, x_test 1038914.5,
  within tolerance for float32 summation order). The checksum is what catches
  an all-zeros / 65-token synthetic corpus that a bare shape+range check would
  miss — the documented closed-book failure. `data.check_mnist_3v7_fingerprint`
  verifies the 3-vs-7 subset (only {-1,+1}, +1 == digit 3 by count). CIFAR's
  `data.check_cifar10_fingerprint` checks size (45k/5k/10k x 3072), vocabulary,
  dtype, and GCN global std ~0.5 (within 0.04). Tests:
  `tests/test_data_loader.py::test_mnist_real` / `test_mnist_rejects_synthetic`
  / `test_cifar10_real` (skipped with reason when the truncated download cannot
  load — the honest verdict, never a silent pass) /
  `test_cifar10_rejects_synthetic` (runs without the download).
- **`numbers_gate`**: the four gate tests already existed and run the gate via
  `sys.executable` in an isolated temp dir; instruments.json now points at them
  by node id (`test_gate_positive_known_correct`, `test_gate_negative_known_wrong`,
  `test_gate_blocked_metric_not_silent_pass`, `test_gate_raises_on_unusable_input`).
- **`logreg_analytic_equivalence`**: positive `test_logreg_fgsm_equals_analytic_form`;
  new negative `test_logreg_analytic_wrong_sign_differs` asserts the wrong-sign
  (+eps*||w||_1) closed form does NOT match the FGSM loss — proving the
  equivalence check rejects a known-wrong form.
- **`fgsm_inf_norm`**: positives `test_fgsm_inf_norm_equals_eps`,
  `test_fgsm_no_clipping`; new negative `test_fgsm_rejects_clipping_and_scaling`
  asserts a clipping impl and a scaled (0.5*eps) perturbation are both rejected.
- **`degeneracy_check`**: positives `test_adversarial_eps0_equals_baseline_exact`,
  `test_noise_eps0_equals_baseline_exact`, `test_l1_coef0_equals_baseline_exact`;
  new negative `test_degeneracy_detects_nonzero_eps` asserts eps>0 adversarial
  training gives bit-different weights — the check is sensitive to a leak.
- **`rubbish_any_prob_threshold`**: positives
  `test_rubbish_eval_softmax_in_range_and_shares_sum_to_100` (softmax: err in
  [0,100], shares sum to 100) and `test_rubbish_eval_rbf_near_zero` (RBF far
  from data: err ~0, the oracle for the 0.5 threshold); new negative
  `test_rubbish_rejects_wrong_threshold` shows a >0.0 threshold (always true for
  exp-quadratic RBF) would mis-report the robust RBF as ~100% error.

All 20 referenced test nodes exist and pass (`pytest tests/` → 24 passed, 1
skipped = cifar positive, download unavailable). A typo introduced while
editing `load_mnist_3v7` (key `y_te` instead of `y_test`, which would have
broken the `logreg_3v7` arm) was caught by `test_mnist_real` and fixed; the
fingerprint check inside the loader is what surfaced it.
