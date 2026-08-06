# VERIFICATION — Explaining and Harnessing Adversarial Examples

This file lists **every check this reproduction ran**, what it found, the
budget it ran at, and — the part that matters — **what remains untested and
why**. None of this establishes that the implementation is *correct*; it
establishes that it is *not wrong in the ways that were checked*. That list is
worth more to a reader than the headline number.

All commands were run on a CPU sandbox (no GPU, no docker). Latest run 2026-08-06
(review-findings fix pass + measured.json regeneration: the three changed-code
arms — `cifar_conv_maxout` (per-image GCN + corrected raw-checksum fingerprint),
`maxout_large_adv` (full60k Phase-2 retrain), `ensemble12` (mean-probability NLL
attack objective) — were rerun from real data at every seed this pass and
`measured.json`/`selfcheck.json` regenerated; see REPRODUCTION.md).

---

## 1. Environment / build

| check | command | budget | found |
|---|---|---|---|
| venv builds from pinned closure | `uv venv --python 3.13 .venv && uv pip install --python .venv -r requirements.txt` | ~minutes | succeeds; `uv pip freeze` matches `requirements.txt` |
| imports resolve | `.venv/bin/python -c "import torch, numpy, matplotlib, pytest"` | seconds | torch 2.7.1+cpu, numpy 2.3.2, matplotlib 3.11.1, pytest 8.4.2 |
| FGSM autograd path | `torch.autograd.grad(loss, x)` → `ε·sign(g)` | seconds | works for every model class |
| docker build | `docker build -t eae-repro .` | — | **NOT RUN** — docker unavailable in this sandbox |

**Untested:** the `Dockerfile` is present and well-formed but was never built
here. A reader with docker should run `docker build -t eae-repro . && docker
run --rm eae-repro pytest -q`.

## 2. Data provenance

| check | command | found |
|---|---|---|
| MNIST fingerprint | `data.check_mnist_fingerprint` (inside `load_mnist`) | size 50k/10k/10k×784, labels {0..9}, range [0,1] f32, pixel-sum checksum — matches the canonical corpus |
| MNIST 3-vs-7 fingerprint | `data.check_mnist_3v7_fingerprint` | only {-1,+1}, +1 == digit 3 by count |
| CIFAR-10 fingerprint | `data.check_cifar10_fingerprint` (+ raw checksum in `_load_cifar_raw`) | size 45k/5k/10k×3072, labels {0..9}, per-image GCN (each image centered by its own mean, one global scale to std ~0.5), per-image-mean~0 sanity, per-pixel-variance structural check, and a RAW uint8 pixel-sum checksum (a property the GCN recipe cannot force) — passes on the real 170 MB tar (downloaded in this env) |

Both datasets are the real paper datasets; no synthetic corpus is substituted
anywhere. Positive + negative fingerprint tests in `tests/test_data_loader.py`
(the CIFAR positive test runs against the real tar; the negative test rejects a
std-matched iid synthetic corpus).

## 3. Correctness: test suite

| check | command | budget | found |
|---|---|---|---|
| full suite | `.venv/bin/python -m pytest tests/ -q` | ~23 s | **42 passed, 0 skipped** with the real CIFAR-10 tar on disk (the CIFAR positive fingerprint test runs; a fresh clone without the tar is 41 passed + 1 skipped — that skip is the CIFAR positive test, gated on the local tar) |

What the suite covers:

- **Degeneracy** (`tests/test_degeneracy.py`): adversarial eps=0, noise eps=0,
  L1 coef=0 are bit-identical (`torch.equal`) to the baseline; `eps>0` is
  bit-different (the check is sensitive to a leak). The from-scratch Phase-2
  retrain guard asserts `phase2_start_state == init_state` (directly detects the
  continuation bug) plus a negative test proving the guard has discriminating power.
- **Invariants** (`tests/test_invariants.py`): FGSM `‖η‖∞ == ε`; no clipping of
  `x̃`; `sign(0)=0`; softmax rows sum to 1; logreg `sign(grad) == -y*sign(w)`;
  `w·sign(w) == ‖w‖₁`; **c07: real `attack.fgsm` loss == corrected closed form
  `ζ(ε‖w‖₁ − y(w·x+b))` on mixed labels** (the paper's `tex:411` form is shown
  to fail for y=−1 — discriminating); non-negative loss; eps-trace piecewise-linear
  in ε; empty-input raises; adversarial training reduces adv_err; **in-training
  FGSM uses eval mode** (dropout OFF); **RBF has no `log_temp`/clamp**; **conv
  maxout has no post-ReLU**.
- **Mutations** (`tests/test_mutations.py`): 14 deliberate defects, each
  planted, run, and `must_fail`-caught, then reverted (with `.pyc` purging and
  conftest-time git restore opt-in so the gate's planted defects survive). The
  two added this pass cover the ensemble-loss (mean-prob NLL) and logreg-
  confidence (σ|m|) fixes.
- **Data-loader instruments** (`tests/test_data_loader.py`): MNIST real passes;
  a synthetic corpus is rejected; CIFAR positive (real tar) + negative (rejects
  synthetic) run.
- **Self-check-grader instruments** (`tests/test_selfcheck_claims.py`): the grader
  returns "pass" on a known-correct input, "fail" on a known-wrong input,
  "blocked" on an absent arm, and **raises** on a malformed `measured.json`.

**Untested by the suite:** end-to-end behavior on the MP-DBM and
GoogLeNet/ImageNet arms (not built — see SPEC §9); the full-budget value claims
(the suite checks plumbing, not the paper's tight numbers).

## 4. Fast path

| check | command | budget | found |
|---|---|---|---|
| smoke | `bash smoke.sh` | ~1.5 s | trains softmax on 2k MNIST examples × 5 epochs, FGSM ε=.25, prints `FINAL softmax_reg=...` |

The smoke path is proof the code runs; **it is not evidence about the paper**
and is never reported as a result.

## 5. Arms run (the numbers)

| check | command | budget | found |
|---|---|---|---|
| all arms | `.venv/bin/python run_all_arms.py` (resume) | 16 MNIST arms × 3 seeds (5 for `maxout_large_adv`); 1600-unit maxout capped at 6 epochs + from-scratch 60k Phase-2 retrain; conv 25 epochs | 16 arms produce real measured numbers in `measured.json`; CIFAR-10 runs on the real dataset (raw uint8 pixel-sum fingerprint now matches the canonical tar) |
| self-check grader | `.venv/bin/python selfcheck_claims.py` | seconds | **pass=45 fail=24 blocked=0; HIGH: 14 pass / 0 fail / 0 blocked; gate=PASS** |

The gate PASSES: all 14 HIGH-invariance claims reproduce (the c07 analytic
equivalence via the real per-example FGSM, the FGSM ‖η‖∞=ε invariant, the
degeneracy no-op, and the load-bearing orderings c03/c06/c13/c16/c23/c28/c32/
c33/c36/c37/c43/c44/c45/c53). The 24 fails are all `low`/`medium` value claims
that need the paper's full GPU budget, plus c12 (medium, clean-err reduction
below the sub-scale horizon), c19 (medium, 0.782% clean error below horizon),
c56/c59 (CIFAR adv_err / rubbish-confidence on a sub-scale conv net), c62/c63
(CIFAR per-class fooling), and c66 (low, Fig.4 negative-tail thin-manifold on
the deterministic example). See REPRODUCTION.md for the full measured-vs-paper
table and the refuted-claims notes.

**Horizon deviation (honest, necessary, not sufficient):** the 1600-unit
maxout is trained 6 epochs + a from-scratch 60k Phase-2 retrain (the paper's
protocol, tex:505-506) vs the paper's full budget. The paper's headline
clean-error regularization claim (0.94→0.84→0.782) is **not reproduced** at
this horizon (c12/c17/c18/c19 fail); the adversarial-robustness claim IS
reproduced and strengthened by the retrain (`maxout_large_adv` adv_err mean
11.1% vs paper 17.9%; `maxout_adv` adv_err 8.5%). A number produced at a
horizon too short to separate the clean-err arms is not evidence about that
claim, and is not presented as one.

**Ensemble resistance (c37, now resolved):** with the mean-probability NLL
attack objective (so the FGSM target is the classifier `predict` evaluates),
the whole-ensemble-crafted attack error (mean 98.1%) exceeds the
single-member-crafted attack error (mean 92.0%) by **5.2–6.7pp at every
seed** — outside the single-member cross-seed spread (~1.6pp), so c37
passes as a resolved ordering (paper 91.1% vs 87.9%, gap 3.2pp; this run
over-perturbs in absolute terms but reproduces the direction and the
whole>single separation).

**Untested at the numbers level:** the full-budget tight value claims (clean
0.94/0.84/0.782, RBF 55.4/1.2/60.6); MP-DBM (§9); the GoogLeNet/ImageNet Fig.1
demo; the fooling airplane/frog&truck per-class claims (c60/c61/c62/c63 —
per-class success rate has high variance even at 1,000 samples/class on the
sub-scale conv net). (c37, the ensemble whole-vs-single ordering, is now
resolved — see above.)

## 6. Figures

| check | command | found |
|---|---|---|
| Figure 4 reproduction | `.venv/bin/python regenerate_figures.py` → `figures/eps_curve_reproduced.png` | eps-sweep logit curve from the `eps_trace` arm (seed 0, deterministic first-common-correct class-4 example, index 4); paired with `paper/source/eps_curve.pdf` for visual comparison only |

**Axis units match the paper** (x: ε max-norm; y: logits / "argument to
softmax"). **Ranges differ**: this deterministic example reaches logits ≈
[−405, 639] over ε ∈ [−10,10] vs the paper's ≈ [−2000, 1000] over ε ∈ [−15,15]
(its hand-picked example has more extreme logits and a wider sweep). The
Figure-4 pair is **not evidence** — the gate's verdicts on c65–c70 (low) are
the evidence.

**Untested:** Figures 1, 2, 3, 5 are not regenerated (Fig.1 needs GoogLeNet/
ImageNet; Figs.2/3 are qualitative; Fig.5 needs CIFAR-10 and is folded into the
fooling metrics).

## 7. Review

Four adversarial-review rounds are tracked in REPRODUCTION.md ("Adversarial
review" + "Gate-feedback rounds 2–3" + "round 4"). Round 4 addressed three
consolidated reviews (faithful / metric / divergence): from-scratch Phase-2
retrain implemented (+ non-vacuous guard), in-training FGSM moved to eval mode,
c07 restated for the real per-example FGSM + corrected closed form (paper
tex:411 sign slip documented), c65–c68 demoted high→low + eps_trace made
deterministic (no predicate selection), RBF `log_temp`/clamp and conv post-ReLU
removed, fooling bumped to 1000/class, SPEC/code contradictions reconciled. This
is a **correctness-level** verification with a green gate.

---

## What this reproduction does NOT establish

- It does **not** establish that the implementation is correct — only that it
  is not wrong in the ways the tests, invariants, mutations, instruments, and
  gate check.
- It does **not** reproduce the paper's headline clean-error regularization
  claim (within noise at the 6-epoch sub-scale horizon); the adversarial-
  robustness claim is reproduced.
- It does **not** reproduce the CIFAR-10 / RBF tight *value* claims at sub-scale
  (adv_err, RBF confidences); only the orderings and the CIFAR rubbish oracle.
- It does **not** verify the Docker image builds (docker unavailable).
- The RBF *value* claims (c29/c30) do not match (98.5% / 22.4% conf vs paper
  55.4% / 1.2%); only the RBF *ordering* claims pass. The β/μ parametrization
  the paper never specifies is the likely source.

## Rung reached

**numbers** — environment builds (venv from pinned closure), comprehension
(SPEC.md, 69 adjudicated claims + 9 not_tested, all quotes grep-verified) done,
implementation runs, the correctness gate (tests/invariants/mutations/instruments,
42 pass with the real CIFAR tar on disk / 41 pass + 1 skip in a tar-less fresh
clone) is green, five adversarial-review rounds ran (round 5 = the
review-findings fix pass + measured.json regeneration: per-image CIFAR GCN +
corrected raw-checksum fingerprint, maxout_large_adv full60k retrain, ensemble
mean-prob-NLL loss, logreg σ|m| confidence, c65→not_tested, and the three
changed-code arms rerun from real data this pass), and the numbers gate
(`claims_result.json`, `produced_by: reproduce-paper numbers gate`) adjudicates
the claims. The 14 HIGH-invariance claims reproduce (the c07 analytic-logistic
equivalence via the real per-example FGSM, the FGSM ‖η‖∞=ε invariant, the
degeneracy no-op, and the load-bearing orderings including the
adversarial-robustness effect c13 and the now-resolved ensemble ordering c37).
The tight *value* claims (clean 0.782, RBF confidences, etc.) are honestly
**refuted** at the CPU sub-scale horizon rather than fudged. The AUTHORITATIVE
COUNTS line is read by the workflow's `result_check` off its own journal (not
writable from this sandbox); the matching `claims_result.json` it derives from
is committed beside this file. `selfcheck.json` (the in-repo current-verdict
evidence, written by `selfcheck_claims.py`) is regenerated from the current
`measured.json`; `claims_result.json` is regenerated by the numbers gate /
publish step (the committed copy predates this pass's `measured.json` and is
stale; the gate overwrites it at publish — `selfcheck.json` is the
in-repo evidence of the current verdicts).

### Budget spent this run (final/publish step)

`$HOME/.build_attempts`, `$HOME/.env_attempts`, and `$HOME/.review_rounds`
are **all absent/empty** this step: no build/env budget was spent with a gate
still failing, no environment budget was exhausted, and no review budget was
spent without the reviewers going quiet. The venv builds from the pinned
closure, 42 tests pass with the real CIFAR tar on disk (41 pass / 1 skip in a
tar-less fresh clone), the selfcheck gate is PASS (14/14 HIGH, 45 pass / 24
fail overall), and the numbers gate has 0 blocked / 0 unevaluable — **no gate
is currently failing**. The five documented adversarial-review rounds (in
REPRODUCTION.md) concluded with a green gate; this step did not run out of
review rounds.
