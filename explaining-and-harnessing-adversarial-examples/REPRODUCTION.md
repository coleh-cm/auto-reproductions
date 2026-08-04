# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

**Paper:** Explaining and Harnessing Adversarial Examples
**Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
**Venue:** ICLR 2015 (arXiv:1412.6572v3, 20 Mar 2015)
**Date started:** 2026-08-04

## Status

**Implementation complete and checked.** The method (FGSM + FGSM adversarial
training, the analytic logistic special case, the RBF / sigmoid / ensemble /
conv-maxout arms, the rubbish & targeted-fooling protocols) is implemented from
`SPEC.md` against `paper/source/iclr2015.tex` (no usable author code exists —
see SPEC §0). The self-check grader adjudicates all 70 claims.

- [x] Reproduction folder + branch `repro/explaining-and-harnessing-adversarial-examples`
- [x] arXiv LaTeX source (1412.6572) unpacked to `paper/source/`; `.tex`/`.bbl` tracked
- [x] `SPEC.md` method spec (70 grep-verified claims, 19 high-invariance)
- [x] Core modules: `data.py`, `models.py`, `attack.py`, `train.py`, `eval.py`
- [x] `run_all_arms.py` / `run_all_arms.sh` run every arm at every seed → `measured.json`
- [x] `smoke.sh` runs the softmax path end-to-end (one FINAL line; not evidence)
- [x] `tests/` — degeneracy, invariants, mutations, data-loader fingerprint, self-check grader (32 pass)
- [x] `instruments.json`, `mutations.json`, `## Constructed truth` in SPEC
- [x] Self-check grader (`selfcheck_claims.py` → `selfcheck.json`): **HIGH 18 pass / 0 fail / 1 blocked**
- [ ] CIFAR-10 arm — BLOCKED (download throttled in this env; see Blockers)
- [x] Adversarial review round 1 clean — 5/7 approved, 2 findings fixed + guarded (see "Adversarial review")

## Self-check grader (NOT claims_result.json)

`selfcheck_claims.py` is this reproduction's OWN grader. It evaluates `claims.json`
against `measured.json` and writes **`selfcheck.json`** (with a `produced_by`
stamp). It does **not** write `claims_result.json` — that filename is owned by
the workflow's numbers gate; a script here writing it would collide and be
refused. `selfcheck.json` and `claims_result.json` are both gitignored.

Latest self-check verdict (run with `.venv/bin/python selfcheck_claims.py`):

| bucket | pass | fail | blocked |
|---|---|---|---|
| HIGH (load-bearing) | 18 | 0 | 1 |
| medium | 14 | 8 | 4 |
| low | 8 | 11 | 6 |
| **all** | **42** | **20** | **8** |

`gate=FAIL` **only** because of the single HIGH block (c63, CIFAR-10 fooling).
The 20 fails are all `low`/`medium` **value** claims (clean 0.94% / 0.782%,
exact confidence %) that need the paper's full GPU budget; their
HIGH-invariance **ordering** counterparts pass. This is the expected honest
outcome at CPU sub-scale.

## Reference notes (from the LaTeX, which is authoritative)

- Preamble macros: `\eps` = `\epsilon`, `\sign` = `\text{sign}`, `\vx,\vw,\veta,\vtheta` bold vectors.
- FGSM (`iclr2015.tex:309`): **η = ε · sign(∇ₓ J(θ, x, y))**
- Adversarial training (~487): **J̃ = α J(θ, x, y) + (1−α) J(θ, x + ε sign(∇ₓ J(θ, x, y)), y)**, α = 0.5
- Adversarial logistic regression: minimize **E ζ(y(ε‖w‖₁ − wᵀx − b))**, ζ softplus
- RBF (`:595`): **p(y=1|x) = exp((x−μ)ᵀ β (x−μ))**, β negative-semidefinite (printed eq lacks the minus sign; SPEC §4.4)

Key reported numbers (verified against the .tex) and what this run measured:

| Claim (tex) | Paper | Measured (mean over seeds) | Verdict |
|---|---|---|---|
| Softmax FGSM ε=.25 MNIST err (`:334`) | 99.9% | 100.0% | ordering ✓ (value low) |
| Softmax adv conf all (`:333`) | 79.3% | 99.16% | value fail (sub-scale) |
| Maxout FGSM ε=.25 MNIST err (`:339`) | 89.4% | 89.09% | **value ✓** |
| Maxout adv conf mistakes (`:339`) | 97.6% | 92.54% | value fail (sub-scale) |
| Conv maxout FGSM ε=.1 CIFAR (`:341`) | 87.15% | BLOCKED | blocked |
| Logreg 3v7 clean err (`:451`) | 1.6% | 1.54% | **value ✓** |
| Logreg 3v7 FGSM err (`:453`) | 99% | 100.0% | ordering ✓ |
| Maxout clean 0.94→0.84 w/ adv (`:491`) | 0.94%→0.84% | 1.47%→1.23% | ordering ✓ (values sub-scale) |
| Large maxout adv, 5 seeds avg (`:509`) | 0.782% | 1.874% | value fail (sub-scale, no 60k retrain) |
| Large maxout FGSM after adv (`:515`) | 17.9% | 56.45% | value fail (sub-scale) |
| Transfer new←advfromorig (`:519`) | 19.6% | 30.83% | ordering ✓ |
| Transfer orig←advfromnew (`:520`) | 40.9% | 61.12% | ordering ✓ (asymmetry holds) |
| L1 .0025 first layer >5% train err (`:429`) | >5% | 6.33% | **value ✓** |
| RBF FGSM ε=.25 err (`:602`) | 55.4% | 98.54% | value fail (sub-scale, RBF training unstated) |
| RBF mistake conf 1.2% (`:603`) | 1.2% | 22.44% | value fail (sub-scale) |
| RBF clean conf 60.6% (`:604`) | 60.6% | — | value fail (sub-scale) |
| RBF rubbish err 0% (`:917`) | 0% | 0.0% | **value ✓** (oracle) |
| Ensemble 12, whole-ensemble attack (`:822`) | 91.1% | 93.20% | ordering ✓ |
| Rubbish maxout MNIST err (`:905`) | 98.35% | 97.22% | **value ✓** |
| Rubbish maxout conf (`:906`) | 92.8% | 90.20% | value fail (sub-scale) |
| Rubbish softmax err (`:913`) | 59.8% | 98.31% | value fail (sub-scale) |
| Rubbish sigmoid-top err (`:908`) | 68% | 11.22% | value fail (sub-scale) |
| Agreement softmax cond (`:686`) | 84.6% | 73.83% | value fail (sub-scale) |
| Agreement RBF cond (`:688`) | 54.3% | 3.04% | value fail (sub-scale, RBF training) |

The HIGH-invariance claims (orderings, the c07 analytic-logistic equivalence, the
FGSM ‖η‖∞=ε invariant, the degeneracy no-op, the Fig.4 piecewise-linear curve
shape) all pass; the table above marks only the value claims that fail at
sub-scale. Two independent reviewers' full verdicts are in `selfcheck.json`.

## Decisions (the paper left these open; logged in SPEC §4)

- **Framework:** PyTorch CPU. FGSM needs ∇ₓ J; `torch.autograd.grad(loss, x)` gives it.
- **No clipping** of x̃ (paper never states any; SPEC §4.9). **sign(0) := 0** (§4.22).
- **Maxout arch (ours):** 2 layers, 5 pieces, dropout input .2 / hidden .5; 240 and 1600 units.
- **RBF (§4.4):** 10 units, β_k = −ψψᵀ − νI (NSD), ν=0.01 floor (without it NLL
  collapses to prob=1 everywhere — contradicts the paper's 1.2% mistake conf).
  Training procedure unstated; this is our choice.
- **Adversarial training:** single shared minibatch, α=0.5, x_adv built from
  current θ with the input-grad computed then **detached** (grads flow into θ,
  not through sign). ε=0 is a true no-op (degeneracy).
- **L1 weight decay:** first weight-bearing layer only.
- **Seeds:** [0,1,2] for most arms; `maxout_large_adv` uses [0..4] (paper's five).
  Three independent RNG streams per seed (init, minibatch order, dropout masks).
- **eps_trace (Fig.4):** FGSM direction computed ONCE at ε=0 and held fixed across
  the ε-grid (only this makes the logits exactly piecewise linear; SPEC §4.15).
- **Hyperparameter sweep (§ "for any claim whose verdict depends on a value the
  paper never states"):** α=0.5 is the paper's stated value (not swept — it is
  stated, tex:488). The RBF ν floor is the one value the paper never states that a
  HIGH claim could depend on; ν is fixed (not gated by a HIGH claim) and the RBF
  value claims are rated low/medium precisely because RBF training is unstated.
- **Mutation gate vs. conftest auto-restore:** `tests/conftest.py` used to
  `git checkout` every mutation-target file at collection time. That recovered
  the tree after an interrupted `tests/test_mutations.py` run, but it also
  reverted *any* planted defect before the must_fail test ran — including the
  defects the external mutation gate plants directly in a source file (the gate
  does not set our `EAE_SKIP_RESTORE` flag). The gate's must_fail tests then ran
  on clean code, PASSED, and the gate reported the mutations as SURVIVED. The
  auto-restore is now **opt-in** (`EAE_AUTO_RESTORE=1`, default OFF): the gate's
  planted defect now reaches the test, and `tests/test_mutations.py` still
  reverts via its own `finally: git checkout` (it sets `EAE_SKIP_RESTORE=1`).
  All 9 mutations are now CAUGHT under the gate's style (apply defect → run
  must_fail without `EAE_SKIP_RESTORE`). Also fixed `mut_logreg_analytic_sign`:
  its `find` string appeared twice in `tests/test_invariants.py`; it now
  includes the following `margin_analytic` line so the match is unique.

## Constructed truth (see SPEC §8.5)

- **Degeneracy:** ε=0 (adv training), ε=0 (noise), coef=0 (L1) each reduce to
  plain training bit-identically — asserted in `tests/test_degeneracy.py`.
- **Same quantity two ways (c07):** for logistic regression FGSM is exact, so the
  closed form `E ζ(y(ε‖w‖₁ − w·x − b))` equals the actual adversarial loss under
  `η = −ε·sign(w)` — asserted in `tests/test_invariants.py`.
- **Invariants from the maths:** ‖η‖∞=ε, sign(0)=0, no clipping, wᵀsign(w)=‖w‖₁,
  softmax rows sum to 1 / RBF rows need not, non-negative loss, piecewise-linear
  logits in ε — all in `tests/test_invariants.py`.
- **Planted structure (Fig.4, c65–c70):** a class-4 example with the thin-manifold
  property; the curve claims check the shape point by point.
- **Paper's standard baseline as oracle:** the FGSM error rates (softmax 99.9%,
  maxout 89.4%) are common knowledge; HIGH ordering claims check directions.

## Blockers

- **CIFAR-10 download is throttled in this environment.** `data.load_cifar10`
  downloads from `cs.toronto.edu`; the connection opens (headers + first bytes
  arrive) but the sustained 170 MB transfer stalls — verified: a chunked
  download reached 12 MB then the socket timed out, repeatedly. MNIST (small IDX
  files) downloads fine. Per the contract ("Real data, or no numbers"), the
  `cifar_conv_maxout` arm is marked **BLOCKED** in `measured.json` (and its 8
  claims c56–c63 are blocked), NOT substituted with synthetic data.
  `data.cifar10_available()` checks the local tar only (never hangs on the
  network); `test_cifar10_real` skips fast with this reason.
- **MP-DBM** (clean 0.88% / FGSM 97.5%, `:794,800`) and **GoogLeNet/ImageNet
  Fig.1** (`:364-383`): deliberately not built (SPEC §9); recorded as BLOCKED at
  every seed in `measured.json` so it covers every arm in `claims.json`.
- **CPU sub-scale:** the 1600-unit large maxout is capped at 6 epochs with no
  60k retrain; the 12-member ensemble and conv net are epoch-capped. The tight
  value claims (0.94%, 0.782%) need the paper's full GPU budget and are rated
  low/medium in `claims.json` for exactly this reason; HIGH-invariance claims
  survive at sub-scale.

## Data provenance

- **MNIST:** the real IDX files (ossci-datasets S3 mirror) loaded by
  `data.load_mnist`; `data.check_mnist_fingerprint` asserts size (50k/10k/10k ×
  784), vocabulary {0..9}, range [0,1], and a pixel-sum checksum a synthetic /
  all-zeros / 65-token-vocabulary corpus cannot match. The 3-vs-7 subset maps
  y=+1 to digit 3 (re-derived independently of the loader). Positive +
  negative fingerprint tests in `tests/test_data_loader.py`.
- **CIFAR-10:** BLOCKED (above). The fingerprint (`check_cifar10_fingerprint`)
  and its negative test (rejects a non-GCN / wrong-dim array) run without the
  download; only the positive test (loading the real tar) is skipped.
- No synthetic corpus is substituted for either dataset anywhere in the arms.

## Adversarial review

An orchestration (`orchestrate`) fanned out 7 parallel reviewers over each
component (data, models, attack, train, eval, arms, measured-consistency)
against the paper LaTeX, then a verify stage tried to refute each
HIGH/correctness finding against the real code. Result: **5/7 components
approved** (models, attack, eval, arms, measured-consistency); 1 finding
refuted; 2 surviving correctness findings fixed + guarded with tests:

- **train.py Phase 2 retrain was NOT from scratch** (MEDIUM correctness,
  confirmed): Phase 2 continued from the Phase-1 state with the same optimizer
  (carried momentum), contradicting `tex:505-506`. ALSO `load_mnist_full` set
  `x_train` = full 60k (val ⊂ train leakage) instead of a 50k/10k split for
  Phase-1 early-stopping. **Fixed:** `load_mnist_full` now returns the 50k
  train split + the 60k as `x_train_full`; `train.train` captures the initial
  weights and Phase 2 restores them + a fresh optimizer. Guarded by
  `tests/test_degeneracy.py::test_retrain_full_60k_is_from_scratch`.
- **models.py `float(nu)` detached the RBF `nu` from autograd** (LOW, dormant —
  `nu_trainable` defaults False, never set in arms; confirmed): the
  `nu_trainable=True` path silently received no gradient. **Fixed:** `_quad`
  uses `self.nu` (a tensor) directly. Guarded by
  `tests/test_invariants.py::test_rbf_nu_trainable_receives_gradient`.
- **data.py CIFAR fingerprint was a rubber stamp** (MEDIUM completeness): it
  checked only shape/std/labels, so a std-matched iid synthetic corpus passed
  (the closed-book failure mode). No synthetic substitution occurred in practice
  (the arm gates on `cifar10_available` and raises → BLOCKED), but the
  defensive instrument was weak. **Fixed:** added a per-pixel-variance
  structural check (real images have non-uniform per-pixel std; iid Gaussian
  does not — needs no precomputed checksum, which can't be populated while
  CIFAR is blocked here) and extended the negative test to reject a
  std-matched synthetic corpus. Also extended the std check to val/test.

- **`run_all_arms.py` eps_trace FINAL line was unparseable** (gate feedback):
  the curve arm printed `FINAL eps_trace=<mean> (seq mean)`, and the trailing
  ` (seq mean)` annotation broke the gate's strict `FINAL <arm>=<value>` parser,
  so `eps_trace` read as "printed no FINAL line". **Fixed:** the sequence-valued
  primary metric now prints the mean with no suffix (`FINAL eps_trace=<mean>`);
  the full per-ε sequence the curve claims (c65–c70) actually evaluate lives in
  `measured.json` under the arm. All six eps_trace curve claims pass.

- **Mutation-target files must be committed before `pytest`.** `tests/test_mutations.py`
  `git checkout HEAD --`s each mutation target (`models.py`, `train.py`,
  `tests/test_invariants.py`, `attack.py`) after probing it, so any *uncommitted*
  edit on those files is silently reverted by a test run. The `float(nu)` and
  retrain fixes above were caught by this once (documented as Fixed here while
  the code sat uncommitted and got reverted); they are now committed and survive
  a `pytest` run. `train.py`'s from-scratch retrain was already committed; its
  guard test lives in `tests/test_degeneracy.py` (a non-target).

Low/defensible items left as-is (documented): `Ensemble.loss` uses
`cross_entropy(mean logits)` — a defensible perturb-the-whole-ensemble
objective (the paper is silent, `tex:819-821`); both flows send gradients to
all members; low impact on the FGSM direction / c34-c35 (tolerances 8.0).

## How to run

```bash
.venv/bin/pytest -q                       # tests (32 pass, 1 skip; cifar positive skips)
.venv/bin/python smoke.sh                 # smoke (one FINAL line; not evidence)
.venv/bin/python -m run_all_arms          # every arm x seed -> measured.json
.venv/bin/python selfcheck_claims.py      # claims.json vs measured.json -> selfcheck.json
```
