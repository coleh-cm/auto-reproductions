# VERIFICATION — Explaining and Harnessing Adversarial Examples

This file lists **every check this reproduction ran**, what it found, the
budget it ran at, and — the part that matters — **what remains untested and
why**. None of this establishes that the implementation is *correct*; it
establishes that it is *not wrong in the ways that were checked*. That list is
worth more to a reader than the headline number.

All commands were run on a CPU sandbox (no GPU, no docker). Dates 2026-08-04.

---

## 1. Environment / build

| check | command | budget | found |
|---|---|---|---|
| venv builds from pinned closure | `uv venv --python 3.13 .venv && uv pip install --python .venv -r requirements.txt` | ~minutes | succeeds; `uv pip freeze` matches `requirements.txt` |
| imports resolve | `.venv/bin/python -c "import torch, numpy, matplotlib, pytest"` | seconds | torch 2.7.1+cpu, numpy 2.3.2, matplotlib 3.11.1, pytest 8.4.2 |
| FGSM autograd path | `torch.autograd.grad(loss, x)` → `ε·sign(g)` | seconds | works for every model class |
| docker build | `docker build -t eae-repro .` | — | **NOT RUN** — docker unavailable in this sandbox |

**Untested:** the `Dockerfile` is present and well-formed but was never built
here, so the image itself (layer install, `CMD pytest -q`) is unverified. A
reader with docker should run `docker build -t eae-repro . && docker run --rm
eae-repro pytest -q`.

## 2. Data provenance

| check | command | found |
|---|---|---|
| MNIST fingerprint | `data.check_mnist_fingerprint` (inside `load_mnist`) | size 50k/10k/10k×784, labels {0..9}, range [0,1] f32, pixel-sum checksum (5133683.0 / 1012586.25 / 1038914.5) — matches the canonical corpus |
| MNIST 3-vs-7 fingerprint | `data.check_mnist_3v7_fingerprint` | only {-1,+1}, +1 == digit 3 by count |
| CIFAR-10 fingerprint | `data.check_cifar10_fingerprint` | code present; **not exercised on real data** — the download stalled |

**Untested:** CIFAR-10. The download from `cs.toronto.edu` stalled at ~11 MB of
~170 MB (throttled to a few MB/min; no reachable S3 pickle mirror). The
`cifar_conv_maxout` arm is BLOCKED and **no synthetic CIFAR substitute was
used**. Fix: obtain the real CIFAR-10 pickle tarball, place it in `./data/`,
re-run `EAE_ONLY=cifar_conv_maxout .venv/bin/python run_all_arms.py`.

## 3. Correctness: test suite

| check | command | budget | found |
|---|---|---|---|
| full suite | `.venv/bin/python -m pytest tests/ -q` | ~13 s | **29 passed, 1 skipped** (the skip is the CIFAR-positive test, download unavailable) |

What the suite covers:

- **Degeneracy** (`tests/test_degeneracy.py`): adversarial eps=0, noise eps=0,
  L1 coef=0 are bit-identical (`torch.equal`) to the baseline — the no-op
  settings truly reproduce the baseline. `test_degeneracy_detects_nonzero_eps`
  asserts eps>0 is bit-different (the check is sensitive to a leak).
- **Invariants** (`tests/test_invariants.py`): FGSM `‖η‖∞ == ε`; no clipping of
  `x̃`; `sign(0)=0`; softmax rows sum to 1; logreg `sign(grad) == -sign(w)`;
  `w·sign(w) == ‖w‖₁`; FGSM loss == analytic closed form (c07, max absdiff
  9.5e-7); non-negative loss; eps-trace piecewise-linear in ε; empty-input
  raises; adversarial training reduces adv_err.
- **Mutations** (`tests/test_mutations.py`): 9 deliberate defects, each
  planted, run, and `must_fail`-caught, then reverted (with `.pyc` purging and
  conftest-time git restore so an interrupted run can never ship a defect).
- **Data-loader instruments** (`tests/test_data_loader.py`): MNIST real passes;
  a synthetic corpus is rejected; CIFAR positive is skipped (download), CIFAR
  negative (rejects synthetic) runs.
- **Numbers-gate instruments** (`tests/test_numbers_gate.py`): the grader
  returns "pass" on a known-correct input, "fail" on a known-wrong input,
  "blocked" on an absent arm (never a silent pass), and **raises** on a
  malformed `measured.json` (never a silent negative verdict).

**Untested by the suite:** end-to-end behavior on CIFAR-10; the MP-DBM and
GoogLeNet/ImageNet arms (not built — see SPEC §9); the full-budget value
claims (the suite checks plumbing, not the paper's tight numbers).

## 4. Fast path

| check | command | budget | found |
|---|---|---|---|
| smoke | `bash smoke.sh` | ~1.5 s | trains softmax on 2k MNIST examples × 5 epochs, FGSM ε=.25, prints `FINAL softmax_reg=99.6000` |

The smoke path is proof the code runs; **it is not evidence about the paper**
and is never reported as a result.

## 5. Arms run (the numbers)

| check | command | budget | found |
|---|---|---|---|
| all arms | `.venv/bin/python run_all_arms.py` (resume) | 15 MNIST arms × 3 seeds (5 for `maxout_large_adv`); 1600-unit maxout capped at **6 epochs** (no 60k retrain); conv 8 epochs | 15 arms produce real measured numbers in `measured.json`; `cifar_conv_maxout` BLOCKED |
| numbers gate | `.venv/bin/python numbers_gate.py` | seconds | **pass=40 fail=19 blocked=11; HIGH: 18 pass / 0 fail / 1 blocked; gate=FAIL** |

The gate is FAIL *only* because of the single HIGH block (c63, CIFAR-10). The
19 fails are low/medium *value* claims that need the paper's full GPU budget;
their HIGH-invariance *ordering* counterparts pass. See REPRODUCTION.md for
the full measured-vs-paper table.

**Horizon deviation (honest, necessary, not sufficient):** the 1600-unit
maxout trained 6 epochs vs the paper's full budget + 60k retrain. The paper's
headline clean-error regularization claim (0.94→0.84→0.782) is **not
reproduced** — at 6 epochs the naive (1.80 ± 0.28) and adv (1.87 ± 0.34) arms
are within noise. A number produced at a horizon too short to separate the
arms is not evidence about the paper's claim, and is not presented as one.

**Untested at the numbers level:** every CIFAR-10 claim (c56-c63); the
full-budget tight value claims (clean 0.94/0.84/0.782, adv 17.9, RBF 55.4/1.2)
— these need the paper's full GPU run; MP-DBM (§9, error 0.88% / adv 97.5%);
the GoogLeNet/ImageNet Fig. 1 demo; the ensemble-of-12 *value* (only the
ordering vs single-member is checked at sub-scale).

## 6. Figures

| check | command | found |
|---|---|---|
| Figure 4 reproduction | `.venv/bin/python regenerate_figures.py` → `figures/eps_curve_reproduced.png` | eps-sweep logit curve from the `eps_trace` arm (seed 0, example 33); paired with `paper/source/eps_curve.pdf` for visual comparison only |
| Figure 4 gate | c65-c68 (HIGH, curve claims) | **pass** — the thin-manifold shape (margin positive near ε=0, large negative far away, piecewise-linear) is reproduced |

**Untested:** Figures 1, 2, 3, 5 are not regenerated (Fig. 1 needs GoogLeNet/
ImageNet; Figs. 2/3 are qualitative weight/perturbation visualizations; Fig. 5
needs CIFAR-10). The Figure-4 pair image is **not evidence** — the gate
verdicts on c65-c68 are the evidence.

## 7. Review

No tracked adversarial-review rounds were run in this reproduction (no
`.review_rounds` journal). Verification instead took the form of: mutation
testing (9 defects caught), instrument positive/negative tests, degeneracy
and invariant tests, and the numbers-gate self-tests. This is a
**correctness-level** verification, not a formal review pass.

---

## What this reproduction does NOT establish

- It does **not** establish that the implementation is correct — only that it
  is not wrong in the ways the tests, invariants, mutations, instruments, and
  gate check.
- It does **not** reproduce the paper's headline clean-error regularization
  claim (within noise at the 6-epoch sub-scale horizon).
- It does **not** reproduce any CIFAR-10 result (arm BLOCKED on data
  download).
- It does **not** verify the Docker image builds (docker unavailable).
- The RBF *value* claims (c29/c30) do not match (98.5% / 22.4% conf vs paper
  55.4% / 1.2%); only the RBF *ordering* claims pass. The β/μ parametrization
  the paper never specifies is the likely source.

## Rung reached

**correctness** — environment builds, comprehension (SPEC) done, implementation
runs, and the correctness gate (tests/invariants/mutations/instruments) is
green. The numbers gate was run (18/19 HIGH pass) but reports **FAIL** on the
single HIGH CIFAR block (c63), so the **numbers** rung is not reached. The
block is an environment/dataset one (CIFAR-10 download stalled), not a method
failure — but it is not an all-BLOCKED arms result either, so it is reported
as the gate failing, not as a clean environment stop.
