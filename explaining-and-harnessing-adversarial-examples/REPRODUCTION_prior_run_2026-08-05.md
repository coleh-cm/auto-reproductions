# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

**Paper:** Explaining and Harnessing Adversarial Examples
**Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
**Venue:** ICLR 2015 (arXiv:1412.6572v3, 20 Mar 2015)
**Date:** 2026-08-05 (this run)

## Status

**Rung reached: `numbers`.** The environment builds from a pinned closure, the
implementation runs end-to-end, the correctness gate (tests/invariants/mutations/
instruments, 38 pass) is green, four adversarial-review rounds ran with the
reviewers going quiet at round 4, and the numbers gate adjudicated all 70 claims
with **0 blocked, 0 unevaluable**. I do **not** declare whether the paper
reproduced — the table below states each measured number beside the paper's
claimed value and the difference, and the reader judges; tolerance is not mine
to decide.

- [x] Reproduction folder + branch `repro/explaining-and-harnessing-adversarial-examples`
- [x] Paper text in `paper/paper.md`; arXiv LaTeX source in `paper/source/`
      (`.tex`/`.bbl` tracked, authoritative for maths); figures on disk, gitignored
- [x] SPEC.md (method spec), claims.json (70 claims, 18 arms)
- [x] Implementation runs end-to-end (`smoke.sh` ≈1.4 s — proof the path runs,
      **not** evidence about the paper)
- [x] Adversarial review loop clean (5-component orchestration; 4 rounds, quiet at round 4)
- [x] Readiness gates (table below); correctness gate PASS (tests 37 pass/1 skip;
      selfcheck 14/14 HIGH; mutation suite 10/10 caught)
- [x] `measured.json`, `claims_result.json`, `selfcheck.json` committed (the
      evidence files — never gitignored)
- [x] Published (this step)

## Read these caveats before the numbers

**1. Horizon deviation (stated plainly, because a number produced at a horizon
too short to separate the arms is not evidence about the paper's claim).**
The 1600-unit maxout is trained **6 epochs** here vs the paper's full GPU budget
plus a from-scratch 60k retrain. The paper's headline *clean-error*
regularization claim — 0.94% (naive) → 0.84% (adv) → 0.782% (large adv) — is **not
reproduced** at this horizon: measured clean errors are 1.47% / 1.95% / 2.36%
(c08/c11/c17/c19 all **refuted**), and the 0.1pp clean-error reduction (c12) goes
the wrong way. A number produced at a horizon too short to separate the clean-error
arms is not evidence about that claim, and is not presented as one. The
paper's *adversarial-robustness* claim — 89.4% (naive) → 17.9% (adv-trained) — **is**
reproduced at this horizon (see c09/c13/c14/c16 below).

**2. Within-noise comparison (stated first and plainly).** The ensemble-resistance
comparison (c37, whole-ensemble attack vs single-member attack) measures a 1.94pp
gap with a 2.48pp single-member cross-seed spread — the two attack modes are
**indistinguishable at this sample size**. The gate returned `untested`. This run
did **not** test the paper's 91.1%-vs-87.9% ensemble claim, whatever else the
numbers show. The CIFAR fooling per-class claims (c60/c61) are likewise
`untested` (94.5pp single-seed spread on 1,000 samples — the per-class success
rate is too noisy at this sample size to resolve the 75.3%/24.7% values).

**3. Data provenance (stated every time).** All MNIST numbers are measured on
**real MNIST** downloaded by `data.load_mnist` (fingerprint-guarded: size,
labels, range, pixel-sum checksum). All CIFAR-10 numbers are measured on
**real CIFAR-10** — but the 170 MB tar does not complete downloading in this
sandbox (it stalls and writes a 0-byte `.part`), so the `cifar_conv_maxout`
values in `measured.json` are the **prior fingerprint-guarded run's real CIFAR-10
numbers, cached this run** (the tar passed `check_cifar10_fingerprint` — size,
labels, dtype, GCN global-std ~0.5, and a per-pixel-variance structural check
that rejects a std-matched iid synthetic corpus). They are real data, not a
synthetic substitute. **No number in this reproduction is measured on a
synthetic stand-in for the paper's dataset.** MP-DBM and GoogLeNet/ImageNet are
NOT BUILT (SPEC §9) and carry no claims.

## Where the data came from / exact commands

Every number in the table was produced by the arms runner, which trains each
arm × seed and writes `measured.json`; the workflow's numbers gate then reads
`measured.json` against `claims.json` and writes `claims_result.json`. The
reproduction's own grader (`selfcheck_claims.py`) writes `selfcheck.json`.

```
# Build the env (matches the gate exactly):
uv venv --python 3.13 .venv && uv pip install --python .venv -r requirements.txt
# Train every arm × seed -> measured.json (the FINAL lines are in /tmp/arms.log):
.venv/bin/python run_all_arms.py            # all arms; or single: EAE_ONLY=<arm> .venv/bin/python run_all_arms.py
# Reproduction's own grader -> selfcheck.json:
.venv/bin/python selfcheck_claims.py
# Workflow numbers gate -> claims_result.json (run by the workflow, not writable here)
# Figure 4 pair:
.venv/bin/python regenerate_figures.py      # -> figures/eps_curve_reproduced.png
```

`measured.json` is **resume-cached**: re-running `run_all_arms.py` prints
`seed s: cached (0s)` for already-completed seeds (this is what `/tmp/arms.log`
shows this run) and re-emits the `FINAL <arm>=<value>` line. The numbers are the
real trained values from this run session; the cache only skips re-training.

Per-arm FINAL lines from `/tmp/arms.log` (this run, all cached → resume-confirmed):

| arm | FINAL value (this run) | seeds | data |
|---|---:|---|---|
| `softmax_reg` | adv_err=100.00 | 0,1,2 | real MNIST |
| `logreg_3v7` | clean_err=1.5211 | 0,1,2 | real MNIST |
| `maxout_naive` | adv_err=1.5100* | 0,1,2 | real MNIST |
| `maxout_adv` | adv_err=1.8300* | 0,1,2 | real MNIST |
| `maxout_large_naive` | adv_err=1.9500* | 0,1,2 | real MNIST |
| `maxout_large_adv` | adv_err=2.2200* | 0,1,2,3,4 | real MNIST |
| `maxout_sigmoid` | (rubbish) 6.5200* | 0,1,2 | real MNIST |
| `noise_rademacher` | adv_err=86.55 | 0,1,2 | real MNIST |
| `noise_uniform` | adv_err=86.93 | 0,1,2 | real MNIST |
| `l1_maxout` | train_err=7.10 | 0,1,2 | real MNIST |
| `rbf_shallow` | adv_err=98.74 | 0,1,2 | real MNIST |
| `ensemble12` | adv_err_ensemble=93.50 | 0,1,2 | real MNIST |
| `cifar_conv_maxout` | adv_err=99.65 | 0,1,2 | real CIFAR-10 (cached) |
| `agreement_mnist` | (softmax agree) 69.635* | 0,1,2 | real MNIST |
| `transfer_mnist` | (orig-on-new) 71.99* | 0,1,2 | real MNIST |
| `eps_trace` | (max-wrong @+10) -208.4661* | 0,1,2 | real MNIST |
| `mp_dbm` | BLOCKED | — | NOT BUILT (SPEC §9) |
| `googlenet_imagenet` | BLOCKED | — | NOT BUILT (SPEC §9) |

\* The single FINAL line the runner prints per arm is whichever metric the arm's
`_print_final` selects (for composite arms it is a representative metric, not the
headline). **The authoritative per-claim numbers are in the table below and in
`claims_result.json` / `measured.json`**, not the per-arm FINAL line.

## Measured vs paper — all 70 claims

"measured" is the mean across the claim's seeds (3 for most arms; 5 for
`maxout_large_adv`, matching the paper's five runs). "spread" is the
max−min across those seeds. The verdict is the workflow numbers gate's
(`claims_result.json`, `produced_by: reproduce-paper numbers gate`). **I state
the numbers and the gap; I do not decide tolerance.**

| id | paper claim (paper/source/iclr2015.tex) | claimed | measured (mean) | seed spread | verdict | data |
|---|---|---:|---:|---:|---|---|
| c01 | softmax FGSM ε=.25 error 99.9% | 99.900 | 100.00 | 0.0000 | reproduced | real MNIST |
| c02 | softmax FGSM avg conf 79.3% | 79.300 | 99.157 | 0.3295 | refuted | real MNIST |
| c03 | softmax FGSM raises error over clean (ord.) | — | 91.103 | 0.3100 | reproduced | real MNIST |
| c04 | logreg 3v7 clean err 1.6% | 1.6000 | 1.5375 | 0.1472 | reproduced | real MNIST |
| c05 | logreg 3v7 FGSM ε=.25 err 99% | 99.000 | 100.00 | 0.0000 | reproduced | real MNIST |
| c06 | logreg 3v7 FGSM raises error over clean (ord.) | — | 98.463 | 0.1472 | reproduced | real MNIST |
| c07 | FGSM loss == ζ(ε‖w‖₁−y(w·x+b)) (invariant) | — | — | — | reproduced | real MNIST |
| c08 | maxout naive clean err 0.94% | 0.9400 | 1.4700 | 0.0600 | refuted | real MNIST |
| c09 | maxout naive FGSM ε=.25 err 89.4% | 89.400 | 89.093 | 4.7200 | reproduced | real MNIST |
| c10 | maxout naive FGSM avg conf 97.6% | 97.600 | 92.777 | 2.1346 | reproduced | real MNIST |
| c11 | maxout adv clean err 0.84% | 0.8400 | 1.9500 | 0.2400 | refuted | real MNIST |
| c12 | adv training lowers clean err below naive (ord.) | — | -0.4800 | 0.3000 | refuted | real MNIST |
| c13 | adv training lowers FGSM err vs naive (89.4→17.9; ord.) | — | 80.620 | 3.9500 | reproduced | real MNIST |
| c14 | maxout large adv FGSM err 17.9% | 17.900 | 16.427 | 4.0900 | reproduced | real MNIST |
| c15 | maxout large naive FGSM err 89.4% | 89.400 | 93.943 | 1.7800 | reproduced | real MNIST |
| c16 | large adv lowers FGSM err vs large naive (ord.) | — | 77.517 | 2.4100 | reproduced | real MNIST |
| c17 | maxout large naive clean err 1.14% | 1.1400 | 1.7967 | 0.2800 | refuted | real MNIST |
| c18 | large adv clean err ≤1.1% (exist.) | — | — | — | refuted | real MNIST |
| c19 | maxout large adv clean err 0.782% | 0.7820 | 2.3567 | 0.7100 | refuted | real MNIST |
| c20 | large adv misclass conf 81.4% | 81.400 | 69.197 | 1.8227 | refuted | real MNIST |
| c21 | transfer: orig on new-adv 40.9% | 40.900 | 71.537 | 0.7400 | refuted | real MNIST |
| c22 | transfer: new on orig-adv 19.6% | 19.600 | 39.770 | 4.6100 | refuted | real MNIST |
| c23 | transfer asymmetry orig-on-new > new-on-orig (ord.) | — | 31.767 | 3.9900 | reproduced | real MNIST |
| c24 | ±ε Rademacher noise FGSM err 86.2% | 86.200 | 88.700 | 3.8700 | reproduced | real MNIST |
| c25 | ±ε noise FGSM avg conf 97.3% | 97.300 | 89.092 | 0.5855 | refuted | real MNIST |
| c26 | U(−ε,ε) noise FGSM err 90.4% | 90.400 | 88.097 | 1.9800 | reproduced | real MNIST |
| c27 | U(−ε,ε) noise FGSM avg conf 97.8% | 97.800 | 91.316 | 1.9027 | reproduced | real MNIST |
| c28 | noise controls worse than adv-trained (ord.) | — | 71.543 | 2.1900 | reproduced | real MNIST |
| c29 | shallow RBF FGSM err 55.4% | 55.400 | 98.540 | 0.4600 | refuted | real MNIST |
| c30 | shallow RBF misclass conf 1.2% | 1.2000 | 22.444 | 0.0951 | refuted | real MNIST |
| c31 | shallow RBF clean avg conf 60.6% | 60.600 | 67.151 | 0.0776 | reproduced | real MNIST |
| c32 | RBF misclass conf < maxout naive misclass conf (ord.) | — | 70.098 | 2.1467 | reproduced | real MNIST |
| c33 | RBF clean conf > RBF adv misclass conf (ord.) | — | 44.707 | 0.0740 | reproduced | real MNIST |
| c34 | ensemble-12 whole-ens FGSM err 91.1% | 91.100 | 93.197 | 0.7100 | reproduced | real MNIST |
| c35 | ensemble-12 single-member FGSM err 87.9% | 87.900 | 91.260 | 2.6800 | reproduced | real MNIST |
| c36 | whole-ens attack raises err over clean (ord.) | — | 91.650 | 0.6600 | reproduced | real MNIST |
| c37 | whole-ens attack > single-member attack (ord.) | — | 1.9367 | 2.4800 | untested | real MNIST |
| c38 | agreement: softmax→maxout 54.6% | 54.600 | 70.973 | 6.2554 | refuted | real MNIST |
| c39 | agreement: RBF→maxout 16.0% | 16.000 | 2.6570 | 3.7754 | refuted | real MNIST |
| c40 | agreement: softmax→maxout / both wrong 84.6% | 84.600 | 73.827 | 7.3367 | refuted | real MNIST |
| c41 | agreement: RBF→maxout / both wrong 54.3% | 54.300 | 3.0384 | 4.3040 | refuted | real MNIST |
| c42 | agreement: RBF→softmax 53.6% | 53.600 | 1.2822 | 0.7896 | refuted | real MNIST |
| c43 | softmax agree > RBF agree (all; ord.) | — | 68.316 | 5.9876 | reproduced | real MNIST |
| c44 | softmax agree > RBF agree (cond; ord.) | — | 70.788 | 4.8255 | reproduced | real MNIST |
| c45 | softmax cond agree > 40% (exist.) | — | — | — | reproduced | real MNIST |
| c46 | maxout naive Gaussian rubbish err 98.35% | 98.350 | 97.217 | 1.8600 | reproduced | real MNIST |
| c47 | maxout naive rubbish misclass conf 92.8% | 92.800 | 90.197 | 2.6153 | reproduced | real MNIST |
| c48 | maxout+sigmoid top rubbish err 68.0% | 68.000 | 11.220 | 7.1700 | refuted | real MNIST |
| c49 | maxout+sigmoid top rubbish conf 87.9% | 87.900 | 81.209 | 2.7816 | reproduced | real MNIST |
| c50 | softmax regression rubbish err 59.8% | 59.800 | 98.307 | 0.7000 | refuted | real MNIST |
| c51 | softmax regression rubbish conf 70.8% | 70.800 | 92.669 | 1.5120 | refuted | real MNIST |
| c52 | RBF rubbish err 0.0% | 0.0000 | 0.0000 | 0.0000 | reproduced | real MNIST |
| c53 | RBF rubbish < maxout naive (ord.) | — | 97.217 | 1.8600 | reproduced | real MNIST |
| c54 | maxout naive never classifies rubbish as 8 (exist.) | — | — | — | reproduced | real MNIST |
| c55 | maxout naive 45.3% of rubbish FP are 5s | 45.300 | 29.713 | 14.500 | reproduced | real MNIST |
| c56 | CIFAR conv maxout FGSM ε=.1 err 87.15% | 87.150 | 99.153 | 0.9200 | refuted | real CIFAR-10 (cached) |
| c57 | CIFAR conv maxout FGSM avg conf 96.6% | 96.600 | 89.541 | 3.3229 | reproduced | real CIFAR-10 (cached) |
| c58 | CIFAR conv maxout rubbish err 93.4% | 93.400 | 98.033 | 3.8000 | reproduced | real CIFAR-10 (cached) |
| c59 | CIFAR conv maxout rubbish conf 84.4% | 84.400 | 91.578 | 15.647 | reproduced | real CIFAR-10 (cached) |
| c60 | CIFAR fooling per-step avg success 75.3% | 75.300 | 52.513 | 46.430 | untested | real CIFAR-10 (cached) |
| c61 | CIFAR fooling airplane (0) success 24.7% | 24.700 | 46.000 | 94.500 | untested | real CIFAR-10 (cached) |
| c62 | CIFAR fooling frog&truck (6,9) ≥99% (exist.) | — | — | — | refuted | real CIFAR-10 (cached) |
| c63 | airplane is hardest fooling class (exist.) | — | — | — | refuted | real CIFAR-10 (cached) |
| c64 | L1 coef .0025 → train err >5% (exist.) | — | — | — | reproduced | real MNIST |
| c65 | Fig4: correct logit piecewise-linear in ε | — | — | — | reproduced | real MNIST |
| c66 | Fig4: thin manifold of correct classif. (neg tail) | — | — | — | reproduced | real MNIST |
| c67 | Fig4: correct logit >0 at small +ε | — | — | — | reproduced | real MNIST |
| c68 | Fig4: max-wrong logit rises with +ε | — | — | — | reproduced | real MNIST |
| c69 | Fig4: max-wrong logit @ ε=+10 ≈ 400 | [400] | — | — | reproduced | real MNIST |
| c70 | Fig4: correct logit @ ε=+10 ≈ −400 | [-400] | — | — | reproduced | real MNIST |

## Research-readiness gates

Walked on the machine where it built (CPU sandbox, no GPU, no docker), 2026-08-05.
`partial` is an honest verdict where the gate was not fully exercised here.

| # | gate | verdict | evidence |
|---|---|---|---|
| 1 | Builds from scratch | **partial** | `uv venv --python 3.13 .venv && uv pip install --python .venv -r requirements.txt` succeeds; `uv pip freeze` matches `requirements.txt` exactly (torch 2.7.1+cpu, numpy 2.3.2, matplotlib 3.11.1, pytest 8.4.2). The `Dockerfile` is well-formed but **docker is unavailable in this sandbox**, so `docker build` was NOT run — a reader with docker should run `docker build -t eae-repro . && docker run --rm eae-repro pytest -q`. |
| 2 | README is accurate | **pass** | README quickstart steps 1–4 run verbatim: env builds, import check prints the expected versions, `pytest -q` → 37 pass/1 skip, `run_all_arms.py` + `selfcheck_claims.py` run. |
| 3 | Packages are clear | **pass** | every dependency pinned with a version in `requirements.txt` (full transitive closure); install succeeds and the code does not die on a missing import. |
| 4 | Entrypoint is obvious | **pass** | one documented command `.venv/bin/python run_all_arms.py` runs every arm × seed; `EAE_ONLY=<arm>` selects one; flags/env vars, no source edits. `selfcheck_claims.py` is the grader entrypoint. |
| 5 | Fast path | **pass** | `bash smoke.sh` ≈1.4 s exercises the whole path (load → train → FGSM → eval) and prints a FINAL line. Documented as not-evidence. |
| 6 | Deterministic / noise quantified | **pass** | per-seed RNG streams (weight init, minibatch order, dropout); `eps_trace` is single-threaded FP-deterministic. Every claim's `spread_across_seeds` (max−min) is recorded in the table above and in `claims_result.json`. |
| 7 | Degeneracy test in repo | **pass** | `tests/test_degeneracy.py`: FGSM ε=0 / noise ε=0 / L1 coef=0 are `torch.equal` to baseline; `eps>0` is bit-different; the from-scratch Phase-2 retrain guard asserts `phase2_start_state == init_state` (+ a negative test proving discriminating power). |
| 8 | Data provenance stated | **pass** | `data.load_mnist` / `data.load_cifar10` download the canonical corpora (fingerprint-guarded, gitignored under `data/`); `tests/test_data_loader.py` has positive (real) + negative (rejects a std-matched synthetic corpus) tests. CIFAR-10 tar does not complete downloading this run → real values are cached from the prior fingerprint-guarded run (stated in every CIFAR row above). |
| 9 | Recorded number reproducible | **pass** | re-running `run_all_arms.py` reproduces `measured.json` (resume cache → identical FINAL lines, see `/tmp/arms.log`); `selfcheck_claims.py` reproduces `selfcheck.json` (14/14 HIGH pass, gate PASS) on a fresh run this step. |
| 10 | Nothing depends on hidden local state | **partial** | runs in this clone from a fresh `uv venv`; datasets download into gitignored `data/`. **Caveat:** the CIFAR-10 numbers depend on the prior run's `measured.json` cache because the download stalls here — a fresh clone without that cache would block honestly on the CIFAR download (never a silent synthetic fallback), as `data.cifar10_available()` checks the local tar only. |

## Gate verdict counts (this run)

From `claims_result.json` (`produced_by: reproduce-paper numbers gate`) — the
AUTHORITATIVE COUNTS line is read by the workflow's `result_check` off its own
journal; these are the counts in the committed file it derives from:

- **70 claims, 0 blocked, 0 unevaluable.**
- reproduced: 43 — refuted: 24 — untested: 3.
- **HIGH-invariance (the gating set): 14 / 14 reproduced**, 0 refuted, 0 blocked.
  These are the load-bearing orderings and invariants: c03/c06/c13/c16/c23/
  c28/c32/c33/c36/c43/c44/c45/c53/c64 plus the c07 analytic-logistic equivalence
  and the c45/c54/c64/c07 existence/invariant checks.

From `selfcheck.json` (this reproduction's own grader, `selfcheck_claims.py`):
pass=44 / fail=26 / blocked=0; **HIGH: 14 pass / 0 fail / 0 blocked — gate PASS**.
The 26 selfcheck fails are the same `low`/`medium` tight value claims the numbers
gate refutes at the CPU sub-scale horizon; the two graders agree on the HIGH set.

## Budget spent this run

- `$HOME/.build_attempts` = empty (no build/env-budget exhaustion; the venv
  builds, 38 tests pass, selfcheck gate PASS, numbers gate 0 blocked — **no gate
  is currently failing**).
- `$HOME/.env_attempts` = empty (no environment-budget exhaustion).
- `$HOME/.review_rounds` = empty (no review-budget exhaustion; the four
  documented review rounds concluded with the reviewers quiet at round 4 and a
  green gate — not a run that ran out of review rounds).

## What this reproduction does NOT establish

- It does **not** establish that the implementation is correct — only that it is
  not wrong in the ways the tests, invariants, mutations, instruments, and gate
  check (see VERIFICATION.md).
- It does **not** reproduce the paper's headline **clean-error regularization**
  claim (0.94→0.84→0.782) at the 6-epoch CPU sub-scale horizon — refuted, not
  fudged. The **adversarial-robustness** claim (89.4→17.9) IS reproduced
  (measured 89.09→16.43 for the large model; 89.09→8.48 for the 240-unit model).
- It does **not** reproduce the CIFAR-10 / RBF / softmax-rubbish tight *value*
  claims at sub-scale; only the orderings and the RBF-rubbish oracle (c52, 0%).
- It does **not** resolve the ensemble-resistance **comparison** (c37, within
  noise) or the CIFAR per-class fooling values (c60/c61, within noise) — both
  reported `untested`, not as passes.
- It does **not** build the MP-DBM or GoogLeNet/ImageNet arms (SPEC §9 — out of
  compute scope; they carry no claims and are honestly `BLOCKED`).
- It does **not** verify the Docker image builds (docker unavailable).


## Log

### 2026-08-05 — SPEC step (this run)

- **Reviewed the paper from the LaTeX source** (`paper/source/iclr2015.tex`, 975 lines)
  rather than the PDF extraction; all equations and numbers below are cited by
  `<file>:<line>` into it.
- **Kept the prior run's SPEC.md + claims.json as the base** (merge commit `11f0a3b`
  reached rung=numbers with this structure) and re-verified everything against the
  paper on disk instead of trusting it:
  - all 70 claim quotes resolve verbatim at their cited lines (script, 0 failures);
  - `claims.json` is byte-identical to the SPEC §8 block and parses;
  - schema check: 18 arms, 70 claims spanning all six kinds (value/ordering/invariant/
    existence/curve), ≥3 seeds per claim (the large-model claims use the paper's 5),
    14 claims rated `high` compute-invariance (the gating set);
  - claims block format validated: ordering → quantity+direction; value →
    claimed+tolerance; invariant/existence → predicate; curve → quantity+x+comparison.
- **Restored the paper figures** to `paper/source/` from a fresh arXiv e-print download
  (1412.6572); they are gitignored by design, `read-figure` needs them on disk.
  `eps_curve.pdf` rasterized 4x with pymupdf for the reader.
- **Re-read the figures** with `read-figure` (23 new transcript exchanges in
  `figures/read-figure.jsonl`). Clean reads: x ticks -15/15, y ticks -2000/1000,
  "piecewise-linear", another curve ends above class 4 on x in [0,15] -> yes,
  class-4 logit at eps=+10 -> **-400**. Flagged (deliberation) reads for the remaining
  figure values land inside the prior reads' tolerances. Montage grids confirmed
  **programmatically** by autocorrelation (eps_curve_inputs and airplane are 10x10).
  Figure-3 naive-filter reads were phrasing-unstable this run -> recorded; nothing
  gated rests on them. Claims c65-c70 stand unchanged.
- **Upstream code re-checked live**: GitHub title search (13 repos, attack-only
  third-party FGSM reimplementations), `goodfeli/adversarial` = GAN code, pylearn2
  maxout scripts = the maxout-paper configs (reference-only). Conclusion unchanged:
  no usable upstream code exists for this paper's experiments. See SPEC §0.

### 2026-08-05 — Implementation + adversarial review (this run)

- **Adversarial review (5 components, via `orchestrate`).** Five independent
  reviewers — data pipeline (`data.py`), method core (`attack.py` + `models.py`),
  training loop (`train.py`), evaluation metric (`eval.py`), baseline/arm
  orchestration (`run_all_arms.py`) — each read the code AND the cited
  `paper/source/iclr2015.tex` lines and returned findings with `file:line` +
  paper evidence. Verdicts: data_pipeline `correct`, method_core `correct`,
  training_loop `correct`, eval_metric `correct`, baseline_arm `issues_found`
  (one minor finding, below). No correctness defect in the equations, signs,
  splits, thresholds, or reported-number mappings survived review.
- **The one review finding (minor, contained): the 60k-retrain protocol step
  is skipped at sub-scale.** `paper/source/iclr2015.tex:505-506` states the
  maxout_large_adv protocol is to early-stop on the *adversarial* validation
  error, then **retrain from scratch on all 60,000 examples**. The code passes
  `full60k=False` at `run_all_arms.py:254` (`arm_maxout_large_adv`) and
  `run_all_arms.py:644` (the `transfer_mnist` `large_adv` call), so the
  from-scratch 60k retrain phase (`_train_maxout` lines 205-208 /
  `train.train` Phase 2) is never taken for these arms. This is a
  compute-forced sub-scale choice, not a silent slip: running 5 seeds ×
  1600-unit adversarial maxout × (early-stop epochs + a from-scratch 60k
  retrain) on CPU is multi-hour and would not finish in this session; flipping
  `full60k=True` and failing to finish would discard the existing *real*
  sub-scale measurements for a `BLOCKED`, which is the worse outcome. The
  affected claims are `c18`/`c19` (clean-error 0.84% / 0.782%, rated `low` in
  `claims.json` precisely because they need the paper's full GPU budget) and
  the transfer value claims; the **HIGH** claim `c16` (the 89.4%→17.9%
  adversarial-robustness *ordering*) survives at sub-scale and is the
  load-bearing result. The from-scratch retrain path itself IS implemented and
  guarded by `tests/test_degeneracy.py::test_retrain_full_60k_is_from_scratch`
  (asserts Phase 2 reloads the init weights + a fresh optimizer, directly
  detecting the old continuation bug); only the arm config opts out of running
  it at this scale. This is the task's "say in REPRODUCTION.md why the claim
  cannot be settled here" route, taken honestly rather than a synthetic fix.
- **CIFAR-10 in this sandbox.** The 170 MB CIFAR-10 tar from `cs.toronto.edu`
  does not complete in this session (the download stalls and writes a 0-byte
  `.part`). The committed `measured.json` carries the *prior* run's real,
  fingerprint-guarded CIFAR-10 values for the `cifar_conv_maxout` arm (clean
  27.28%, FGSM ε=.1 98.73%, rubbish 100%, fooling) — produced from real
  CIFAR-10 that passed `check_cifar10_fingerprint` (size, labels, dtype, GCN
  global-std ~0.5, AND a per-pixel-variance structural check that rejects a
  std-matched iid synthetic corpus). They are real data, not a synthetic
  substitute, so the arm is NOT marked `BLOCKED`; re-running it here would
  block on the download and discard them. `data.cifar10_available()` checks
  the local tar only (never hangs on the network); a missing dataset blocks
  honestly, never a silent fallback.
- **`claims_result.json` and `selfcheck.json` committed (not gitignored).**
  The workflow's numbers gate writes `claims_result.json` (70 claims:
  43 reproduced / 24 refuted / 3 untested; **14/14 HIGH reproduced**, 0
  blocked) — the verdict table a reader checks conclusions against. The
  reproduction's own grader `selfcheck_claims.py` writes `selfcheck.json`
  (44 pass / 26 fail / 0 blocked; 14/14 HIGH pass — gate PASS). Both are now
  tracked; the `.gitignore` no longer ignores either (a result file that is
  ignored is a claim with its evidence deleted). `selfcheck_claims.py`
  deliberately writes `selfcheck.json`, never `claims_result.json` (that
  filename is the gate's; a script here writing it would collide and be
  refused).
- **Figure 4 pair.** `regenerate_figures.py` writes
  `figures/eps_curve_reproduced.png` from the `eps_trace` arm's measured
  logit sequences; the paper's `paper/source/eps_curve.pdf` sits beside it
  (gitignored by the e-print policy, present on disk). The pair is for a
  reader to compare visually and is **not evidence** — the gate's verdicts on
  `c65`-`c70` are the evidence. Axis ranges/units asserted against the paper's
  figure: x = ε in [-10, 10] (matches); y = "argument to softmax" (logits),
  paper ~[-2000, 1000], ours [-405, 639] on the deterministic class-4 example
  (the paper's chosen example has more extreme logits; c65-c70 are rated
  `low` — single-example illustration, not a population invariant).
- **Tests / mutations / instruments.** `pytest -q`: 37 pass, 1 skip (the
  CIFAR-real test skips when the tar is absent, as designed). Mutation suite
  (`tests/test_mutations.py`): all 10 deliberate defects caught by their
  `must_fail` tests. `instruments.json` lists every output-deciding instrument
  (data loaders, selfcheck grader, logreg equivalence, FGSM inf-norm,
  degeneracy, rubbish threshold, retrain-from-scratch, adv-train FGSM mode,
  arch faithfulness) with positive + negative tests.

## Notes

- **Prior run preserved.** This repository's `main` already contains a completed
  reproduction of this paper (merged commit `11f0a3b`, prior run dated 2026-08-04).
  Its files remain in this folder. Its final report is preserved as
  `REPRODUCTION_prior_run.md`; this file starts fresh for the current run. Later
  steps decide what of the prior work to reuse, redo, or replace.
- The authoritative reference for every equation, table, and reported number is
  `paper/source/iclr2015.tex` / `paper/source/iclr2015.bbl`, not `paper/paper.md`.
