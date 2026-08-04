# Reproduction: Confidence-Weighted Self-Distillation for Learning under Label Noise

- **Paper:** Confidence-Weighted Self-Distillation for Learning under Label Noise
- **Authors:** A. Bergstrom, M. Oyelaran, K. Vasquez
- **Year:** unknown
- **Date started:** 2026-08-04
- **paper_ref:** ce7a63e8-2c90-4516-887d-14515c8f4516
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Rung reached:** `numbers` (on-disk numbers gate 9/9 `reproduced`, no build /
  environment / review budget file records a still-failing gate at publish)

## Source of the numbers

The paper's arxiv_id is `unknown`, so no arXiv LaTeX source could be fetched.
All equations, tables, and numbers below come from the PDF-extracted text in
`paper/paper.md` (reliable for prose, **not** for maths) and from running the
implementation. The data is the paper's own `sklearn.datasets.load_digits`
corpus (1797 8×8 digits, K=10, pixels ÷16 → [0,1]), fingerprinted by
`instruments.json` — **not a synthetic stand-in**. The full 4000-step
training horizon the paper states (§3) was used at every seed; the horizon was
**not** shortened to fit the machine (one full arm runs in ~0.8 s on CPU).

## Measured vs. claimed

The paper reports a **single seed-0 run** (paper §3, Table 1). The arms just
ran at the paper's full configuration at seeds {0,1,2}; their `FINAL` lines
are in `/tmp/arms.log`. Seeds 1 and 2 are a robustness check the paper did
**not** perform, so only the seed-0 row is directly comparable to the paper's
claimed values.

**Headline comparison (seed 0, the seed the paper ran):**

| Arm | λ | Paper claims | Measured (seed 0) | Δ (measured − claimed) | Exact command |
|---|---|---|---|---|---|
| Cross-entropy (baseline) | 0 | 0.9370 | 0.9370 | +0.0000 | `python run_experiment.py --lambda 0.0 --seed 0` |
| CWSD (ours) | 1 | 0.9620 | 0.9611 | −0.0009 | `python run_experiment.py --lambda 1.0 --seed 0` |
| Gap (CWSD − baseline) | — | +0.0250 | +0.0241 | −0.0009 | (difference of the two above) |

The two `FINAL` lines for the headline comparison, exactly as printed:

```
$ python run_experiment.py --lambda 0.0     # paper §5 bare command
FINAL accuracy=0.9370
$ python run_experiment.py --lambda 1.0     # paper §5 bare command
FINAL accuracy=0.9611
```

The grid runs the same program with `--seed` and `--metrics-out` (the exact
invocation `run_all_arms.sh` uses, per `claims.json`):

```
python run_experiment.py --lambda 0.0 --seed 0 --metrics-out <tmp>   # baseline, seed 0
python run_experiment.py --lambda 0.0 --seed 1 --metrics-out <tmp>   # baseline, seed 1
python run_experiment.py --lambda 0.0 --seed 2 --metrics-out <tmp>   # baseline, seed 2
python run_experiment.py --lambda 1.0 --seed 0 --metrics-out <tmp>   # CWSD, seed 0
python run_experiment.py --lambda 1.0 --seed 1 --metrics-out <tmp>   # CWSD, seed 1
python run_experiment.py --lambda 1.0 --seed 2 --metrics-out <tmp>   # CWSD, seed 2
```
(collected by `./run_all_arms.sh`, which writes `measured.json` and prints one
`FINAL <arm>=<value>` line per arm-seed to stdout).

**All measured values (from `/tmp/arms.log` → `measured.json`):**

| Arm | seed 0 | seed 1 | seed 2 | mean | spread (max−min) |
|---|---|---|---|---|---|
| baseline (λ=0) | 0.9370 | 0.9407 | 0.9315 | 0.9364 | 0.0092 |
| CWSD (λ=1)     | 0.9611 | 0.9481 | 0.9556 | 0.9549 | 0.0130 |
| gap (CWSD−base)| +0.0241 | +0.0074 | +0.0241 | +0.0185 | — |

The CWSD − baseline gap is **positive at every seed** (never reversed): the
two arms are distinguishable across the three seeds the grid ran. The paper's
own seed (0) separates the arms cleanly — the +0.0241 gap there exceeds both
arms' within-seed spreads (baseline 0.0092, CWSD 0.0130). At seed 1, however,
the gap (+0.0074) is **smaller than the baseline arm's own within-seed spread
(0.0092)**, so at that one extra seed the two arms are within noise of each
other and that single seed does not by itself test the paper's comparison.
Whether this constitutes a reproduction is the reader's call; this report
states the numbers and the differences and asserts no tolerance. (The arms
are **not** within noise of each other overall — CWSD leads at every seed —
so the grid as a whole does test the paper's comparison.)

## Numbers gate (`claims_result.json`)

The gate adjudicates 9 claims declared in `claims.json` against
`measured.json`. Verdict counts: **reproduced 9, refuted 0, untested 0,
blocked 0** (`summary.gate_pass = true`). The 6 high compute-invariance
claims (structural: λ=0==CE degeneracy, gate weight ∈ [0,λ], convex target,
stop-gradient, single-network/no-extra-params; and the sign-only
CWSD>baseline ordering) all pass; the 3 low claims (exact Table-1
magnitudes, seed-0-only in the paper) pass within seed-widened tolerances.
Two claims are recorded in `claims.json` `not_tested` (the §4 *attribution*
claim and Table-1 magnitudes at seeds ≠ 0), not adjudicated as reproductions.

## Research-readiness gates

| # | Gate | Verdict | Evidence |
|---|---|---|---|
| 1 | Builds from scratch | **partial** | A fresh `uv venv --python 3.13` + `uv pip install -r requirements.txt` from a clean dir builds, runs `pytest -q` → 48 passed, and reproduces `--lambda 1.0` → 0.9611. The `Dockerfile` is present and self-contained but `docker` is not installed in this environment, so `docker build/run` was **not** exercised. |
| 2 | README is accurate | **pass** | Quickstart followed verbatim in a clean checkout: `uv venv --python 3.13 --clear .venv` + `uv pip install --python .venv -r requirements.txt` + `./run_all_arms.sh` works; the single-arm commands `.venv/bin/python run_experiment.py --lambda 0.0` / `--lambda 1.0` print the documented `FINAL accuracy=<float>` lines. (Test-count references in README/SPEC corrected from 47 to 48 this run.) |
| 3 | Packages are clear | **pass** | `requirements.txt` pins every dependency with a version (numpy 2.5.1, scikit-learn 1.9.0, scipy 1.18.0, joblib 1.5.3, threadpoolctl 3.6.0, narwhals 2.24.0, pytest 9.1.1 + its pins); the fresh-venv install succeeds and the code then imports and runs with no missing-import death. |
| 4 | Entrypoint is obvious | **pass** | One documented command `python run_experiment.py --lambda {0.0,1.0}` runs the experiment; all hyperparameters are flags (no source edits). Paper §5 gives exactly these two bare commands. |
| 5 | Fast path | **pass** | `./smoke.sh` runs the whole path (load → corrupt → init → train → evaluate → print) at 50 steps in <1 s, printing one `FINAL smoke=0.8370` line (deliberately labelled `smoke=`, not `accuracy=`, so it is not mistaken for a result). |
| 6 | Deterministic / noise quantified | **pass** | Same seed → same number: `./run_all_arms.sh` re-run this pass reproduced `measured.json` byte-identical and the same six `FINAL` lines. Run-to-run spread is quantified per arm (baseline 0.0092, CWSD 0.0130) in the table above. |
| 7 | Degeneracy test in the repo | **pass** | `tests/test_degeneracy.py` (4 tests): the λ=0 path is bitwise identical to an independently written cross-entropy routine, per-step (loss + every grad) and end-to-end (300-step SGD loop, accuracy), swept over `s ∈ {0.01…10.0}`. This is the paper's own verification gate (λ=0 ⇒ `t=y` ⇒ Eq. 4 is plain CE). |
| 8 | Data provenance stated | **pass** | `instruments.json` fingerprints the data as the paper's own `sklearn.datasets.load_digits` (n_total=1797, 64 features, K=10, Xtr 1257×64, Xte 540×64, SHA-256 of the split arrays); `tests/test_instruments.py` asserts the fingerprint. Pixel max is 16 (÷16 → [0,1]), matching paper §3. |
| 9 | Recorded number is reproducible | **pass** | The exact commands recorded beside the numbers above, run again this pass, produce the same values within the quantified noise (the re-run was byte-identical). |
| 10 | Nothing depends on hidden local state | **pass** | Fresh-clone build (gate 1) reproduces the numbers; the only inputs are `paper/paper.md`, `requirements.txt`, the source, and `sklearn`'s bundled `load_digits` (no manual wheel, no home-dir state). |

**Gates: 9 pass, 1 partial (gate 1, Docker unexercised), 0 fail.**

## Budget at publish

- `$HOME/.build_attempts`: **does not exist** — no build budget recorded as
  spent; the build is clean (`run_all_arms.sh` writes a well-formed
  `measured.json`, the gate returns 9/9, `pytest -q` → 48 passed).
- `$HOME/.env_attempts`: **does not exist** — the environment reproduces from
  the pinned `requirements.txt` (fresh venv verified, gate 1).
- `$HOME/.review_rounds`: **does not exist** — the prior adversarial
  component review (5 reviewers, all approved) went quiet; no review budget
  was spent without resolution in this run.

## Log

- 2026-08-04 — Ingest: cloned repo (shallow over HTTPS), branch
  `repro/confidence-weighted-self-distillation-for-learning-under-label`
  created at `main` HEAD and pushed; paper text saved to `paper/paper.md`
  (byte-identical to this run's objective text); this file started. No arXiv
  fetch (`arxiv_id: unknown`).
- 2026-08-04 — Implementation/verify pass: the prior run left a complete,
  tested reproduction on the branch. This pass re-ran it end to end and
  adversarially reviewed each independent component against the paper via an
  orchestration of 5 review subagents (data pipeline, method core Eqs 1-4,
  training loop, evaluation metric + FINAL contract, baseline arm +
  degeneracy gate), each prompted to find failures and cite `file:line`. All
  5 components were approved with no blocker/major issues. Three nit/minor
  findings were acted on (SPEC `train()` signature, a misleading test comment,
  and a new `test_train_leaves_test_set_clean` regression test). No
  code-behavior change; `measured.json` unchanged.
- 2026-08-04 — Numbers pass (this run): every arm re-run at the paper's full
  4000-step budget at seeds {0,1,2}; `/tmp/arms.log` records the six `FINAL`
  lines (baseline 0.9370/0.9407/0.9315, CWSD 0.9611/0.9481/0.9556), identical
  to the committed `measured.json` and reproduced byte-identical on a fresh
  `./run_all_arms.sh`. Numbers gate `claims_result.json`: reproduced 9 / 0 /
  0 / 0, `gate_pass=true`. Fresh-venv build (gate 1) verified: 48 tests
  pass, `--lambda 1.0` → 0.9611. README/SPEC test-count references corrected
  47 → 48. No build/env/review budget file exists; no gate is failing at
  publish. The measured-vs-claimed table and the readiness-gate table above
  are this pass's contribution; the numbers are stated, not judged.
