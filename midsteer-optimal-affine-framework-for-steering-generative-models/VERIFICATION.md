# VERIFICATION — MidSteer reproduction (arXiv:2605.05220)

This file lists every check this reproduction actually ran, what each found, the
budget each ran at, and — the part that matters — what remains untested and why.

**Nothing here establishes that the implementation is correct.** It establishes
that the implementation is not wrong in the specific ways that were checked. The
list of checks that were *not* run is worth more to a reader than the headline
number, because the headline number this run produced is "BLOCKED" for every
one of the paper's empirical claims.

## Headline

- paper_ref: `0e6756c9-d827-4749-a8d7-3140a8c98a25`
- project_id: `06910d54-1d99-4864-8bff-ab3007a0c70e`
- Rung reached: **environment** (all four comparison arms BLOCKED — see below).
- Budget spent: `$HOME/.build_attempts` = **7** (build/environment budget
  exhausted while the arms gate kept failing). `$HOME/.env_attempts` and
  `$HOME/.review_rounds` were not written (no separate env-attempt or review-round
  budget was consumed).
- Authoritative counts: **`result_check` never printed an `AUTHORITATIVE COUNTS`
  line in this run's journal**, which is why this is not rung `numbers`. The
  verdict counts that *do* exist are from `claims_result.json`
  (produced by `evaluate_claims.py`, `generated_by='workflow_subagent'`):
  reproduced=3, refuted=0, untested=0, blocked=19. The 3 reproduced are
  closed-form invariants on synthetic data; the 19 blocked are every model-arm
  claim. These are the reproduction's own evaluator output, not the
  `result_check` authoritative line, and are reported as such.

## Checks that ran

### 1. Environment / import gate
- **Command:** `uv venv --python 3.13 .venv && uv pip install --python .venv -r requirements.txt` then `.venv/bin/python -m pytest tests/ -q`
- **Budget:** CPU, ~8 s, no network for the install step (pinned wheels already cached), no GPU.
- **Found:** 117 passed. Every pinned dependency imports; every `core` module imports; `clip` resolves to `clip-anytorch` (OpenAI-CLIP fork), not the clipboard tool; the whitening identity $W^+W = I$ holds at `atol=1e-6`.
- **Establishes:** the pinned environment resolves and the vendored upstream + closed-form `midsteer_core` import cleanly. **Does not establish** the model arms run — they need CUDA + `HF_TOKEN`.

### 2. Closed-form affine core unit tests
- **Command:** `tests/test_midsteer_sign_matches_eq23.py`, `tests/test_invariants.py`, `tests/test_core.py`, `tests/test_degeneracy.py` (run as part of `pytest tests/ -q`).
- **Budget:** CPU, seconds.
- **Found:** the `midsteer_core` affine maps match Eqs. 6/13/19/22/23 at the literal `β=1` setting; the erasure special case (MidSteer≡LEACE when Z2 constant) and the vanilla / vanilla-switch special cases hold; the degeneracy/no-op settings reproduce the baseline.
- **Establishes:** the closed-form math is implemented to the equations. **Does not establish** the maps are correct *as steering operators on real model activations* — that needs the model arms.

### 3. E1 closed-form invariant checks (REAL, the only real numbers this run produced)
- **Command:** `.venv/bin/python experiments/run_e1_synth.py` (run inside `bash run_all_arms.sh`); writes `results/e1_synth.json`; `assemble_measured.py` flows the per-seed residuals into `measured.json` under the `e1_synth` arm.
- **Budget:** CPU, seconds, 3 seeds × 16 feasible-set perturbations per claim, synthetic Gaussian data of known covariance (seeded RNG — provenance stated in the script docstring).
- **Found:** C1, C2, C3 all `pass` at every seed. Worst residuals across seeds: C1 constraint ≤8.29e-13, C2 flip ≤1.66e-12, C3 matched-cov ≤1.48e-12; minimal-disturbance gap = 0 at every seed; special-case residuals ≤2.66e-15 (C1/C2) and 0 (C3 erasure). The single FINAL line `FINAL e1_synth=1.655510e-12` is the max C2 flip residual.
- **Establishes:** on **synthetic data of known covariance**, the closed-form LEACE / LEACE-Switch / MidSteer maps satisfy their constraints and are minimally disturbing over the constraint-preserving feasible set, and reduce to the paper's vanilla special cases. **This says nothing about the paper**, whose claims are about Llama-2-7B-chat and SDXL activations, not synthetic Gaussians. It is the cheapest possible check that the affine algebra is built to the equations the paper proves.

### 4. Claims evaluator (verdict table)
- **Command:** `.venv/bin/python evaluate_claims.py` → `claims_result.json`; `.venv/bin/python selfcheck_claims.py` → `selfcheck.json`.
- **Budget:** CPU, seconds; operates on the already-assembled `measured.json`.
- **Found:** reproduced=3 (C1–C3), refuted=0, untested=0, blocked=19 (C4–C22). `selfcheck.json` agrees. C1–C3 predicates are executable boolean expressions over `measured.e1_synth.*` residual thresholds; C20–C22 quantities are plain `measured.<arm>.<metric>` refs to per-x sequences (BLOCKED here). The evaluator was rebuilt with positive **and** negative tests for every verdict path (reproduced/refuted/untested/blocked for ordering, value, invariant, curve).
- **Establishes:** the verdict table is produced by a deterministic evaluator over measured evidence, not hand-written. **Does not establish** any model-arm claim — they are blocked by the missing arms, which is the correct blocked result, not a defect in the evaluator.

### 5. Mutation suite (self-checking the instruments)
- **Command:** `tests/test_mutations.py` (part of `pytest tests/ -q`).
- **Budget:** CPU, seconds.
- **Found:** 5 deliberate defects injected into the affine core / instruments are all caught by the committed tests.
- **Establishes:** the tests are not vacuous — they fail when the code is wrong. **Does not establish** the tests cover every way the model arms could be wrong, because the model arms never ran.

### 6. Smoke (fast path, not evidence)
- **Command:** `bash smoke.sh` (sets `MIDSTEER_SMOKE=1`).
- **Budget:** CPU, seconds; writes only to gitignored `results/e1_synth_smoke.json`.
- **Found:** `FINAL e1_synth_smoke=9.12e-14` — a measured residual, not a verdict word; proves the E1 code path runs at smoke scale.
- **Establishes:** the E1 path runs end-to-end at small scale. **Explicitly not evidence about the paper** (smoke output is never reported as a result, and is gitignored so it cannot clobber the canonical `results/e1_synth.json` — guarded by `tests/test_smoke_no_clobber.py`).

### 7. Inline adversarial review of the six fixed claims (C1–C3, C20–C22)
- **Command:** inline reading of the paper `.tex` at each claim's cited lines, byte-for-byte (the `orchestrate` backend returned HTTP 500 on a trivial ping this session, so the parallel adversarial-review workflow could not run; the review was done directly).
- **Budget:** no agent budget consumed (orchestrate unavailable); reviewer rounds file `$HOME/.review_rounds` was not written.
- **Found:** C1 (`guardedness.tex:73-85`), C2 (`main.tex:343-367`), C3 (`main.tex:418-448`), C20 (`switching_suppl.tex:21`), C21 (`experiments.tex:121`), C22 (`suppl.tex:642`) all faithful to the cited equations/tables; one latent scalar/curve name collision (`bertp_mmlu`) found and fixed by renaming C22's curve metric to `c22_bertp_mmlu`.
- **Establishes:** the six claims that were made gate-evaluable are faithful to the paper text. **Does not establish** the model-arm claims (C4–C19) were reviewed against live model output — they could not be, because the arms are BLOCKED. **The review budget was not spent on rounds of reviewer disagreement** (orchestrate was down), so this is not a "reviewers went quiet" pass; it is a single inline read.

## What remains untested, and why

- **Every model-arm claim (C4–C22) is untested.** Reason: the sandbox is CPU-only
  (`torch.cuda.is_available()` is False, no `nvidia-smi`) and has no `HF_TOKEN`,
  so the Llama-2-7B-chat arms (E2, E3) and SDXL arms (E4, E5) cannot run. They
  print `FINAL <arm>=BLOCKED` and write BLOCKED partials; `midsteer_core/data`
  raises `BlockedException` and never falls back to synthetic data. The paper's
  scalar targets (70.7, 0.281, 6.0, the C22 plateau) and all 16 ordering
  comparisons between the four arms were therefore not measured. This is an
  environment blocker, not an implementation defect.
- **The paper's dataset was not used.** E1 runs on synthetic Gaussian data of
  known covariance. A number measured on a synthetic stand-in for the paper's
  activations says nothing about the paper; it is reported here only as a check
  of the closed-form algebra, and explicitly so.
- **The training/generation horizon was not shortened to fit the machine.** No
  model arm ran at all, so there is no "shortened horizon" caveat to record —
  the issue is total absence of the arms, not a reduced one.
- **Docker build was never exercised.** `docker` is not installed in this
  sandbox. The `Dockerfile` (CUDA 12.4 base) is committed but unverified; a
  reader with Docker must run `docker build -t midsteer-repro . && docker run
  --rm midsteer-repro python -m pytest tests/ -q`.
- **Reviewer-rounds pass was not achieved.** The `orchestrate` backend was down
  (HTTP 500) this session, so the prescribed parallel adversarial-review workflow
  did not run. The review of the six fixed claims was done inline in one pass; no
  `$HOME/.review_rounds` budget was spent and no reviewer-disagreement loop ran.
  This is not a clean review pass and is not reported as one.
- **Curve claims (C20–C22) figures were not regenerated.** The underlying model
  metrics are BLOCKED, so regenerating Figs. 2/5 would fabricate evidence; none
  is produced. The paper's figure image files are gitignored out of the repo
  (only `.tex` is kept), so there is no paper figure to commit a pair beside.
- **Out-of-scope items (recorded in `claims.json` `not_tested`):** Appendix L
  erasure-model tables (covered synthetically by C3's erasure special case);
  Qwen2.5-7B/14B and SANA arms (SPEC §8 — only Llama-2-7B-chat and SDXL are in
  scope); the GPT-4o-mini judge; exact BERT-P/F1 magnitudes; the full per-β
  tables; the unbalanced class-prior reading of Σ_XZi (G3).

## What this verification does NOT establish

It does not establish that MidSteer reproduces the paper. It establishes that:
the pinned environment builds and imports; the closed-form affine maps match
Eqs. 6/13/19/22/23 and satisfy their constraints and special cases on synthetic
data; the verdict table is produced by a deterministic evaluator over measured
evidence with positive and negative tests for every path; the tests fail when
the code is wrong (mutation suite); and the smoke path runs without clobbering
evidence. The paper's actual empirical comparison — MidSteer vs vanilla vs
LEACE-Switch vs base on Llama-2-7B-chat and SDXL — was not run in this sandbox
and so is, in the honest sense, unverified.
