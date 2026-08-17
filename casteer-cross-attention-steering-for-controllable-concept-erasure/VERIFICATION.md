# VERIFICATION.md — CASteer reproduction (arXiv 2503.09630)

This file records **every check this reproduction ran**, what it found, the
budget it ran at, and — the part that matters — **what remains untested and
why**. None of this establishes that the implementation is correct; it
establishes that it is not wrong in the ways that were checked. That list is
worth more to a reader than the headline number.

- paper_ref: `d0ab97b5-e9bf-41af-9846-d19f846eef80`
- project_id: `06910d54-1d99-4864-8bff-ab3007a0c70e`
- Rung reached: **`environment`** (all-BLOCKED arms — see §A).
- Numbers gate `claims_result.json` AUTHORITATIVE COUNTS: **reproduced=1, refuted=0, untested=0, blocked=16**.

## A. Budget and the gate that is still failing

| Budget file | Value | Meaning |
| --- | --- | --- |
| `~/.build_attempts` | 4 | The build/arms (numbers) gate ran 4 times; it is still failing. |
| `~/.env_attempts` | (empty) | No environment-attempts budget was spent. |
| `~/.review_rounds` | 4 | The review gate ran 4 rounds; the budget was spent and the reviewers did not go quiet. |

**The gate still failing is the build/arms (numbers) gate.** Its last output
(captured in `/tmp/arms.log`) is 27 lines, every one `FINAL <arm>=BLOCKED`
(9 arms × 3 seeds {42,1234,2024}), followed by
`wrote …/measured.json`. `measured.json` records `BLOCKED` for every diffusion
metric; `measured_blocked_reasons.json` carries the reason:

> CPU-only host; SD-1.4/SDXL at the paper full config (50 steps, I2P=4703
> prompts / snoopy+other=800 per concept / COCO=3000 captions × 3 seeds) is
> infeasible without a GPU. The paper used 8×V100 (`supplementary.tex:30`).

An all-BLOCKED arms result is the `environment` rung and is a legitimate
outcome: the host cannot produce a number that separates the arms, so the
paper's comparison was not tested. The review gate also failed (rejected, 4
outstanding findings — §F); that is a *separate* failure and is reported as
such, not folded into the environment rung.

## B. Checks that ran and what they found

### B1. Environment / import gate
- Command: `python -m pytest tests/test_environment.py -q`
- Result: **pass**. Every pinned dependency imports; every `core` module
  imports; the Householder / unit-norm invariants and the closed-form
  `(I − 2ssᵀ)c` check hold.
- Budget: CPU-only, seconds.

### B2. Method-core invariants (the paper's maths)
- Command: `python -m pytest tests/test_method_core.py tests/test_degeneracy.py tests/test_invariants.py -q`
- Result: **pass**. Eq.6 (matrix == dot-product), Eq.7 (clip only positive
  projections), Eq.4 (constant α), Eq.9 (multi-concept average, no
  renormalization), CFG conditional-half-only steering, unit-norm
  construction, Householder norm preservation, and β=0 bit-exact degeneracy
  all hold.
- Budget: CPU-only, seconds.

### B3. Data-loader fingerprints
- Command: `python -m pytest tests/test_data.py -q`
- Result: **pass**. ImageNet 50/`tench`, 80 CLIP templates, 30,000 COCO
  captions, 4,703 I2P prompts (live `AIML-TUDA/i2p` fetch) all fingerprint to
  the expected counts/shapes. COCO-30k reference loader RAISES when not
  vendored (no synthetic substitution).
- Budget: CPU + one network fetch for I2P.

### B4. Evaluation-instrument tests
- Command: `python -m pytest tests/test_eval.py -q`
- Result: **pass**. CLIP score positive/negative, FID identical/different/
  empty, NudeNet/Q16/LPIPS raise `MissingEvaluatorError` when their tool is
  absent, no-OK-on-empty. Confirms the metrics raise rather than fabricate.

### B5. Runner / driver dispatch tests
- Command: `python -m pytest tests/test_runner.py tests/test_driver.py -q`
- Result: **pass**. Per-task steering-vector mapping (`TASK_VECTOR`),
  per-model `vector_dir` isolation (sdxl ≠ sd14), I2P per-prompt `sd_seed`,
  `n_per` template expansion to declared counts (800/concept, 200 style),
  bare-concept CLIP reference text, Eq.9 7-concept average for I2P-overall,
  `arm_metrics_from_claims` predicate parsing, nudity full-set scaling +
  inconclusive-below-floor.

### B6. Mutation suite (do the tests actually catch defects?)
- Command: apply each of 16 defects in `mutations.json`, run its `must_fail`
  node, confirm it FAILS on the mutated code (i.e. the test is real
  evidence). Full harness re-run after every fix pass.
- Result: **16/16 caught, 0 surviving** on the correct implementation; all
  65 tests pass on the correct implementation.
- Budget: CPU-only, ~minutes.

### B7. Smoke run (full path on real SD-1.4, tiny config)
- Command: `bash smoke.sh` (= `.venv/bin/python scripts/diffusion/smoke.py`)
- Result: `FINAL casteer_clip=0.526408` (snoopy_cs on 1 steered image, 4
  steps / 256² / seed 42); `results/smoke/{casteer_clip,sd14}.png` written.
  Reproduces on re-run. **NOT evidence about the paper** (tiny subset,
  reduced steps/resolution).

### B8. Full arm sweep (the numbers gate input)
- Command: `bash run_all_arms.sh`
- Result: 27 `FINAL <arm>=BLOCKED` lines; `measured.json` with every
  diffusion metric `BLOCKED`; `measured_blocked_reasons.json` with the
  CPU-only reason. The one CPU-runnable metric (`house`) is filled:
  `casteer_noclip.<seed>.house_pass=1`, `house_max_norm_err=3.55e-14`
  (seeds 42/1234) / `2.84e-14` (seed 2024).

### B9. Self-check evaluator (our own, NOT the gate)
- Command: `python selfcheck_claims.py` → `selfcheck.json`
- Result: `house` PASS; 16 diffusion claims BLOCKED. This is our evaluator;
  the workflow's numbers gate produces `claims_result.json` independently.

### B10. Numbers gate (workflow, not us)
- Output: `claims_result.json` (`produced_by: reproduce-paper numbers gate`).
- Verdicts: `house` → `reproduced`; 16 diffusion claims → `blocked`
  (every referenced value BLOCKED). AUTHORITATIVE COUNTS: reproduced=1,
  refuted=0, untested=0, blocked=16.
- Budget: the gate ran against `measured.json`; it settled all 17 claims
  (0 `unevaluable`).

## C. What is established

- The method core is paper-faithful at the equation level (Eq.4/6/7/9,
  Householder, unit-norm, CFG half-only), verified by invariants + a
  mutation suite where every defect is caught.
- The plumbing runs end-to-end on real SD-1.4 (smoke): estimate → steer →
  generate → score, on CPU.
- The one CPU-runnable claim (`house`) is reproduced to ~1e-14 at 3/3 seeds.
- No synthetic stand-in was substituted for any paper dataset: the COCO-30k
  FID reference raises when absent; NudeNet/Q16/LPIPS raise when their tools
  are absent.

## D. What remains untested and why

- **All 16 diffusion-number claims** (nudity, I2P-overall, COCO-FID, Snoopy
  erasure/preservation, style LPIPS, SDXL-distilled transfer, the two
  constant-α ablations, the two curve figures) — **not measured**. Reason:
  CPU-only host; the paper's full config (50 steps × ≥800–4,703 prompts × 3
  seeds × 9 arms, 8×V100, `supplementary.tex:30`) is infeasible on CPU. A
  number produced at a horizon short enough to fit this machine would not
  separate the arms and is not evidence about the paper's claim; none was
  produced.
- **The two curve figures** (`fig_clip_shape`, `fig_fid_shape`) were not
  regenerated because each plots BLOCKED CS/FID points; fabricating a figure
  from BLOCKED numbers was declined.
- **`house` is a maths invariant, not an empirical result** — it certifies
  the Householder operator on synthetic vectors, not the production
  steering/generation/eval path. "1 reproduced" does not mean the paper's
  empirical claims were reproduced.

## E. Gates not provable in this sandbox

- **Docker build** (`docker build -t repro . && docker run …`): `docker` is
  not installed here; the `Dockerfile` is well-formed and the CPU env builds
  via `uv pip install`, but the image build was not executed.
- **GPU run** (any diffusion arm at full config): no CUDA device available.

## F. Review gate — outstanding objections (review budget spent)

The review budget (4 rounds) was spent and the reviewers did not go quiet.
The last review round returned `rejected` with four non-cosmetic findings,
each verified firsthand against the repo and the paper's LaTeX:

1. **Style-eval prompt leak + protocol divergence.** Van Gogh style-eval
   generation prompts are byte-identical to the *positive* halves of the
   steering-vector *estimation* prompts (`run_all_arms.py` vs
   `core/construct_prompts.py`), inflating LPIPS_e in the method's favour,
   and the prompt set diverges from the SAFREE procedure the paper defers
   to (`experiments.tex:127-128`). Not registered in SPEC/claims
   sensitivities. **Unfixed.**
2. **Headline evaluators unreachable as shipped, even on GPU.** `nudenet`
   and `lpips` are absent from `requirements.txt`; `q16_inappropriate_count`
   raises by construction though the paper names the Q16 detector
   (`experiments.tex:47-49`); the COCO-30k FID reference is not vendored.
   `nudity_total`, `vangogh_lpips_e`, `i2p_overall_pct`, `coco_fid30k` can
   never settle in the pinned environment. **Unfixed.**
3. **Others-concept aggregation degrades to a subset mean.** Per-concept
   scoring failures are silently skipped and the mean taken over
   survivors, not the paper's mean-over-five. **Unfixed.**
4. **`house` is disconnected from the implementation it certifies** — a
   synthetic-vector plumbing invariant, not the production path. Recorded
   here so "1 reproduced" is not over-read.

Rounds 1–2 produced fix commits (`790d081`, `91011ce`) resolving F1/F2/F3,
A1/A3, the unevaluable-claim SyntaxErrors, the surviving mutations, and the
driver/eval-path corrections. Rounds 3–4 produced the four findings above,
which were **not** fixed before the review budget ran out. The `orchestrate`
tool was unavailable in this sandbox (Internal Server Error on every call),
so review was performed by the workflow's `workflow-step-review` agent
against the repo and the paper's `.tex` — recorded for provenance.

## G. Reproducibility of this verification

All CPU checks re-run from a fresh clone:

```
git clone -b repro/casteer-cross-attention-steering-for-controllable-concept-erasure \
    https://github.com/coleh-cm/auto-reproductions.git
cd auto-reproductions/casteer-cross-attention-steering-for-controllable-concept-erasure
uv venv --python 3.13 .venv && uv pip install --python .venv -r requirements.txt
.venv/bin/python -m pytest tests/ -q        # 65 passed
bash smoke.sh                               # FINAL casteer_clip=0.526408
bash run_all_arms.sh                        # 27× BLOCKED + measured.json (house filled)
.venv/bin/python selfcheck_claims.py        # selfcheck.json: house PASS, 16 BLOCKED
```

The GPU path (which would un-block the 16 diffusion claims) additionally
needs a CUDA device, `HF_TOKEN`, the un-vendored COCO-30k FID reference,
and — per review finding 2 — `nudenet`/`lpips` added to `requirements.txt`
and a pinned Q16 checkpoint. None of those were available in this run.
