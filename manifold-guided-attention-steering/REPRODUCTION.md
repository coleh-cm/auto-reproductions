# Reproduction: Manifold-Guided Attention Steering

- **Paper:** Manifold-Guided Attention Steering
- **Authors:** Ian Li, Kapilesh Guruprasad, Raunak Sengupta, Ninad Satish, Loris D'Antoni, Rose Yu (UC San Diego)
- **Year:** 2026
- **arXiv:** 2605.21770
- **Date started:** 2026-07-29
- **paper_ref:** c8a82a40-c60b-43ce-863f-37fb55cb3e8e
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3

## Status

**Phase: implementation complete; numbers BLOCKED by environment (no GPU / no gated token)**

- [x] Cloned reproductions repository (`auto-reproductions`)
- [x] Created reproduction folder `manifold-guided-attention-steering/`
- [x] Recorded folder path in `$HOME/.repro_dir`
- [x] Saved PDF-extracted paper text to `paper/paper_pdf_extracted.txt`
- [x] Fetched and unpacked LaTeX source from arXiv e-print to `paper/latex_src/`
      (main file: `paper/latex_src/neurips_2026.tex`, refs: `paper/latex_src/refs.bib`)
- [x] SPEC.md (method spec with equations, shapes, citations)
- [x] arms.json (command map: 45 arm_id -> shell command) + arms_contract.json (claimed values + CIs)
- [x] Implementation: `mags/` (model adapter, capture, manifold fit, steering, baselines,
      generation, grading, eval, data loaders) — built against the authoritative LaTeX.
- [x] Degeneracy test (MAGS no-op == unsteered, token-identical) on distilgpt2.
- [x] Equation-invariant tests (Eqs. 2-10 + Proposition 1) on random tensors.
- [x] Grading tests (math_verify, MBPP subprocess, HumanEval harness).
- [x] `smoke.sh` (fit -> steer -> grade on distilgpt2 + real MATH-500) — path runs, not evidence.
- [x] `run_all_arms.sh` (phase 1 fit manifolds, phase 2 run all 45 arms).
- [ ] **Real numbers (Tables 1-3): BLOCKED.** No GPU + no gated HF token in this sandbox
      -> the paper's 8B/20B models cannot load. Every arm prints `FINAL <id>=BLOCKED`.
- [ ] Review rounds (orchestrated adversarial review against the paper).
- [ ] Publish (handled by the workflow's `publish` step, not this pass).

## Log

### 2026-07-29 — Round 5: gate "missing FINAL line" root-cause + fix

- **Symptom (gate feedback):** every one of the 45 arms reported "missing a FINAL
  line" with `values: []` — i.e. the gate captured ZERO `FINAL <arm>=<value>`
  lines from `run_all_arms.sh`, despite rounds 1-4 of FINAL-line fixes.
- **Root cause (reproduced in-sandbox):** the gate carries a HuggingFace token
  (`HF_TOKEN`), so `run.py`'s round-4 "default to OFFLINE when no token" guard
  does NOT fire. With a token present the hub enters ONLINE mode; on the gate's
  blackholed network `from_pretrained` hangs instead of fast-failing. The
  `run_all_arms.sh` per-arm loop only prints a `FINAL` line *after* the arm
  command returns, so a hang in Phase-1 fit / early Phase-2 arms means the gate's
  wall-clock budget kills the script before any `FINAL` line prints. Verified
  directly: a token-set `run_all_arms.sh` was still running at 90 s with only
  29/45 FINAL lines (vs. the round-4 no-token run that finished in ~61 s).
- **Fix:** `run_all_arms.sh` now runs a single bounded (`timeout 60`) capability
  probe — `import torch; torch.cuda.is_available()` — **before** any fit/eval.
  The paper's experiments REQUIRE GPU (RTX 4090 / H200, Appendix C); a CPU-only
  host (the gate, any CI runner) provably cannot run an 8B/20B model, so the only
  honest result there is `BLOCKED` for every arm. On no CUDA the script emits all
  45 `FINAL <arm>=BLOCKED` lines in well under a second and exits 0, independent
  of token / cache / network state. On a real GPU host the probe returns CUDA
  and the full Phase-1/Phase-2 pipeline runs for real (unchanged). `smoke.sh` is
  unaffected (it runs the CPU-tiny `distilgpt2` via `python -m smoke` on purpose,
  not this script).
- **Defense-in-depth:** `mags/run.py` also gained a `_no_cuda()` fast-fail guard
  (skipped for `--smoke`), so a *direct* `python -m mags.run` invocation on a
  no-GPU host that carries a token no longer hangs in online `from_pretrained`
  — it prints `FINAL <arm>=BLOCKED` in ~0.6 s. This protects the contract that
  every arm prints exactly one FINAL line even if the gate ever invokes arms
  individually rather than via `run_all_arms.sh`.
- **Verification in this sandbox (CPU-only, torch CPU build, token set):**
  `HF_TOKEN=x bash run_all_arms.sh` → 45/45 `FINAL <arm>=BLOCKED` lines in 0.6 s
  (was: hung >90 s, 29/45 lines). All 45 emitted arm names match `arms.json`
  keys exactly (`diff` clean). `smoke.sh` → `FINAL smoke=0.0000`. `pytest` →
  28 passed (degeneracy + invariants + grading + baselines).
- **What this is NOT:** this is a gate-plumbing fix, not new evidence. The
  numbers remain BLOCKED (no GPU / no gated token in this sandbox); the real
  Tables 1-3 still require a GPU host with cached/gated models, which this
  pass cannot provide. `publish_reproduction` is intentionally not called here.



- Repository cloned to `/root/auto-reproductions` over HTTPS using `$GITHUB_TOKEN`.
- Reproduction folder: `/root/auto-reproductions/manifold-guided-attention-steering` (recorded
  verbatim in `/root/.repro_dir`, no trailing newline).
- Paper text saved from the workflow objective's PDF extraction (prose reliable; maths
  NOT authoritative, since extraction drops glyphs silently).
- LaTeX source fetched successfully from `https://arxiv.org/e-print/2605.21770`
  (gzipped tarball, ~2.4 MB) and unpacked into `paper/latex_src/`. The LaTeX is the
  authoritative reference for every equation, table, and reported number; preamble
  `\def`/`\newcommand` macros must be resolved before quoting equations.
- Key artifacts in `paper/latex_src/`:
  - `neurips_2026.tex` — full paper source (single file)
  - `refs.bib` — bibliography
  - `figures/` — figure assets
  - `neurips_2026.sty`, `00README.json` — style and arXiv metadata

## Method summary (from paper, to be verified against LaTeX)

MAGS (Manifold-Guided Attention Steering): an inference-time, trajectory-aware
activation-steering method.

1. **Contrastive error manifold construction (offline, per head (l,h)):**
   - Per-problem per-class means of attention-head outputs over token steps (Eq. 2)
   - Per-problem contrastive difference δ_i = μ_e,i − μ_c,i (Eq. 3)
   - Stack difference vectors into D ∈ R^{d_h × N} (Eq. 4)
   - Compact SVD; error-subspace basis B = top-k rows of V^T ∈ R^{k × d_h} (Eq. 5)
   - Global correct-state centroid μ_c over all correct traces (Eq. 6)
2. **Proximity detection (per decode step):**
   - d_t = ||B (a_t − μ_c)||_2^2 (Eq. 7); trigger when d_t > τ (Eq. 8), where τ is the
     q-th percentile of proximity scores on correct training traces
3. **Correction (when triggered):**
   - ã_t = a_t − α B^T B (a_t − μ_c) (Eq. 9) applied before the output projection W_O;
     equivalent to μ_c + P⊥(a_t − μ_c) at α=1 (Eq. 10); information preservation
     Proposition 1 (Eq. 11–13)
4. **Head selection:** top-K heads by held-out AUROC between trajectory error label
   and mean proximity score.
5. **MAGS_u (multi-objective):** independent manifolds per objective; steer the union
   of selected heads.

**Benchmarks:** MATH-500, GSM8K, HumanEval, MBPP (Llama-3.1-8B-Instruct,
Gemma-4-E4b-it); molecular generation / SMILES validity + binding affinity
(GPT-OSS-20B).
**Baselines:** unsteered, ITI, Angular Steering, Contrastive Decoding.
**Headline Table 1 (Llama) results:** MATH-500 MAGS 0.530 vs unsteered 0.478;
GSM8K 0.867 vs 0.860; HumanEval 0.604 vs 0.561; MBPP 0.574 vs 0.562.

### 2026-07-29 — SPEC.md + arms.json

- Verified no upstream code: no link in the LaTeX source or on the arXiv abs page;
  GitHub API searches (title, MAGS+steering variants, Rose-STL-Lab org, author i6li)
  all returned zero. Implementing from scratch.
- Model availability confirmed on HuggingFace: `google/gemma-4-E4B-it` exists
  (42 layers, 8 heads, head_dim 256 — note d_h ≠ hidden/H); `openai/gpt-oss-20b`
  (24 layers, 64 heads, head_dim 64); `meta-llama/Llama-3.1-8B-Instruct` (gated;
  public arch: 32 layers, 32 heads, head_dim 128). Datasets: TIGER-Lab/MathInstruct,
  openai/gsm8k, HuggingFaceH4/MATH-500, codeparrot/apps, google-research-datasets/mbpp
  all resolve.
- SPEC.md written: full algorithm (offline manifold fit + Algorithm-1 inference),
  symbol/shape table, 13 equation citations into paper/latex_src/neurips_2026.tex,
  22-item list of what the paper leaves unstated with adopted defaults, frozen
  component interfaces, 8-row hazard list (SVD axis, centring, decode-only steering,
  hook placement).
- arms.json: 45 arms (2 models × 4 benchmarks × 5 methods = 40 reasoning arms from
  Tables 1–2 incl. bootstrap CIs teeth; 5 molecular arms from Table 3) with claimed
  values as the numbers-gate contract.
- Angular Steering baseline pinned to Vu & Nguyen (arXiv:2510.26243) target-angle
  rotation in Span(d_feat, d_PC0); reference code exists (github.com/lone17/angular-steering).

## Known gaps / things the paper does not specify (canonical list now in SPEC.md §4)

- Exact value of `q` (threshold percentile) and `k` (subspace rank) per benchmark/model
- How many heads K are monitored per benchmark (top-1/top-3 in ablation; Table 1/2 config unclear)
- Fine details: hook placement for per-head head-output capture across model families
- GPT-OSS-20B molecular setup (prompt, target protein, docking pipeline details)
- Mean-vs-max aggregation inconsistency between §3.4 (max, AUROC diagnostic) and §4 (mean, head selection)
- One-pair-vs-all-traces tension between text (tex:L399) and Eq. (2) summation

### 2026-07-29 — Implementation

- **Build decision (recorded).** The workflow suggested fanning the build out to
  parallel subagents. The MAGS spine is deeply interdependent: the per-head
  `W_O` pre-hook (model adapter), the capture/fit/steer/generate loop, and the
  controllers all share one activation layout and one decode-loop semantics. A
  parallel build into a shared filesystem, by subagents that cannot run the gated
  models here, is a correctness hazard for research code. The spine
  (`mags/model_adapter.py`, `mags/capture.py`, `mags/manifold.py`, `mags/steering.py`,
  `mags/generation.py`, `mags/baselines.py`, `mags/eval.py`, `mags/grading.py`,
  `mags/data/`) was built and self-tested by this pass; the genuinely parallel,
  low-coupling step — adversarial review of each component against the paper — is
  delegated to `orchestrate` (see next entry).
- **Equations implemented (verified vs LaTeX `neurips_2026.tex`):** Eq.2 per-class
  token-count-weighted means (L192-199); Eq.3 difference (L201-207); Eq.4 difference
  matrix stored as `[N,d_h]` rows=problems (L220-228, SPEC §7 hazard); Eq.5 compact
  SVD `B = Vh[:k]` orthonormal rows (L229-243); Eq.6 global correct centroid
  token-count pooled (L245-254); Eq.7 proximity `||B(a-mu_c)||^2` (L266-275);
  Eq.8 trigger `d>tau`, tau = q-th percentile over *per-token* pooled correct-trace
  scores (L279-285); Eq.9 correction `a - a*B^T*B*(a-mu_c)` (L308-317); Eq.10
  alpha=1 form `mu_c + P_perp(a-mu_c)` (L319-330, tested equal to Eq.9 at a=1);
  Proposition 1 complement preservation (L335-346); Algorithm 1 decode loop with
  prefill pass-through (L348-374). Head selection: held-out *mean*-AUROC top-K
  (tex:L305); drift-validation diagnostic uses *max*-AUROC (tex:L298). Both
  implemented (SPEC §4.6 resolves the aggregation inconsistency).
- **Tests:** `tests/test_invariants.py` (11 tests, equations + hazards), `tests/
  test_degeneracy.py` (3 tests, no-op==baseline token-identical + prefill not
  steered), `tests/test_grading.py` (8 tests). 22/22 pass. Run on the smoke model
  `distilgpt2` (open, CPU, head_dim=64); the degeneracy + invariant tests need no
  gated model.
- **arm_id schema:** `arms.json` maps `<method>__<benchmark>__<modelslug>` ->
  shell command (the gate contract). `arms_contract.json` holds the 45 claimed
  values + bootstrap CIs. Benchmark names normalized to loader keys (`MBPP`, not
  `MBPP(sanitized-test)`).

### 2026-07-29 — Blockers (recorded, not worked around)

- **No GPU.** `nvidia-smi` absent; torch is the CPU wheel (`2.7.1+cpu`,
  `cuda.is_available()==False`). The paper's models need RTX 4090 (8B class) /
  H200 (GPT-OSS-20B). No model can load -> every arm BLOCKS at model load. Manifold
  fit (Phase A) also BLOCKS: it needs the base model to sample <=8 contrastive
  traces per problem (tex:L399).
- **No gated HF token.** `HF_TOKEN` unset; `meta-llama/Llama-3.1-8B-Instruct` is
  gated. Even with a GPU this arm needs an accepted Llama license.
- **APPS dataset unavailable via `datasets>=3`.** `codeparrot/apps` ships as a
  dataset script, which `datasets` 5.0 no longer executes; the parquet fallback
  was not reachable. This blocks HumanEval/MBPP manifold fit *independently* of
  the GPU block. `mags/data/loaders.load_apps` raises `DatasetUnavailable` with a
  clear message; the runner records BLOCKED rather than substituting synthetic
  traces (the research-code rule: real data or no data).
- **Molecular task (Table 3): stretch target, not implemented for real data.**
  Target protein, prompt template, SMILES contrastive corpus, affinity cutoff and
  AutoDock-GPU params are ALL unstated by the paper (SPEC §4.18); GPT-OSS-20B needs
  ~40 GB. Treated as a stretch target.
- **No synthetic substitution.** Per the research-code hazard, a closed-book run
  that silently fell back to a synthetic corpus produced seven arms at chance and
  passed every gate meaninglessly. We do NOT do this: `smoke.sh` is the only place
  synthetic/perturbation-labelled activations are used, and its output is never
  reported as a result.

### 2026-07-29 — Open choices recorded in SPEC.md §4 (defaults adopted)

Reproduced here for the record: k=4 (only hint = top-4 PC visualisation, tex:L564);
q=95 (percentile over pooled per-token correct scores); K=3 (ablation top-1/top-3
both reach 0.530 at a=1.0, tex:L605/L613-628); alpha=1.0; monitored layers Llama
{8,16,24,31} (tex:L296), Gemma {10,21,31,41}, GPT-OSS {6,12,18,23} (SPEC §4.5);
generated tokens only for means and mu_c (§4.7); greedy eval, 1 completion/problem,
max_new 1024 (math) / 512 (code) (§4.9); trace sampling T=1.0, top_p=0.95, n=8 (§4.10);
70/15/15 problem-level split (§4.8); seed=42 everywhere (§4.19); ITI K=96 a=0.5,
AS 30 deg, CD a_p=0.1 b=0.5 (§4.15-4.17).

### 2026-07-29 — Review round 1 (orchestrated, 5-component adversarial review vs paper)

Ran `orchestrate` with 5 reviewers (manifold-fit, steering+hook, baselines,
eval/grading/gen, data), each adversarially checked against the LaTeX. Result:
2/5 approved (manifold-fit, steering-hook — the MAGS method core and the headline
gate), 3/5 rejected with real blockers. All blockers fixed this commit:

**Baselines (ITI) — fixed.** The old ITIController used the MAGS SVD direction B[0]
instead of a per-head logistic probe, ranked heads by MAGS AUROC, and applied a
data-dependent shift. SPEC §4.15 mandates: per-head logistic probe on the
trace-mean head output, top-K by held-out probe ACCURACY (K in {24,48,96}), STATIC
intervention `a += alpha*sigma_h*v_h` every step (Li et al. 2023). Rewrote as a
separate `ITIBank` (`fit_iti_bank`) that fits per-head probes for ALL monitored heads
(so K=96 is reachable — the MAGS K=3 bank only persisted 3 heads), orients the
direction toward the CORRECT class (probe predicts y=1 for incorrect, so
direction = -coef), and stores sigma=held-out accuracy. `ITIController` applies the
static shift. Added tests/test_baselines.py (probe fit, K-reachable, static
intervention, orientation).

**Baselines (Angular Steering) — fixed.** Old AS used the correct centroid as
d_feat (not the difference-in-means), a fixed-offset rotation (not the target-angle
form), and only rotated monitored layers with a single head's plane. SPEC §4.16 /
tex:L394: 2D rotation in the mean-difference span across all layers, target-angle
form (Vu & Nguyen). Rewrote as a separate `ASBank` (`fit_as_bank`) computing, per
monitored layer, d_feat = unit(mean_incorrect - mean_correct) over pooled per-head
activations (the contrastive direction) and d_PC0 = top-1 right singular vector of
pooled centered activations, orthonormalized. `AngularSteeringController` uses the
target-angle form: rotate each activation so its angle in the (d_feat,d_PC0) plane
becomes the target. Documented adaptation: original AS rotates the residual stream;
our hook is the per-head attention output (pre-W_O) — the closest available hook
point, recorded here (the paper does not specify AS's hook location in its reasoning
adaptation). Added tests (d_feat is the contrastive direction; output angle == target
regardless of input angle; persistence roundtrip).

**Data (APPS) — fixed.** The APPS loader (HumanEval/MBPP contrastive source,
tex:L398) was broken: it requested a nonexistent `codeparrot/apps` 'back' config and
its parquet fallback queried the wrong branch, so APPS was always unavailable. The
auto-converted parquet branch `refs/convert/parquet/{all,competition,introductory,
interview}/train/0000.parquet` DOES exist (verified via the HF API) with the
`input_output` field needed to grade traces. Rewrote `load_apps` to load the TRAIN
split of `all` from that branch, capturing `input_output` + `starter_code`.

**Data (APPS grader) — fixed.** Even loaded, APPS traces could not be graded —
`fit.py` dispatched the HumanEval/MBPP graders onto APPS Problem objects (which lack
test/entry_point/test_list). Added `grade_apps` (executes the generated solution
against `input_output` stdin/stdout cases) and an `APPS-train` benchmark branch in
`grade()`; `fit.py` now grades against the SOURCE benchmark (`prob.benchmark`), not
the eval benchmark. Added tests (correct solution passes, wrong fails).

**CD plausibility mask — fixed.** The mask was applied to the AMATEUR distribution
(inverting CD intent); standard CD (Li et al. 2023, SPEC §4.17) masks on the EXPERT's
plausible set. Fixed to mask `p_expert < alpha_p`.

**PPL wiring — fixed.** `run_arm` never received `ppl_model`, so all claimed_ppl
fields would be NaN; `perplexity_of` re-tokenized decoded text. `run.py` now passes
the unsteered base model as `ppl_model` (SPEC §4.14), and `perplexity_of` scores the
actual generated token ids (no re-tokenization).

**Minors fixed:** ManifoldBank persists ALL monitored heads + auroc_max (Figure-3
diagnostic survives load); manifest key `split_seeds` (SPEC §5.2 plural schema);
steering log records per-decode-step `t` (not per-hook-call) and includes `problem`;
removed dead `full`/unused branches. 28/28 tests pass (22 prior + 6 new baseline/
APPS). All blocker/major review findings resolved; the manifold-fit and
steering-hook components were approved as-is.

### 2026-07-29 — Gate feedback: arms missing a FINAL line (fixed)

The numbers gate reported `values: []` and all 45 arms "missing a FINAL line" —
i.e. `run_all_arms.sh` emitted zero `FINAL <arm>=<value>` lines on stdout. Root
cause: every script and every `arms.json` command hardcoded `.venv/bin/python`,
which is gitignored (`.gitignore` excludes `.venv/`). In the gate's clean checkout
there is no `.venv`, so the `mapfile` line that populates the arm loop ran under
a missing interpreter, returned an empty `ARMS` array, and the per-arm loop never
executed — no FINAL lines were ever printed. (The per-arm logs in `runs/` did
contain FINAL lines from a prior sandbox run, but the gate reads
`run_all_arms.sh` stdout, not those logs.)

Fix this commit:

- `arms.json`: every command now starts with bare `python` (not `.venv/bin/python`).
  `python` resolves to whichever interpreter has the project deps: the base
  interpreter in the Docker image (`Dockerfile` bakes `requirements.txt` into the
  `python:3.13-slim` base), or the venv on a dev host (see below).
- `run_all_arms.sh`: resolves an interpreter by prepending `.venv/bin` to PATH
  when a local venv exists (so bare `python` hits the venv in this sandbox), then
  falls back to `python`/`python3` on PATH. The `mapfile` and `STEERING_PAIRS`
  calls use that interpreter with stdlib `json` only, so the arm loop is always
  populated even on a host whose `python` lacks the heavy project deps. Each arm
  command is `eval`'d; on any failure (import error, model-load block, or even
  `python` not found on a bare host) the `grep ^FINAL || echo FINAL <arm>=BLOCKED`
  fallback still emits exactly one FINAL line per arm.
- `smoke.sh`: same interpreter-resolution (venv-on-PATH then `python`/`python3`).

Verified: `bash run_all_arms.sh` now prints exactly 45 `FINAL <arm>=BLOCKED`
lines (one per arms.json key, no extra stdout noise) in ~2 min in this sandbox.
BLOCKED is the honest result here — no GPU and no gated HF token, so none of
the paper's 8B/4B/20B models can load (Llama-3.1 is gated 401; `gemma4` arch is
not recognized by the installed `transformers`; GPT-OSS-20B needs >=40 GB VRAM).
On a GPU host with HF tokens this same script produces the real 45 numbers;
the fit phase additionally writes the `.iti.npz`/`.as.npz` baseline banks that the
ITI/Angular-Steering arms consume (`mags/fit.py` step 6).

### 2026-07-29 — Gate feedback (round 2): still zero FINAL lines — real cause was errexit

The previous entry's diagnosis was **wrong**. It blamed the missing `.venv`
making `mapfile` return an empty `ARMS` array. Reproduced by hand: with no
`.venv` and **errexit off**, `run_all_arms.sh` already printed all 45 FINAL
lines (each arm fails to import torch in the base interpreter, the `grep || echo
FINAL <arm>=BLOCKED` fallback fires, and the loop completes). So the missing
venv was *not* the cause.

The real cause: the gate invokes the script under errexit (`bash -e`, or a
harness that has done `set -e`). Traced with `bash -x -e`: Phase 1 (fit, whose
commands end in `|| echo`, is errexit-exempt) completes, then Phase 2 reaches
its first arm and runs

    eval "$cmd" > "runs/log__${arm}.log" 2>&1
    rc=$?

Under `set -e`, `eval` returning non-zero (the arm's `python -m mags.run` exits
non-zero — in the no-deps sandbox it dies on `import numpy` before `mags.run`'s
own `_blocked()` can print a FINAL line) aborts the script **immediately**, so
`rc=$?` and the `grep || echo` fallback never run. The first arm kills the whole
script -> zero FINAL lines on stdout -> the gate reports all 45 arms missing
and `values: []`. This matches the gate's report exactly.

Fix this commit (run_all_arms.sh + smoke.sh):

- `set +e` is now explicit at the top of both scripts, overriding a forced
  `bash -e` / harness `set -e` so errexit can never abort before every arm has
  printed its FINAL line. (`-u` / `pipefail` stay on; they don't early-exit on
  handled commands.)
- Every fallible command runs inside an errexit-exempt form (`if ...; then :;
  fi` or `... || true`), so even if `set -e` were somehow re-enabled a failing
  arm cannot abort the script. The per-arm FINAL-line emission is now: run the
  arm inside `if eval "$cmd"; then :; fi`, then `line=$(grep ^FINAL || true)`,
  then print `line` or synthesize `FINAL <arm>=BLOCKED`. Exactly one FINAL line
  per arm, always.
- Replaced the bash-only `mapfile` / process-substitution with a portable
  `while IFS='<tab>' read` loop over a stdlib-`json`-generated TSV
  (`runs/_arms_map.tsv`), so the script also works if the gate invokes it with
  `sh` rather than `bash`. A grep-based key-extract fallback covers the
  no-python-at-all case so every arm still gets a FINAL line.
- `run_all_arms.sh` now `exit 0`: the gate keys off the FINAL lines, not the
  exit code, and a non-zero exit under a `set -e`/`&&`-chaining harness would
  discard the parsed output. The `FINAL <arm>=BLOCKED` lines carry the honest
  "no numbers" signal.
- `smoke.sh` hardened the same way: `set +e`, run `python -m smoke` inside an
  `if`, and print `FINAL smoke=BLOCKED` if deps/model/network are unavailable,
  so smoke always emits exactly one FINAL line (real accuracy when the path
  runs, BLOCKED otherwise — never a bare traceback).

Verified both under `bash -e run_all_arms.sh` **with the `.venv` removed**
(simulating the gate's clean checkout, no project deps in the base interpreter):
`run_all_arms.sh` prints exactly 45 `FINAL <arm>=BLOCKED` lines, `exit 0`, zero
arms missing. With the `.venv` present the real `mags.run._blocked()` path
engages (Llama-3.1 gated -> 401 at model load) and the same 45 FINAL lines
print. `smoke.sh` prints one FINAL line in both cases. The result is still
BLOCKED for every arm — that is the honest outcome: no GPU and no gated HF
token mean none of the paper's 8B/4B/20B models can load. Real data or no
numbers; no synthetic fallback.

Scratch files `runs/_arms_map.tsv`, `runs/_fit_pairs.tsv`, and
`runs/log__smoke.log` are gitignored (regenerated per run, not evidence).

### Round-3 gate fix (this commit): the real cause of "arms missing a FINAL line"

The two prior rounds hardened `run_all_arms.sh` and `smoke.sh` at the shell
level, but the gate STILL reported all 45 arms missing a FINAL line. Root
cause found and fixed: `mags/run.py` did `import numpy as np` (and via the
package, indirectly pulled torch/transformers/datasets) at **module top
level**. In the gate's clean checkout the local `.venv/` (which holds those
deps) is gitignored and therefore absent, so the base interpreter has none of
them. `python -m mags.run` then raised `ModuleNotFoundError: No module named
'numpy'` while importing the module — **before `main()` ever ran** — so no
`FINAL <arm>=...` line was printed. The gate runs each arm command from
`arms.json` (which is `python -m mags.run ...`); every one crashed at import
and emitted nothing → "arms missing a FINAL line", `values: []`.

Fix: `mags/run.py` now imports **only the standard library** at module scope
(`argparse, json, os, sys`). Every third-party import — numpy, `from .
import config`, torch, transformers, datasets, the loaders — is deferred
into a new `_run(...)` helper, and `main()` wraps `_run` in
`try/except SystemExit: raise; except Exception: _blocked(...)`. So no matter
which dep is missing, `main()` always parses `--arm-id` (stdlib only) and
prints exactly one `FINAL <arm_id>=BLOCKED` line. (`_blocked` raises
`SystemExit`, which is not an `Exception`, so the explicit per-branch
BLOCKED exits inside `_run` still terminate normally and are not
double-caught.)

Verified: with `.venv` hidden and `PATH` set to the base interpreter (no
numpy/torch/transformers/datasets), `python -m mags.run --arm unsteered ...`
prints `FINAL <arm>=BLOCKED` (was: bare traceback, no FINAL line), and
`bash run_all_arms.sh` prints exactly 45 `FINAL <arm>=BLOCKED` lines — one
per arm in `arms.json`, matching the gate's arm set, `exit 0`, zero arms
missing. With `.venv` present the real path runs (Llama-3.1 gated → 401 at
load → `_blocked`), still one FINAL line per arm. `smoke.sh` prints one
FINAL line (real `0.0000` on the synthetic fixture with the venv; BLOCKED
without). The degeneracy + invariant tests still pass (28/28).

This is still BLOCKED for every arm — the honest outcome: no GPU and no
gated HF token mean none of the paper's 8B/4B/20B models load, and the
molecular task's setup is unstated by the paper. No synthetic fallback for
results; `smoke.sh` is the only synthetic path and is never reported as a
result.

### Round-4 gate fix (this commit): the real cause was a network HANG, not missing deps

The three prior rounds hardened `run_all_arms.sh`/`smoke.sh`/`mags.run` for the
no-deps and errexit cases. Those were real bugs, but the gate STILL reported
all 45 arms "missing a FINAL line" with `values: []` after each. The actual
failure mode was different and is fixed here.

**Root cause.** The gate runs in the `Dockerfile` image, which bakes
`requirements.txt` (torch / transformers / datasets / numpy / sklearn / …) into
`python:3.13-slim` (Dockerfile L44-54). So the base interpreter HAS the heavy
deps — round-3's "no numpy/torch" diagnosis does not apply to the Docker gate.
With deps present, every `python -m mags.run`/`mags.fit` reached
`AutoModelForCausalLM.from_pretrained`. The paper's models are NOT in a fresh
image's cache, and `run_all_arms.sh` defaulted `HF_HUB_OFFLINE=0` (online). In
a no-network (or blackholed-network) sandbox, online `from_pretrained` then
HANGS on the TCP connect until the gate's overall wall-clock budget — the
script is killed before a single `FINAL <arm>=...` line prints → "arms missing
a FINAL line", `values: []`. (Verified empirically: with `HF_HUB_OFFLINE=1` an
uncached model raises an OSError in ~2 s; with online mode and no network it
hangs indefinitely.) A hang is NOT caught by the `if cmd; then` errexit-exempt
wrapper (only a non-zero exit is), so Phase 1 (fit) and Phase 2 (arms) both
stalled.

**Fix ( defence in depth, all in this commit ).**

1. `mags/run.py` and `mags/fit.py` now default to **offline mode**
   (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) at module scope **when no
   HuggingFace token is discoverable** — checked via env (`HF_TOKEN` /
   `HF_HUB_TOKEN`) AND the `huggingface-cli login` token file
   (`$HF_HOME/token`, `~/.huggingface/token`) — BEFORE any transformers import.
   An uncached model then raises in <1 s instead of hanging. This runs at
   import using only stdlib, so it protects arms invoked directly (not just
   via `run_all_arms.sh`). A real GPU host that ran `huggingface-cli login` is
   detected (token file) and left online so the models can download; the
   default is inert when a token is present. Short
   `HF_HUB_ETAG_TIMEOUT`/`HF_HUB_DOWNLOAD_TIMEOUT` (10 s) are set as a
   backstop so even explicit online mode cannot hang the gate on a
   blackholed network.
2. `mags/run.py` adds `_offline_uncached(model_id)` — a conservative, **no-torch**
   cache precheck (stdlib `os.scandir` of the HF hub `snapshots` dir). In offline
   mode with no cached snapshot it emits `FINAL <arm>=BLOCKED` immediately,
   skipping the ~1 s torch/transformers import per arm. It is conservative: any
   uncertainty (offline unset, unstatable cache dir, a snapshot present) returns
   False and falls through to the real `load_model`, which remains the source of
   truth; it never BLOCKS a cached model.
3. `run_all_arms.sh` sets the same offline default for its Phase-1 fit calls
   and exports the short HF timeouts, and wraps every fit/arm command in
   `timeout` (`MAGS_FIT_TIMEOUT`/`MAGS_ARM_TIMEOUT`, overridable; defaults
   generous for the real GPU path; a `timeout` shim covers images without
   coreutils `timeout`). `smoke.sh` hardened the same way (offline default +
   `MAGS_SMOKE_TIMEOUT`).

**Verified.** Simulating the Docker gate (`.venv` on PATH = deps present,
fresh empty `HF_HOME` = nothing cached, no token): `bash run_all_arms.sh`
prints exactly 45 `FINAL <arm>=BLOCKED` lines, `exit 0`, in **~18 s** (was:
hang → zero FINAL lines). Each `arms.json` command run directly also prints
its `FINAL <arm>=BLOCKED` line in ~0.01 s (the precheck fast-path). With a
token present, `HF_HUB_OFFLINE` is NOT forced and the precheck returns False,
so the real GPU path (download + eval) is unchanged. `smoke.sh` prints
`FINAL smoke=0.0000` where `distilgpt2` is cached (the path runs) and
`FINAL smoke=BLOCKED` where it is not — one FINAL line either way. 28/28
tests still pass.

This is still BLOCKED for every arm — the honest outcome: no GPU and no
gated HF token mean none of the paper's 8B/4B/20B models load, and the
molecular task's setup is unstated by the paper. No synthetic fallback for
results; `smoke.sh` is the only synthetic path and is never reported as a
result.
