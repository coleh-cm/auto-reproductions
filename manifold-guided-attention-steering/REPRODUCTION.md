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

### 2026-07-29 — Setup

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
