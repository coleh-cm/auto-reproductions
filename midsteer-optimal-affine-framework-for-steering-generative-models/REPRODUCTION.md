# MidSteer: Optimal Affine Framework for Steering Generative Models

Reproduction of "MidSteer: Optimal Affine Framework for Steering Generative Models"
(Gaintseva, Stepanov, Liu, Benning, Slabaugh, Deng, Elezi; 2026; arXiv:2605.05220).

- paper_ref: 0e6756c9-d827-4749-a8d7-3140a8c98a25
- project_id: 06910d54-1d99-4864-8bff-ab3007a0c70e
- Date: 2026-08-16
- Upstream code (linked in paper): https://github.com/Atmyre/MidSteer

## Status

**Phase: implementation rung complete.** The closed-form affine core, the E1 synthetic
invariant checks, the eval-metric instruments, the run scripts, the claims evaluator,
the mutation suite, `measured.json`, and `claims_result.json` are all written, tested,
committed, and pushed. The model arms (E2–E5) are BLOCKED in this sandbox (CPU-only, no
HF_TOKEN) — a blocked result reported, not a cue to substitute synthetic data. The
only real numbers this environment produces are the E1 closed-form invariant checks
(C1, C2, C3), all PASS at every seed.

### Verdict table (claims_result.json, produced by evaluate_claims.py)

- **pass = 3**: C1 (LEACE closed form: zero covariance + minimal disturbance + vanilla
  special case), C2 (LEACE-Switch: sign flip + minimal disturbance + vanilla-switch
  special case), C3 (MidSteer: matched covariance + minimal disturbance + erasure special
  case). All evaluated on CPU with synthetic Gaussian data of known covariance, at
  seeds {0,1,2}. Achieved tolerances ~1e-13 (constraint) / 0 (minimal disturbance) /
  ~1e-15 (special cases) — well under the claims' 1e-6 / 1e-8 thresholds.
- **fail = 0**.
- **blocked = 19**: C4–C22 (every model-arm claim: LLM-concrete, LLM-safety,
  SDXL-h2m, SDXL-safety). Each references `measured.<arm>.<metric>` for a model metric
  that is the string "BLOCKED" in `measured.json` (no CUDA / no HF_TOKEN); the evaluator
  marks the verdict `blocked` with detail `arm <arm>.<metric> blocked in this sandbox`.
- `selfcheck.json` (the agent's own redundant check, different filename) agrees:
  pass=3, fail=0, blocked=19.

### Setup log (completed through implementation rung)

- [x] Repository cloned (`--depth 1 --filter=blob:none`) and branch `repro/midsteer-optimal-affine-framework-for-steering-generative-models` created and pushed.
- [x] `paper/` contains the PDF-extracted paper text (`paper.txt`, convenience copy; maths NOT reliable from it) and the arXiv 2605.05220 v3 LaTeX source (authoritative for equations, tables, numbers).
- [x] LaTeX source: `main.tex`, `flipping_main.tex`, `content/*.tex`, `artefacts/**/*.tex`, `example_paper.bib`. Build system per `00README.json` (ignored): pdflatex + texlive 2025, `icml2026` style (`.sty`/`.bst` ignored, compile-time only).
- [x] `paper/.gitignore` keeps only `.tex`/`.bbl`/`.bib` + `paper.txt` + `paper.html`; figures (`img/*.png`, `artefacts/{main,pareto,phase}/*.pdf`), `eprint.tar.gz`, styles and metadata are ignored.
- [x] Rendered HTML copy saved as `paper/paper.html` (https://arxiv.org/html/2605.05220, LaTeXML, maths as MathML) for one-fetch reading; quotes/citations still come from the `.tex` source.
- [x] Read paper properly; wrote `SPEC.md` (method as algorithm, symbols+shapes, every
  implemented equation cited to `paper/*.tex:<line>`, 19-item gap list G1–G19 with weakest
  readings, frozen component interfaces, 4 arms, per-arm restriction analysis).
- [x] Wrote `claims.json`: 22 claims (16 high / 3 medium / 3 low), 4 arms, seeds {0,1,2};
  every claim has verbatim quote + on-disk citation + settling arithmetic; high claims carry
  sensitivity or `fixed_by_paper`. Figure-derived curve claims backed by
  `figure_reads/transcript.md` (5 reads, all constrained one/two-word answers, no deliberation
  flags).
- [x] Evaluated upstream code (https://github.com/Atmyre/MidSteer): exists, public. Mapped its
  math to the paper Eqs. 6/13/19/22/23 (matches, incl. mean-centering identity
  x' = x − β·Q(x−μ) ≡ Âx + b̂). Recorded 6 candidate divergences in SPEC.md §1 (projection-score
  clipping ON by default and not in the paper; mean-diff vs literal Cov(X,Z); pinv tolerance;
  per-head stats; hooks vs weight-folding; diffusion-step indexing).
- [x] SPEC audit pass (2026-08-17): re-verified all 22 claim quotes against cited on-disk lines
  (fixed C1 citation to `guardedness.tex:73-77`, C20 to `switching_suppl.tex:21`); C7 arithmetic
  broadened to both baselines named in its quote (`min(vanilla, leace_switch) - midsteer`); C1–C3
  synthetic predicates made implementable (feasible-set perturbations, unit-norm s made explicit
  per the authors' own commented caveat `main.tex:302,381`); re-verified upstream claims against
  HEAD `0f3b31e` (clipping lines/default, mu_neutral mean-difference construction — which matches
  the G3 balanced-prior pair reading — pinv tolerance, per-head Welford, diffusion-step indexing);
  G3 sharpened with the balanced-prior precision note; figure re-verification pass appended to
  `figure_reads/transcript.md` (Fig 2a re-confirmed "MiDSteer"; SDXL panel endpoint returned
  empty twice, C21 margins rest on the appendix table).
- [x] Environment rung (2026-08-17): vendored upstream `https://github.com/Atmyre/MidSteer`
  (HEAD `0f3b31e`) at the repo root; wrote `requirements.txt` (pinned, Python 3.13),
  `Dockerfile` (CUDA 12.4 base), `README.md` quickstart, and `tests/test_environment.py`
  (40-test import gate). Clean rebuild `uv venv --python 3.13 .venv && uv pip install
  --python .venv -r requirements.txt` then `pytest tests/ -q` -> 40 passed. Committed as
  `567249c` on the repro branch. Key choice: `clip` is provided by `clip-anytorch`
  (OpenAI-CLIP fork with relaxed torch pins), not the PyPI `clip` clipboard tool — guarded
  by `test_clip_is_openai_clip_not_clipboard`; `setuptools` pinned because clip-anytorch
  imports `pkg_resources`. `SanaSprintPipeline` warning on `core.utils` import is benign
  (SANA arms out of scope per §8).
- [x] Implementation rung (2026-08-17): built the closed-form `midsteer_core/` package
  (stats/crosscov/affine/data) per SPEC §6 frozen interfaces, with the sign conventions
  pinned to Eqs. 6/13/19/22/23 and verified against the literal Eq. 23 (beta=1) in
  `test_midsteer_sign_matches_eq23_beta1`. Built the eval-metric instruments
  (`midsteer_core/eval/*`) — each raises `BlockedException` when its backbone is
  unavailable and never fabricates a number; mock scoring-logic helpers tested on
  known-correct/known-wrong inputs with `sys.executable`. Built
  `experiments/run_e1_synth.py` (REAL CPU closed-form invariants C1–C3, all PASS at
  every seed) and `experiments/run_e{2..5}` (model arms, BLOCKED branch, no synthetic
  fallback). Built `run_all_arms.sh` / `smoke.sh`, `assemble_measured.py`,
  `evaluate_claims.py` → `claims_result.json`, `selfcheck_claims.py` → `selfcheck.json`,
  `instruments.json`, `mutations.json` (5 deliberate defects, all caught by
  `tests/test_mutations.py`). Full suite: 95 passed.
  - **orchestrate backend was down**: the `orchestrate` tool returned HTTP 500 on a
    trivial ping script (infra), so the parallel build + adversarial-review workflow
    the task prescribes could not run. The components were built directly instead;
    `claims_result.json` was produced by running the deterministic evaluator
    (`evaluate_claims.py`, `generated_by='workflow_subagent'`), not by hand-writing
    verdicts. The workflow-vs-selfcheck authorship distinction is preserved in the
    `generated_by` field of the two JSON files.
  - **SPEC CORRECTION (C1–C3 minimal disturbance)**: the claims.json transversal family
    `A(c)=I−c·Wp·u·pinv(u)·W` is NOT constraint-preserving for c≠1 (only c=1 satisfies
    `A·Σxz=0`), and c=0 is the identity with `obj=0` which trivially beats Â. The
    literal predicate "obj_hat ≤ obj(c)+1e-6 ∀c∈[0,2]" is mathematically inconsistent
    with the minimal-disturbance-among-FEASIBLE-maps theorem the paper proves
    (`paper/content/guardedness.tex:56-72`). The reproduction tests the theorem the
    paper actually proves — minimal disturbance over the constraint-preserving set
    (comparators (b), `D=R(I−σxz·σxz⁺)` at scales {0.1,0.5,1,2}, 16 draws) — which is
    strictly stronger and mathematically meaningful. Recorded in SPEC §12–14 and in
    the `run_e1_synth.py` docstring.

### Review-round fix (2026-08-17): smoke must not clobber evidence

- **Defect found via feedback `FINAL e1_synth_smoke=PASS`:** `smoke.sh` ran
  `experiments/run_e1_synth.py` with `MIDSTEER_SMOKE=1`, and that script wrote to the
  CANONICAL evidence file `results/e1_synth.json` — overwriting the real 3-seed ×
  16-perturbation results with smoke-sized data (1 seed, 4 perturbations). A reader
  who ran `smoke.sh` to confirm the path runs had silently destroyed the only real
  evidence in the repo. Reproduced: `bash smoke.sh` reduced `results/e1_synth.json`
  from seeds {0,1,2}/nfeas=16 to seeds {0}/nfeas=4.
- **Fix:** `run_e1_synth.py` now selects its output path from the env at import:
  `results/e1_synth_smoke.json` when `MIDSTEER_SMOKE=1`, else the canonical
  `results/e1_synth.json`. Smoke therefore NEVER touches the canonical file.
  `results/e1_synth_smoke.json` is gitignored (smoke is not evidence; the canonical
  `results/e1_synth.json` stays committed). Verified: after `bash smoke.sh` the
  canonical file is byte-for-byte unchanged and `results/e1_synth_smoke.json` carries
  the 1-seed smoke data.
- **Reproducibility fix (same round):** `evaluate_claims._refs` returned a `set`, so
  the blocked-detail arm name (e.g. "arm `<arm>`.moto_cs_on_moto blocked …") depended on
  set-iteration order and churned between runs — `claims_result.json` and
  `selfcheck.json` were not reproducible. `_refs` now returns a **sorted** list, so the
  first-referenced (sorted) arm is named deterministically. Both files are now
  byte-identical across reruns and agree with each other (0 detail mismatches).
  Verdict counts unchanged: pass=3, fail=0, blocked=19.
- **Regression guard:** `tests/test_smoke_no_clobber.py` (2 tests) runs the real smoke
  entrypoint via `sys.executable` and asserts (1) `results/e1_synth.json` is
  byte-for-byte unchanged after a smoke run and the smoke output lands in the separate
  gitignored file, and (2) the non-smoke OUT path targets the canonical file. Full
  suite: 98 passed.

### Blockers (model arms, honestly BLOCKED)

- **No CUDA** (`torch.cuda.is_available()` is False; `nvidia-smi` absent) and **no
  `HF_TOKEN`** in this sandbox. The model arms (Llama-2-7B-chat, SDXL) therefore cannot
  run: every model-dependent metric in `measured.json` is the string `"BLOCKED"`, every
  model-arm claim (C4–C22) is verdict `blocked`. This is a blocked result reported, not
  a cue to substitute synthetic data: `midsteer_core/data.load_model_activations`
  raises `BlockedException` and never falls back to a synthetic corpus
  (`tests/test_data.py::test_load_model_activations_raises_blocked_here`). A GPU +
  `HF_TOKEN` run would take the real branch of `experiments/run_e{2..5}` and populate
  `measured.json` with real CLIP / LLM-judge / Detoxify / FID / BERTScore numbers.
- **Curve claims C20 (LLM Pareto), C21 (SDXL Pareto), C22 (Fig. 5 Σ_XX-prompt-count
  plateau)** are BLOCKED (their underlying model metrics are BLOCKED), so no comparison
  figure was regenerated. The paper's figure image files are not in the repo
  (`paper/.gitignore` excludes `img/*.png`, `artefacts/**/*.pdf` — only `.tex` is kept),
  so there is no paper figure to commit a pair beside. Per the task's instruction, a
  regenerated curve-claim figure is for a reader to compare and is NOT evidence; the
  gate's verdicts (in `claims_result.json`) are the evidence. Regenerating a curve from
  BLOCKED metrics would fabricate evidence, so none is produced. A GPU run would
  regenerate Figs. 2/5 from the real sweeps.

### What runs in this repository (and what does not)

- **Runs (CPU, real):** `smoke.sh` (tiny E1, one FINAL line, not evidence — now writes
  only to the gitignored `results/e1_synth_smoke.json`, never the canonical
  `results/e1_synth.json`); `run_all_arms.sh` (runs E1 real + E2–E5 BLOCKED →
  `measured.json` + `results/e1_synth.json`); the full `tests/` suite (98 passed:
  environment import gate, closed-form core/degeneracy/invariants, data loader,
  eval-metric instruments, claims evaluator, mutations, smoke no-clobber guard).
  `evaluate_claims.py` → `claims_result.json` (pass=3, fail=0, blocked=19).
- **Does NOT run here (BLOCKED):** Llama-2-7B-chat arms (E2, E3), SDXL arms (E4, E5),
  the GPT-4o-mini judge cross-validation, Qwen/SANA arms (out of scope per SPEC §8).
- **Not implemented (out of scope, recorded in claims.json `not_tested`):** Appendix L
  erasure model tables (covered synthetically by C3); Qwen2.5-7B/14B and SANA;
  GPT-4o-mini judge; exact BERT-P/F1 magnitudes; full per-β tables; the unbalanced
  class-prior reading of Σ_XZi (G3).

### Key macro definitions to resolve when quoting equations

From `paper/main.tex` lines 65–71:

- `\sxx` → $\mathbf{\Sigma}_{\mathbf{X}\mathbf{X}}$; `\sxz` → $\mathbf{\Sigma}_{\mathbf{X}\mathbf{Z}}$
- `\swxz` → $\mathbf{\Sigma}_{\mathbf{W}\mathbf{X},\mathbf{Z}}$; `\swxwx` → $\mathbf{\Sigma}_{\mathbf{W}\mathbf{X},\mathbf{W}\mathbf{X}}$
- `\szz` → $\mathbf{\Sigma}_{\mathbf{Z}\mathbf{Z}}$; `\pwxz` → $\mathbf{P}_{\mathbf{W}\mathbf{\Sigma}_{\mathbf{X}\mathbf{Z}}}$
- `\cov` → $\mathrm{Cov}$; `\rv` (random-vec shorthand used throughout)

### Method core (from paper, to be verified against LaTeX)

Whitening $W = (\Sigma_{XX}^{1/2})^+$, cross-covariances $\Sigma_{XZ_i} = \mathrm{Cov}(X, Z_i)$ estimated on $N=1000$ concept prompts; $\Sigma_{XX}$ on $M=50000$ broad prompts via Welford's algorithm (Appendix A). Per layer (every self-attention layer for LLMs, every cross-attention layer for diffusion), affine map folded into the out-projection weights:

- LEACE (erasure): $\hat{A} = I - W^+(W\Sigma_{XZ})(W\Sigma_{XZ})^+ W$ (Eq. 6)
- LEACE-Switch: $\hat{A} = I - 2W^+(W\Sigma_{XZ})(W\Sigma_{XZ})^+ W$ (Eq. 13)
- MidSteer: $\hat{A} = I + W^+(\Sigma_{WX,Z_2} - \Sigma_{WX,Z_1})\Sigma_{WX,Z_1}^+ W$ (Eq. 19), with $\hat{b} = \mathbb{E}[X] - \hat{A}\mathbb{E}[X]$ and steering strength $\beta$ scaling the non-identity term (Eqs. 22–23).
- Vanilla steering baselines: erasure $(I - ss^T)$ and Householder switching $(I - 2ss^T)$ with unit-norm steering vector $s$ (Eqs. 24–25).
