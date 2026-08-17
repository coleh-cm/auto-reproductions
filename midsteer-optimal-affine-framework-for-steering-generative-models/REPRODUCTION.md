# MidSteer: Optimal Affine Framework for Steering Generative Models

Reproduction of "MidSteer: Optimal Affine Framework for Steering Generative Models"
(Gaintseva, Stepanov, Liu, Benning, Slabaugh, Deng, Elezi; 2026; arXiv:2605.05220).

- paper_ref: 0e6756c9-d827-4749-a8d7-3140a8c98a25
- project_id: 06910d54-1d99-4864-8bff-ab3007a0c70e
- Date: 2026-08-16
- Upstream code (linked in paper): https://github.com/Atmyre/MidSteer

## Status

**Rung reached: environment.** Every arm ran this session (see "Arms run" below).
The four **model comparison arms** the paper's claims are about — `base`,
`vanilla`, `leace_switch`, `midsteer` — are **all BLOCKED** (no CUDA, no
`HF_TOKEN` in this sandbox), so this run produces no evidence for or against the
paper's empirical claims (C4–C22). The only real numbers are the E1 closed-form
invariant checks (C1, C2, C3), all PASS at every seed — but those are measured on
synthetic Gaussian data, so they verify the *math* of Eqs. 6/13/19/22/23 and say
nothing about the paper. The build/environment budget was spent
(`$HOME/.build_attempts` = 7) with the arms gate still failing; an all-BLOCKED
arms result is the environment rung and a legitimate outcome. `result_check`
never printed an `AUTHORITATIVE COUNTS` line, so `numbers` was not reached. See
`VERIFICATION.md` for the full check list and what remains untested.

The closed-form affine core, the E1 synthetic invariant checks, the eval-metric
instruments, the run scripts, the claims evaluator, the mutation suite,
`measured.json`, and `claims_result.json` (reproduced=3, refuted=0, untested=0,
blocked=19) are all written, tested, committed, and pushed. The model arms
(E2–E5) are BLOCKED in this sandbox (CPU-only, no HF_TOKEN) — a blocked result
reported, not a cue to substitute synthetic data: `midsteer_core/data` raises
`BlockedException` and never falls back to a synthetic corpus.

### Verdict table (claims_result.json, produced by evaluate_claims.py)

- **reproduced = 3**: C1 (LEACE closed form: zero covariance + minimal disturbance + vanilla
  special case), C2 (LEACE-Switch: sign flip + minimal disturbance + vanilla-switch
  special case), C3 (MidSteer: matched covariance + minimal disturbance + erasure special
  case). All evaluated on CPU with synthetic Gaussian data of known covariance, at
  seeds {0,1,2}. The C1–C3 `predicate` is an executable boolean expression over
  `measured.e1_synth.*` residual thresholds (the per-seed residuals flow into
  `measured.json` from `results/e1_synth.json`), so the verdict is judged from measured
  evidence, not a precomputed pass bool. Achieved tolerances ~1e-13 (constraint) / 0
  (minimal disturbance) / ~1e-15 (special cases) — well under the claims' 1e-6 / 1e-8
  thresholds.
- **refuted = 0**.
- **untested = 0**.
- **blocked = 19**: C4–C22 (every model-arm claim: LLM-concrete, LLM-safety,
  SDXL-h2m, SDXL-safety, and the three curve claims C20–C22). Each references
  `measured.<arm>.<metric>` for a model metric that is the string "BLOCKED" in
  `measured.json` (no CUDA / no HF_TOKEN); the evaluator marks the verdict `blocked`
  with detail `arm <arm>.<metric> blocked in this sandbox`. C20–C22 `quantity`/`against`
  are plain `measured.<arm>.<metric>` refs to per-x SEQUENCES (BLOCKED here), so the gate
  resolves them and returns `blocked` rather than `unevaluable` on a list-comprehension.
- `selfcheck.json` (the agent's own redundant check, different filename) agrees:
  reproduced=3, refuted=0, untested=0, blocked=19.

### Arms run (2026-08-17): measured numbers beside the paper's claims

Every arm was run this session. The gate's spread was printed in `/tmp/arms.log`;
the four **model comparison arms** the paper's claims are about — `base`,
`vanilla`, `leace_switch`, `midsteer` — are **all BLOCKED**. A measurement that
cannot tell those four arms apart has not tested the paper's comparison, whatever
else it shows, so this run produces **no evidence for or against** the paper's
empirical claims (C4–C22). The only real numbers this sandbox produced are the
E1 closed-form invariant checks, which verify the *math* of Eqs. 6/13/19/22/23 on
synthetic Gaussian data of known covariance — not the paper's Llama-2-7B / SDXL
results, and not on the paper's dataset. State that plainly: **the E1 numbers
below are measured on a synthetic stand-in for the paper's activations, so they
say nothing about the paper; they only check that the closed-form affine maps are
built to the equations the paper proves.**

Exact command that produced every number below (run this session, deterministic
across reruns — the re-run reproduced `FINAL e1_synth=1.655510e-12` byte-for-byte):

```bash
bash run_all_arms.sh          # E1 real (CPU) + E2–E5 BLOCKED → results/e1_synth.json + measured.json
.venv/bin/python evaluate_claims.py   # → claims_result.json (verdict table over measured.json)
```

Final arm lines (`/tmp/arms.log`, this run):

| Arm | FINAL value | Meaning |
| --- | --- | --- |
| `base` | `BLOCKED` | Llama-2-7B / SDXL base model — no CUDA / no HF_TOKEN |
| `vanilla` | `BLOCKED` | vanilla steering arm — no CUDA / no HF_TOKEN |
| `leace_switch` | `BLOCKED` | LEACE-Switch arm — no CUDA / no HF_TOKEN |
| `midsteer` | `BLOCKED` | MidSteer arm — no CUDA / no HF_TOKEN |
| `e1_synth` | `1.655510e-12` | worst (max) C1/C2/C3 covariance-constraint residual across seeds {0,1,2}, synthetic data |

Per-claim measured vs. paper-claimed (from `claims_result.json`, produced by
`evaluate_claims.py` over `measured.json`). `measured` is verbatim from the
verdict table; the paper-claimed value is the `claimed` field (`None` for
ordering / invariant claims, which have no scalar target).

| Claim | Kind | Paper claims | Measured (this run) | Tolerance |
| --- | --- | --- | --- | --- |
| C1 | invariant | Eq. 6 closed form holds (constraint=0, minimal disturbance over feasible set, vanilla special case) | `true` at every seed; constraint ≤8.29e-13, min-disturb gap=0, vanilla special ≤2.66e-15 | 1e-6 / 1e-8 |
| C2 | invariant | Eq. 13 closed form holds (flip constraint, minimal disturbance, vanilla-switch special) | `true` at every seed; flip ≤1.66e-12, gap=0, special ≤2.66e-15 | 1e-6 / 1e-8 |
| C3 | invariant | Eq. 19 closed form holds (matched-cov constraint, minimal disturbance, erasure special) | `true` at every seed; matched-cov ≤1.48e-12, gap=0, erasure special=0 | 1e-6 / 1e-8 |
| C4 | ordering | MidSteer keeps "motorcycle" intact (≥ vanilla) | `BLOCKED` (model metric) | — |
| C5 | ordering | MidSteer keeps "motorcycle" intact (≥ LEACE-Switch) | `BLOCKED` | — |
| C6 | value | `moto_cs_on_moto` = **70.7** (`flipping_main.tex:46-48`) | `BLOCKED` | ±3.0 |
| C7 | ordering | min(vanilla, leace_switch) horse→moto ≥ MidSteer | `BLOCKED` | — |
| C8 | ordering | MidSteer RTP ≤ LEACE-Switch RTP | `BLOCKED` | — |
| C9 | ordering | MidSteer RTP ≤ base RTP (0.371 → 0.281) | `BLOCKED` | — |
| C10 | value | `rtp` = **0.281** (`safety_table.tex:71`) | `BLOCKED` | ±0.05 |
| C11 | ordering | MidSteer helpfulness ≥ vanilla | `BLOCKED` | — |
| C12 | ordering | MidSteer unrelated-CS > vanilla (8.46 → 3.11) | `BLOCKED` | — |
| C13 | ordering | LEACE-Switch MMLU BERT-F1 ≥ MidSteer | `BLOCKED` | — |
| C14 | ordering | MidSteer violence-CS ≤ vanilla | `BLOCKED` | — |
| C15 | ordering | vanilla violence-CS ≥ base | `BLOCKED` | — |
| C16 | ordering | MidSteer peace-CS ≥ LEACE-Switch | `BLOCKED` | — |
| C17 | value | `viol_cs` = **6.0** (`safety_table.tex:83`) | `BLOCKED` | ±5.0 |
| C18 | ordering | LEACE-Switch FID = min interventions | `BLOCKED` | — |
| C19 | ordering | MidSteer target-CS ≥ max(vanilla, leace_switch) | `BLOCKED` | — |
| C20 | curve | MidSteer src-CS below both baselines at β∈{3,4,5} (LLM pairs) | `BLOCKED` (per-x sequence) | — |
| C21 | curve | MidSteer horse src-CS below both baselines at β∈{1..5} (SDXL) | `BLOCKED` (per-x sequence) | — |
| C22 | curve | BERT-Precision plateau ≈ **[0.9578, 0.9605, 0.961]** over M∈{5000,10000,20000} (`suppl.tex:642`) | `BLOCKED` (per-x sequence) | ±0.03 |

E1 residuals per seed (`results/e1_synth.json`, produced by
`experiments/run_e1_synth.py`):

| Seed | C1 constraint | C2 flip | C3 matched-cov | min-disturb gap | special case |
| --- | --- | --- | --- | --- | --- |
| 0 | 5.15e-13 | 1.66e-12 | 5.35e-13 | 0 | 2.66e-15 / 0 |
| 1 | 8.29e-13 | 1.44e-12 | 4.89e-13 | 0 | 2.66e-15 / 0 |
| 2 | 6.98e-13 | 1.29e-12 | 1.48e-12 | 0 | 2.66e-15 / 0 |

The C1–C3 residuals are ~1e-12 to 1e-15, well inside the claims' 1e-6 / 1e-8
thresholds, so the closed-form math of the paper's three affine maps is
reproduced on synthetic data. **None of the paper's scalar targets (70.7, 0.281,
6.0, the C22 plateau) could be measured**, and **none of the 16 ordering
comparisons between the four arms could be resolved**, because all four
comparison arms are BLOCKED by the sandbox (no CUDA GPU, no `HF_TOKEN`). The
difference between this run and the paper is therefore not a number to report —
it is the absence of a number, and that absence is the result.

### research-readiness gates

| # | Gate | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | Builds from scratch (Docker) | **partial** | `Dockerfile` (CUDA 12.4 base) written and committed, but `docker` is not installed in this sandbox so `docker build && docker run` was never exercised. A reader with Docker must verify it. |
| 2 | README is accurate | **partial** | Quickstart (`uv venv ... && uv pip install -r requirements.txt && pytest tests/ -q`) followed verbatim works → 117 passed (README now states 117). Docker quickstart block unverified here (no docker). |
| 3 | Packages are clear | **pass** | `requirements.txt` pins every dependency with a rationale per group; `tests/test_environment.py` (40-case import gate) imports every dep + every `core` module; install then run does not die on a missing import. |
| 4 | Entrypoint is obvious | **pass** | One command `bash run_all_arms.sh` runs every arm at the paper config across seeds {0,1,2}; `smoke.sh` is the fast path; no source edits required. |
| 5 | Fast path | **pass** | `smoke.sh` runs the whole E1 path at smoke scale in ~seconds, writes only to gitignored `results/e1_synth_smoke.json`, prints one `FINAL e1_synth_smoke=<float>` line. |
| 6 | Deterministic / noise quantified | **partial** | E1 closed-form invariants are deterministic across reruns (re-run reproduced `1.655510e-12` byte-for-byte; `claims_result.json` is byte-identical across reruns after the sorted-`_refs` fix). The model arms are BLOCKED so their run-to-run noise is unmeasured. |
| 7 | Degeneracy test in repo | **pass** | E1 `run_e1_synth.py` checks the erasure / vanilla special cases (MidSteer≡LEACE when Z2 constant; LEACE special when β→0) — the method's no-op settings reproduce the baseline — as committed tests (`tests/test_midsteer_sign_matches_eq23_beta1`, `tests/test_invariants.py`), runnable by anyone. |
| 8 | Data provenance stated | **partial** | E1 uses synthetic Gaussian data of known covariance (provenance: generated in-script with seeded RNG, stated in `run_e1_synth.py` docstring). The paper's real datasets (Llama-2-7B-chat concept prompts, SDXL horse/motorcycle) are NOT fetched here — the data loader raises `BlockedException` rather than downloading; a GPU run would obtain them via the vendored upstream `scripts/`. |
| 9 | Recorded number is reproducible | **partial** | The E1 numbers reproduce exactly on rerun with the recorded command. The paper's recorded numbers (70.7, 0.281, 6.0, C22 plateau) are NOT reproduced — they are BLOCKED, so their reproducibility is untested. |
| 10 | Nothing depends on hidden local state | **pass** | Fresh-clone recipe in README works; `.venv/`, `__pycache__/`, `.pytest_cache/`, smoke output, hidden states, steering vectors, datasets are all gitignored; no home-dir or manually-fetched wheel dependency. |

### Rung reached

This run reached the **environment** rung, not `numbers`. The build/environment
budget was spent (`$HOME/.build_attempts` = 7) and the arms gate is still failing
at publish time: `run_all_arms.sh` prints `FINAL base=BLOCKED / vanilla=BLOCKED /
leace_switch=BLOCKED / midsteer=BLOCKED` for all four comparison arms — an
**all-BLOCKED arms result**, which is the environment rung and a legitimate
outcome. The blocker is the sandbox: `torch.cuda.is_available()` is False, no
`nvidia-smi`, and no `HF_TOKEN`, so the Llama-2-7B-chat and SDXL model arms
cannot run and emit no synthetic fallback. The `e1_synth` auxiliary arm (closed
form on synthetic data) did run real, but it is not one of the paper's
comparison arms and was measured on a synthetic stand-in, so it does not lift the
rung. The `result_check` gate never printed an `AUTHORITATIVE COUNTS` line (no
such line exists in this run's journal), which is consistent with `numbers` not
being reached. The comprehension (SPEC.md, claims.json), implementation
(`midsteer_core/`, instruments, run scripts) and an inline correctness review of
the six fixed claims (C1–C3/C20–C22) were completed, but those rungs cannot be
promoted past the all-BLOCKED arms gate — a number the environment cannot
produce is not evidence about the paper.

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

### Review-round fix (2026-08-17, pass 2): smoke must not emit a verdict word

- **Defect found via feedback `FINAL e1_synth_smoke=PASS`:** after the clobber fix
  above, `smoke.sh` still printed `FINAL e1_synth_smoke=PASS`. `PASS`/`FAIL` are
  **verdict words** that belong only to `claims_result.json` (the verdict table). The
  smoke is explicitly "not evidence about the paper … never report its output as a
  result", so emitting a verdict word made the smoke log read as a paper result — a
  free-floating PASS not recorded in `measured.json` or `claims_result.json`, where
  the E1 invariant verdicts actually live (`measured: "PASS"`, `verdict: "pass"`).
  Note the FULL `run_e1_synth.py` (non-smoke) run legitimately prints
  `FINAL e1_C1=PASS`/`FINAL e1_C2=PASS`/`FINAL e1_C3=PASS`: those are the **measured
  values** of the invariant claims C1–C3 (they are evidence and flow into
  `claims_result.json`). The smoke is not that run and must not mimic it.
- **Fix:** `run_e1_synth.py`'s smoke branch now prints exactly one line
  `FINAL e1_synth_smoke=<float>` whose value is the **worst (max) covariance-constraint
  residual** across C1/C2/C3 at smoke scale (`max(C1.constraint,
  C2.flip_constraint, C3.matched_cov_constraint)`). This is a **measured number**, not
  a verdict: a tiny residual (~1e-13) proves the closed-form affine maps were built and
  their constraints were evaluated — i.e. the code path ran — without asserting
  anything about the paper. Empty result still raises (no success path reports OK on
  empty). Verified: `bash smoke.sh` → `FINAL e1_synth_smoke=9.12e-14`.
- **Regression guard:** `tests/test_smoke_no_clobber.py` now asserts the smoke FINAL
  line is a finite non-negative float matching `^FINAL e1_synth_smoke=[0-9.eE+-]+$`,
  and explicitly that   the old `=PASS`/`=FAIL` verdict forms are absent. Full suite:
  98 passed.

### Review-round fix (2026-08-17, pass 3): instruments.json schema

- **Defect found via feedback `instruments.json declares no instruments and
  gives no reason`:** `instruments.json` was a **flat object with named keys**
  (`data_loader`, `judge_cs`, …), each entry carrying `name`/`what_it_decides`/
  `positive_test`/`negative_test`. The numbers-gate / reviewer expects a
  **top-level `instruments` array** (the canonical schema used by every
  sibling reproduction — `confidence-weighted-self-distillation-…`,
  `explaining-and-harnessing-adversarial-examples`). With neither an
  `instruments` array nor a `not_applicable`+reason at the top level, the gate
  read the file as "declares no instruments and gives no reason" — even
  though every instrument was present under a named key. A reader chasing the
  verdict table could not find the instrument registry in the shape the gate
  documents.
- **Fix:** `instruments.json` is now a top-level `{"instruments": [...],
  "not_applicable": {"applies": false, "reason": "…"}}` document. All nine
  instruments are preserved verbatim as array entries (one data loader, six
  eval-metric scorers `judge_cs`/`clip_cs`/`fid`/`detoxify_score`/`armorm_score`/
  `bertscore_mmlu`, the `closed_form_invariants` checker, and the
  `evaluate_claims` verdict writer), each with `name`, `what_it_decides`,
  `positive_test`, `negative_test`, `additional_tests`, `requires_tools`, and
  citations. `not_applicable.applies=false` with a reason documents that this
  reproduction has nine instruments, so the nothing-judges case does not apply
  (per the task's instruction to say so with `not_applicable` and a reason
  only when nothing here judges an output).
- **`requires_tools` is now descriptive** (was a boolean): each instrument
  lists the external tools it needs (`CUDA GPU`, `HF_TOKEN`,
  `transformers (Llama-3.1-8B-Instruct)`, `diffusers`, `clean-fid`, `detoxify`,
  `bert-score`, …) or `[]` for the pure-CPU instruments
  (`closed_form_invariants`, `evaluate_claims`). A reader can now tell which
  verdicts are blocked by this CPU sandbox (the six scorers + the data loader)
  vs measured here (the two CPU instruments).
- **Regression guard:** `tests/test_instruments.py` (7 tests) validates the
  schema itself — the top level is an `instruments` array (or a
  `not_applicable` with a reason), each entry has the four required fields,
  `not_applicable.applies` is consistent with the instrument count, every
  `positive_test`/`negative_test`/`additional_tests` node is a real
  collectible pytest node (or an existing script path) in this repo, a
  `data_loader` instrument fingerprinting the paper's dataset is present, a
  degeneracy/equivalence/invariant instrument is present, and
  `requires_tools` is declared per instrument. This prevents the named-key
  regression from recurring. Full suite: 105 passed.

### Review-round fix (2026-08-17, pass 4): not_applicable must be a single sentence, not a dict

- **Defect found via feedback `not_applicable excuses the WHOLE reproduction,
  so it must be one sentence saying why nothing here judges an output. It was
  a dict.`:** `instruments.json` carried a top-level
  `not_applicable: {"applies": false, "reason": "..."}` dict alongside a
  non-empty `instruments` array. The gate reads `not_applicable` as the
  nothing-judges-an-output case that excuses the WHOLE reproduction, so it
  must be ONE sentence (a string) and must only appear when there are NO
  instruments. A dict form was ambiguous: a list-of-exemptions reading could
  silently excuse every grader and data loader in the file. To exempt a
  single instrument, `not_applicable` with its reason goes ON that instrument;
  the top-level form is all-or-nothing and has no place next to a real
  instrument list.
- **Fix:** removed the top-level `not_applicable` dict from
  `instruments.json` entirely (this reproduction has nine instruments, so the
  nothing-judges case does not apply). The `_comment` now documents the
  convention: not_applicable is a single sentence present only when no
  instruments exist; per-instrument exemptions live on the instrument. The
  nine instruments and all their fields are unchanged.
- **Regression guard:** `tests/test_instruments.py` was rewritten to enforce
  the new convention — `test_not_applicable_is_a_single_sentence_and_only_
  when_no_instruments` asserts that if `not_applicable` is present it is a
  non-empty string and there are no instruments, and if instruments exist
  `not_applicable` is absent. A new
  `test_rejects_dict_not_applicable_alongside_instruments` reproduces the
  feedback's exact failure mode (a `{applies, reason}` dict next to a
  non-empty `instruments` array) and asserts the validator rejects it.
  `test_instruments_json_top_level_is_instruments_array` now accepts either
  an `instruments` array or a top-level `not_applicable` sentence (the
  nothing-judges case), but not both. Full suite: 106 passed.

### Review-round fix (2026-08-17, pass 5): make the 6 unevaluable claims gate-evaluable

The numbers-gate feedback marked C1, C2, C3, C20, C21, C22 `unevaluable` because the
gate evaluates each claim's `predicate` (invariant) / `quantity`+`against` (curve)
directly as Python over `measured` with a restricted builtin set (abs/all/any/bool/
float/int/len/max/min/round/sorted/sum) and NO `x` in scope — and those fields were
natural-language descriptions of the math (the C1–C3 predicates) or list-comprehensions
in `x` with trailing prose (the C20–C22 quantities). A claim that cannot be evaluated is
a defect in the reproduction, not a result.

Fixed without weakening any claim:
- **C1–C3 (`invariant`):** `predicate` is now an executable boolean expression over
  `measured.e1_synth.*` residual thresholds, e.g.
  `measured.e1_synth.c1_n_feasible > 0 and measured.e1_synth.c1_constraint < 1e-6 and
  measured.e1_synth.c1_minimal_disturbance_gap <= 1e-6 and
  measured.e1_synth.c1_vanilla_special < 1e-8`. A new `e1_synth` arm (CPU, no model)
  carries the per-seed invariant residuals in `measured.json` (assembled from
  `results/e1_synth.json` by `assemble_measured.py`), so the gate judges the verdict
  from measured evidence. The verbatim prose predicate is preserved in
  `predicate_description`. The `e1_synth` metric keys are claim-prefixed
  (`c1_constraint`, `c2_flip_constraint`, `c3_matched_cov_constraint`, …) to be
  unambiguous across the three claims.
- **C20–C22 (`curve`):** `quantity`/`against` are now plain `measured.<arm>.<metric>`
  refs to PER-X SEQUENCES (the gate provides no `x`; it treats these refs as the stored
  sequences and compares elementwise). C20/C21 use new derived metrics
  `c20_min_baseline_src` / `c21_min_baseline_horse` = elementwise min of the two
  baselines' per-beta sequences (the weakest-dominance reference), computed by
  `assemble_measured.py` so the elementwise-min is correct on a real host (a naive
  `min(list, list)` would be lexicographic, not elementwise). C22 uses the existing
  `bertp_mmlu` sequence. The verbatim list-comprehensions are preserved in
  `quantity_description` / `against_description`.
- **`evaluation_rule`** rewritten in the gate's vocabulary (reproduced / refuted /
  untested / blocked) with the executable-predicate and per-x-sequence model stated
  explicitly; `evaluate_claims.py` emits `generated_by='workflow_subagent'` (the
  WORKFLOW's table; the gate's `produced_by` form is never what this repo commits).
- `evaluate_claims.py` now evaluates invariants by executing the predicate over
  `measured.json` at every seed, and curves by elementwise per-x comparison; the
  ordering path gained the within-noise `untested` verdict (|mean| <= cross-seed
  spread). `tests/test_claims_eval.py` rewritten with positive AND negative tests for
  every path (reproduced/refuted/untested/blocked for ordering, value, invariant,
  curve), a test that a predicate referencing a disallowed name (`x`) is blocked (not a
  rubber stamp), and a test that `generated_by=='workflow_subagent'` (not the gate's
  `produced_by`). `instruments.json` `evaluate_claims` entry updated to reference the
  new positive/negative tests. Full suite: 117 passed.

Result: `claims_result.json` now has reproduced=3 (C1–C3), blocked=19 (C4–C22),
untested=0, refuted=0 — no `unevaluable`. `selfcheck.json` agrees. The model arms
remain BLOCKED (no CUDA / no HF_TOKEN), which is the correct blocked result for this
sandbox, not a defect.

**Adversarial review of the 6 fixed claims against the paper** (the orchestrate backend
was down — HTTP 500 on a trivial ping — so the review was done inline, reading the
paper `.tex` at each claim's cited lines byte-for-byte):
- C1 (`paper/content/guardedness.tex:73-85`): A_hat = I − W⁺(WΣ_XZ)(WΣ_XZ)⁺W,
  b_hat = E[X] − A_hat E[X], W = (Σ_XX^{1/2})⁺; constraint Cov(AX+b,Z)=0 (l.71);
  objective E‖AX+b−X‖² (l.64-67). The predicate thresholds all three sub-claims
  (constraint, minimal disturbance over the feasible set, vanilla special case
  Cor. 4.1 `main.tex:297-315` incl. the authors' `s`-normalisation caveat l.302).
  Faithful.
- C2 (`main.tex:343-367`): A_hat = I − 2W⁺(WΣ_XZ)(WΣ_XZ)⁺W (l.365); flip constraint
  Cov(AX+b,Z) = −Cov(X,Z) (l.359); Cor. 4.3 `main.tex:377-394` (vanilla-switch
  special case, unit-norm s per l.381). Faithful.
- C3 (`main.tex:418-448`): A_hat = I + W⁺(Σ_{WX,Z2}−Σ_{WX,Z1})Σ_{WX,Z1}⁺W (l.447);
  matched-covariance constraint Cov(AX+b,Z1)=Cov(X,Z2) (l.441); erasure special case
  Z2 constant ⇒ MidSteer≡LEACE (`main.tex:461`). Faithful.
- C20 (`switching_suppl.tex:21` + `llm_flip_tables_noclip.tex`): verified
  MidSteer source-CS below BOTH baselines at β∈{3,4,5} for both pairs
  (horses→motorcycles: MidSteer 1.7/1.6/1.4 vs vanilla 2.9/3.0/3.0 & LEACE
  3.7/3.5/3.4; dogs→cats: MidSteer 1.9/1.6/1.6 vs vanilla 4.8/4.6/4.4 & LEACE
  3.7/3.6/3.6). min-of-averages and average-of-mins both give dominance; margins ≥1.2.
  Faithful.
- C21 (`experiments.tex:121` + `diffusion_flip_tables_noclip.tex`,
  tab:flip_sdxl_noclip_horse_to_motorcycle): MidSteer horse src-CS 51.2/50.0/49.2/48.8/48.3
  below both CASteer 70.0/52.1/51.4/51.0/50.7 and LEACE 65.0/51.2/50.5/50.2/49.9 at
  β∈{1..5}; margins ≥1.2 (1.2 at β=2). Faithful.
- C22 (`suppl.tex:642`): "stabilises around 5,000 prompts" encoded as the
  β-averaged BERT-Precision sequence over M∈{5000,10000,20000} matching the Fig.5
  plateau values within tolerance. Review found a latent scalar/curve NAME
  COLLISION: `bertp_mmlu` was both C22's 3-element ablation curve and the e2-concrete
  scalar (unused by any claim) — on a GPU host the scalar would clobber the curve.
  Fixed by renaming C22's curve metric to `c22_bertp_mmlu` (distinct), leaving e2's
  scalar `bertp_mmlu` separate. Faithful after the rename.

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
  `results/e1_synth.json`);   `run_all_arms.sh` (runs E1 real + E2–E5 BLOCKED →
  `measured.json` + `results/e1_synth.json`); the full `tests/` suite (117 passed:
  environment import gate, closed-form core/degeneracy/invariants, data loader,
  eval-metric instruments, claims evaluator with positive/negative tests for every
  verdict path, mutations, smoke no-clobber guard, instruments.json schema guard).
  `evaluate_claims.py` → `claims_result.json` (reproduced=3, refuted=0, untested=0,
  blocked=19; `generated_by='workflow_subagent'`).
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
