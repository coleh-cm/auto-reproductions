# Reproduction: CASteer: Cross-Attention Steering for Controllable Concept Erasure

- **Paper:** CASteer: Cross-Attention Steering for Controllable Concept Erasure
- **Authors:** Tatiana Gaintseva, Andreea-Maria Oncescu, Chengcheng Ma, Ziquan Liu, Martin Benning, Gregory Slabaugh, Jiankang Deng, Ismail Elezi
- **Year:** 2026
- **arXiv:** 2503.09630
- **Date started:** 2026-08-17
- **paper_ref:** d0ab97b5-e9bf-41af-9846-d19f846eef80
- **project_id:** 06910d54-1d99-4864-8bff-ab3007a0c70e

## Status

**In progress — implementation + correctness complete; diffusion-number claims BLOCKED on CPU.**

- [x] Reproduction folder created; branch `repro/casteer-cross-attention-steering-for-controllable-concept-erasure` pushed.
- [x] arXiv e-print 2503.09630 fetched and unpacked under `paper/` — `.tex`/`.bib` committed, figure images and style files gitignored (no `.bbl` present in the source tarball).
- [x] Rendered HTML saved as `paper/paper.html` and committed.
- [x] SPEC.md — method written up as an algorithm with symbol shapes and cited equations; unstated-items analysis (U1–U14, U14 = LPIPS_e direction contradiction); arms, restrictions, 17-claim `claims.json`; §11 Constructed truth, §12 unstated-value sweeps, §13 environment constraint.
- [x] claims.json validator round 1 fixed: `arms` (9) + `arm_configs` declared; per-arm `restrictions` map (all `narrows_situations`); curve claims carry `x` + `against`; every `sensitivity` numeric `plausible`/`survives` with non-degenerate survival ratio (i2p claim reframed to images-per-prompt with full-width survival + inconclusive-below-3000 note); Snoopy CS claims moved to the paper's own per-seed normalized CS (`experiments.tex:75`) so the CLIP-checkpoint sensitivity is honestly survivable across [512,768].
- [x] Upstream code search — **official code exists**: `abstract.tex:36` → https://github.com/Atmyre/CASteer (pinned at `135912a555a8606c01f55843c738cf8062319ed7`); vendored verbatim at repo root (commit `6f3b97f`).
- [x] Implementation / environment setup — `requirements.txt`, `Dockerfile`, `README.md`, `pyproject.toml`, `.gitignore`, import gate `tests/test_environment.py`.
- [x] **Method core implemented** (`core/controller.py`): added constant-α Eq.4 mode (`steering_mode='constant'`, SPEC U13) for the `const_a*` arms; added Eq.9 multi-concept average (`average_concept_vectors`, no re-normalization, U9); fixed a CPU bug where `forward` hardcoded `vector.half()` (now preserves the caller's dtype — `float16` on GPU, `float32/float64` on CPU); kept the dot-product Eq.6/7 path numerically identical for `float16`. `tests/test_method_core.py` (8 invariants) + `tests/test_degeneracy.py` (β=0 bit-exact identity on both modes, inactive-control identity) pass.
- [x] **Data pipeline implemented** (`core/data.py`): fingerprinted loaders for ImageNet classes (50, `tench`), CLIP templates (80), COCO captions (30,000), I2P prompts (4,703 via `AIML-TUDA/i2p`), and the real COCO-30k FID reference (RAISES if not vendored — never substitutes synthetic). `tests/test_data.py` passes (incl. live I2P fingerprint).
- [x] **Evaluation metrics implemented** (`core/eval/metrics.py` + device-agnostic `core/eval/clip.py`): CLIP score (ViT-B/32, CPU-runnable), FID (clean-fid, `device='cpu'`), and NudeNet/Q16/LPIPS which RAISE `MissingEvaluatorError` when their tool is not installed. Fixed two real bugs in `core/eval/clip.py`: hardcoded `device="cuda"`/`.cuda()` (crashed on CPU) and the `range(len(images)//50)` loop that silently dropped the trailing <50 images. `tests/test_eval.py` (11 instrument tests) passes.
- [x] **Runner implemented** (`core/runner.py`, `scripts/diffusion/run_all_arms.py`, `scripts/diffusion/smoke.py`): CPU-friendly pipeline loader (fp32 on CPU, fp16 on GPU, no `device_map`); per-arm config mirrored from `claims.json`; `is_arm_feasible` drives the BLOCKED decision on CPU; fail-loud-on-empty everywhere.
- [x] **smoke runs end-to-end on CPU**: `smoke.sh` loads SD-1.4 (fp32), estimates 80 steering vectors from 2 prompt pairs (Algorithm 1, shared seed 0, patch-mean, pair-mean, pos−neg, unit-norm), generates 1 steered (casteer_clip, β=2, Eq.7 clip) + 1 vanilla image at 4 steps/256²/seed 42, then scores the steered image with the paper's `snoopy_cs` metric (mean CLIP score on a Snoopy prompt, ViT-B/32) via the same eval code path the full arms use. Prints exactly one stdout line `FINAL casteer_clip=<snoopy_cs>` (arm name from claims.json, real numeric measurement; provenance metadata to stderr), ~41 s once SD-1.4 is cached. NOT evidence about the paper.
- [x] **Degeneracy + invariants tests + mutations**: 33 tests pass; `mutations.json` (6 deliberate defects) each caught by its `must_fail` test (verified by applying find→replace and watching the named node fail). `instruments.json` records every correctness-deciding instrument with positive/negative tests.
- [x] **`selfcheck_claims.py` → `selfcheck.json`**: `house` PASS (max norm error 3.5e-14, unit-norm OK); 16 diffusion claims BLOCKED. This is our own evaluator; `claims_result.json` is produced by the workflow's numbers gate, not by us.
- [x] **`measured.json`**: all diffusion metrics `BLOCKED` with `measured_blocked_reasons.json` sidecar (CPU-only host; paper full config infeasible without GPU, `supplementary.tex:30`).
- [ ] Adversarial review loop clean (the `orchestrate` tool returned Internal Server Error on every call in this sandbox, including trivial probes; components were built directly with self-review against the paper and verified by tests + mutations — see "Orchestration" note below).
- [ ] Readiness gates.
- [ ] Publish.

## What this run reproduces

- **`house` (the Householder norm-preservation invariant)** — fully reproduced (pure math, no generation). High compute-invariance.
- **The method core is correct and tested**: Eq.6 (matrix == dot-product), Eq.7 (clip only positive projections), Eq.4 (constant α), Eq.9 (multi-concept average, not re-normalized), degeneracy at β=0 (bit-exact baseline), CFG conditional-half-only steering, unit-norm construction (U1).
- **The code path runs end-to-end** (smoke): estimate → steer → generate, on the real SD-1.4, on CPU.

## What this run CANNOT reproduce (BLOCKED)

- All 16 diffusion-number claims. This sandbox is CPU-only; the paper's full config (50 steps × ≥1,000 prompts × 3 seeds × multiple arms, on 8×V100, `supplementary.tex:30`) is infeasible on CPU (weeks of wall time). `measured.json` records `BLOCKED` for every diffusion metric; `run_all_arms.sh` emits `FINAL <arm>=BLOCKED` per arm/seed. The external evaluators (NudeNet, Q16, LPIPS) are not installed and the real COCO-30k FID reference is not vendored — both are documented blockers, not synthetic substitutes.

## Orchestration

The workflow's implementation step was to call `orchestrate` to build the 5 components in parallel and review each against the paper. The `orchestrate` tool returned **Internal Server Error on every call** in this sandbox, including a trivial `META + return {}` probe and a one-agent `agent('say pong')` probe. The tool is unavailable here. The components were therefore built directly, with self-review against the paper (`paper/content/*.tex`, cited in SPEC.md), and verified objectively by the test suite (33 passing), the mutation suite (6/6 caught), and the live smoke run. This deviation is recorded here for provenance.

## Curve-claim figures

The two curve claims (`fig_clip_shape`, `fig_fid_shape`) come from the paper's figures `paper/content/images/snoopy_clip_vs_clip_2.png` and `snoopy_clip_vs_fid_2.png` (axes read in `figure_transcript.md`: normalized Snoopy CS vs mean other-concepts CS / vs mean other-concepts FID). The paper's figure PNGs are gitignored under `paper/.gitignore` (style/asset policy) but remain on disk for a reader to inspect. **These figures cannot be regenerated here**: each plots measured CS/FID points for the steered arms, which are BLOCKED on this CPU-only host (§ above). Per the reproduction rule, a regenerated figure + the paper's figure are a pair for a reader to compare and are **not** evidence — the gate's verdicts (in `claims_result.json`, produced by the numbers gate) are the evidence. We did not fabricate a figure from BLOCKED numbers; the curve claims stand at BLOCKED with the declared `against` constants recorded in `claims.json` (SPM/SAFREE/DoCo norm-CS = [0.7758, 0.6969, 0.6255]; Receler/ESD mean-other-FID = [77.5, 78.88, 106.64]) so a GPU run can complete the comparison. The axis ranges/units the figures use: x = normalized Snoopy CS ∈ ~[0.5, 0.85] (paper's own per-seed normalization, `experiments.tex:75`); y = mean CLIP score of other concepts (fig 4) / mean FID of other concepts (fig 5).

## Figure readings

- `figure_transcript.md` — `read-figure` Q&A on the two Snoopy scatter plots (`snoopy_clip_vs_clip_2.png`, `snoopy_clip_vs_fid_2.png`). Note: first two calls returned empty answers under a too-small `--max-tokens` (reasoning budget consumed); re-asked with `--max-tokens 2000`. Findings: Ours is left of SPM/SAFREE/DoCo on both plots; above Receler/ESD on clip-vs-clip; below Receler/ESD on clip-vs-fid; Ours ≈ (0.59, 0.983) — consistent with table-derived (0.583, 0.982). Emitted as curve claims `fig_clip_shape` / `fig_fid_shape`.

## Source layout

- `paper/paper.html` — reading surface (MathML rendering).
- `paper/iclr2026_conference.tex` — main file; `paper/math_commands.tex` — preamble macros (resolve these before quoting equations).
- `paper/content/*.tex` — abstract, intro, method (`method_2.tex`), experiments, related work, conclusion, supplementary.
- `paper/content/<model>_tables/*.tex` — results tables per model (SD 1.4, SD 1.4 step-0, SD 1.5, SDXL, Sana, multi-concept, strength, UNet-padding, constant-strength ablations).
- Quotes use `<file on disk>:<line>` citations against this `.tex` source.

## Log

- 2026-08-17: Repo cloned shallow/blobless. Branch `repro/casteer-cross-attention-steering-for-controllable-concept-erasure` created and pushed. Both arXiv copies fetched successfully (e-print tarball + HTML); no fetch failures to record.
- 2026-08-17: Read full paper (method, experiments, all tables, supplementary + algorithms). Inspected upstream `core/{controller,diffusion_steering,vector_dump,utils,construct_prompts}.py` + `estimate_steering_vectors.py` to pin interfaces and confirm ambiguity resolutions (unit-norm f_norm; attn2-module output hook; conditional-CFG-branch-only steering; SD-1.4 at 50 steps/gs-default/512²; shared global seed 0 for vector construction). Wrote `SPEC.md` + `claims.json` (17 claims, 10 high-compute-invariance) and committed figure transcripts. Key internal inconsistency recorded as U1: paper prints `f_norm(v)=v/||v||²` (method_2.tex:81, supplementary.tex:60) but its own projection-length (method_2.tex:125) and Householder (experiments.tex:21-22) statements plus upstream code all require unit norm — adopted unit norm, flagged for the sweep.
- 2026-08-17: instruments.json fix pass. The gate flagged `positive_test`/`negative_test` entries that were prose rather than `<file>::<test>` node references (fid_scorer, degeneracy_equivalence_check, householder_invariant_check, nudenet_detector, q16_classifier, lpips_scorer). Rewrote every entry as a JSON array of real pytest nodes. Added two missing negative-test nodes that prove the checks *reject* a known-wrong input: `tests/test_degeneracy.py::test_degeneracy_rejects_nonzero_offset` (the bit-exact check rejects a `+0.001` offset at β=0, the `degeneracy_offset` defect) and `tests/test_method_core.py::test_householder_rejects_half_strength` (the norm-preservation invariant rejects `(I - 1·s sᵀ)c`, the `half_strength_projection` defect). For the not-installed graders (NudeNet/Q16/LPIPS) the `positive_test` is the raises-when-unavailable node and the `negative_test` is `test_metrics_no_ok_on_empty_dirs` (no OK-on-empty). Full suite: 35 passed.
