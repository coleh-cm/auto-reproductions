# Reproduction: CASteer: Cross-Attention Steering for Controllable Concept Erasure

- **Paper:** CASteer: Cross-Attention Steering for Controllable Concept Erasure
- **Authors:** Tatiana Gaintseva, Andreea-Maria Oncescu, Chengcheng Ma, Ziquan Liu, Martin Benning, Gregory Slabaugh, Jiankang Deng, Ismail Elezi
- **Year:** 2026
- **arXiv:** 2503.09630
- **Date started:** 2026-08-17
- **paper_ref:** d0ab97b5-e9bf-41af-9846-d19f846eef80
- **project_id:** 06910d54-1d99-4864-8bff-ab3007a0c70e

## Status

**In progress — setup complete.**

- [x] Reproduction folder created; branch `repro/casteer-cross-attention-steering-for-controllable-concept-erasure` pushed.
- [x] arXiv e-print 2503.09630 fetched and unpacked under `paper/` — `.tex`/`.bib` committed, figure images and style files gitignored (no `.bbl` present in the source tarball).
- [x] Rendered HTML saved as `paper/paper.html` and committed.
- [x] SPEC.md — method written up as an algorithm with symbol shapes and cited equations; unstated-items analysis (U1–U13); arms, restrictions, 17-claim `claims.json`.
- [x] Upstream code search — **official code exists**: `abstract.tex:36` → https://github.com/Atmyre/CASteer (pinned at `135912a555a8606c01f55843c738cf8062319ed7`); plan: vendor upstream, run its entrypoints, log every change needed.
- [ ] Implementation / environment setup.
- [ ] Smallest end-to-end run producing a parsed number.
- [ ] Adversarial review loop clean.
- [ ] Readiness gates.
- [ ] Publish.

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
