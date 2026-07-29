# Reproduction: Manifold-Guided Attention Steering

- **Paper:** Manifold-Guided Attention Steering
- **Authors:** Ian Li, Kapilesh Guruprasad, Raunak Sengupta, Ninad Satish, Loris D'Antoni, Rose Yu (UC San Diego)
- **Year:** 2026
- **arXiv:** 2605.21770
- **Date started:** 2026-07-29
- **paper_ref:** c8a82a40-c60b-43ce-863f-37fb55cb3e8e
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3

## Status

**Phase: workspace setup**

- [x] Cloned reproductions repository (`auto-reproductions`)
- [x] Created reproduction folder `manifold-guided-attention-steering/`
- [x] Recorded folder path in `$HOME/.repro_dir`
- [x] Saved PDF-extracted paper text to `paper/paper_pdf_extracted.txt`
- [x] Fetched and unpacked LaTeX source from arXiv e-print to `paper/latex_src/`
      (main file: `paper/latex_src/neurips_2026.tex`, refs: `paper/latex_src/refs.bib`)
- [ ] SPEC.md (method spec with equations, shapes, citations)
- [ ] Implementation
- [ ] Review rounds
- [ ] Readiness gates
- [ ] Publish

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

## Known gaps / things the paper does not specify (to track in SPEC.md)

- Exact value of `q` (threshold percentile) and `k` (subspace rank) per benchmark/model
- How many heads K are monitored per benchmark (top-1/top-3 in ablation; Table 1/2 config unclear)
- Fine details: hook placement for per-head head-output capture across model families
- GPT-OSS-20B molecular setup (prompt, target protein, docking pipeline details)
