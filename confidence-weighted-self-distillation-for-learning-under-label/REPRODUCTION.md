# Reproduction: Confidence-Weighted Self-Distillation for Learning under Label Noise

- **Paper:** Confidence-Weighted Self-Distillation for Learning under Label Noise
- **Authors:** A. Bergstrom, M. Oyelaran, K. Vasquez (Institute for Applied Learning Systems)
- **Year:** unknown
- **arxiv_id:** unknown
- **Date started:** 2026-08-04
- **paper_ref:** ce7a63e8-2c90-4516-887d-14515c8f4516
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Branch:** `repro/confidence-weighted-self-distillation-for-learning-under-label`

## Status

**Ingest complete.** Work is in `setup` stage; no code has been (re)validated by this run yet.

- [x] Reproduction workspace set up from `main`, branch
      `repro/confidence-weighted-self-distillation-for-learning-under-label` pushed
- [x] Paper text saved to `paper/paper.md`
- [ ] Comprehension (SPEC)
- [ ] Implementation / verification of existing code against the paper
- [ ] Adversarial review rounds clean
- [ ] Readiness gates
- [ ] Publish

## Source notes

- The objective's `arxiv_id` is `unknown`, so **no arXiv LaTeX source could be
  fetched** (`https://arxiv.org/e-print/<id>` is not applicable). Per the
  ingest instructions, this is recorded here and work proceeds with the
  PDF-extracted text at `paper/paper.md`, which is reliable for prose but
  **not** for maths — some symbols may be silently missing (e.g. the gate
  sharpness `s` in Eq. (2) has no stated value in the extracted text).
- This slug's folder already exists on `main`: it holds the complete output of
  a prior reproduction run of the same paper (its `REPRODUCTION.md` reported
  rung `numbers`). Those artifacts (`run_experiment.py`, tests, `SPEC.md`,
  `VERIFICATION.md`, `measured.json`, etc.) are left in place as starting
  material; this run re-verifies them against the paper rather than trusting
  them. The prior run's stale remote branch tip (`9a07636`) was fast-forwarded
  to current `main` — the slug folder content was identical (`git diff` empty).

## Claimed results (paper Table 1, §4)

| Method | λ | Test accuracy |
|---|---|---|
| Cross-entropy (baseline) | 0 | 0.9370 |
| CWSD (ours) | 1 | 0.9620 |

Setup (§3): scikit-learn `load_digits` (1797 8×8 digits, K=10, pixels ÷16),
30% stratified test split at seed 0, 20% symmetric label noise, 64-hidden-unit
ReLU MLP, SGD lr 0.1, batch 64, 4000 steps; λ=1, τ=0.9, T=2; single seed-0 run.
