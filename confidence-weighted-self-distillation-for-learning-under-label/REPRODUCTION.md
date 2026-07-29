# Reproduction: Confidence-Weighted Self-Distillation for Learning under Label Noise

- **Paper:** Confidence-Weighted Self-Distillation for Learning under Label Noise
- **Authors:** A. Bergstrom, M. Oyelaran, K. Vasquez
- **Year:** unknown
- **Date started:** 2026-07-29
- **paper_ref:** ce7a63e8-2c90-4516-887d-14515c8f4516
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3

## Status

**Spec written.** `SPEC.md` fixes the algorithm, shapes, equations with citations, the
frozen interfaces for `run_experiment.py`, and the full list of things the paper leaves
unstated (headline: the gate sharpness `s` in Eq. (2) has no value anywhere in the paper).
No upstream code exists (checked the paper text, GitHub repo/code search). Next step:
implementation.

## Running log

- 2026-07-29: Cloned reproductions repository, created reproduction folder, saved paper
  text verbatim to `paper/paper.md`, started this log.
- 2026-07-29: Wrote `SPEC.md` (method as algorithm with shapes, equation citations into
  `paper/paper.md`, unstated-items list §4, frozen component interfaces §5, upstream-code
  search record §6). Installed numpy 2.5.1 + scikit-learn 1.9.0; verified split arithmetic
  empirically (1797 → 1257 train / 540 test).
- 2026-07-29: Verified the reproducible environment end to end. Recreated the venv from
  scratch with `uv venv --python 3.13 .venv` (removed a stale `.venv` first so the bare
  command succeeds idempotently) and `uv pip install --python .venv -r requirements.txt`;
  all 11 pinned packages installed. `import run_experiment` resolves; `pytest -q` → 5
  passed. Both arms run and print the `FINAL accuracy=<float>` contract line:
  baseline (λ=0) `0.9315`, CWSD (λ=1) `0.9519` — the ~2-point CWSD-over-baseline
  improvement is reproduced (Table 1's absolute numbers are not bitwise-reproducible
  because the paper omits the RNG stream layout, weight init, and gate sharpness `s`;
  see SPEC §4).

## Target numbers

| Method | λ | Test accuracy |
|---|---|---|
| Cross-entropy (baseline) | 0 | 0.9370 |
| CWSD (ours) | 1 | 0.9620 |
