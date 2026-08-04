# Reproduction: Confidence-Weighted Self-Distillation for Learning under Label Noise

- **Paper:** Confidence-Weighted Self-Distillation for Learning under Label Noise
- **Authors:** A. Bergstrom, M. Oyelaran, K. Vasquez
- **Year:** unknown
- **Date started:** 2026-08-04
- **paper_ref:** ce7a63e8-2c90-4516-887d-14515c8f4516
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3

## Status

**Ingest done.** This run's paper text saved to `paper/paper.md`.

- **arXiv source: unavailable.** The objective gives `arxiv_id: unknown`, so the
  LaTeX source could not be fetched and there is no authoritative math source on
  disk. All equations, tables, and numbers must therefore come from the
  PDF-extracted text in `paper/paper.md`, which is reliable for prose but **not
  for maths** (extraction silently drops some symbols). Every equation used
  downstream is reconstructed from that extraction and must be sanity-checked
  against the prose that surrounds it (e.g. Eq. (2) must be a convex weight in
  `[0, λ]`, Eq. (3) must be a convex combination, Eq. (4) must reduce to
  standard cross-entropy at `λ = 0` exactly as the paper states).
- **Paper text verified for this run.** The objective's paper text was written
  out and diffed byte-for-byte against the committed `paper/paper.md`:
  identical. The on-disk paper is exactly this run's input, not a leftover.
- **Prior runs exist.** This same paper (identical paper_ref and project_id)
  was reproduced in earlier runs. The immediately prior run's final state is on
  `main` at commit `536f58f` (its own REPRODUCTION.md, preserved in git
  history at that commit, reports reaching the `review` rung with a 9/9
  passing on-disk numbers gate, and references an even earlier run at
  `3f47d2c`). That run's code (`run_experiment.py`, `tests/`, `SPEC.md`,
  `measured.json`, etc.) is still present in this folder. This run starts its
  own tracking here; later steps will re-verify or rebuild that code against
  the paper rather than trusting it.

## Log

- 2026-08-04 — Ingest: cloned repo (shallow `--depth 1 --filter=blob:none`,
  over HTTPS using existing credentials; clone completed without the HTTP/2
  stream failures seen in earlier runs), branch
  `repro/confidence-weighted-self-distillation-for-learning-under-label`
  created at `main` HEAD (`536f58f`) and pushed; the prior remote tip
  `fb7714f` of this branch is an ancestor of `main` (already merged), so the
  push fast-forwards it. Paper text saved to `paper/paper.md` (byte-identical
  to this run's objective text); this file started. No arXiv fetch
  (`arxiv_id: unknown`), so no eprint artifacts and nothing to gitignore
  beyond the folder's existing Python ignores.
