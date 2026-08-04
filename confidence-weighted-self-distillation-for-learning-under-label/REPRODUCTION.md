# Reproduction: Confidence-Weighted Self-Distillation for Learning under Label Noise

- **Paper:** Confidence-Weighted Self-Distillation for Learning under Label Noise
- **Authors:** A. Bergstrom, M. Oyelaran, K. Vasquez
- **Year:** unknown
- **Date started:** 2026-08-04
- **paper_ref:** ce7a63e8-2c90-4516-887d-14515c8f4516
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3

## Status

**Ingest done.** Paper text saved to `paper/paper.md`.

- **arXiv source: unavailable.** The objective gives `arxiv_id: unknown`, so the
  LaTeX source could not be fetched and there is no authoritative math source on
  disk. All equations, tables, and numbers must therefore come from the
  PDF-extracted text in `paper/paper.md`, which is reliable for prose but **not
  for maths** (extraction silently drops some symbols). Every equation used
  downstream is reconstructed from that extraction and must be sanity-checked
  against the prose that surrounds it (e.g. Eq. (2) must be a convex weight in
  `[0, λ]`, Eq. (3) must be a convex combination, Eq. (4) must reduce to
  standard cross-entropy at `λ = 0` exactly as the paper states).
- **Prior run exists.** This same paper (identical paper_ref and project_id)
  was reproduced in an earlier run; its final state is on `main` at commit
  `3f47d2c` and it reported reaching the `review` rung (numbers-gate build
  budget spent). That run's code (`run_experiment.py`, `tests/`, `SPEC.md`,
  etc.) is still present in this folder. This run starts its own tracking here;
  later steps will re-verify or rebuild that code against the paper rather than
  trusting it.

## Log

- 2026-08-04 — Ingest: cloned repo (shallow, blobless), branch
  `repro/confidence-weighted-self-distillation-for-learning-under-label`
  created and pushed; paper text saved; this file started.
- 2026-08-04 — SPEC step (this run): rewrote `SPEC.md` and `claims.json` from
  `paper/paper.md` alone, then verified them against the actual code and a fresh
  environment (numpy 2.5.1, scikit-learn 1.9.0 installed for this workstation).
  Verified, not trusted from the prior run: (1) every grep anchor cited in
  SPEC §1/§3 and claims.json resolves to the cited line; (2) all 11 `quote`
  fields are verbatim under the documented normalisation (hyphen-join +
  whitespace collapse, PDF token spacing kept: `2 . 5`, `[0 , 1]`) — a checker
  one-liner is embedded in `claims.json.evaluation.quote_policy`; (3) the
  frozen CLI/function interfaces in SPEC §5 match `run_experiment.py`'s
  argparse and signatures; (4) split is 1257/540 as stated; (5) fresh runs
  reproduce the numbers SPEC asserts: baseline seed 0 = 0.9370 (exact match,
  Table 1), CWSD seed 0 = 0.9611, baseline seed 1 = 0.9407, CWSD seed 2 =
  0.9556 — identical to `measured.json`; `pytest -q tests` = 47 passed;
  (6) upstream-code search re-run (GitHub repos for the title / topic /
  institution, users for the authors): all `total_count: 0`; no links in the
  paper. Paper has no figures (grep over `paper/paper.md`), so no curve
  claims. Key unstated item confirmed: the gate sharpness `s` in Eq. (2) has
  no value anywhere in the paper; `s = 0.15` stands as the calibrated default
  with the sensitivity sweep recorded in SPEC §4 item 1.
