# Reproduction log: Distributionally Robust Linear Regression With Block Lewis Weights

- **paper_ref:** 0a8cf406-bd9b-4ebe-8a3f-c9e3d62a2a94 · **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **arXiv:** 2607.00252 (Manoj & Patel, 2026)

## Status

### 2026-07-30 — COMPREHENSION (this commit)
- Fetched arXiv e-print LaTeX source (authoritative for maths) into `paper/`; fetched the
  companion paper MO25 (arXiv 2311.10013) whose Algorithm 2 the paper's Lewis-weight step
  invokes, stored grep-ably as `paper/mo25_main.tex`.
- No upstream code exists: no links in paper/arXiv page; GitHub search empty; the
  "included Jupyter notebook" cited at `paper/experiments.tex:38` is absent from the arXiv bundle.
- Wrote `SPEC.md` (method as algorithms, symbol shapes, equation citations `paper/<file>:<line>`,
  register of 18 unstated items, frozen component interfaces, arm list) and `arms.json`
  (8-arm numbers-gate contract).
- Scope decision: numbers gate = paper's Section 8 empirical evaluation (synthetic + ACS Income).
  The theory algorithms are cited in SPEC §5 as a stretch goal; the paper's own experiments ran
  the unaccelerated trust-region variant (`paper/experiments.tex:78`).
- Environment probe: pip works (numpy 2.5.1 installed); census.gov reachable for folktables.

## Open items for next steps
- Resolve SPEC §6 items 1 (synthetic recipe), 4 (1%-gap reference), 12 (ACS target scale) empirically.
- Then: data pipeline → objectives/lewis → 7 arms + reference → harness → run → adversarial review.
