# Reproduction log — Distributionally Robust Linear Regression With Block Lewis Weights

- paper_ref: 0a8cf406-bd9b-4ebe-8a3f-c9e3d62a2a94
- project_id: d7735ece-02c4-4228-985c-00834c92b8f3
- arXiv: 2607.00252 · Manoj & Patel (2026) · ICLR 2026

## Status: SPEC frozen (step: read + spec)

- LaTeX source on disk at `paper/arxiv-2607.00252-src/` — used as authoritative for all
  equations (PDF extraction was not trusted for maths).
- Read in full: intro/technical overview/Alg. 1 (`body.tex`), block Lewis weights
  (`other_proofs.tex`), mirror descent (`mirror_descent.tex`), MS acceleration Alg. 3
  (`improved_ms.tex`), interpolation/Alg. 4/Alg. 5 (`interpolation.tex`), experiments
  (`experiments.tex`).
- Upstream code: NONE found (arXiv page has no code link; GitHub repo/code searches empty;
  both authors' GitHub accounts have zero public repos; the "included Jupyter notebook"
  referenced at experiments.tex:38 is absent from the arXiv tarball). Implementing from scratch.
- External algorithmic dependency resolved: MO25 ("[MO25, Algorithm 2]") source fetched from
  arXiv:2311.10013 and its algorithm re-derived concretely in SPEC.md (E8).
- SPEC.md written: shapes, equations with grep-able citations, unstated-items list U1–U18,
  frozen interfaces, arms list (7 method arms × 2 datasets + OPT reference) and gate targets
  (T1–T5).

## Open risks handed to later steps
- folktables ACS download needs network (D2); synthetic D1 is self-contained.
- Python scientific stack not yet installed (numpy/scipy/cvxpy absent).
- Paper's benchmarked arms are the unaccelerated ball-oracle variant; grid values unstated and
  disclosed as our choice in SPEC.md §7 (U2/U3).
