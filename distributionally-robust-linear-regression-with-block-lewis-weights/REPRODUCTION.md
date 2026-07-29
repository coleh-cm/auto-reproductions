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

## Status: IMPLEMENTED + RUN (step: implement + review)

### What runs
- `gdr/` package: `problem.py` (folding, E1), `objectives.py` (E3/E4/E5 — verified vs
  finite-diff + Lemma 6.1), `lewis.py` (E6/E7/E8/E9/E11 — overestimate & sandwich
  verified numerically), `reference.py` (E20 CVXPY epigraph), `runner.py`.
- `gdr/solvers/`: subgradient (E22), smoothed_gd/_hb/_nesterov (E3/E22), ipm (E21,
  decreasing-μ central path with inner damped-Newton centering), ball_oracle
  (E19, Moré–Sorensen trust-region Newton; geometry ∈ {euclidean, lewis} with
  the E11 reset), opt_reference (E20).
- `gdr/data_synthetic.py` (D1) and `gdr/data_acs.py` (D2).
- `run_arm.py` (per-arm entrypoint, OPT=1 normalization U15, cached OPT),
  `arms.json`, `run_all_arms.sh`, `smoke.sh`.
- `tests/`: 12 tests pass — degeneracy (Lewis-at-reset == Euclidean bit-identical;
  p=2 objective == least squares) + invariants (Lemma 6.1, E6 overestimate &
  ‖w‖₁≤2(d+1), E7 residual sandwich, smoothed grad/Hess finite-diff & PSD,
  p-grad finite-diff, subgradient validity, ball-oracle monotonicity).

### Numbers (committed in `results/`)

Synthetic (D1, seed=0, cond(AᵀA)=1.40e5, ERM/robust ratio 1.47) — T4 qualitative:
subgradient=NR, smoothed_gd/_hb/_nesterov=NR (plateau), ipm=4,
ball_oracle_euclidean=4, ball_oracle_lewis=4. Matches the paper: IPM & both BO
arms reach the 1% target; first-order methods plateau above it; IPM reaches
the lowest final loss (experiments.tex:106–109).

ACS Income (D2, seed=6, California worst, ERM mean 107.3) — T1 gate:
ball_oracle_euclidean=1, ball_oracle_lewis=1, ipm=10, smoothed_hb=10,
smoothed_gd=45, smoothed_nesterov=10, subgradient=3.
- BO arms ≤ 2 ✓ (paper 1).
- iters(BO)=1 < iters(IPM)=10 ≤ iters(HB)=10 ✓ (ordering holds; paper 1 < 8 < 47).
- subgradient reaches 1% in 3 ✗ (paper: never reaches).
T3 (report-only): ERM mean 107.3 (paper 108.2 ±5 ✓), worst state California
(paper ✓), robust Max/Mean 1.030 (paper 1.02 ✓); but ERM worst 112.7 (paper
138.1 ✗), ERM Max/Mean 1.051 (paper 1.28 ✗), CA decrease 2.04 (paper 24.3 ✗).

### Blocker B1 (ACS heterogeneity, U4/U7)
The reproduced ACS ERM-robust gap is ~1.8% vs the paper's ~25%, so the
subgradient reaches 1% (paper: NR) and IPM/HB counts are compressed. The gap
cannot be enlarged with the faithful d=10 / no-intercept / log1p / 200-per-state
setup (40-seed scan: max ERM worst 116.8, Max/Mean ≤ 1.09). The paper's
per-state heterogeneity (Max/Mean 1.28) traces to an undisclosed ACS
preprocessing/seed (U4/U7). Structure reproduces (CA worst, robust Max/Mean
1.03, ERM mean 107.3); magnitude does not. This is a partial-reproduction
blocker on the T1 subgradient-NR condition and the T3 magnitudes; the BO<IPM≤HB
ordering and BO=1 hold.

### Decisions recorded in SPEC.md §8A
U1 synthetic recipe, U2 grids, U4 ACS preprocessing, U6 OPT-relative gap, U7
seeds (synthetic 0, ACS 6 = California worst), U11 (f̂ not used — §8 is
unaccelerated), U15 (OPT=1 normalization, required for the IPM), U16 (E11 reset
implemented + degeneracy test). Stretch accelerated arms (E16/E17/E18) not
implemented (no §8 number exercises them; U3).
