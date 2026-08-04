# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

- **Paper:** EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
- **Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
- **Year:** 2015 (ICLR 2015 conference paper)
- **arXiv:** [1412.6572](https://arxiv.org/abs/1412.6572) (v3, 20 Mar 2015)
- **Date reproduction started:** 2026-08-04

## Status

- [x] Reproduction workspace set up; repo cloned at `~/auto-reproductions` (shallow,
  blobless), working on branch `repro/explaining-and-harnessing-adversarial-examples`
- [x] Paper text saved to `paper/paper.md` (PDF-extracted prose)
- [x] arXiv LaTeX source fetched from `https://arxiv.org/e-print/1412.6572` and unpacked
  to `paper/source/`; `iclr2015.tex` and `iclr2015.bbl` committed (identical to the
  previously committed copies), figures/style files gitignored per folder `.gitignore`
- [x] Paper read properly (full `iclr2015.tex` re-read this run); `SPEC.md` written
  (algorithm, symbol shapes, equation citations, unstated details — carried over from the
  prior completed run on this branch and extended this run, see below)
- [x] Upstream code search recorded (re-run 2026-08-04: 13 third-party partial repos,
  author search 0 repos, pylearn2 link HTTP 200 — unchanged; SPEC.md §7)
- [x] Figures read with `read-figure` and transcript committed (`paper/figure_transcripts.md`);
  Figure-4 reads turned into `curve` claims fc1–fc3 (SPEC.md §12); new arm
  `experiments/f4_eps_curve.py`; `numbers_gate.py` gained curve evaluation; claims.json grew
  34 → 37 claims / 13 → 14 arms / 16 → 19 high; not_tested 10 → 9 (Fig. 4 now tested)
- [x] Environment rebuilt this sandbox (`uv venv .venv` + pinned requirements on CPython
  3.12.13); MNIST downloaded; **89/89 tests pass**
- [x] Numbers gate re-run: `make_measured.py` (f4 arm × seeds 0,1,2) + `--assemble-only` +
  `numbers_gate.py` → **34 pass / 3 fail / 0 blocked; HIGH 19/19 pass; gate=PASS**
  (the 3 fails are the annotated low/magnitude claims c03/c12/c13 needing paper scale)
- [x] Figure a curve claim came from regenerated beside the paper's
  (`experiments/f4_plot.py` → `results/figures/f4_eps_curve_repro{,_seed1,_seed2}.png`
  from the committed curve DATA in `results/f4_eps_curve.json` + per-seed files; axis
  ranges ε∈[-15,15] and y-unit "argument to softmax" asserted against the paper's
  figure inside the plotter, `tests/test_f4_figure.py`). The pair is for a reader to
  compare and is NOT evidence — the gate's verdicts on the curve data are.
- [ ] Implementation runs smallest end-to-end case
- [ ] Adversarial review rounds clean
- [ ] Readiness gates walked and recorded
- [ ] Numbers compared to paper and published

## Notes

- The LaTeX source under `paper/source/` (`iclr2015.tex`) is authoritative for every
  equation, table and reported number; preamble `\def`/`\newcommand` macros must be
  resolved before quoting any equation from it. The PDF-extracted text in
  `paper/paper.md` is reliable for prose only.
- **Prior run:** this slug already has a completed reproduction in this branch's history
  (branch tip `40237e7` at ingest time, landed on `main` as `4f8171c`): full `SPEC.md`,
  implementation under `src/`/`experiments/`, `arms.json`, `claims.json`,
  `measured.json`, `VERIFICATION.md`, `numbers_gate.py`, and the prior run's 54 KB
  `REPRODUCTION.md` log (see git history of this branch — it was replaced by this file
  at this run's ingest commit). This run re-uses and re-verifies that work rather than
  redoing it from scratch.
- **2026-08-04 (this run):** the prior SPEC's figure coverage was the gap — every figure
  claim sat in `not_tested` and no read-figure transcript existed. This run read all six
  figures with the vision tool (`paper/figure_transcripts.md`, 12 exchanges), extracted
  Figure 4's axis semantics (ε ∈ [−15, 15], "argument to softmax", class 4 overtaken at
  ε≈0.5), added the `f4_eps_curve` arm (naive maxout, FGSM-direction logits sweep, 61
  sample points), taught `numbers_gate.py` the `curve` kind (crosses/above/below/
  increasing/decreasing/matches over per-seed sequences; measured.json stays scalar-only),
  added three high-invariance curve claims (fc1–fc3, all pass at every seed), added
  claims-integrity tests (every quote is verbatim at its cited line) and curve-evaluator
  instrument tests, and updated SPEC/VERIFICATION/README/arms_metadata/instruments
  coherently. Gate: 34 pass / 3 fail / 0 blocked; HIGH 19/19 PASS.
- **2026-08-04 (instruments.json `not_applicable` fix):** the prior
  `instruments.json` carried a top-level `not_applicable` *list* naming the two
  out-of-scope instruments (MP-DBM generative inference, CIFAR-10 loader). That
  shape is the global "nothing here judges an output" marker, so a list there can
  be misread as excusing the whole reproduction. Per the contract a top-level
  `not_applicable` must be one sentence for the global-N/A case (absent here,
  because 14 instruments DO judge outputs), and a per-instrument exemption goes
  ON the instrument itself as `"not_applicable": {"reason": ...}` so the other
  instruments still run. Moved `mp_dbm_generative_inference` and `cifar10_loader`
  into `instruments` with `positive_test`/`negative_test` = null and their reason
  inline; removed the top-level list; updated `_doc` and README. 92/92 tests pass;
  gate unaffected (no consumer parsed the field).
- **2026-08-04 (instruments.json declaration-format fix):** the contract requires every
  `positive_test`/`negative_test` to be a strict `<file>::<test>` node. Three instruments
  declared non-conformant shapes: `numbers_gate_scalar_evaluator.positive_test` carried
  trailing prose; `numbers_gate_curve_evaluator` used a brace-list `{a,b,c}` for both
  fields; `claims_quote_verifier.positive_test` carried trailing prose and its
  `negative_test` was a free-form sentence, not a test node at all. The brace/prose
  shapes are not parseable as a single test node, so the declaration checker read them as
  "declared but absent". Fixed: each field is now one `<file>::<test>` naming a canonical
  known-correct (positive) and known-wrong (negative) case. For
  `claims_quote_verifier` this required a NEW negative test
  (`test_claims_verifier_rejects_bad_quote_count_line_pointer`) that proves the verifier
  REJECTS each of the four bad-input kinds the contract names — a non-verbatim quote
  (empty start-line set), a stale compute-invariance count (count mismatch), a shifted
  citation line (line not among real starts), and an unresolvable metric pointer (walk
  finds no key) — so a corrupted `claims.json` could not slip through silently. The other
  curve-gate tests (below/increasing/matches/every-seed/missing-file) remain in
  `tests/test_curve_gate.py` as additional coverage; the named pair is the canonical
  crosses pair (fc1). 93/93 tests pass; gate unaffected (34/3/0, HIGH 19/19 PASS).
- **2026-08-04 (setup pass, this session ~08:40 UTC):** re-ran the ingest step of
  a fresh workflow dispatch for this paper. Re-cloned `coleh-cm/auto-reproductions`
  to `~/auto-reproductions` shallow+blobless (`--depth 1 --filter=blob:none`, clone
  healthy, no tarball fallback needed). `$HOME/.repro_dir` =
  `/root/auto-reproductions/explaining-and-harnessing-adversarial-examples` and
  `$HOME/.repro_branch` = `repro/explaining-and-harnessing-adversarial-examples`
  written (no trailing newlines) and asserted. Continued the existing remote branch
  at tip `5fb3af9` — today's ingest (06:19) and subsequent figure/instruments/
  adversarial-review work (through 07:55) already sit on this ref, having started
  this `REPRODUCTION.md` fresh; nothing from that work was discarded, and the prior
  completed reproduction remains in this branch's history and on `main` (landed as
  `4f8171c`). Branch pushed (up-to-date). Re-fetched the arXiv e-print
  `https://arxiv.org/e-print/1412.6572` (704 KB tarball): the committed
  `paper/source/iclr2015.tex` and `iclr2015.bbl` are byte-identical to the fresh
  download — no fetch failure to record. Full source unpacked on disk under
  `paper/source/` (10 `.png`, `eps_curve.pdf`, `fancyhdr.sty`, `natbib.sty`,
  `iclr2015.sty/.bst`); only `*.tex`/`*.bbl`/`*.bib` are tracked, the rest are
  gitignored per the folder `.gitignore`. `paper/paper.md` verified to carry this
  run's objective text (spot-checked distinctive strings incl. "Potemkin village",
  "0.782", "45.3%", "accidental steganography"). LaTeX remains authoritative for
  every equation, table and number; resolve preamble `\def`/`\newcommand` macros
  before quoting.
- **2026-08-04 (figure-regeneration pass):** the contract requires that any
  figure a curve claim came from be regenerated beside the paper's, with axis
  ranges and units asserted against the paper's figure. Figure 4 (the eps-sweep
  panel, fc1–fc3) was the gap: `experiments/f4_eps_curve.py` recorded the curve
  DATA (and the gate settles the claims against that data) but never drew the
  panel. This pass added `matplotlib==3.11.1` (+8 transitive wheels, pinned in
  `requirements.txt`) — figures only, Agg backend, NOT on the numbers path —
  and a plotter (`experiments/f4_eps_curve._plot_figure4`, surfaced via
  `experiments/f4_plot.py`) that regenerates the Figure 4 LEFT panel from the
  COMMITTED curve data (no retraining): x-axis ε∈[-15,15] with ticks
  -15,-10,-5,0,5,10,15 and label "ε"; y-axis "argument to softmax" (the raw
  pre-softmax logits — the SAME units the paper plots; the paper's numeric
  y-range [-2000,1000] reflects a fully-trained maxout, this sub-scale run is
  smaller-magnitude — the curve claims are SHAPE claims (crosses/below/
  increasing) and scale-invariant, so the magnitude gap is expected and noted
  here, not hidden). The plotter ASSERTS the x-range and y-unit match the
  paper's figure before drawing, so a right-shaped curve on a different scale
  cannot read as a match. Three seed panels committed
  (`results/figures/f4_eps_curve_repro{,_seed1,_seed2}.png`); the paper's panel
  is `paper/source/eps_curve.pdf` (gitignored as a non-reproduced-from binary,
  present on disk). `run_all_arms.sh` now regenerates the panel after
  `make_measured.py` (idempotent, safe to re-run). 7 new tests
  (`tests/test_f4_figure.py`): positive (axis-match writes a real PNG), four
  negatives (tampered x-range / y-unit / eps-span / wrong-data raise loudly),
  custom-filename, and committed-PNG-present + committed-data-axis-contract.
  104/104 tests pass; gate unchanged (34/3/0, HIGH 19/19 PASS). The PNG is for
  a reader to compare and is NOT evidence — the gate's verdicts on the curve
  data are the evidence.
- **2026-08-04 (adversarial component review pass):** ran an orchestrated
  adversarial review of all six components (data pipeline, attacks, objectives,
  training loop, eval metrics, harness) against the paper's `iclr2015.tex`, each
  finding independently verified by a referee subagent that tried to refute it.
  5/6 components approved clean; the eval-metric component surfaced two CONFIRMED
  latent no-success-on-empty contract violations (SPEC §11): `eval_clean_confidence_rbf`
  returned `nan` on a zero-length input (no empty guard; the companion
  `eval_fgsm_rbf` raised correctly via `_eval_from_probs_pred`), and the three
  `eval_rubbish*` graders (`eval_rubbish` / `eval_rubbish_rbf` /
  `eval_rubbish_sigmoid`) returned `AttackEval(error_rate=nan, n=0)` on `n<=0`.
  These are LATENT — the real MNIST test set is non-empty so no reported M8/M9
  number was affected — but a vacuous nan/0.0 verdict silently propagating is
  exactly the failure the no-success-on-empty oracle exists to catch. Fixed:
  all four graders now raise `ValueError` on empty input, matching `eval_clean`
  and `_eval_from_probs_pred`. Added 5 tests
  (`test_eval_clean_confidence_rbf_raises_on_empty`,
  `test_eval_rubbish*_raises_on_empty`) and a `grader_must_raise_on_empty`
  instrument entry. Verified empirically all four raise; verified the
  regenerated per-seed result files (m1/m2/m3 × seeds 0,1,2) are BYTE-IDENTICAL
  to the committed copies (the guards never fire on real data, so `measured.json`
  is unchanged and current). 97/97 tests pass; gate unchanged (34/3/0, HIGH 19/19
  PASS). The 3 low/magnitude fails (c03/c12/c13) remain the annotated
  paper-scale claims, unchanged.
