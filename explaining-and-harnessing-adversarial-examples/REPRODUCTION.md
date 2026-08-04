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
