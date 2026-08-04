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
- [x] **2026-08-04 unevaluable-claims fix-up pass:** the prior gate run left 9 claims
  `unevaluable` (defects in the claims contract, not results). All 9 are now evaluable:
  - c12/c13/c17/c34 used `mean(...)` / `min(a,b)` / `max-min`, but the numbers
    gate's expression evaluator exposes no `min`/`max`/`mean` builtins, so each
    raised `NameError`. Rewritten to reference **precomputed derived scalars**
    (new `derived:` metrics computed by `make_measured.py`'s new
    `_compute_derived` step, because the gate can't aggregate inline):
    `m5_maxout1600_advtrain.per_seed_spread` (c13, max−min over seeds),
    `m6_robustness_transfer_eval.min_noise_control_fgsm_error` (c17, min of the
    two m7 noise-control FGSM rates), `m9_rubbish_evals.min_linear_rubbish_error`
    (c34, min of the two linear-model rubbish rates); c12 now uses the existing
    per-seed `mean_test_error` (= `per_seed_test_errors` at sub-scale). c12/c13
    honestly **refute at sub-scale** (1.49% mean, 0.10% spread vs the paper's
    0.782% / 0.06% — both LOW, expected to need 1600-unit/5-seed paper scale);
    c17/c34 **reproduce**.
  - c20/c21 referenced metric names containing dots (`l1_0.0025_train_error`,
    `l1_2.5e-05_test_error`); the gate tokenizes `measured.<arm>.<metric>` by
    splitting on `.`, so the dotted metric parsed as the invalid decimal literal
    `0025_train_error` (`SyntaxError`). Renamed the metric KEYS to dot-free
    `l1_coeff0025_train_error` / `l1_coeff000025_test_error` (the JSON-pointer
    path `arms.l1_0.0025.clean_train_error` — the result-file key — is
    unchanged). c20/c21 **reproduce**.
  - fc1/fc2/fc3 (Figure-4 curve claims) referenced `measured.f4_eps_curve.<curve>`
    but the curve sequences lived only in per-seed files, not `measured.json`,
    so the gate saw "no measurement for correct_logit". `make_measured.py` now
    resolves `curve_metrics` and stores the sequences (61-point ε∈[-15,15] grid,
    same length/order at every seed) verbatim into `measured.json` so a `curve`
    claim reads them point by point. fc1/fc2/fc3 **reproduce**.
  - Tests updated: `test_measured_resolver` exempts `curve_metrics` from the
    scalar-only assertion; `test_claims_integrity` skips `derived:` pointers in
    the file-resolution test; SPEC.md §10 embedded claims.json re-embedded to
    stay byte-identical. **104/104 tests pass; gate 34 pass / 3 fail / 0
    blocked; HIGH 19/19; gate=PASS.**
- [x] Figure a curve claim came from regenerated beside the paper's
  (`experiments/f4_plot.py` → `results/figures/f4_eps_curve_repro{,_seed1,_seed2}.png`
  from the committed curve DATA in `results/f4_eps_curve.json` + per-seed files; axis
  ranges ε∈[-15,15] and y-unit "argument to softmax" asserted against the paper's
  figure inside the plotter, `tests/test_f4_figure.py`). The pair is for a reader to
  compare and is NOT evidence — the gate's verdicts on the curve data are.
- [x] Implementation runs smallest end-to-end case (`smoke.sh`, ~3 s, prints `FINAL adversarial=…`)
- [x] Adversarial review rounds clean (5/6 components approved; the one finding — 4 latent no-success-on-empty graders — fixed and tested; no review-round budget spent, no `$HOME/.review_rounds`)
- [x] Readiness gates walked and recorded (see "Research-readiness gates" table below: 8 pass / 2 partial / 0 fail)
- [x] Numbers compared to paper and published (see "Measured numbers vs paper claims" tables A–C below; `publish_reproduction` called)

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

## Measured numbers vs paper claims (final run, this session)

Every number below is the **mean over 3 seeds (0, 1, 2)** of the arm's headline
metric(s), taken from `measured.json` (rebuilt this session by
`./run_all_arms.sh` → `make_measured.py`; the per-seed values live in
`results/_per_seed/*.json` and are referenced by the `FINAL <arm>=…` lines in
`/tmp/arms.log`). The **exact command** is the arm's `command_per_seed` from
`claims.json`, run once per seed via `make_measured.py` (m5 carries the
documented sub-scale flags `--units 240 --epochs 12`; see VERIFICATION.md §0).
The "paper claim" column is the value the paper states at the cited line of
`paper/source/iclr2015.tex`. **Differences are stated, not adjudicated** — this
table reports measured minus claimed; whether the difference is "in tolerance"
is the reader's call, not this reproduction's. All data is **real MNIST**
(`src/fgsm_repro/data.py`, fingerprinted); no synthetic stand-in was used for
any arm (CIFAR-10 / ImageNet / MP-DBM arms are in `not_tested`, not
substituted — VERIFICATION.md §4).

### Table A — headline per-arm: measured vs claimed

| Arm (command) | Paper claim (tex line) | Measured (mean, 3 seeds) | Δ (measured − claimed) |
|---|---|---|---|
| `m1_softmax_regression`<br>`python experiments/m1_softmax.py --seed {seed}` | FGSM error 99.9 %, conf 79.3 % (tex:333) | error **99.993 %**, conf **95.96 %** | error +0.09 pp; conf +16.7 pp |
| `m2_logistic_3v7`<br>`python experiments/m2_logreg.py --seed {seed}` | clean 1.6 %, FGSM error 99 % (tex:454–456) | clean **1.93 %**, FGSM **99.25 %** | clean +0.33 pp; FGSM +0.25 pp |
| `m3_maxout240_clean`<br>`python experiments/m3_maxout_fgsm.py --seed {seed}` | FGSM error 89.4 %, conf 97.6 % (tex:338–339) | error **96.66 %**, conf **91.51 %** | error +7.26 pp; conf −6.09 pp |
| `m4_maxout240_advtrain`<br>`python experiments/m4_adversarial.py --seed {seed}` | baseline 0.94 % → adv 0.84 % (tex:492–494) | baseline **1.69 %** → adv **1.38 %** | both +0.75 / +0.54 pp (sub-scale; direction holds at every seed) |
| `m5_maxout1600_clean` (sub-scale 240 units / 12 epochs)<br>`python experiments/m5_large_advtrain.py --seeds {seed} --units 240 --epochs 12` | baseline 1.14 % (tex:499) | clean **1.95 %** | +0.81 pp (sub-scale) |
| `m5_maxout1600_advtrain` (same command, adv arm)<br>`python experiments/m5_large_advtrain.py --seeds {seed} --units 240 --epochs 12` | mean 0.782 % over 5 seeds (tex:509) | mean **1.49 %** over 3 seeds | +0.71 pp (sub-scale; needs 1600-unit / 5-seed budget — claim c12 rated `low`) |
| `m6_robustness_transfer_eval`<br>`python experiments/m6_robustness_transfer.py --seed {seed}` | own FGSM 17.9 %, orig→adv 19.6 %, adv→orig 40.9 %, conf 81.4 % (tex:514–523) | own **10.93 %**, orig→adv **34.82 %**, adv→orig **66.49 %**, conf **65.03 %** | own −6.97 pp; orig→adv +15.2 pp; adv→orig +25.6 pp; conf −16.4 pp |
| `m7_maxout_noise_sign`<br>`python experiments/m7_noise_controls.py --seed {seed}` | Bernoulli-noise control FGSM error 86.2 %, conf 97.3 % (tex:555–557) | error **99.69 %**, conf **86.64 %** | error +13.5 pp; conf −10.7 pp |
| `m7_maxout_noise_uniform` (same command, uniform arm)<br>`python experiments/m7_noise_controls.py --seed {seed}` | Uniform-noise control FGSM error 90.4 %, conf 97.8 % (tex:555–557) | error **99.96 %**, conf **87.29 %** | error +9.56 pp; conf −10.5 pp |
| `m8_rbf_shallow`<br>`python experiments/m8_rbf.py --seed {seed}` | FGSM error 55.4 %, conf-on-error 1.2 %, clean conf 60.6 % (tex:600–603) | error **95.05 %**, conf **24.92 %**, clean conf **36.58 %** | error +39.6 pp; conf +23.7 pp; clean conf −24.0 pp |
| `m9_rubbish_evals`<br>`python experiments/m9_rubbish.py --seed {seed}` | maxout+softmax 98.35 %, sigmoid-top 68 %, softmax-reg 59.8 %, RBF 0 % (tex:905–922) | maxout **88.60 %**, sigmoid **68.20 %**, softmax-reg **84.93 %**, RBF **0.00 %** | maxout −9.75 pp; sigmoid +0.20 pp; softmax-reg +25.1 pp; RBF 0.0 (=paper) |
| `e1_ensemble12_maxout`<br>`python experiments/e1_ensemble.py --base-seed {seed}` | ensemble-targeted 91.1 %, single-member 87.9 % (tex:822–823) | targeted **99.88 %**, single **99.71 %** | targeted +8.8 pp; single +11.8 pp |
| `m_l1_weight_decay`<br>`python experiments/m_l1_weight_decay.py --seed {seed}` | coeff 0.0025 → >5 % train error; smaller → no benefit (tex:429–431) | coeff-0.0025 train error **88.64 %**; coeff-2.5e-5 test **1.84 %** (vs baseline test 1.69 %) | 0.0025 stuck (✓ >5 %); smaller coeff gives no test benefit (+0.15 pp) |
| `f4_eps_curve`<br>`python experiments/f4_eps_curve.py --seed {seed}` | Fig. 4: correct class (4) crossed by a wrong class at small ε≈0.5–1 (figure read, tex:762–769) | ε crossover **+0.5** at every seed (curve claims fc1–fc3 pass) | matches the figure-read crossover |

### Table B — M6 transfer sub-numbers (tex:514–523), full detail

These are the four sub-numbers the paper gives for the adversarially-trained
model's robustness and cross-model transfer; the headline `m6` arm measures all
four.

| Quantity | Paper (tex) | Measured (mean, 3 seeds) | Δ |
|---|---|---|---|
| own-FGSM error rate | 17.9 % (tex:514) | 10.93 % | −6.97 pp |
| orig→adv transfer (orig FGSM on adv-trained model) | 19.6 % (tex:521) | 34.82 % | +15.2 pp |
| adv→orig transfer (adv FGSM on orig model) | 40.9 % (tex:521) | 66.49 % | +25.6 pp |
| mean confidence on a misclassified adversarial example | 81.4 % (tex:523) | 65.03 % | −16.4 pp |

### Table C — M8 cross-model agreement sub-numbers (tex:681–687)

| Quantity | Paper (tex) | Measured (mean, 3 seeds) | Δ |
|---|---|---|---|
| RBF predicts maxout's class (over maxout errors) | 16.0 % (tex:681) | 29.55 % | +13.6 pp |
| softmax predicts maxout's class (over maxout errors) | 54.6 % (tex:681) | 62.29 % | +7.7 pp |
| softmax predicts maxout's class (over both-wrong) | 84.6 % (tex:684) | 63.95 % | −20.7 pp |
| RBF predicts maxout's class (over both-wrong) | 54.3 % (tex:684) | 45.87 % | −8.4 pp |
| RBF predicts softmax's class (over maxout errors, secondary on softmax adv) | 53.6 % (tex:687) | 37.35 % | −16.3 pp |

### Where the data came from / what separates the arms

- **Real MNIST throughout.** `src/fgsm_repro/data.py` loads the 4 IDX gz files
  from pinned mirrors; `tests/test_data_fingerprint.py` locks the raw sha256,
  label vocabulary, and 50000/10000 split. No arm used a synthetic stand-in for
  the paper's dataset.
- **The two headline arms are NOT within noise of each other.** For the M4
  comparison the method (adversarial-training) arm's clean test error is below
  the baseline arm's at **every one of the three seeds** (1.47 < 1.78,
  1.25 < 1.82, 1.43 < 1.48 %). A measurement that cannot tell the arms apart
  would not test the paper's comparison; this one does test the *direction*
  (adversarial training reduces clean error), even though the *magnitudes* are
  sub-scale (dropout OFF + 5000 steps vs dropout ON + convergence). The
  magnitudes are rated `compute_invariance=low` and must NOT be read as the
  paper's 0.94 %→0.84 % numbers.
- **Sub-scale training horizon.** M5 was run at 240 units / 12 epochs (paper:
  1600 units / patience-100 early stopping / 5 seeds) because a single
  1600-unit seed did not finish within the sandbox CPU budget. A number
  produced at a horizon too short to separate the M5 arms would not be evidence
  about the paper's claim; the M5 arms **are** separated at this horizon
  (advtrain 1.49 % < baseline 1.95 % at every seed), so the direction is
  tested, but the magnitude (paper 0.782 %) is not, and is not reported as
  such.

## Research-readiness gates

Walked this session. `partial` is used where the honest answer is partial.
The 2 `partial` (gates 1 and 10) rest solely on `docker` not being installed in
this sandbox; the non-Docker evidence for both (fresh `.venv` from
`requirements.txt`, 104/104 tests, smoke + gate run) is verified.

| # | Gate | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Builds from scratch | **partial** | `Dockerfile` present and well-formed; `docker` not installed in this sandbox so `docker build`/`run` not exercised. Equivalent fresh-`venv` build (`uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt`) IS verified — imports clean, 104/104 tests pass. |
| 2 | README is accurate | **pass** | README `## Quickstart` followed verbatim this session in a fresh `.venv`: env-OK import line prints, `pytest -q` → 104 passed, `./run_all_arms.sh` prints the 14 `FINAL` lines, `./smoke.sh` prints its `FINAL` line (~3 s). No gap papered over from memory. |
| 3 | Packages are clear | **pass** | `requirements.txt` pins every dependency with a version (torch 2.7.1, numpy 2.3.2, pytest 8.4.2, matplotlib 3.11.1, …); install from clean succeeds; code then imports without a missing-import death. |
| 4 | The entrypoint is obvious | **pass** | One documented command `./run_all_arms.sh` runs every arm at every seed via flags (`--seed`/`--units`/`--epochs`); no source-edit required to run any experiment. |
| 5 | There is a fast path | **pass** | `smoke.sh` exercises the full path (data→model→FGSM input-grad probe→mixed loss→SGD→eval) in 200 steps / ~3 s, printing `FINAL adversarial=0.11349999904632568`. Path-prover only, never a paper result. |
| 6 | Deterministic, or noise quantified | **pass** | `torch.manual_seed` set before model construction; per-module dropout + batch-shuffle generators seeded from `cfg.seed`. Same seed → bit-identical output. Baseline re-run this session reproduces the committed `results/gate_result.json`. Run-to-run spread is recorded (the per-seed columns in Tables A–C). |
| 7 | Degeneracy test is in the repo | **pass** | `tests/test_degeneracy.py` locks `--lambda 0` (method at its no-op) reproducing `--baseline` bit-for-bit at cost / train-step / CLI level. The strongest cheap plumbing check, shippable so a reader can verify without trusting us. |
| 8 | Data provenance is stated | **pass** | `src/fgsm_repro/data.py` downloads the 4 raw IDX gz from pinned mirrors (storage.googleapis.com/cvdf-datasets/mnist → ossci-datasets.s3.amazonaws.com, 3 tries/mirror, 10 s timeout), cached under `data/mnist/`; `tests/test_data_fingerprint.py` locks raw sha256 + 50000/10000 split. |
| 9 | The recorded number is reproducible | **pass** | `make_measured.py --assemble-only` rebuilds `measured.json` byte-identically from the committed per-seed files (deterministic); the exact command per arm is recorded in `claims.json` `command_per_seed` and shown in Table A. |
| 10 | Nothing depends on hidden local state | **partial** | Runs in a fresh clone of the repo (this branch). Docker-in-a-fresh-container is the untested piece (docker absent); the fresh-`.venv`-from-`requirements.txt` path is verified. `.venv/`, `data/`, `mnist/`, caches are gitignored and recreated. |

**Readiness summary: 8 pass / 2 partial / 0 fail.** No gate failed; the two
`partial` are the Docker-not-installed caveat on gates 1 and 10, with the
non-Docker evidence for both verified.
