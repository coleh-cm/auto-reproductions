# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

- **Paper:** EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
- **Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
- **Year:** 2015 (ICLR 2015 conference paper)
- **arXiv:** [1412.6572](https://arxiv.org/abs/1412.6572) (v3, 20 Mar 2015)
- **paper_ref:** 43e841f0-05f8-4249-ba14-14c1a586f6ee
- **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Date reproduction started:** 2026-08-04
- **Reproduction dir:** `/root/auto-reproductions/explaining-and-harnessing-adversarial-examples`
- **Branch:** `repro/explaining-and-harnessing-adversarial-examples`

## Status

- [x] Reproduction workspace set up: repo `coleh-cm/auto-reproductions` cloned to
  `~/auto-reproductions` (shallow, blobless: `--depth 1 --filter=blob:none`), folder
  `explaining-and-harnessing-adversarial-examples/` in use.
- [x] Folder path written to `$HOME/.repro_dir` (no trailing newline); branch name
  written to `$HOME/.repro_branch`.
- [x] Branch `repro/explaining-and-harnessing-adversarial-examples` created locally from
  current `main` and pushed to origin.
- [x] Paper PDF-extracted text at `paper/paper.md` (verified to match this run's
  objective text; already on disk from prior merged run).
- [x] arXiv LaTeX source re-fetched fresh from `https://arxiv.org/e-print/1412.6572`
  (gzipped tarball, 704,379 bytes) and unpacked to `paper/source/`. `iclr2015.tex`
  (61,082 B) and `iclr2015.bbl` (6,795 B) are tracked; no `.bib` exists in the source
  (references live in the `.bbl`). Figures (`.png`/`.pdf`), style files
  (`.sty`/`.bst`) and the tarball are gitignored per folder `.gitignore`; they remain
  unpacked on disk for reference only. The freshly fetched `.tex`/`.bbl` are
  byte-identical to the previously committed copies (v3 source, unchanged).
- [ ] Paper read (method sections, equations from `paper/source/iclr2015.tex` with
  preamble macros `\eps` = ε, `\sign` = sign resolved); `SPEC.md` written.
- [ ] Upstream code search done.
- [ ] Implementation runs end to end on the smallest case.
- [ ] Adversarial review loop clean.
- [ ] Readiness gates walked.
- [ ] Numbers measured and published.

## Notes for later steps

**Workspace history.** This folder already contained a complete, merged reproduction
from an earlier run of this workflow (merge commit `536f58f` on `main`: "final EAE
reproduction"), including `SPEC.md`, `README.md`, `src/`, `experiments/`, `tests/`,
`results/`, claims/gate tooling, and a 308-line `REPRODUCTION.md` log. This run's log
starts fresh above; the prior log is recoverable from git history
(`git log main -- REPRODUCTION.md`). The prior implementation is treated as
pre-existing content of this workspace, not as output of this run; every claim about
"this run reproduced X" in later steps must come from work re-executed and
re-verified during this run.

**Paper math reference.** The LaTeX in `paper/source/iclr2015.tex` is authoritative.
Key macros from the preamble: `\eps` = ε, `\sign` = sign, `\veta`/`\vtheta`/`\vx` =
bold vectors. The headline FGSM equation (tex line 309) is
\(\eta = \epsilon\,\mathrm{sign}(\nabla_x J(\theta, x, y))\) — the PDF-extracted text
renders the epsilon as a small square that can be misread.

**arXiv fetch.** Succeeded this run via plain `curl`; no GnuTLS/HTTP-2 failure.
The repo clone (step 1) likewise succeeded shallow+blobless, so no tarball fallback
was needed.

## Implementation pass (this run)

**Method.** This run decomposed the implementation into the five independent
components the SPEC's frozen interfaces already fix — data pipeline, method
core (FGSM + objectives), training loop, evaluation metric, and the baseline
arm + harness — and ran an adversarial review of each against the paper's
LaTeX (`paper/source/iclr2015.tex`) and the SPEC contract, with each flagged
defect then independently verified by a second subagent whose job was to
refute it. Only refutation-surviving findings were acted on. (See the
orchestration run `eae-component-review` for the per-component verdicts.)

**Gate-authorship fix (acted on this pass).** `claims_result.json` is the
contract's evidence table and must be written BY the numbers gate, never by
hand — a hand-authored table replaces four honest verdicts with pass/fail,
which is the one report worse than a failure (two prior runs did exactly this
and both tables were unusable). The gate (`numbers_gate.py`) did not stamp
the file, so a gate-written and a hand-written table were indistinguishable.
Fix: the gate now writes a top-level `produced_by: "numbers_gate.py"` stamp
and its docstring records why. `tests/test_claims_integrity.py::
test_claims_result_json_is_gate_authored_with_produced_by_stamp` asserts the
shipped file carries the stamp AND that re-running the gate re-stamps it (so
the on-disk file is not a stale hand-edit the gate would overwrite). Re-ran
the gate this pass: 34 pass / 3 fail / 0 blocked, 19/19 HIGH pass, gate PASS
— identical verdicts to before, now with the authorship stamp.

**Status of the three `low`-invariance fails.** `c03_softmax_fgsm_confidence_value`,
`c12_m5_advtrain_mean_magnitude`, `c13_m5_seed_spread_invariant` are all
`compute_invariance=low` and fail at this run's CPU sub-scale (the M5 paper-full
config is 1600 units / patience 100 / 5 seeds, infeasible here; `make_measured.py`
runs M5 at 240 units / 12 epochs). They are informational and expected at
sub-scale; the gate's load-bearing verdict is over HIGH claims (19/19 pass).
This is recorded in `claims_result.json['summary']['note']` and in SPEC §9.

## Review-pass fix (this run)

The consolidated adversarial review (faithful + metric + divergence) found the
implementation math faithful to the paper at every equation traced, but flagged
one blocking artifact-integrity defect and several non-blocking doc/notes
issues. All acted on this pass; verdicts unchanged (34 pass / 3 fail / 0
blocked, 19/19 HIGH pass, gate PASS).

**BLOCKER fixed: `claims_result.json` re-authored by the gate.** The working
tree had a hand-/outer-tool-rewritten `claims_result.json` (stamp
`produced_by: "reproduce-paper numbers gate"`, foreign verdict vocabulary
`reproduced/refuted/untested`, dropped `summary`/`not_tested` blocks) that the
repo's own provenance test
(`tests/test_claims_integrity.py::test_claims_result_json_is_gate_authored_with_produced_by_stamp`)
rejected (suite 104/105). Fix: re-ran `.venv/bin/python numbers_gate.py` against
the shipped `measured.json` and committed its verbatim output (stamp
`produced_by: "numbers_gate.py"`, 34 pass / 3 fail / 0 blocked, 19/19 HIGH,
`gate_pass: true`, full `summary` + 9-entry `not_tested` ledger). This restores
the four verdicts the rewrite had mis-adjudicated: `c09` and `c21` (downgraded
to "untested" under a mean-vs-cross-seed-spread rule absent from the spec —
both are pure orderings whose direction holds at 3/3 seeds) and the Fig. 4
curve claims `fc2`/`fc3` (marked "refuted" at grid indices 29/30 = ε=−0.5/0.0,
outside/at the boundary of the claims' declared `x_range`; the gate's
`_restrict` correctly restricts and passes both at 3/3 seeds). The working-tree
`measured.json` differed from HEAD only in run-log timing lines, not in any
metric value (verified leaf-key by leaf-key); the per-seed `results/_per_seed/`
files were untouched. Suite back to 105/105.

**Non-blocking fixes (this pass):**

- *Stale claim notes corrected.* `c09` cited old single-run numbers
  (2.12→1.71 / 1.98→1.64); replaced with the current per-seed M4 values
  (1.78→1.47 / 1.82→1.25 / 1.48→1.43, adv−base = −0.31/−0.57/−0.05 pp) plus an
  explicit power caveat (the per-seed margins are ~31/57/5 test examples of 10k,
  t≈−2.1, df=2 — consistent in direction, small, comparable to the paper's own
  ~0.10 pp effect; settled by the spec'd per-seed ordering, not by a margin
  large vs seed noise). `c28` said "Sub-scale (4 members): 99.76%" but the run
  used 12 members; corrected to the per-seed ensemble-targeted FGSM error
  (99.87 / 99.89 / 99.89%). `c29` predicted the ordering "REVERSED (99.76 vs
  99.79)" but the measured per-seed gaps are +0.16/+0.19/+0.19 pp — positive at
  3/3 seeds, in the paper's direction; corrected the note (the gap is thin at
  saturation, hence `low`, but not reversed).
- *`c21` power caveat added.* The small-coefficient L1 benefit ordering holds at
  3/3 seeds (+0.04/+0.08/+0.32 pp, ~4/8/32 test examples of 10k) but the margins
  are small; recorded that the claim is settled by the spec'd per-seed ordering
  rather than by a margin large relative to seed noise.
- *SPEC §6 item 9 free-β paragraph corrected.* The paragraph described the
  superseded design ("β is a free parameter with a negative-definite init
  (−0.01·I)"), contradicting `models.py:298-323`, the SPEC table, and item 31
  (β is `−diag(a_k)`, `a_k = softplus(raw_k) > 0`, negative-definite BY
  CONSTRUCTION). A free β with neg-def init drifts positive under softmax-CE
  training (verified), leaving the RBF family and breaking the off-manifold
  confidence-decay mechanism; the construction constraint is what makes the
  paper's RBF immunity phenomenology structurally reachable. The paragraph now
  matches the code.
- *`c11` (M5) arm-name cross-reference added.* The arms are named
  `m5_maxout1600_*` after the paper's 1600-unit configuration, but the sub-scale
  run used `--units 240 --epochs 12` (recorded in `measured.json
  _meta.subscale_overrides`). The note now states this explicitly so a
  name-only reader is not misled; the name is retained for stable claim/metric
  pointers.

**M5 early-stopping horizon caveat (sub-scale limit).** The paper's M5
protocol (tex:501-512) selects the number of epochs by early stopping on the
*adversarial* validation error, then retrains on all 60k. At this sub-scale
(`select_max_epochs=12`), the adversarial-validation early-stopper reaches the
epoch cap rather than plateauing: the adversarial arm's best adv-valid epoch is
index 11/12 at seeds 0 and 1 (the final epoch) and 9/12 at seed 2. The
directional claim (c11) still measures — the arms genuinely diverge (adv−clean
= −0.38/−0.30/−0.68 pp at 3/3 seeds) — but the protocol's signature feature
(plateau-driven early stopping) is only partially exercised at this 12-epoch
horizon; at the paper's patience-100 scale it would fire well before the cap.

**Untouched (intentionally).** The review's strongest divergence objection —
that the RBF β negative-definite-by-construction "manufactures" the paper's RBF
phenomenology — was dismissed as not a divergence: E8 (tex:595) prints a decaying
probability that is valid only for neg-semi-definite β, the paper's immunity
claim is expressly architectural (tex:600-604), and a free β drifting positive
leaves the RBF family entirely. The structural nature of the 0%-rubbish result
is already documented (`claims.json` c33 note; SPEC items 9/31). The optional
`nt08` MNIST rubbish per-class tally (45.3% classified as 5s / 0% as 8s,
tex:924-926) is implementable from the existing M9 machinery but was left for a
later pass to avoid destabilizing the gate.

## Second review-pass fix (this run)

A follow-up adversarial re-read (faithful + metric + divergence) confirmed the
implementation math and the gate-authored `claims_result.json` were correct
(re-running `.venv/bin/python numbers_gate.py` against the shipped
`measured.json` reproduces the committed file byte-for-byte: 34 pass / 3 fail /
0 blocked, 19/19 HIGH, `gate_pass: true`; `measured.json` vs HEAD differs only
in run-log timing strings, zero metric drift; suite 105/105; mutations 6/6
verified). It flagged only stale `claims.json` notes that predated the latest
re-run and misstated measured values (verdicts unaffected — the gate reads
`measured.json`, not the notes), plus an epoch count and tolerance-inflation
disclosures. All acted on this pass.

**Stale notes refreshed (seven claims).** The notes are documentation; the
gate verdicts are unchanged. Each note now states the current per-seed
measured values and the 3-seed mean:

- `c01` — "100.0% / 99.1% / 99.8%" → per-seed FGSM error softmax
  99.99/100.00/99.99%, logistic 99.12/98.82/99.80%, maxout 97.55/96.09/96.33%
  (mean 96.66%); all far above the 0.5 existence threshold.
- `c07` — "sub-scale 99.8%" → fgsm error per-seed 97.55/96.09/96.33% (mean
  96.66%).
- `c08` — "Sub-scale 88.7%" → mean confidence-on-errors per-seed
  91.33/91.92/91.27% (mean 91.51%).
- `c30` — "Sub-scale: 82.1%" → maxout-softmax rubbish error per-seed
  89.12/88.45/88.24% (mean 88.60%).
- `c31` — "Sub-scale: 81.7%" → softmax-regression rubbish error per-seed
  83.15/85.76/85.87% (mean 84.93%).
- `c32` — "Sub-scale 79.2%" → sigmoid-top rubbish error per-seed
  67.57/67.46/69.58% (mean 68.20%, vs paper 68% — genuinely close, not saved
  by the tolerance).
- `c09` — "12 epochs" corrected to "20 epochs" (the M4 run used
  `max_epochs=20`, `epochs_run=20` at all 3 seeds, per
  `results/_per_seed/m4_adversarial__seed{0,1,2}.json`); the per-seed M4
  error numbers in the note were already correct.

**Cross-model agreement notes refreshed (three claims).** `c25`/`c26`/`c27`
notes quoted seed-0 values where the gate evaluates per-seed; replaced with
the per-seed ranges and means (c25: 62.77/63.46/60.63% vs 30.50/30.25/27.92%,
means 62.29% vs 29.55%; c26: 64.48/64.81/62.55% vs 47.34/46.11/44.15%, means
63.95% vs 45.87%; c27: 38.69/37.23/36.12%, mean 37.35%).

**Tolerance-inflation disclosed (four `low` value claims).** `c18`/`c19`/
`c23`/`c24` pass only because their tolerances are wide relative to the gap
from the paper's value at this sub-scale; their notes now say so explicitly
and point to the load-bearing ordering claims that carry the real content
(c17 for the noise controls; c22 for the RBF). c18 measured 99.69% mean vs
paper 86.2% (tol 0.15); c19 99.96% vs 90.4% (tol 0.15); c23 24.92% vs 1.2%
(tol 0.25, gap 0.237 ≈ 20× the target); c24 95.05% vs 55.4% (tol 0.45, gap
0.396 ≈ half the [0,1] scale). All four are `compute_invariance: low`
(informational, off the load-bearing gate); the direction claims they
support (c17, c22) hold with real margins.

**SPEC.md re-embedded.** `SPEC.md`'s section-10 embedded `claims.json` was
re-synced to the updated file; `test_spec_embedded_claims_json_is_byte_identical`
passes (suite 105/105).
