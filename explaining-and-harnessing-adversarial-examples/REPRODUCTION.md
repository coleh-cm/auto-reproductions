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
