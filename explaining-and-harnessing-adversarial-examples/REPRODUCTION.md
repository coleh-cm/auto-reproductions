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

## Final numbers pass (this run)

Every arm of `claims.json` was re-run this session at **3 seeds** (`[0,1,2]`)
by `make_measured.py` (the harness `run_all_arms.sh` delegates to), writing
`measured.json` and one `FINAL <arm>=<headline>` line per arm to `/tmp/arms.log`.
The numbers gate was then re-run: `.venv/bin/python numbers_gate.py` →
`claims_result.json` (`produced_by: "numbers_gate.py"`, **34 pass / 3 fail /
0 blocked**, HIGH **19/19 pass**, `gate_pass: true`). Re-running the arms
reproduced the committed `measured.json` **metric values byte-identically**
(verified leaf-key by leaf-key; the only diff vs HEAD was run-log timing
strings inside `_meta`). **All data is real MNIST** loaded from the pinned IDX
mirrors by `src/fgsm_repro/data.py` and fingerprinted by
`tests/test_data_fingerprint.py` — no synthetic stand-in was used for any arm.
The CIFAR-10 / ImageNet / GoogLeNet / MP-DBM arms are in `not_tested` (§4 of
`VERIFICATION.md`), **not substituted**: a number measured on a synthetic
stand-in for those datasets would say nothing about the paper, so none was
produced.

### A. Measured vs paper-claimed, per arm (MNIST core)

All values are fractions in [0,1] unless marked `%`. "Per seed" = seeds 0/1/2.
The exact command is `claims.json`'s `command_per_seed` with `{seed}` ∈ {0,1,2};
m5's paper-full config is infeasible on this CPU so `make_measured.py` appends
`--units 240 --epochs 12` (the established sub-scale, recorded in
`measured.json['_meta']['subscale_overrides']`).

| Arm | Metric (paper claim, tex) | Paper | Measured mean (per-seed) | Exact command |
|-----|---------------------------|-------|--------------------------|---------------|
| m1 softmax regression | FGSM error rate ε=0.25 (tex:333, c02) | 99.9 % | **99.993 %** (99.99 / 100.00 / 99.99) | `python experiments/m1_softmax.py --seed {seed}` |
| m1 softmax regression | mean conf on errors (tex:333, c03 low) | 79.3 % | **95.96 %** (96.32 / 95.40 / 96.16) | same |
| m2 logistic 3-vs-7 | clean error (tex:454-456, c04) | 1.6 % | **1.93 %** (2.01 / 2.01 / 1.77) | `python experiments/m2_logreg.py --seed {seed}` |
| m2 logistic 3-vs-7 | FGSM error rate (tex:454-456, c06) | 99 % | **99.25 %** (99.12 / 98.82 / 99.80) | same |
| m3 maxout 240×2 clean | FGSM error rate (tex:338-339, c07 low) | 89.4 % | **96.66 %** (97.55 / 96.09 / 96.33) | `python experiments/m3_maxout_fgsm.py --seed {seed}` |
| m3 maxout 240×2 clean | mean conf on errors (tex:338-339, c08 low) | 97.6 % | **91.51 %** (91.33 / 91.92 / 91.27) | same |
| m4 maxout 240 advtrain | baseline clean error (tex:492-494) | 0.94 % | **1.69 %** (1.78 / 1.82 / 1.48) | `python experiments/m4_adversarial.py --seed {seed}` |
| m4 maxout 240 advtrain | adversarial clean error (tex:492-494, c09) | 0.84 % | **1.38 %** (1.47 / 1.25 / 1.43) | same |
| m5 maxout 1600 clean | clean error (tex:500, sub-scale) | 1.14 % | **1.95 %** (1.82 / 1.80 / 2.22) | `python experiments/m5_large_advtrain.py --seeds {seed} --units 240 --epochs 12` |
| m5 maxout 1600 advtrain | clean error mean (tex:506-512, c12 low) | 0.782 % | **1.49 %** (1.44 / 1.50 / 1.54) | same |
| m6 robustness/transfer | own-FGSM error (tex:514, c14) | 17.9 % | **10.93 %** (10.39 / 12.89 / 9.50) | `python experiments/m6_robustness_transfer.py --seed {seed}` |
| m6 robustness/transfer | orig→adv transfer (tex:518, c15) | 19.6 % | **34.82 %** (33.98 / 34.67 / 35.80) | same |
| m6 robustness/transfer | adv→orig transfer (tex:519, c15) | 40.9 % | **66.49 %** (67.30 / 63.97 / 68.19) | same |
| m6 robustness/transfer | conf on own-FGSM errors (tex:522, c16) | 81.4 % | **65.03 %** (65.08 / 64.49 / 65.52) | same |
| m7 noise-sign control | FGSM error (tex:555, c18 low) | 86.2 % | **99.69 %** (99.58 / 99.88 / 99.62) | `python experiments/m7_noise_controls.py --seed {seed}` |
| m7 noise-uniform control | FGSM error (tex:556, c19 low) | 90.4 % | **99.96 %** (99.98 / 99.97 / 99.92) | same |
| m8 shallow RBF | FGSM error (tex:601, c24 low) | 55.4 % | **95.05 %** (94.99 / 95.08 / 95.07) | `python experiments/m8_rbf.py --seed {seed}` |
| m8 shallow RBF | conf on errors (tex:603, c23 low) | 1.2 % | **24.92 %** (24.89 / 24.93 / 24.93) | same |
| m8 shallow RBF | clean confidence (tex:605) | 60.6 % | **36.58 %** (36.57 / 36.58 / 36.59) | same |
| m9 rubbish | maxout+softmax (tex:905, c30) | 98.35 % | **88.60 %** (89.12 / 88.45 / 88.24) | `python experiments/m9_rubbish.py --seed {seed}` |
| m9 rubbish | sigmoid-top (tex:909, c32 low) | 68 % | **68.20 %** (67.57 / 67.46 / 69.58) | same |
| m9 rubbish | softmax regression (tex:912, c31) | 59.8 % | **84.93 %** (83.15 / 85.76 / 85.87) | same |
| m9 rubbish | RBF (tex:912, c33) | 0 % | **0.00 %** (0 / 0 / 0) | same |
| e1 ensemble 12 maxout | ensemble-targeted FGSM (tex:814, c28) | 91.1 % | **99.88 %** (99.87 / 99.89 / 99.89) | `python experiments/e1_ensemble.py --base-seed {seed}` |
| e1 ensemble 12 maxout | single-member-targeted (tex:818, c29 low) | 87.9 % | **99.71 %** (99.71 / 99.70 / 99.70) | same |
| m_l1 weight decay | L1 coeff 0.0025 train error (tex:486, c20) | >5 % | **88.64 %** (88.64 / 88.64 / 88.64) | `python experiments/m_l1_weight_decay.py --seed {seed}` |
| m_l1 weight decay | L1 coeff 2.5e-5 test error (tex:490, c21) | no benefit | baseline−this = +0.04/+0.08/+0.32 pp (3/3 ≥0) | same |
| f4 eps-curve | crossover ε (figure, fc1) | ~0.5–1 | **0.5** (0.5 / 0.5 / 0.5) | `python experiments/f4_eps_curve.py --seed {seed}` |
| f4 eps-curve | frac correct on ε∈[4,15] (fc2) | 0 | **0.00** (0 / 0 / 0) | same |
| f4 eps-curve | max-wrong logit ε=0→15 (fc3) | increasing | 6.92→872.7 / 5.31→480.9 / 6.22→467.6 | same |

### B. Direction (ordering) claims — the load-bearing comparison

The paper's comparison claims are *directional* (adversarial training reduces
clean error; noise controls are weaker regularizers than adversarial training;
RBF is less fooled than linear; softmax agrees with maxout more than RBF does;
transfer is asymmetric; ensemble not resistant). These are the HIGH
`compute_invariance` claims and they are what the sub-scale run can actually
test. Each holds at **3/3 seeds**:

| Claim | Ordering | Per-seed sign (3/3) |
|-------|----------|---------------------|
| c09 M4 advtrain reduces clean error | adv < baseline | −0.31 / −0.57 / −0.05 pp |
| c11 M5 advtrain beats baseline | adv < clean | −0.38 / −0.30 / −0.68 pp |
| c14 advtrained robust vs naive | own-FGSM < naive FGSM | −87.16 / −83.20 / −86.83 pp |
| c15 transfer asymmetry | adv→orig > orig→adv | +33.32 / +29.30 / +32.39 pp |
| c17 noise controls weaker than advtrain | min-noise-FGSM − own-FGSM > 0 | +89.19 / +86.99 / +90.12 pp |
| c22 RBF low confidence when fooled | conf-on-error − clean-conf < 0 | −11.68 / −11.66 / −11.66 pp |
| c25 softmax agrees > RBF agrees (all errors) | softmax−RBF > 0 | +32.27 / +33.21 / +32.71 pp |
| c26 softmax agrees > RBF agrees (both wrong) | softmax−RBF > 0 | +17.14 / +18.69 / +18.40 pp |
| c28 ensemble not resistant | ensemble-targeted error high | True / True / True |
| c30/c31 maxout & softmax-reg fooled by rubbish | error high | True ×6 |
| c34 RBF less fooled than linear on rubbish | RBF − min-linear < 0 | −83.15 / −85.76 / −85.87 pp |
| fc1/fc2/fc3 Figure 4 curve | crosses / below / increasing | 3/3 each |

### C. Honest caveats — what this measurement can and cannot tell the reader

**Shortened training horizon.** This run is at a **strict sub-scale** of the
paper's configuration (m5: 240 units / 12 epochs / 3 seeds vs the paper's 1600
units / patience-100 / 5 seeds, tex:500–512; m4: 5000 SGD steps with dropout OFF
for degeneracy validity vs the paper's dropout-ON training to convergence,
tex:492; e1: 12 members at 5 epochs/patience 10 vs 12 fully-converged nets). A
number produced at a horizon too short to separate the arms would not be
evidence about the paper's claim, so the *magnitudes* in table A are **not**
presented as the paper's numbers — they are rated `compute_invariance=low` in
`claims.json` precisely because they need the paper's full budget. The
**directional** comparison (table B) is what the sub-scale run can test, and the
two headline arms (m4, m5) **do** separate directionally at every seed, so the
paper's comparison is genuinely tested, not washed out by the short horizon.

**The m4 / m5 margins are small relative to seed noise.** The m4 per-seed
adversarial-minus-baseline gaps are −0.31 / −0.57 / **−0.05 pp** — the seed-2
margin is ~5 test examples of 10 000, i.e. at the edge of what 3 seeds resolve.
The direction is consistent (3/3, sign-stable) but the *magnitude* of the
effect is comparable to seed-to-seed variation; the gate settles c09 by the
spec'd per-seed ordering, not by a margin large vs seed noise. The same applies
to c21 (L1 small-coefficient "no benefit": +0.04 / +0.08 / +0.32 pp) and c29
(ensemble-targeted vs single-member: +0.16 / +0.19 / +0.19 pp at saturation).
These are **not** "arms within noise of each other" — the sign is consistent at
3/3 — but a reader should not read the per-seed margins as precise effect sizes.

**Magnitudes far from the paper at sub-scale (informational, low).** The noise
controls (c18/c19: 99.69 %/99.96 % vs 86.2 %/90.4 %), the RBF FGSM error and
confidence (c23/c24: 24.92 %/95.05 % vs 1.2 %/55.4 %), the m6 transfer numbers,
and the m5 mean (1.49 % vs 0.782 %) all miss the paper's magnitudes. These pass
the gate only because their tolerances are wide relative to the sub-scale gap
(tolerance-inflation disclosed per-claim in `claims_result.json` notes); the
load-bearing direction claims (c17 for the noise controls, c22 for the RBF)
carry the real content with real margins. **c03** (softmax FGSM confidence
79.3 %), **c12** (m5 mean 0.782 %) and **c13** (m5 seed-spread 0.0006) **fail**
at sub-scale and are the 3 `low` failures — all expected, all need the paper's
full scale.

**Data provenance, stated.** Real MNIST throughout (no synthetic fallback for
any evaluated arm). No CIFAR-10 / ImageNet / GoogLeNet / MP-DBM number was
produced; those arms are in `not_tested`, not substituted with a stand-in.

### D. The numbers-gate verdict (machine-graded, gate-authored)

`claims_result.json` is written **by** `.venv/bin/python numbers_gate.py`
(`produced_by: "numbers_gate.py"`), never by hand. Re-running the gate against
the shipped `measured.json` reproduces the committed file. Verdict counts:

| compute_invariance | pass | fail | blocked | total |
|--------------------|------|------|---------|-------|
| high (load-bearing) | 19 | 0 | 0 | 19 |
| medium | 6 | 0 | 0 | 6 |
| low (informational) | 9 | 3 | 0 | 12 |
| **all** | **34** | **3** | **0** | **37** |

`FINAL gate=PASS` (gate passes iff every HIGH claim is `pass` with none
blocked). The 3 failures are all `low` and all expected at sub-scale (c03, c12,
c13; see §C above). The verdict is **not** "the paper reproduced" — it is
"every load-bearing directional claim holds at 3/3 seeds at this sub-scale;
the magnitudes do not match the paper and are not claimed to." Tolerance is
the gate's, not a human judgement call; the reader is given the measured and
claimed numbers side by side (table A) to judge.

## Research-readiness gates

Walked this run. `partial` is used where the honest answer is partial. No gate
is marked pass that a fresh checkout would break.

| # | Gate | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Builds from scratch (Dockerfile/env spec) | **partial** | `Dockerfile` present and well-formed; `docker` is not installed in this sandbox so `docker build/run` was not executed. The equivalent from-scratch build (`uv pip install -r requirements.txt` into a fresh `.venv`) IS verified: clean imports, 105/105 tests pass. |
| 2 | README accurate (quickstart verbatim in clean checkout) | **pass** | `README.md` Quickstart run verbatim this run: `uv venv` + `uv pip install -r requirements.txt` → `env OK`; `pytest -q` → 105 passed; `./smoke.sh` → `FINAL adversarial=0.11349999904632568`; `./run_all_arms.sh` → 14 `FINAL <arm>=…` lines + gate. (Fixed a stale line this run that claimed `run_all_arms.sh` prints only `baseline`/`adversarial`.) |
| 3 | Packages clear (versions declared; install succeeds; no missing import) | **pass** | `requirements.txt` pins the full transitive closure (torch 2.7.1 CPU, numpy 2.3.2, pytest 8.4.2, matplotlib 3.11.1). Fresh `.venv` install succeeds; `import torch,numpy,pytest` clean. |
| 4 | Entrypoint obvious (one documented command, flags not source edits) | **pass** | `./run_all_arms.sh` runs every arm x seed and the gate via flags; `experiments/*.py` take `--seed`/`--out`/`--units`/`--epochs` flags. No constants edited to run. |
| 5 | Fast path (smoke config <2 min exercising the whole path) | **pass** | `./smoke.sh` runs data→model→FGSM input-grad probe→mixed loss→SGD→eval in ~3 s, one `FINAL adversarial=…` line. Explicitly a path-prover, not a paper result. |
| 6 | Deterministic, or noise quantified | **pass** | Same seed → bit-identical output; `make_measured.py --assemble-only` rebuilds `measured.json` metrics byte-identically; `tests/test_degeneracy.py` locks `--lambda 0 == --baseline`. This run fixed a rare order-dependent suite flake by adding `tests/conftest.py` (pins `torch.set_num_threads(1)` + `torch.manual_seed(0)` per test); 8/8 consecutive full-suite runs now 105/105. |
| 7 | Degeneracy test in the repo | **pass** | `tests/test_degeneracy.py` (5 tests) asserts the method's no-op (`--lambda 0`) reproduces `--baseline` bit-for-bit at cost/train-step/CLI level; `verify_mutations.py` M2 confirms the no-op floor is guarded (defect breaks it, clean passes). |
| 8 | Data provenance stated | **pass** | `src/fgsm_repro/data.py` downloads 4 IDX gz files from pinned mirrors (cvdf-datasets → ossci-datasets) with retries/timeout; `tests/test_data_fingerprint.py` locks raw sha256, label vocabulary + histogram, 50000/10000 split; missing/corrupt data raises loudly. |
| 9 | Recorded number reproducible (command beside the number, rerun matches within noise) | **pass** | Table A records the exact command beside each number; re-running the arms this session reproduced every metric byte-identically (only run-log timing strings drifted). |
| 10 | Nothing depends on hidden local state (fresh clone, fresh container) | **partial** | Runs in a fresh clone from the committed sources (`.venv/` and `data/mnist/` are gitignored, regenerated). The truly-fresh-*container* run (gate 1) is the untested piece — `docker` not installed. |

**Verdict: 8 pass / 2 partial / 0 fail.** Both `partial` rest solely on `docker`
not being installed in this sandbox; the non-Docker evidence for both is
verified.

## Adversarial review rounds (disclosure)

`$HOME/.review_rounds` records that **4 review rounds** were spent on this
reproduction. The review loop went quiet on substance: the final review round
flagged only non-blocking documentation issues (stale `claims.json` notes that
predated the latest re-run, an epoch-count typo, and tolerance-inflation
disclosures), all of which were acted on this run (see "Second review-pass fix"
above); **no blocking objections remained**. This is reported plainly so a
reader does not mistake a round-budget-exhausted run for an unreviewed one: the
reviewers raised no outstanding blocker, but the round budget was spent getting
there, not zero.
