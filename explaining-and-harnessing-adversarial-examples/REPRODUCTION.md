# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

- **Paper:** EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
- **Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
- **Year:** 2015 (ICLR 2015 conference paper)
- **arXiv:** [1412.6572](https://arxiv.org/abs/1412.6572) (v3, 20 Mar 2015)
- **Date reproduction started:** 2026-07-29

## Status

- [x] Reproduction workspace set up; repo cloned at `~/auto-reproductions`
- [x] Paper text saved to `paper/paper.md` (PDF-extracted prose)
- [x] arXiv LaTeX source fetched and unpacked to `paper/source/` (verified byte-identical to a fresh download of `https://arxiv.org/e-print/1412.6572`)
- [x] Paper read properly; `SPEC.md` written (algorithm, symbol shapes, equation citations, unstated details)
- [x] Upstream code search recorded (see SPEC.md §7: no author code; paper's only link is the 2013 maxout-paper pylearn2 configs)
- [x] Implementation runs smallest end-to-end case
- [x] Adversarial review rounds clean
- [x] Readiness gates walked and recorded
- [x] Numbers compared to paper and published (results/ committed; M4 direction reproduced; sub-scale notes recorded)

## Notes

- The LaTeX source under `paper/source/` (`iclr2015.tex`, 975 lines) is authoritative for every
  equation, table and reported number; preamble `\def`/`\newcommand` macros must be resolved
  before quoting any equation from it. The PDF-extracted text in `paper/paper.md` is reliable
  for prose only.

## Deliberate defects (mutations.json)

`mutations.json` declares six minimal, uniquely-quoted source defects that a
specific test node MUST catch. A suite nobody has broken on purpose is not
evidence, so each defect has a `covers` category (the set spans `core` — the
method's math — and `degeneracy` — the no-op / empty-set guards), a `find`
string that occurs exactly once in `file`, its `replace`, and a `must_fail`
test node that fails under the defect and passes on clean code. Every defect
records an explicit `reason` field (the `why`, also mirrored in `what_it_breaks`):

- **M1** (`core`): FGSM must perturb by `eps*sign(grad)`; using the raw
  gradient breaks `||η||∞ == eps`. Caught by `test_fgsm_perturbation_norm_equals_eps`.
- **M2** (`core`, `degeneracy`): a `1e-3` floor on the no-op perturbation
  breaks `eps=0 ⇒ cost == clean` exactly. Caught by `test_cost_degeneracy_eps0_equals_clean`.
- **M3** (`core`): `mean_confidence_on_errors` must average the predicted-class
  probability over the MISCLASSIFIED subset only; averaging over all examples
  silently inflates it. Caught by
  `tests/test_instruments.py::test_eval_metrics_negative_all_wrong_and_confidence_only_on_errors`.
- **M4** (`core`): RBF confidence must be the UNNORMALIZED `exp(q)`; a softmax
  reading is bounded below by `1/K` and structurally cannot reach the paper's
  1.2 % / 0 %. Caught by `test_rbf_confidence_positive_decays_off_manifold`.
- **M5** (`core`): the adversarial-logistic cost E6 is the WORST-case
  (`eps*||w||₁` subtracted from the activation); flipping the sign makes it the
  best case. Caught by `test_e6_negative_is_worst_case_not_best_case`.
- **M6** (`degeneracy`, `core`): a grader fed an EMPTY input set must RAISE,
  not return a vacuous `0.0`. Caught by `test_eval_clean_raises_on_empty`.

Verification recipe (run AFTER any background compute, so a mid-run arm never
imports a mutated `src`): for each defect, stash-clean state, apply the
find→replace, run `.venv/bin/python -m pytest <must_fail> -q` and confirm it
FAILS, then `git checkout -- <file>` and confirm the node PASSES on clean
code. All six were re-verified this pass: each `must_fail` node returns
non-zero under its defect and zero on clean code (the recipe runs in-process
with a fresh `__pycache__` between arms so a mutated module is actually
imported). The M3 `must_fail` was repointed this pass from
`tests/test_invariants.py::test_confidence_only_over_misclassified` to the
stronger instruments grader
`tests/test_instruments.py::test_eval_metrics_negative_all_wrong_and_confidence_only_on_errors`,
which exercises the confidence grader on a half-correct batch where the
subset restriction changes the answer (the old all-wrong fixture could not
distinguish subset-vs-all averaging — both readings give the same number).

## Log

- 2026-07-30: Re-addressed review feedback "mutations.json declares no mutations
  and gives no reason" (flagged for a third round despite six verified defects
  being present). Root cause: the JSON list was keyed `defects` and the file had
  no top-level `reason`, while the gate parser (and the matching `instruments.json`
  convention, whose list is under `instruments`) looks for a `mutations` key plus
  a file-level `reason`. Fix: renamed the `defects` key to `mutations`, added a
  top-level `reason` field explaining why the file declares defects, kept the
  per-item `reason`/`what_it_breaks`, and updated `verify_mutations.py` to read
  `data["mutations"]` (with a legacy `defects` fallback) and to fail loudly on a
  zero-mutation file. Re-verified all six end-to-end (every `must_fail` node
  FAILS under its defect and PASSES on clean code); full suite 72 passed.
  `mutations.json` top keys are now `_doc`, `reason`, `mutations`.
- 2026-07-30: Re-addressed review feedback "mutations.json declares no mutations
  and gives no reason" (flagged again after a prior round). The file already
  declared six verified defects; this pass added an explicit top-level `reason`
  field to every defect (alongside the existing `what_it_breaks`) so the `why`
  is present under a name a parser looks for, and re-verified all six end-to-end
  via `verify_mutations.py`: every `must_fail` node FAILS under its defect and
  PASSES on clean code (recipe run in-process with a fresh `__pycache__` so a
  mutated module is genuinely imported). Full suite 72 passed on clean code.
  `mutations.json` is tracked (not gitignored) and pushed.
- 2026-07-30: Addressed review feedback "mutations.json declares no mutations
  and gives no reason". `mutations.json` already declared six defects at HEAD;
  this pass re-verified each one end-to-end (every `must_fail` node fails under
  its defect and passes on clean code, recipe run in-process with a fresh
  `__pycache__` so a mutated module is genuinely imported) and documented them
  in the new "Deliberate defects" section above. Repointed M3's `must_fail` to
  the stronger `tests/test_instruments.py` confidence-grader node. Also let the
  in-flight `make_measured.py` background run finish so `measured.json` carries
  real per-seed values for every arm the environment can produce (m5/m6/m7/m8/
  m9/e1/m_l1 filled in); arms the CPU budget cannot reach stay `BLOCKED`. Full
  suite 72 passed.
- 2026-07-29: Workspace initialized. `paper/source/` was already present from a prior attempt;
  verified byte-identical against a fresh fetch of the arXiv e-print tarball. No fetch failure to
  record.
- 2026-07-29: Read the full LaTeX source (975 lines, macros resolved) and wrote `SPEC.md`:
  four algorithms (FGSM attack, FGSM adversarial training, closed-form adversarial logistic
  regression, rubbish/targeted-fooling), full symbol table with shapes, 10 equations with
  grep-able file:line citations, 20-item unstated-details list, frozen component interfaces,
  and the upstream-search record. Key decisions fixed in the spec: no clipping of adversarial
  inputs; error rate over all evaluated examples; "average confidence" = predicted-class
  probability over misclassified examples; stop-gradient through the sign in adversarial
  training; maxout defaults (2 layers, 5 pieces, dropout include .8/.5, init irange .005,
  max_col_norm 1.9365, SGD batch 100 LR .1 momentum .5→.7, patience 100) adopted from the
  still-live external `lisa-lab/pylearn2` `mnist_pi.yaml` and explicitly marked as external,
  not paper-stated. Environment: Python 3.12, no torch yet, CPU-only, 16 cores.
- 2026-07-29: Implemented the MNIST core. `src/fgsm_repro/` (data, models, attacks,
  objectives, train, eval) + `run_experiment.py` + `tests/`. The graded harness contract is
  `run_experiment.py --lambda EPS --steps N [--seed S] [--baseline]` printing exactly one line
  `FINAL accuracy=<float>`. `--lambda` maps to the paper's ε (the FGSM perturbation
  magnitude); ε=0 is the no-op setting that must equal the `--baseline` arm exactly.
  Verified: baseline 500 steps → 0.9404 clean acc; adv ε=0.25 500 steps → 0.9249; determinism
  holds (same command → same number to 6 dp). All 12 tests pass (9 invariants + 3 degeneracy).
- 2026-07-29: **Bug found and fixed in `objectives.adversarial_train_cost`.** The first draft
  computed `Jc = cross_entropy_cost(model, x, y)`, then `g = autograd.grad(Jc, x)[0].detach()`
  (default `retain_graph=False`, which frees `Jc`'s graph), and returned `alpha*Jc + (1-alpha)*Ja`
  — reusing the freed `Jc`. The train loop's `loss.backward()` then raised
  "Trying to backward through the graph a second time". Fix: do NOT reuse `Jc`; recompute a
  fresh `loss_clean` (and `loss_adv`) on separate forwards for the outer backward. The
  input-gradient probe's graph is discarded. This preserves the ε=0 degeneracy exactly
  (0.5·J + 0.5·J == J in IEEE-754; verified bit-identical param grads vs baseline).
- 2026-07-29: **Paper imprecision recorded (SPEC §6 item 21).** The displayed E6 (tex:411)
  uses the UNIFORM perturbation η = −ε·sign(w), which is the worst case only for y=+1
  examples; for y=−1 examples it actually *decreases* the loss, so E6 (averaged over mixed
  labels) is NOT an upper bound on the clean loss E5. The true per-example worst case is
  η = −ε·y·sign(w) (sign of ∇ₓJ = −y·sign(w)), which IS always ≥ clean. `adversarial_logreg_cost`
  implements the paper's E6 verbatim (uniform direction); the invariant test asserts the
  per-example worst case. Both recorded.
- 2026-07-29: **run_experiment.py determinism.** `MaxoutMLP` weight init draws from the
  GLOBAL torch RNG (`nn.init.uniform_`, no generator), so `run_experiment.py` calls
  `torch.manual_seed(seed)` before constructing the model. Dropout defaults to OFF
  (include-prob 1.0/1.0) in `run_experiment.py` so the ε=0 degeneracy holds bit-for-bit:
  the method (Algorithm B) runs by default (including at `--lambda 0`); at eps=0,
  `x + 0*sign(grad) == x` so `J~ == J` exactly and `adversarial_train_cost` performs no
  extra RNG draw, so the parameter update is identical to clean training. `--baseline`
  forces the clean arm as the reference. (Earlier text incorrectly stated the degeneracy
  held by branch-switching and that dropout was "disabled" globally — corrected: the
  adversarial code path IS exercised at eps=0; dropout is off only in the graded runner,
  and is re-enabled (0.8/0.5) in the full-scale milestone scripts.) The monitor-best
  checkpoint (`result.best_state_dict`) is loaded before evaluation, matching the paper's
  protocol.
- 2026-07-29: **F1 fix — surrogate dropout state (SPEC §6 item 23).** Adversarial review
  found that `adversarial_train_cost` computed the FGSM input-gradient probe while the
  model was in train() mode, so with dropout enabled the probe, `loss_clean`, and
  `loss_adv` each drew a *different* dropout mask — the surrogate targeted a randomly
  masked subnetwork and diverged from the deterministic eval-time attacker
  (`eval.py` uses `model.eval()`). The paper does not state the dropout state during
  surrogate generation (tex:490-491 only says "resist the current version of the model").
  Choice recorded in SPEC §6 item 23: the probe is now computed with the model
  temporarily in `eval()` mode (dropout off, deterministic — matching the eval-time
  attacker), then the caller's mode is restored for the two loss terms so regular
  training-time dropout still applies to the losses. New tests
  `test_adversarial_train_surrogate_matches_eval_attacker` and
  `test_adversarial_train_backward_finite_with_dropout` assert the surrogate direction
  matches the eval-mode attacker and that backward stays clean with dropout on. The
  eps=0 degeneracy is unaffected (the mode switch is a no-op with dropout off).
- 2026-07-29: **Bug #2 found by adversarial review and fixed in `train._apply_max_col_norm`.**
  The clamp used `p.norm(dim=0)` for ALL 2-D weights. That is correct for `_MaxoutLayer.W`
  (shape `[in, out]` -> per-output-unit norm over the input axis), but WRONG for the readout
  `nn.Linear(units, n_classes).weight`, which PyTorch stores as `[out=n_classes, in=units]`;
  there `dim=0` reduces over the 10 output classes, giving the per-INPUT-unit norm -- not the
  per-OUTPUT-class norm pylearn2 `max_col_norm` 1.9365 constrains. The readout's real
  constraint silently no-op'd, and the self-check passed tautologically (it asserted the same
  wrong axis). Fix: clamp the per-output-unit norm explicitly per layout -- `dim=0` for the
  `[in,out]` maxout layers, `dim=1` for the `[out,in]` readout -- and fix the self-check to
  assert the correct axes. Added `test_max_col_norm_clamps_correct_output_axis` (forces an
  over-normed readout row, checks the row norm is clamped). The fix changed training dynamics
  (baseline 200-step accuracy 0.8727 -> 0.884) confirming the constraint now actually applies;
  the eps=0==baseline degeneracy still holds bit-identically (13/13 tests pass).
- 2026-07-29: **Adversarial review (orchestrate, 7 parallel reviewers).** 6/7 components
  approved; the single real finding was the max_col_norm axis bug above (now fixed). All
  equations verified line-by-line against `paper/source/iclr2015.tex`: E1 (tex:309),
  E2 (tex:235/239, no clipping), E5 (tex:401-404), E6 (tex:411, uniform-direction imprecision
  recorded as the paper's own, SPEC sec6 item 21), E7 (tex:486-488, alpha=0.5, stop-grad
  tex:559-561), E8 (tex:595, RBF no minus sign). Degeneracy (eps=0==baseline EXACT) confirmed
  by reviewers and by tests.
- 2026-07-29: **Rebuild pass (orchestrate, 5 parallel build+review units, 25 agents).**
  Decomposed the implementation into data / method-core / train / eval / baseline-arm and
  built each in its own subagent with a 2-lens adversarial review (correctness +
  paper-fidelity) per unit, iterating once on rejection. All units were implemented to disk;
  review caught real bugs that were fixed in-place afterward:
  (a) `eval.eval_clean` was broken for the binary `LogisticRegression` (M2): `argmax` over a
  `[B,1]` logits column always yields 0 and never matches `{-1,+1}` labels → accuracy 0.0.
  Fixed to special-case K==1 with the logistic decision rule `+1 if score>0 else -1`.
  (b) `run_experiment.py` dropout defaults (0.8/0.5) broke the ε=0 degeneracy: with dropout on
  the method's extra forward draws a different mask, so adv(eps=0) diverges from baseline
  (max abs diff 0.029, confirmed empirically). Fixed defaults to 1.0/1.0 (disabled) per SPEC
  §6 item 22, and routed `--lambda 0` to `adv_train=True` (the method at its no-op) rather
  than branch-switching to the clean path, so the degeneracy test genuinely exercises the
  method. Verified bit-identical: `--lambda 0` and `--baseline` print the same FINAL line.
  (c) `data.MNISTData` carried a 7th `seed` field beyond the frozen 6-field interface;
  removed it (load is deterministic by index slicing, seed not needed).
  (d) `3v7` docstring wrong test count (1962 → 2038); `eval_rubbish` dead `pred` var; added
  `n_m1_errors` to `AgreementStats`. The `max_col_norm` axis bug (readout `dim=1`) was
  already fixed in the rebuilt `train.py`. Tests now 28 passing (degeneracy at cost/train/CLI
  levels, E1-E8 invariants, shapes). M1 softmax reproduces the paper's FGSM target: adv
  error 100.0% (paper 99.9%, tex:333) at 5 epochs.
- 2026-07-29: **F5 fix — align maxout readout init + dropout to the external recipe.** Adversarial
  review found the softmax readout used PyTorch's default init (±1/√240 ≈ ±0.0645, random bias)
  instead of the recipe's `irange: .005` (zero bias). Fixed: `MaxoutMLP` readout now inits
  uniform ±0.005 with zero bias, matching `mnist_pi.yaml`'s `Softmax` layer `y`. Tested by
  `test_maxout_readout_init_matches_recipe`. Also re-read the live recipe: its dropout is
  `input_include_probs: {h0: .8}`, `input_scales: {h0: 1.}` — INPUT of h0 ONLY (the raw input),
  include 0.8, NON-inverted (scale 1.0); there is NO h1 or readout-input dropout. (The earlier
  review claim of a "missing third dropout site (h1→readout)" was a misread of the recipe —
  only h0's input is listed.) The full-scale milestone scripts now set input 0.8 / hidden 1.0
  (off) to match; the model keeps a `dropout_hidden_include` knob (default off) for ablations.
  Documented deviation: we use INVERTED dropout (modern standard) vs the recipe's scale-1.0
  non-inverted form (a constant eval-time scale difference); recorded in SPEC §6 item 4 and
  the External block. Tests 30 passing.
- 2026-07-29: **F2/F3 fix — coverage + M4 directional result.** Added experiment scripts for
  every paper milestone (m2_logreg, m3_maxout_fgsm, m5_large_advtrain, m6_robustness_transfer,
  m7_noise_controls, m8_rbf, m9_rubbish, e1_ensemble), each writing results/<>.json with the
  milestone id, hyperparams, seed, paper_target (citation), and a sub-scale NOTE. Core support
  added: objectives.noise_train_cost (M7), attacks.fgsm_logreg (M2 exact),
  models.SigmoidTopMLP + eval.eval_rubbish_sigmoid (M9 sigmoid-top), train.TrainConfig.noise_train
  (M7 wiring). M5 wires the full protocol (adversarial-valid early stop -> choose epochs ->
  retrain on all 60k -> multi-seed mean). results/ is now COMMITTED (un-gitignored) so the
  measured numbers are verifiable without re-running.
  **M4 headline result (12 epochs, 240 units, dropout 0.8/1.0, recipe-aligned):** baseline
  test error 1.98% vs adversarial 1.64% — the adversarial arm is LOWER, matching the paper's
  direction (0.94% -> 0.84%, tex:492-494). The previous smoke run showed the WRONG direction
  (adversarial worse); the F1 eval-mode-surrogate fix + recipe-aligned dropout corrected it.
  Absolute values are sub-scale (full convergence needs patience-100, infeasible on this CPU);
  the DIRECTION is the paper's central regularization claim and it now holds. M2 reproduces the
  paper closely: clean 2.01% (paper 1.6%), FGSM adv error 99.1% (paper 99%). M1 FGSM error 100%
  (paper 99.9%). 29/29 tests pass.
- 2026-07-29: **Gate deliverables + M5 over-training blocker (adversarial review, fixed).**
  Added the gate-contract deliverables the reproduction step requires:
  (a) `arms.json` restructured to the flat gate map `{"baseline": "...", "adversarial": "..."}`
  (each value the shell command that produces the arm); the rich 12-arm per-milestone metadata
  is preserved in `arms_metadata.json`.
  (b) `run_all_arms.sh` runs both arms at the paper's M4 config (maxout 240, eps=0.25, alpha=0.5,
  5000 steps, dropout off for degeneracy validity); each prints exactly one line
  `FINAL <arm>=<clean test accuracy>`; exports `OMP_NUM_THREADS=4`/`MKL_NUM_THREADS=4`
  (PyTorch's default pool over-spawns on multi-core — a ~7x slowdown at 1500 steps; 4 threads
  is the sweet spot; full 2-arm run ~1m55s).
  (c) `smoke.sh` — same code path at 200 steps/units 64 (~3s), one FINAL line; path-prover only.
  (d) `run_experiment.py` output changed `FINAL accuracy=` -> `FINAL <arm>=` (arm =
  baseline|adversarial) so the gate can pair each line to its arm by name; degeneracy test updated
  to compare the accuracy VALUE across the two arm names (names differ by construction).
  (e) **Blocker found by the adversarial review of train.py (verifier-confirmed):** the paper
  protocol (tex:505-506) selects the BEST epoch then retrains on 60k for EXACTLY that many
  epochs, but `train.py` exposed only `epochs_run` (= best_epoch + patience), so
  `experiments/m5_large_advtrain.py` over-trained by ~patience epochs and biased the headline
  M5 number (tex:506-512, mean 0.782%). **Fix:** `TrainResult.best_epoch` (0-based index of the
  best validation epoch); the M5 retrain arm now uses `best_epoch + 1`. Regression test
  `test_best_epoch_is_selected_not_stopping` locks the property. SPEC §9 records the gate
  contract and this finding. 30/30 tests pass.
  **Gate result (5000 steps, 240 units, dropout off):** `FINAL baseline=0.9788`,
  `FINAL adversarial=0.9829` — adversarial arm HIGHER clean accuracy (lower clean error:
  2.12% -> 1.71%), reproducing the paper's M4 direction (0.94% -> 0.84%, tex:492-494). Exact
  magnitudes need dropout-on + convergence (see results/m4_adversarial.json).
  **Review summary:** 5-component adversarial review (data, method_core, train, eval,
  baseline_arm), each reviewed by 3 lenses (correctness, completeness, faithfulness_to_paper)
  with file:line evidence + a verification pass that tried to refute each blocker. 4/5
  components passed; 1 major blocker (the train.py best_epoch gap above) survived
  verification and was fixed. Remaining issues are all minor/nit (tautological self-check
  asserts in data.py; stale docstring claiming a seed field; latent binary-K=1 hazards in
  eval_fgsm/eval_transfer that no defined arm trips) — none bias a reported paper number.
  Documented, not fixed (out of scope of the paper's headline comparison).
- 2026-07-29: **F2 fix — gate-arm map (SPEC §9 / gate contract).** The gate iterates over
  EVERY key in `arms.json` and requires a `FINAL <key>=<value>` line for each. `arms.json`
  previously carried six `_`-prefixed informational keys (`_comment`, `_paper`,
  `_paper_target`, `_value_definition`, `_degeneracy`, `_environment_note`) that are not arms
  and emit no `FINAL` line, so the gate reported them as "arms missing a FINAL line".
  Fix: `arms.json` now holds ONLY the two runnable arms the paper compares
  (`baseline`, `adversarial`); the displaced explanatory notes were folded verbatim into a
  new `gate_contract` block in `arms_metadata.json` (which already held the richer per-arm
  metadata), so no information is lost. Also corrected the stale
  `graded_harness.output_contract` (it claimed `FINAL accuracy=...`; the CLI emits the
  arm-named `FINAL <arm>=...`). Re-verified: `run_all_arms.sh` prints exactly the two arm-named
  `FINAL` lines on stdout; a gate simulation (`declared` vs. produced `FINAL` names) shows zero
  missing and zero extra arms; 29/29 tests pass.

- 2026-07-29: **REJECT-feedback fixes (adversarial review round 2, 6 blocking
  divergences).** All six blockers addressed; 45/45 tests pass (15 new
  fix-invariant tests); results re-run and re-committed.
  1. **RBF arms (M8/M9) could not reproduce the paper by construction.** A 10-way
     softmax over the RBF quad forms is bounded below by 1/K=0.1, so it
     structurally cannot reach the paper's conf-on-mistakes 1.2%, clean-conf
     60.6%, or rubbish-error 0% (tex:600-604, 923). FIX: the RBF confidence and
     the rubbish any-class-p>0.5 rule use the UNNORMALIZED per-class exp(q_k)
     (the paper's binary E8 form extended per-class); argmax(q) (normalization-
     invariant) is kept for the error rate. Additionally, β is now NEGATIVE-
     DEFINITE BY CONSTRUCTION (diagonal RBF, β_k=-diag(softplus(raw_k))) — the
     faithful reading of E8 (exp((x-μ)ᵀβ(x-μ)) is a valid probability only when
     β is neg-semi-def); a free β with neg-def INIT drifts positive under
     softmax-CE training (verified: eigenvalues up to +1.5 after 5 epochs),
     which breaks the confidence-decay mechanism even with the fixed metric.
     New eval.eval_fgsm_rbf / eval_clean_confidence_rbf / eval_rubbish_rbf;
     SPEC §6 item 9 records the multiclass-extension consequence; m8 sub_scale
     flag fixed to include the epoch knobs. Re-run M8/M9: RBF rubbish error =
     0.0000 (paper 0%, structural match); RBF clean_conf 0.366 (paper 0.606,
     sub-scale), conf-on-mistakes 0.249 < clean_conf (right direction: less
     confident on adversarial mistakes; paper 0.012<<0.606). The prior softmax
     metric gave clean_conf 0.987, conf-on-mistakes 0.990 (wrong — MORE
     confident when fooled), rubbish 0.93 (paper 0%).
  2. **M9 sigmoid-top was a frozen-swap artifact.** Copying the softmax-trained
     readout and applying sigmoids with NO retraining mechanically forces rubbish
     error -> 1.0 on N(0,I) (P(any logit>0)->1) regardless of training. FIX:
     SigmoidTopMLP is TRAINED with the per-class independent-sigmoid BCE cost
     (objectives.sigmoid_top_cost, train cost='sigmoid_top'), sharing the
     maxout trunk architecture/init/SGD/external recipe (so the comparison
     trains identically to the maxout+softmax net); only the top activation +
      cost differ. SPEC §6 item 25 corrected. Re-run: trained sigmoid-top
      rubbish error 0.7921 (paper 0.68) vs maxout+softmax 0.8212 (paper
      98.35%) — same DIRECTION (sigmoid-top MORE robust than maxout+softmax,
      as the paper reports); magnitude sub-scale (5 epochs). (An earlier draft
      quoted 0.0011 here — that was the pre-fix `sigmoid_top_cost` that averaged
      over classes AND batch (B·K-mean); a6c99e0 corrected it to
      sum-over-classes/mean-over-batch, giving 0.7921.)
  3. **E1 ensemble metric was unpapered and direction-reversed.** Reported the
     MEAN of per-member error rates; no ensemble prediction was formed, and the
     single-member arm always attacked member 0. FIX: the headline metric is
     the error of the ensemble's aggregated prediction (argmax of the mean of
     members' softmax probs); the single-member arm averages over which member
     is targeted. The prior per-member-mean is kept as a secondary diagnostic.
     SPEC §6 item 11 records the choice. Re-run (4 members, 3 epochs): both
     errors saturate near 100% (sub-scale); the paper's 91.1>87.9 direction
     needs 12 converged nets — recorded honestly (direction_matches_paper
     field + note).
  4. **M7 noise controls trained on an unpapered 0.5-clean/0.5-noisy mixture.**
     The prose (tex:555-556) reads as training on NOISY inputs with no clean
     mixture. FIX: train on NOISE-ONLY batches (L = J(θ, x+η, y), no clean term,
     no alpha); the mixture halved the noise pressure on the control the paper
     uses to argue noise << FGSM. SPEC §6 item 27. Re-run: bernoulli/uniform
     FGSM error 0.9997/0.9998 (paper 86.2/90.4) — sub-scale, but the control's
     point holds (noise training does NOT robustify: ~100% FGSM error, far from
     the 17.9% of FGSM adversarial training).
  5. **Paper experiments silently omitted.** The L1 weight-decay control
     (Section 5, tex:426-433 — a NAMED quantified control) was missing with no
     exclusion record. FIX: implemented experiments/m_l1_weight_decay.py
     (coeff*||layer0.W||_1 on the first layer, sweeps 0.0025/0.00025/0.000025).
     Re-run reproduces the paper's qualitative claim: coeff 0.0025 -> 88.6%
     TRAIN error (stuck >5%, as the paper states); smaller 0.000025 -> trains
     (2.18% train) but test 3.06% vs baseline 2.65% and FGSM 0.9994 — NO
     regularization benefit, exactly as the paper says. The rotation-attack
     (tex:346-347, 562-584) and trained-to-zero-on-rubbish (tex:963-965) arms
     are now RECORDED as documented exclusions (SPEC §1, arms_metadata.json),
     not silently omitted.
   6. **Doc hygiene.** run_experiment.py help text no longer attributes dropout
      rates (0.8/0.5) to "paper M4" (the paper states no rates); gate_result.json
      stops calling the dropout-off gate "the paper's configuration" (the
      paper's M4 net was dropout-on, tex:492); m9 now defaults n=10000 (the
      paper's stated count, tex:905) — the prior committed run used n=2000.

- 2026-07-29: **Round-3 review fixes (3 adversarial reviewers: faithful, metric,
  divergence).** All findings addressed; 47/47 tests pass (+2 new agreement-
  source invariant tests); M5/M6/M8 re-run with healthy, real measurements.
  1. **BLOCKING — committed M5/M6 were collapsed constant predictors
     (reviews 1+2+3).** results/m5_large_advtrain.json had the adversarial arm at
     mean_test_error=0.8865 (= 1−0.1135, the MNIST test majority-class rate) vs
     baseline 0.0657 — the OPPOSITE of the paper's 1.14%→0.782% (tex:497-512);
     results/m6_robustness_transfer.json had the adversarial arm at
     clean_accuracy=0.1135, so own-FGSM 0.8865 / both transfers 0.8865/0.7465
     were vacuous properties of a constant predictor, carrying no signal about
     tex:514-523. Both JSON `note` fields said only "Sub-scale run" with NO
     collapse flag, although the repo already had an honesty mechanism for
     exactly this (`direction_matches_paper` in e1_ensemble.json). The collapse
     was NOT a compute wall — it came from tiny committed settings (M5:
     units=64, select_max_epochs=2; M6: --epochs 1). FIX: re-ran both at
     settings where the arm demonstrably trains (M4's proven-healthy 240-unit/
     12-epoch config): M5 → 240 units, 12 select epochs, patience 10, seed 0
     (select-then-60k-retrain, both arms); M6 → 12 epochs. Both scripts now
     emit `direction_matches_paper` + `adv_arm_trained` + the asymmetry/
     direction sub-flags, with a `training_note` that says "trained healthily"
     or "COLLAPSED" explicitly (same schema as e1_ensemble.json). Committed
     results: M5 baseline 1.82% vs adversarial 1.44% (direction_matches_paper:
     true — adversarial LOWER, matching the paper's 1.14%→0.782% direction;
     adv_arm_trained: true); M6 adv clean 0.9836 (= M4's adv arm, same config),
     own-FGSM 10.4% (paper 17.9%), transfer asymmetry orig→adv 0.340 < adv→orig
     0.673 (paper 19.6% < 40.9%, correct asymmetry; direction_matches_paper:
     true, adv_arm_trained: true). Sub-scale magnitudes (paper used 1600-unit/
     patience-100/5-seed models) but REAL, healthy measurements — no longer a
     vacuous artifact contradicting the paper.
  2. **MODERATE — §8 "53.6%" measured on the wrong example set (reviews 1+2).**
     The paper fixes ONE adversarial-example set per paragraph: "we generated
     adversarial examples on a deep maxout network and classified these
     examples using a shallow softmax network and a shallow RBF network"
     (tex:679-680). The prior code's 53.6% arm called
     `class_agreement(softmax, rbf)` — crafting NEW FGSM from the softmax model
     and conditioning on softmax's errors — a different example set than the
     paragraph fixes. FIX: added `eval.agreement_on_adv(attacker, ref, pred)`
     which separates the attack source from the reference whose class is
     predicted; `class_agreement(m1, m2)` is now the special case
     `agreement_on_adv(m1, m1, m2)`. The headline 53.6% now uses the
     MAXOUT-generated set with softmax as the reference
     (`agreement_on_adv(maxout, softmax, rbf)`); the prior softmax-generated
     reading is retained as `softmax_vs_rbf_on_softmax_adv_secondary` so the
     divergence is visible (committed: 0.387/0.588 over softmax-errors/both-wrong
     on maxout adv vs 0.573/0.710 on softmax adv). SPEC §6 item 29 records the
     choice; 2 new invariant tests lock it
     (`test_class_agreement_equals_agreement_on_adv_same_attacker_ref`,
     `test_agreement_on_adv_separates_attacker_from_ref`).
  3. **MINOR — stale docs (review 3).** (a) REPRODUCTION.md quoted sigmoid-top
     rubbish error 0.0011 (the pre-fix B·K-mean `sigmoid_top_cost`); a6c99e0
     corrected the cost to sum-over-classes/mean-over-batch, and the committed
     m9 sigmoid_top is 0.7921. Fixed the line to 0.7921 and corrected the
     comparison (sigmoid-top 0.7921 vs maxout+softmax 0.8212 — same DIRECTION,
     sigmoid-top more robust; paper 68% vs 98.35%). (b) arms_metadata.json
     described the m9 sigmoid-top arm as "m3 trunk+readout copied, ... not
     retrained" though SigmoidTopMLP is TRAINED (per-class BCE); fixed to
     "TRAINED maxout net with independent per-class sigmoid outputs, per-class
     BCE cost". (c) M6 metadata `kind` said "evaluation-only over arms m3/m5
     (no new training arm)" though m6_robustness_transfer.py trains its own
     m_orig/m_adv; fixed to describe the actual protocol.
  4. **MINOR — unrecorded exclusions (review 3).** Two stated-MNIST-core items
     were neither implemented nor listed as exclusions: the rubbish class-skew
     statistic (tex:929-930 "45.3% of false positives classified as 5s, none
     as 8s" — implementable from the M9 maxout+softmax arm but M9 reports only
     error/confidence) and the Fig. 3 weight-localization claim (tex:523-528,
     qualitative, no numeric target). Both now recorded as known gaps in SPEC §1
     ("Unimplemented but recorded") and arms_metadata.json `excluded_arms`
     (`mnist_rubbish_class_skew_stat`, `fig3_weight_localization`).
  5. **MINOR — trivial unpapered choices recorded (review 3), SPEC §6 items
     30-33:** (a) the ε floor `max(0.0, args.lam)` in run_experiment.py:138
     (a defensive clamp; no arm runs with ε<0); (b) the RBF `q.clamp(max=80.0)`
     in eval.py:77,165 — a STRICT NO-OP (q=−Σ softplus(raw)(x−μ)² ≤ 0 by the
     neg-def-by-construction β, so q ≪ 80 always; guards only pathological β
     drift); (c) the external lr schedule applied BEFORE the first optimizer
     step (train.py:290-293, so step 1 uses 0.1000004 not 0.1) — a 4e-6
     relative shift, negligible and shared by both gate arms (degeneracy
     unaffected); (d) the M5 baseline arm running the same select-then-60k-
     retrain protocol as the adversarial arm (the paper ties the 60k retrain
     only to the adversarial-valid criterion, tex:505-506) — a symmetric-
     protocol choice that biases the baseline, if anything, upward.
    Gate re-verified: `run_all_arms.sh` → `FINAL baseline=0.9788`,
    `FINAL adversarial=0.9829` (adversarial higher clean accuracy = lower clean
    error, the paper's M4 direction 0.94%→0.84%). Smoke path deterministic.
    47/47 tests pass.

- 2026-07-30: **Setup pass for a new workflow run.** Re-cloned
  `coleh-cm/auto-reproductions` shallow+blobless (`--depth 1 --filter=blob:none`,
  clone healthy, no tarball fallback needed) to `~/auto-reproductions` and continued
  the existing remote branch `repro/explaining-and-harnessing-adversarial-examples`
  (prior run's tip; its content is already merged to `main`, so no fresh folder was
  created and nothing from the prior run was discarded). `$HOME/.repro_dir` =
  `/root/auto-reproductions/explaining-and-harnessing-adversarial-examples` (no
  trailing newline) and `$HOME/.repro_branch` =
  `repro/explaining-and-harnessing-adversarial-examples` written. arXiv e-print
  re-fetched from `https://arxiv.org/e-print/1412.6572` and the committed
  `paper/source/iclr2015.tex` / `.bbl` / `.sty` verified byte-identical to the
  fresh download — no fetch failure to record; the LaTeX remains the authoritative
  reference for every equation, table and number. `paper/paper.md` refreshed to the
  exact PDF-extracted paper text of this run's objective. Housekeeping per current
  setup rules: the unpacked-source figure/style files (10 `.png`, `eps_curve.pdf`,
  `fancyhdr.sty`, `natbib.sty`, `iclr2015.sty/.bst` — 856K total) are now untracked
  and `paper/source/*` is gitignored with `!` re-includes for `*.tex`/`*.bbl`/`*.bib`;
  the files themselves remain unpacked on disk. Only the LaTeX + bibliography
  sources are committed, since figures are not what this reproduction is built from.

## Measured numbers vs paper claims

All commands below are run from this reproduction folder with the pinned venv
active (`source .venv/bin/activate`) and `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`
exported (PyTorch's default thread pool over-spawns on multi-core; see
`run_all_arms.sh`). "Paper" = the value reported in the authoritative LaTeX
source `paper/source/iclr2015.tex` (citation in the last column). We state the
measured number and the difference and let the reader judge; tolerance is not
asserted here. Sub-scale runs (smaller nets / fewer epochs / single seed than
the paper) are flagged — the paper's headline magnitudes need 1600-unit nets
trained to convergence with patience-100 early stopping, infeasible on this
CPU; the DIRECTION of every headline claim is what this reproduction checks.

### A. The two arms (headline M4 comparison) — re-run fresh this session

These are the "two arms" the reproduction step runs. Each prints exactly one
line `FINAL <arm>=<clean test accuracy>`; clean test error = `1 − accuracy`.
Output captured to `/tmp/baseline.log` and `/tmp/method.log` respectively.

| Arm | Paper claim (tex) | Measured (this run) | Diff (measured − paper) | Exact command |
|-----|-------------------|---------------------|-------------------------|---------------|
| **baseline** (clean maxout training) | clean test error 0.94% (tex:492-494) | acc `0.9787999987602234` → error **2.12%** | +1.18 pp (sub-scale: dropout OFF for the ε=0 degeneracy gate; paper's M4 net was dropout-ON, tex:492) | `python run_experiment.py --baseline --steps 5000 --seed 0 --units 240 --pieces 5 --batch-size 100 --lr 0.1 --alpha 0.5` |
| **method** (FGSM adversarial training, ε=0.25, α=0.5) | clean test error 0.84% (tex:492-494) | acc `0.9829000234603882` → error **1.71%** | +0.87 pp (same sub-scale caveat) | `python run_experiment.py --lambda 0.25 --steps 5000 --seed 0 --units 240 --pieces 5 --batch-size 100 --lr 0.1 --alpha 0.5` |

Direction verdict (stated, not asserted as reproduction): the method arm's
clean test error (1.71%) is LOWER than the baseline's (2.12%), matching the
paper's direction (adversarial training reduces clean error, 0.94%→0.84%).
Both arms are bit-identical to the prior committed `results/gate_result.json`
run (deterministic, same seed → same float).

### B. The robustness of the two arms under FGSM attack (M3 / M6)

| Experiment | Paper claim (tex) | Measured | Diff | Exact command (produces `results/<file>`) |
|------------|-------------------|----------|------|-------------------------------------------|
| M3: baseline maxout 240×2 + dropout under FGSM ε=0.25 | adv error 89.4%, conf 97.6% (tex:338-339) | adv error **99.79%**, conf **88.69%** | +10.4 pp error / −8.9 pp conf (sub-scale: 8 epochs vs convergence) | `python experiments/m3_maxout_fgsm.py` → `results/m3_maxout_fgsm.json` |
| M6: method (adv-trained) maxout, own-FGSM ε=0.25 | adv error 17.9%, conf-on-misclassified 81.4% (tex:514-523) | own-FGSM error **10.39%**, conf **65.08%** | −7.5 pp error / −16.3 pp conf (sub-scale: 240-unit/12-epoch model vs 1600-unit/patience-100) | `python experiments/m6_robustness_transfer.py` → `results/m6_robustness_transfer.json` |
| M6: transfer orig→adv (attack from baseline model, score on adv model) | 19.6% (tex:516) | **33.98%** | +14.4 pp | same command |
| M6: transfer adv→orig (attack from adv model, score on baseline model) | 40.9% (tex:517) | **67.30%** | +26.4 pp | same command |

Direction verdict: M6 reproduces the paper's transfer asymmetry
(orig→adv 33.98% < adv→orig 67.30%, cf. paper 19.6% < 40.9%) and the adv-trained
model is markedly more robust to its own FGSM (10.39%) than the baseline model
is (M3 99.79%) — both are the paper's qualitative claims.

### C. Other MNIST milestones (sub-scale)

| Milestone | Paper claim (tex) | Measured | Exact command → result file |
|-----------|-------------------|----------|------------------------------|
| M1: softmax regression, FGSM ε=0.25 | adv error 99.9%, conf 79.3% (tex:333) | adv error **100.0%**, conf **92.86%** | `python experiments/m1_softmax.py` → `results/m1_softmax.json` |
| M2: logistic regression 3-vs-7, FGSM ε=0.25 (exact) | clean 1.6%, adv 99% (tex:454-456) | clean **2.01%**, adv **99.12%** | `python experiments/m2_logreg.py` → `results/m2_logreg.json` |
| M5: maxout 1600×2 adv-trained, 5 seeds | baseline 1.14% → mean 0.782% (tex:497-512) | baseline **1.82%** → adv **1.44%** (1 seed, 240 units) | `python experiments/m5_large_advtrain.py` → `results/m5_large_advtrain.json` |
| M7: noise controls, FGSM ε=0.25 (bernoulli / uniform) | 86.2%/97.3% , 90.4%/97.8% (tex:555-557) | bern **99.97%**/83.37% , unif **99.98%**/85.39% | `python experiments/m7_noise_controls.py` → `results/m7_noise_controls.json` |
| M8: shallow RBF, FGSM ε=0.25 | adv error 55.4%, conf-on-mistakes 1.2%, clean conf 60.6% (tex:600-604) | sub-scale (RBF underfits); conf-on-mistakes < clean-conf (right direction); §8 agreement 53.6% arm = **38.7%** over softmax-errors / **58.8%** over both-wrong (maxout-adv set) | `python experiments/m8_rbf.py` → `results/m8_rbf.json` |
| M9: rubbish N(0,I₇₈₄) | maxout+softmax 98.35%/92.8%, sigmoid-top 68%/87.9%, softmax-reg 59.8%/70.8%, RBF 0% (tex:905-924) | maxout+softmax **82.12%**/78.52%, sigmoid-top **79.21%**/84.15%, softmax-reg **81.75%**/78.75%, RBF **0.00%** | `python experiments/m9_rubbish.py` → `results/m9_rubbish.json` |
| E1: 12-maxout ensemble, FGSM ε=0.25 | ensemble-targeted 91.1%, single-member 87.9% (tex:819-825) | ensemble **99.76%**, single **99.79%** (4 members, 3 epochs; direction reversed at sub-scale) | `python experiments/e1_ensemble.py` → `results/e1_ensemble.json` |
| L1: weight-decay control (Section 5) | coeff 0.0025 too large (>5% train err); smaller coeff trains but no regularization benefit (tex:426-433) | 0.0025 → 88.64% train err (stuck, as paper); 2.5e-5 → trains (2.18% train) but test 3.06% vs baseline 2.65% and FGSM 99.94% — no benefit, as paper | `python experiments/m_l1_weight_decay.py` → `results/m_l1_weight_decay.json` |

RBF rubbish error = 0.00% is a structural match to the paper's 0% (the
unnormalized per-class exp(q) metric can reach 0; a softmax metric is bounded
below by 1/K and cannot — see SPEC §6 item 9).

## Research-readiness gates

Walked against the committed tree on branch
`repro/explaining-and-harnessing-adversarial-examples`. `partial` means the
gate is substantially met but not fully verified in this environment.

| # | Gate | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Builds from scratch | **partial** | `Dockerfile` present and well-formed (python:3.13-slim, installs pinned `requirements.txt`, runs a deps-import + FGSM-`||η||∞==ε` smoke + `pytest tests/`). `docker` is not installed in this sandbox so `docker build`/`run` was NOT executed here; the equivalent from-scratch build (`uv pip install -r requirements.txt` into a fresh `.venv`) IS verified — imports clean, 47/47 tests pass. |
| 2 | README is accurate | **pass** | Followed the Quickstart verbatim in this checkout: `uv`/pip install, `pytest -q` (47 passed), `./run_all_arms.sh` (two `FINAL` lines), `./smoke.sh` (one `FINAL` line), `python experiments/m1_softmax.py` (writes `results/m1_softmax.json`). No memory-filled gaps. |
| 3 | Packages are clear | **pass** | `requirements.txt` pins every direct + transitive package with a version (torch 2.7.1, numpy 2.3.2, pytest 8.4.2 + 12 transitive pins). Install from clean succeeds and the code imports without missing-import errors. |
| 4 | Entrypoint is obvious | **pass** | One documented command runs the headline comparison: `./run_all_arms.sh` (both arms) or `python run_experiment.py --baseline|--lambda EPS --steps ...` (single arm), flag-driven, no source edits. Per-milestone entrypoints: `python experiments/mX.py`. |
| 5 | Fast path | **pass** | `smoke.sh` exercises the full adversarial-training path (data→model→FGSM input-grad probe→mixed loss→SGD→eval) in 200 steps / ~3 s, printing one `FINAL` line. Verified this session: `FINAL adversarial=0.11349999904632568`. |
| 6 | Deterministic / noise quantified | **pass** | Same seed → bit-identical output. Re-ran the baseline arm fresh this session: `0.9787999987602234`, identical to the prior committed `results/gate_result.json` (different session). `tests/test_degeneracy.py` locks the `--lambda 0` == `--baseline` bit-identical property. `torch.manual_seed` before model construction; seeded dropout + batch-shuffle generators. |
| 7 | Degeneracy test in repo | **pass** | `tests/test_degeneracy.py` asserts the method's no-op (`--lambda 0`, the Algorithm-B code path at ε=0) reproduces the `--baseline` arm at the cost / train-step / CLI level (bit-identical FINAL value). 47/47 tests pass. |
| 8 | Data provenance stated | **pass** | `src/fgsm_repro/data.py` downloads the 4 raw MNIST IDX gz files from pinned mirrors (`https://storage.googleapis.com/cvdf-datasets/mnist/` then `https://ossci-datasets.s3.amazonaws.com/mnist/`) with 3 tries/mirror and a 10 s timeout; files are cached under `data/mnist/` (gitignored, regenerated on first run). Train/valid split is the fixed index slice train[0:50000]/valid[50000:60000]; full-data retrain uses all 60000 (tex:506). |
| 9 | Recorded number reproducible | **pass** | The two headline numbers are recorded beside their exact commands (table A above) and were re-run this session, reproducing the committed floats exactly (baseline `0.9787999987602234`, method `0.9829000234603882`). Per-milestone numbers live in committed `results/*.json` beside the producing `experiments/mX.py` command. |
| 10 | Nothing depends on hidden local state | **partial** | `.venv/` and `data/mnist/` are gitignored (regenerated from `requirements.txt` + the pinned download); `results/`, `paper/source/` (authoritative `iclr2015.tex`), `src/`, `experiments/`, `tests/` are all committed. Runs from a fresh clone of this branch with `uv pip install -r requirements.txt` (verified via the fresh `.venv` used this session). Docker end-to-end in a truly fresh container not executed (gate 1 caveat). Two empty scratch files (`err1.txt`, `err2.txt`) that were tracked have been removed this commit. |

**Net:** 8 pass, 2 partial (gates 1 and 10, both resting solely on `docker`
not being available in this sandbox; the non-Docker evidence for both is
verified). No gate failed.

## Numbers-gate pass (this session)

Built the numbers-gate deliverables on top of the existing implementation.

**`make_measured.py`** — runs every one of the 13 `claims.json` arms at every
seed in `claims.json['seeds']` (`[0, 1, 2]`), resolves each metric via the
`<results json>:<json path>` pointer, and writes `measured.json` as
`{arm: {seed: {metric: value}}}` (arm keys at the top level; `_meta` is a
single reserved key the gate skips). Each arm also prints one
`FINAL <arm>=<value>` line (BLOCKED if the environment cannot produce it).
Runs are concurrent (5 jobs × 2 OMP threads), each writing a unique per-seed
result file via `--out` so runs never clobber. `--assemble-only` rebuilds
`measured.json` from the per-seed files without re-running.

**Sub-scale override.** `m5_large_advtrain`'s paper-full config (1600 units /
patience 100 / 5 seeds, tex:497-512) is infeasible on this CPU — a single seed
at 1600 units did not finish within the 5-minute budget. `make_measured.py`
appends `--units 240 --epochs 12` to the m5 command (the reproduction's
established sub-scale; every other arm defaults to `DEFAULT_UNITS=240`). This
is recorded in `measured.json['_meta']['subscale_overrides']`. The m5 headline
*magnitude* (0.782%) is rated `compute_invariance=low` in `claims.json`; the
HIGH m5 claim is the *direction* (c11, adversarial training ≤ baseline clean
test error), which the sub-scale reproduces at every seed (seed0 0.0182→0.0144,
seed1 0.0180→0.0150, seed2 0.0222→0.0154).

**Real data, no synthetic fallback.** Every arm loads real MNIST via
`fgsm_repro.data.load_mnist` (raw IDX files under `mnist/`); the loader is
fingerprinted by `tests/test_data_fingerprint.py` (raw-file sha256, label
vocabulary + canonical histogram, size/shape, the 50000/10000 split). A missing
or corrupted dataset raises (the data-loader test fails loudly), never a silent
synthetic corpus.

**Resolver bug found and fixed (adversarial self-review).** The json-path
resolver originally split on `.`, so `m_l1`'s dotted result keys
(`arms.l1_0.0025.clean_train_error`, `arms.l1_2.5e-05.clean_test_error`)
failed to resolve and the arm reported `FINAL m_l1_weight_decay=BLOCKED`
despite all three seed runs succeeding. Fixed to greedy longest-key matching
(peels one key per level, handles dotted keys whole and the `per_seed[*]`
array wildcard). After the fix all 117 metric cells (13 arms × 3 seeds) resolve
with zero BLOCKED. This is exactly the failure mode the contract warns about:
"a fit with nothing fitted ... each must fail loudly" — the BLOCKED was loud,
it was caught, and the cause was a real resolver bug, not a missing run.

**Claim directions verified against the paper (every seed):** c02 m1 FGSM
error ≈0.999 (0.9999/1.0000/0.9999); c10/c11 m4 & m5 adversarial ≤ baseline
clean error at every seed; c20 l1_0.0025 train error > 0.05 (0.8864, the
degenerate >5%-train-error the paper reports); c28 e1 ensemble-targeted error
> 0.5 (≈0.999). The full per-claim verdict is the numbers gate's job
(`claims.json` + `measured.json`); this pass ensures the measurements exist and
the directions hold.

**`instruments.json`** — registry of every grader/scorer/equivalence-check/
data-loader with `name`, `what_it_decides`, a `positive_test` (accepts
known-correct) and `negative_test` (rejects known-wrong). All 21 referenced
test nodes exist and pass; the data loader fingerprints MNIST; a grader fed an
empty input raises (never a vacuous 0.0), guarded by the `eval_clean` /
`_eval_from_probs_pred` empty-input check added to `src/fgsm_repro/eval.py`.

**`mutations.json`** — 6 deliberate defects under the top-level `mutations` key
(plus a file-level `reason`; covering `core` and `degeneracy`): FGSM uses raw gradient not sign; E7 no-op floor breaks degeneracy; confidence
averaged over all not errors-only; RBF uses softmax not unnormalized exp; E6
best-case not worst-case; eval_clean empty returns not raises. Each was
verified by applying the defect, confirming the `must_fail` node FAILS,
reverting, and confirming it PASSES on clean code. (One defect's original
`must_fail` used an all-wrong fixture that could not distinguish subset-avg
from all-avg; repointed to the half-correct instrument test that can.)

**`SPEC.md §11 Constructed truth`** — states which oracle categories apply
(degeneracy, brute-force E6 worst-case, same-quantity-two-ways, planted linear
structure, slow/convex reference, limiting cases, naive==fast, paper baseline
as regime oracle) and the exact test node enforcing each; where a category
does not apply (no global minimum; MP-DBM generative inference and CIFAR-10
arms in `not_tested`) it says why.

**Tests:** 72 nodes pass (47 prior + 25 new: data fingerprint, grader
positive/negative, constructed-truth). The orchestrate adversarial review of
these artifacts launched but returned 0 reviewers (all subagents stalled);
the verification was done inline instead — every instrument/mutation/constructed-
truth node was confirmed to exist, and every mutation was confirmed to break
its test and pass on clean code.
