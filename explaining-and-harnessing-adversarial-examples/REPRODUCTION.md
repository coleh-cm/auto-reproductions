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

## Log

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
     rubbish error 0.0011 (paper 0.68) — same DIRECTION (sigmoid-top far more
     robust than the softmax-top net's 0.82); magnitude sub-scale (5 epochs).
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
