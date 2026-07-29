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
- [ ] Readiness gates walked and recorded
- [ ] Numbers compared to paper and published

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
  `torch.manual_seed(seed)` before constructing the model. Dropout is DISABLED
  (include prob 1.0) in `run_experiment.py` specifically so the ε=0 degeneracy holds
  bit-for-bit: dropout masks would diverge the RNG between the adv arm (extra forward for the
  input gradient) and the baseline arm. The full milestone experiments may re-enable dropout;
  this runner does not. The monitor-best checkpoint (`result.best_state_dict`) is loaded
  before evaluation, matching the paper's protocol.
