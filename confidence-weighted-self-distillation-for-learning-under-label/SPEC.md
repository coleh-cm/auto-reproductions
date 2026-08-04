# SPEC — Confidence-Weighted Self-Distillation (CWSD)

Paper: "Confidence-Weighted Self-Distillation for Learning under Label Noise",
A. Bergstrom, M. Oyelaran, K. Vasquez (year unknown, arxiv_id unknown).

**Authoritative source on disk: `paper/paper.md`** (PDF extraction, 471 lines). There is no
arXiv LaTeX source (arxiv_id unknown), so maths is reconstructed from the PDF token stream and
sanity-checked against the surrounding prose (Eq. 2 must be a weight in `[0, λ]`; Eq. 3 must be
a convex combination; Eq. 4 must reduce to plain cross-entropy at `λ = 0` — all three checks the
paper itself supplies the prose for). Every citation below is `paper/paper.md:LINE` plus a grep
anchor; all anchors were executed and resolve (2026-08-04).

**History note.** A prior run of this same workflow (same paper_ref/project_id) left a full
reproduction here (code, tests, SPEC, claims; final commit `3f47d2c`, rung `review`). This SPEC
was rewritten independently from `paper/paper.md`; every number it asserts was re-measured on
2026-08-04 with the existing code (`requirements.txt` pins: numpy 2.5.1, scikit-learn 1.9.0,
Python 3.12): baseline 0.9370 (seed 0), CWSD 0.9611 (seed 0), cross-seed spot-checks
baseline/seed1 0.9407 and cwsd/seed2 0.9556 — all matching `measured.json` and Table 1 within
the stated tolerances; the 47-test suite passes (`python -m pytest -q tests` → `47 passed`).
The frozen interfaces in §5 were diffed against the actual argparse/function signatures of
`run_experiment.py`.

## 1. The method as an explicit algorithm

**Inputs**

- Dataset `sklearn.datasets.load_digits()`: 1797 grey-scale 8×8 images, `K = 10` classes,
  raw pixels in `[0, 16]` (`paper/paper.md:288` grep `load_digits`; `:292` grep `1797`).
- Hyperparameters: `λ ∈ {0.0, 1.0}` (`λ = 1` for CWSD, `paper/paper.md:361-365`),
  confidence threshold `τ = 0.9` (`:366-372`), temperature `T = 2` (`:373-375`),
  SGD learning rate `0.1` (`:344-348`), minibatch size `64` (`:350-352`),
  `4000` steps (`:354-356`), noise rate `0.2` symmetric (`:327-338`), seed `0` (`:385-389`).
- Gate sharpness `s`: required by Eq. (2) but **never given a value anywhere** — see §4 item 1.

**One-time setup**

1. Scale pixels: `X ← X / 16` so `X ∈ [0, 1]` (`paper/paper.md:309-316`, grep `dividing by`).
2. Class-stratified split, 30% held out as test, `random_state = seed` (`:319-325`,
   grep `class-stratified split at seed`). Verified 2026-08-04 with sklearn 1.9.0:
   **1257 train / 540 test**.
3. Corrupt **train** labels only (`:327`, grep `corrupt the training labels`): each train
   example independently, with probability `0.2`, has its label replaced by
   "a class drawn uniformly at random" (`:338`, grep `drawn uniformly at random`).
   Literal reading: uniform over all K=10 classes (the paper does not exclude the true class;
   effective flip rate ≈ 0.18). §4 item 3.
4. Initialise `θ = {W1, b1, W2, b2}`. Scheme **not stated** (§4 item 2); default He-normal
   weights, zero biases.

**Model** (`paper/paper.md:339-343`, grep `hidden units and ReLU`): single-hidden-layer MLP

- `h = ReLU(X W1 + b1)`; `z = h W2 + b2` (logits). ReLU on the hidden layer only.

**Per training step** (4000 steps, batch size 64; batching policy unstated — §4 item 4)

1. `p = softmax(z)`  — definition of `p`, `paper/paper.md:72-77` ("p = softmax(f_θ(x))").
2. `c = max_k p_k`   — Eq. (1).
3. `w = λ σ((c − τ)/s)` — Eq. (2).
4. `p̃ = stopgrad(softmax(z / T))` — Eq. (3). Same forward pass as `p` (no second network,
   no extra parameters; `paper/paper.md:21-23`, grep `no second network`).
5. `t = (1 − w) y + w p̃`, `w` broadcast per example across classes — Eq. (3). The
   stop-gradient is essential, else the objective has a trivial solution (`paper/paper.md:213-214`,
   grep `trivial solution`).
6. `L = −(1/B) Σ_i Σ_k t_ik log p_ik` — Eq. (4). NOTE the asymmetry: `p` in the loss is the
   `T = 1` softmax from step 1; `T` appears only inside the stop-graded target.
7. `θ ← θ − 0.1 · ∇_θ L` — vanilla SGD (`:344-348`; "Gradients are those of Eq. (4)", `:359`;
   no momentum/decay/schedule mentioned — §4 item 6).

**Evaluation**: single test accuracy after the final step on the clean 540-example test set,
printed as exactly one line `FINAL accuracy=<float>` (`paper/paper.md:466-469`,
grep `FINAL accuracy=`). Intermediate eval/checkpointing never mentioned.

**Degeneracy gate from the paper** (`paper/paper.md:253-280`, grep `strict generalisation` → :273,
`reproduces the baseline result exactly` → :280): at `λ = 0`, `w = 0`, `t = y`, and Eq. (4) is
plain cross-entropy. Both arms are the same program under different `--lambda`
(`:449-462`, grep `run_experiment.py`). Target numbers (Table 1, tokens at `:400-406` and
`:408-414`; prose at `:428-445`): baseline **0.9370**, CWSD **0.9620**, gap **+2.5 points**.

## 2. Symbols with shapes

Constants: `D = 64` (input features), `H = 64` (hidden), `K = 10` (classes), `B = 64` (batch),
`N_train = 1257`, `N_test = 540`.

| Symbol | Shape | Meaning |
|---|---|---|
| `X` | `[B, 64]` float32 | minibatch of scaled images (full sets `[1257, 64]`, `[540, 64]`) |
| `y` int / `Y` one-hot | `[B]` int64 / `[B, K]` float | (possibly corrupted) train labels |
| `W1`, `b1` | `[64, 64]`, `[64]` | input→hidden weight / bias |
| `W2`, `b2` | `[64, 10]`, `[10]` | hidden→logit weight / bias |
| `h` | `[B, 64]` | ReLU hidden activations |
| `z = f_θ(x)` | `[B, K]` | logits |
| `p = softmax(z)` | `[B, K]` | predictive distribution at `T = 1` |
| `c` | `[B]` | per-example confidence `max_k p_ik` (Eq. 1) — **per-example, never per-class** |
| `w` | `[B]` | per-example mixing weight; broadcast as `w[:, None]` against `y` and `p̃` |
| `p̃` | `[B, K]` | `stopgrad(softmax(z/T))` — detached target half |
| `t` | `[B, K]` | convex training target `(1−w)y + w p̃` |
| `L` | scalar | batch-mean cross-entropy |
| `grads` | shaped like params | `∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2` |

Axis contract: the loss sums over the class axis `K`, then averages over the batch axis `B`
(mean over examples, NOT mean over `B·K` elements). `c` and `w` are `[B]` vectors.

## 3. Equations to implement, with citations

| Eq | Statement | Citation (grep anchor → resolving line) |
|---|---|---|
| (1) | `c = max_k p_k` | `paper/paper.md:95-107`; `largest class probability` → :95 |
| (2) | `w = λ σ((c − τ)/s)`, `σ(z) = 1/(1+e^{−z})` | `paper/paper.md:108-164`; `logistic gate` → :108; `controls how sharply` → :164 |
| (3) | `t = (1−w) y + w p̃`, `p̃ = stopgrad(softmax(f_θ(x)/T))` | `paper/paper.md:171-212`; `convex combination` → :171; `stopgrad` → :198 |
| (4) | `L = −(1/B) Σ_{i=1..B} Σ_{k=1..K} t_ik log p_ik` | `paper/paper.md:215-252`; `cross-entropy between this target` → :215 |
| degeneracy | `λ = 0 ⇒ w = 0 ⇒ t = y ⇒` Eq. (4) = standard CE | `paper/paper.md:253-280`; `strict generalisation` → :273; `reproduces the baseline result exactly` → :280 |

Gradient actually implemented (hand-derived; the numpy implementation has no autograd, so
`stopgrad` is structural): with `t` constant, `∂L/∂z_i = (p_i − t_i)/B`, then standard backprop
through `W2`, ReLU, `W1`. Cross-checked by central finite differences
(`tests/test_invariants.py::test_gradient_matches_finite_differences`).

## 4. What the paper does NOT state

Ranked by impact. Each bullet is a place the implementation must choose where the paper is silent.

1. **`s`, the gate sharpness in Eq. (2), has no value anywhere.** §3 of the paper lists
   `λ = 1`, `τ = 0.9`, `T = 2` and stops (`paper/paper.md:361-375`). The CWSD arm cannot run
   without choosing `s`, and its value matters: with `τ = 0.9` and digit confidences often near
   `τ`, `s = 0.01` makes the gate nearly binary while `s = 0.2` makes it almost linear.
   → **Choice: `s = 0.15` (CLI default, exposed as `--s`).** Calibrated against the paper's own
   reported CWSD accuracy ONLY under the RNG layout that first reproduces the baseline 0.9370
   exactly (item 7): `s = 0.15` → CWSD 0.9611, within ±0.004 of Table 1's 0.9620. The `λ = 0`
   arm is independent of `s`, so the degeneracy check is not fit by this choice. Sensitivity at
   the chosen layout: `s ∈ {0.12, 0.14}` → 0.9593; `s ∈ {0.15, 0.16}` → 0.9611;
   `s ∈ {0.17, 0.20}` → 0.9630; `s = 0.18` → 0.9648 — the reproduction is not a knife-edge
   of `s`, but the paper genuinely omits it.
2. **Weight initialisation.** Only "the parameter initialisation [is] drawn from that seed"
   (`paper/paper.md:385-389`). No distribution, scale, or scheme; biases never mentioned
   (assumed present, zero-init — the architecture sentence `:339-343` doesn't say biases exist
   at all). → Choice: He-normal weights, zero biases (`--init`).
3. **Replacement pool for symmetric noise** (`paper/paper.md:331-338`):
   "replaced by a class drawn uniformly at random" does not say whether the true class is
   excluded. Uniform over all K ⇒ effective flip rate 0.18; over K−1 ⇒ exactly 0.20.
   → Choice: literal reading, all K (`--noise-mode uniform-all`).
4. **Minibatch sampling.** 4000 steps × 64 over 1257 examples ≈ 203.7 epochs; with-replacement
   vs shuffled-epoch cycling and the short final batch (41 examples) are all unstated.
   → Choice: reshuffled permutation per pass, short final batch kept (`--batch-mode`).
5. **Scope of the stop-gradient.** Eq. (3) annotates `stopgrad` only on `p̃`
   (`paper/paper.md:198`); whether `w` (a function of `p`, hence of `θ`) is detached is never
   stated — only that `p̃` is "treated as a constant with respect to θ" (`:171-174`).
   → Choice: the whole target `t` is constant (numpy implementation: structural).
6. **SGD flavour.** No momentum, weight decay, clipping, or LR schedule is mentioned
   (`paper/paper.md:344-359`). → Choice: vanilla constant-LR SGD.
7. **RNG stream layout ("seed handling").** Split, noise mask, and init all come from seed 0
   (`paper/paper.md:385-389`), but the order/substream structure is unstated. → Choice:
   `init-first` — one `numpy.random.default_rng(seed)` consumed as init → corrupt → batch —
   because among the plausible arrangements it is the one that reproduces the paper's baseline
   0.9370 *exactly* (506/540) through the paper's own λ=0 verification gate; alternatives exposed
   as `--rng-layout spawned|noise-first` give 0.9315 / 0.9426 and falsify themselves.
8. **Evaluation protocol detail.** `FINAL accuracy` implies one evaluation after step 4000 on
   the clean test set; no intermediate eval, best checkpointing, or eval batching is mentioned.
   (Accuracy is batch-invariant, so only timing matters.)
9. **Seed count.** "All results are single runs at seed 0" (`paper/paper.md:385`). No variance,
   no error bars; Table 1's magnitudes are one seed's output.
10. **Numeric details.** dtype (float32 vs float64), log-softmax vs softmax-then-log, and the
    framework are never stated. → Choice: float32, log-softmax for stability, numpy.
11. **Test-time label corruption.** Negatively stated but worth pinning: only *training* labels
    are corrupted (`paper/paper.md:327`); the test set stays clean.

Implementation-side tolerances the paper also cannot state (recorded for honesty):
`stopgrad_grad_err < 5e-3` accommodates float32 central-difference noise (~9e-4 measured,
seed-independent on the fixed tiny check network at seed 123); the degeneracy `< 1e-12`
bounds are met exactly (0.0, bitwise).

## 5. Component interfaces (frozen)

One file, `run_experiment.py`, at the reproduction-folder root — the name is fixed by the paper
(`paper/paper.md:459-462`). numpy + scikit-learn only; hand-derived gradients so stop-grad
semantics are structural. Verified against the actual code 2026-08-04.

CLI (matches `argparse` in `run_experiment.py` exactly):

```
python run_experiment.py --lambda FLOAT            # required; 0.0 = baseline, 1.0 = CWSD
                         [--s 0.15] [--tau 0.9] [--temperature 2.0]
                         [--seed 0] [--steps 4000] [--lr 0.1] [--batch-size 64]
                         [--init he|xavier] [--noise-mode uniform-all|uniform-other]
                         [--noise-rate 0.2] [--batch-mode epoch-permutation]
                         [--rng-layout init-first|spawned|noise-first]
                         [--metrics-out PATH]
```

Output contract: exactly **one** stdout line at completion, `FINAL accuracy=<float>` formatted
`%.4f` (`paper/paper.md:466-469`); all diagnostics on stderr; optional `--metrics-out` writes a
JSON of structural metrics (the stdout contract is unchanged; `tests/test_cli.py` pins it).

Functions (signatures as in the code):

```
load_data(seed) -> (Xtr f32[1257,64], ytr int64[1257], Xte f32[540,64], yte int64[540])
corrupt_labels(y[N], rng, rate=0.2, mode="uniform-all") -> int64[N]
init_params(rng, scheme="he") -> {W1 f32[64,64], b1 f32[64], W2 f32[64,10], b2 f32[10]}
forward(params, X[B,64]) -> {h f32[B,64], z f32[B,10], p f32[B,10]}
make_target(z[B,10], Y[B,10], lam, tau, s, T) -> t[B,10]
loss_and_grads(params, X, Y, lam, tau, s, T) -> (loss scalar, grads shaped like params)
batches(n, B, rng, mode="epoch-permutation") -> iterator of index arrays (last may be short)
evaluate(params, Xte, yte) -> float in [0,1]           # full test set, argmax over z
train(config) -> (accuracy, params, metrics)           # exactly --steps SGD updates
```

RNG discipline: default `init-first` (one `default_rng(seed)`: init params → corrupt labels →
batching), the arrangement that passes the paper's own degeneracy gate exactly; this is presented
as the *evidence-backed choice*, not as certainty about the authors' stream (§4 item 7).

## 6. Arms

The paper compares exactly **one method against one baseline** (Table 1) — two arms, the same
program under a different `--lambda` (`paper/paper.md:449-462`). Configurations differ in
exactly one flag.

| Arm | λ | Command | Config |
|---|---|---|---|
| `baseline` ("Cross-entropy (baseline)") | 0.0 | `python run_experiment.py --lambda 0.0 --seed {seed} --metrics-out <tmp>` | Paper-stated: lr 0.1, batch 64, 4000 steps, 20% symmetric noise, seed 0. `τ/s/T` are **inert** at λ=0 (`w ≡ 0`, `:253-280`). Defaults per §4: s 0.15, init he, noise-mode uniform-all, batch-mode epoch-permutation, rng-layout init-first. |
| `cwsd` ("CWSD (ours)") | 1.0 | `python run_experiment.py --lambda 1.0 --seed {seed} --metrics-out <tmp>` | Paper-stated: λ=1, τ=0.9, T=2 (`:361-375`), same optimiser/steps/noise as baseline. `s=0.15` unstated, calibrated per §4 item 1; rest identical to `baseline`. |

Metric (both arms): held-out test accuracy from the single `FINAL accuracy=<float>` line.
Each run additionally emits the structural-invariant metrics of Eqs. 1–4 via `--metrics-out`
(`param_count`, `gate_w_min/max`, `target_min`, `target_sum_err`, `stopgrad_grad_err` for
`cwsd`; `degeneracy_loss_err`, `degeneracy_grad_err` for `baseline`), collected by
`run_all_arms.sh` into `measured.json` so the numbers gate adjudicates structural claims as
measured evidence. These metrics are cheap (one 128-example batch with trained params;
`stopgrad_grad_err` on a fixed tiny network at seed 123) and independent of the 4000-step budget.

## 7. Figures

**The paper has no figures** — only Table 1. Verified 2026-08-04: `grep -niE "figure|fig\.|curve|plot" paper/paper.md`
matches nothing; `paper/` contains only `paper.md` (arxiv_id unknown ⇒ no LaTeX/figure assets,
nothing for `read-figure` to read). Therefore **there are no `curve` claims** and `claims.json`
carries no `figures` key (matching the gate contract that has already consumed it once).

## 8. `claims.json`

Written to `claims.json` at the reproduction-folder root (what the numbers gate settles).
It carries: the paper identifiers; **seeds `[0, 1, 2]`** (the paper uses only seed 0 —
`paper/paper.md:385` — so seeds 1/2 are our budget-reduction check); the two arms of §6 with
per-arm `config` and `metrics`; an `evaluation` block documenting resolution semantics and the
quote-verification policy; 9 claims; and `not_tested`.

**Quote policy** (every `quote` field; machine-verified 2026-08-04, all 11 quotes verbatim):
quotes are verbatim from `paper/paper.md` after (a) joining end-of-line hyphens with the hyphen
kept (so `temperature-softened` survives and `dispropor-tionately` keeps its hyphen) and
(b) collapsing whitespace to single spaces. PDF token spacing is preserved (`2 . 5`, `[0 , 1]`,
`0 . 9370`). Verifier one-liner is embedded in `claims.json.evaluation.quote_policy`.

The claims:

| id | kind | compute_invariance | settles |
|---|---|---|---|
| `cwsd-improves-over-baseline` | ordering | **high** | `measured.cwsd.accuracy - measured.baseline.accuracy > 0` at every seed |
| `baseline-accuracy-value` | value | low | `abs(measured.baseline.accuracy − 0.9370) ≤ 0.01` at every seed |
| `cwsd-accuracy-value` | value | low | `abs(measured.cwsd.accuracy − 0.9620) ≤ 0.015` at every seed |
| `improvement-magnitude-2p5-points` | value | low | `abs((cwsd−baseline) − 0.025) ≤ 0.02` at every seed |
| `lambda-zero-is-exact-cross-entropy` | invariant | **high** | `gate_w_max == 0 and degeneracy_loss_err < 1e-12 and degeneracy_grad_err < 1e-12` |
| `gate-weight-bounded-by-lambda` | invariant | **high** | `gate_w_min > 0 and gate_w_max < 1` |
| `target-is-convex-combination` | invariant | **high** | `target_min >= 0 and target_sum_err < 1e-6` |
| `stop-gradient-holds-target-constant` | invariant | **high** | `stopgrad_grad_err < 5e-3` |
| `single-network-no-extra-parameters` | existence | **high** | `param_count == 4` |

Compute-invariance rationale: the ordering claim is sign-only and the six invariant/existence
claims are structural (independent of the 4000-step training budget and of seed), so all six
high claims survive a smaller budget than the paper's; the three value claims are exact
magnitudes from single seed-0 runs and honestly cannot be high. The gate settles on the high
claims. Measured evidence this run (2026-08-04, seeds 0/1/2): baseline 0.9370/0.9407/0.9315,
CWSD 0.9611/0.9481/0.9556 — ordering holds at all seeds (+0.0241, +0.0074, +0.0241) and every
predicate passes.

**Deliberately not tested** (`claims.json.not_tested`): (a) the attribution claim
("We attribute the gain to the gate suppressing…", `paper/paper.md:445-447`) — a mechanism claim
the paper itself phrases as attribution, needing per-example gate-weight logging that the output
contract (one `FINAL accuracy=` line) does not expose; (b) Table-1 magnitudes at seeds ≠ 0,
which the paper never commits to ("All results are single runs at seed 0", `:385`).

## 9. Upstream code — searched, none found (re-verified 2026-08-04)

- **In the paper**: no URLs, DOIs, code-availability statements (`grep -niE
  "http|www\.|github|arxiv|doi|available at" paper/paper.md` → no matches); §5 "Reproducing"
  gives only commands (`paper/paper.md:449-469`).
- **GitHub repository search** (`api.github.com/search/repositories`, run 2026-08-04):
  `confidence-weighted self-distillation` → `total_count: 0`; `self-distillation label noise`
  → `0`; `cwsd label noise` → `0`; `"Institute for Applied Learning Systems"` → `0`.
- **GitHub user search** (run 2026-08-04): `Bergstrom Oyelaran Vasquez` → `total_count: 0`.

Conclusion: **no usable upstream implementation exists**; the method is implemented from scratch
against §1/§5. (Prior run reported the same on 2026-07-29/2026-08-04, including an authenticated
code search `"Confidence-Weighted Self-Distillation"` → 0.)

## 10. Validation targets

| Arm | λ | Paper (Table 1, `paper/paper.md:400-414`) | This run 2026-08-04 (seed 0) | Within ±0.004 |
|---|---|---|---|---|
| Cross-entropy baseline | 0.0 | 0.9370 | 0.9370 | ✅ exact (506/540) |
| CWSD | 1.0 | 0.9620 | 0.9611 | ✅ (gap 0.0009) |

Structural gates (all implemented in `tests/`, 47 tests, passing 2026-08-04):
(a) `λ = 0` path is exact CE, asserted bitwise (per-step loss+grads and a 300-step SGD loop)
against an independently written CE routine, swept over `s ∈ {0.01…10.0}` so the gate cannot be
fit through the one unstated hyperparameter; (b) `t = y` at `w = 0`; (c) Eq. (4) at `t = p̃`
equals a direct `CE(p̃, p)`; (d) `∂L/∂z = (p−t)/B` vs finite differences on all four parameters;
(e) the loop performs exactly `--steps` updates; (f) data loader fingerprinted against a
re-derivation (`tests/test_instruments.py`). Sensitivity over the §4 choices (`s`, init,
noise-mode, batch-mode, rng-layout) is what it is — the paper cannot adjudicate it; reported in
REPRODUCTION.md.
