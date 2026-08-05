# SPEC — Confidence-Weighted Self-Distillation (CWSD)

Paper: "Confidence-Weighted Self-Distillation for Learning under Label Noise",
A. Bergstrom, M. Oyelaran, K. Vasquez (year unknown, arxiv_id unknown).

**Authoritative source on disk: `paper/paper.md`** (PDF extraction, 471 lines). There is no
arXiv LaTeX source (arxiv_id unknown), so maths is reconstructed from the PDF token stream and
sanity-checked against the surrounding prose (Eq. 2 must be a weight in `[0, λ]`; Eq. 3 must be
a convex combination; Eq. 4 must reduce to plain cross-entropy at `λ = 0` — all three checks the
paper itself supplies the prose for). Every citation below is `paper/paper.md:LINE` plus a grep
anchor; all anchors were re-executed and resolve (2026-08-04).

**Verification note (this SPEC pass, 2026-08-05, post-adversarial-review fix).**
A prior run of this same workflow left a complete reproduction in this folder; an
adversarial review REJECTED it on one blocking, outcome-determinative finding:
the repo's headline conclusion — "the paper-LITERAL gradient (stopgrad only on
`p_tilde`) does NOT reproduce Table 1 at any `s`; the headline is reachable only
under the DETACHED variant" — was empirically FALSE. The reviewer re-executed
CWSD under `--grad-mode literal` at the unstated sharpness `s = 2.0` and got
0.9648/0.9481/0.9611 at seeds 0/1/2 (vs baseline 0.9370/0.9407/0.9315), so the
`cwsd > baseline` ordering, the value and the magnitude claims ALL pass at every
seed. The mechanism is structural: the gate-path gradient term is proportional
to `1/s`, so the LITERAL gradient converges to the DETACHED one as `s` grows;
the prior pass's own sensitivity sweep stopped at `s = 0.30`, exactly before the
regime that falsifies the universal. This pass fixes that: the s-sweep is
extended to `s = 5.0` across seeds 0/1/2 in BOTH grad modes (`sweep_s.py` ->
`s_sweep.json`), every "at any `s`" / "only under detached" statement is
corrected to the s-dependent truth, and the cwsd arm is re-adjudicated under a
STATED `s` policy (the sharp-gate default `s = 0.15`, prose-aligned and provably
non-tuning — the headline FAILS there under literal, so `s = 0.15` was not
chosen to pass). The default `--grad-mode literal` still implements the paper's
equations as written; the `detached` variant is kept as a documented
counterfactual. The `param_count` metric is COMPUTED from the params dict. The
`ordering` claim's verdict rule is now declared once (a spread heuristic: a
within-noise ordering is `untested`, not `refuted`), matching the gate. Everything
below was re-verified in this pass:

- **Quotes:** all 11 `claims.json` quotes re-checked verbatim against `paper/paper.md` under the
  §8 normalization (11/11 PASS); all 24 grep anchors cited below re-executed and resolve to the
  stated lines.
- **Eq. (2)'s `s` is genuinely valueless:** the file's only two bare `s` tokens resolve to
  `paper/paper.md:139` (inside Eq. (2)) and `:162` (prose) — no assignment anywhere; the §3
  hyperparameter sentence (`:361-375`) lists exactly `λ = 1`, `τ = 0.9`, `T = 2` and stops.
  Eq. (2)'s stacked token layout (`τ` at `:138` over `s` at `:139` inside the parentheses)
  confirms the fractional reading `(c − τ)/s`.
- **Arms re-executed** on the pinned env (numpy 2.5.1, scikit-learn 1.9.0, Python 3.13.5):
  seeds 0/1/2 → baseline `0.9370` / `0.9407` / `0.9315` (grad-mode-independent at λ=0),
  CWSD-LITERAL `0.9407` / `0.9296` / `0.9333` at the gated sharp-gate `s = 0.15` —
  byte-identical to `measured.json`. At the gated `s = 0.15` the paper-LITERAL CWSD
  arm does NOT reproduce Table 1 at any seed; the ordering `cwsd > baseline` is
  within noise (holds at seeds 0/2 by +0.0037/+0.0019 = +2/+1 test examples,
  FLIPS at seed 1 by −0.0111), so under the declared spread heuristic the
  ordering verdict is `untested` (arms not separated), not refuted. This is
  `s`-DEPENDENT, not a universal: the full sweep (`sweep_s.py` -> `s_sweep.json`,
  §4 item 1) shows the headline REPRODUCES under the literal gradient for
  shallow gates — ordering holds at every seed for `s ≥ ~0.7`, value and
  magnitude for `s ≥ ~2.0` (e.g. `s = 2.0`: 0.9648/0.9481/0.9611, gaps
  +0.0278/+0.0074/+0.0296) — because the gate-path term (`∝ 1/s`) vanishes and
  literal → detached. The DETACHED counterfactual (`--grad-mode detached`)
  reproduces Table 1 already at the default `s = 0.15` (0.9611/0.9481/0.9556);
  it is reported in `selfcheck.json` / REPRODUCTION.md, not as the gated arm.
- **Structural metrics re-measured** (`--metrics-out`, seeds 0/1/2): baseline `gate_w_max = 0.0`,
  `degeneracy_loss_err = 0.0`, `degeneracy_grad_err = 0.0` (bitwise, holds under literal too —
  the gate-path term is `lam*...=0` at λ=0); CWSD-LITERAL `param_count = 4` (COMPUTED via
  `len(params)`), `gate_w_min > 0`, `gate_w_max < 1`, `target_min > 0`,
  `target_sum_err = 1.19e-07 < 1e-6`, `stopgrad_grad_err = 1.08e-03 < 5e-3` — every high
  invariant/existence claim's predicate passes. `stopgrad_grad_err` now finite-differences the
  loss with `p_tilde` FROZEN (w recomputed) — the literal gradient — and discriminates BOTH
  failure modes: a no-stopgrad (through `p_tilde`) FD diverges (~2.1) and a detached analytic
  (no gate path) diverges (~4.0) on the peaked net.
- **Calibration evidence re-measured:** baseline RNG layouts `spawned` → 0.9315 and
  `noise-first` → 0.9426 (falsify themselves against Table 1's 0.9370; `init-first` reproduces
  it exactly); CWSD-LITERAL s-sweep (seeds 0/1/2, `sweep_s.py` -> `s_sweep.json`,
  full table in §4 item 1): `s = 0.05→0.9537/0.9315/0.9426, 0.10→0.9370/0.9389/0.9333,
  0.15→0.9407/0.9296/0.9333, 0.20→0.9352/0.9352/0.9370, 0.30→0.9426/0.9389/0.9370,
  0.50→0.9519/0.9407/0.9444, 0.70→0.9556/0.9444/0.9500, 1.00→0.9611/0.9444/0.9537,
  2.00→0.9648/0.9481/0.9611, 5.00→0.9648/0.9500/0.9611`. The literal gap is `s`-dependent:
  the ordering `cwsd > baseline` flips at seed 1 for small `s` (sharp gate) and
  becomes positive at every seed for `s ≥ 0.7`; the value/magnitude pass at every
  seed for `s ≥ 2.0`. The gate-path gradient term is `∝ 1/s`, so the literal
  gradient converges to the detached one as `s` grows — the prior pass truncated
  this sweep at `s = 0.30` and wrongly concluded the literal gradient never
  reproduces. The `λ = 0` arm is bitwise insensitive to `s` and to `grad-mode`
  (s=0.01/s=10.0, literal/detached all → 0.9370).
- **Test suite:** `python -m pytest -q tests` → 53 passed (was 51; +2 from the M6 detached /
  M7 no-stopgrad-on-p_tilde mutations, both caught by the rewritten frozen-`p_tilde` FD check).
- **Figures:** `paper/` contains only `paper.md`; `grep -niE "figure|fig\.|curve|plot"
  paper/paper.md` matches nothing → no figures, no `curve` claims possible (§7).
- **Upstream code:** paper URL/code grep → no matches; GitHub repository search
  (`confidence-weighted self-distillation`, `self-distillation label noise`, `cwsd label
  noise`, `"Institute for Applied Learning Systems"`) and user search (author trio) →
  `total_count: 0` throughout (§9).
- **Interfaces:** SPEC §5 diffed clean against `run_experiment.py`
  (`--lambda --grad-mode --s --tau --temperature --seed --steps --lr --batch-size --init
  --noise-mode --noise-rate --batch-mode --rng-layout --metrics-out`; `load_data, corrupt_labels,
  init_params, forward, make_target, _gate_path_grad, loss_and_grads, batches, evaluate, train`).

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
   no momentum/decay/schedule mentioned — §4 item 6). **`∇_θ L` is the paper-LITERAL
   gradient** (default `--grad-mode literal`, §3/§4 item 5): stopgrad ONLY on `p̃`, so the
   gate weight `w` is differentiable in `z` and `dL/dz = (p − t)/B + gate-path term`. The
   `--grad-mode detached` variant (whole target constant) drops the gate-path term.

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

Gradient actually implemented (hand-derived; the numpy implementation has no autograd).
The DEFAULT `--grad-mode literal` is the paper-faithful gradient: stopgrad ONLY on `p̃`
(Eq. 3 as written, `paper/paper.md:198, :171-174`); the gate weight `w = λσ((c−τ)/s)` is
differentiable in `z` through `c = max_k p_k`, so the gradient INCLUDES the `L→t→w→c→z`
gate path:

    dL/dz_ik = (p_ik − t_ik)/B − (1/B)·λσ(u_i)(1−σ(u_i))·(1/s)·p_im*·(δ_{m*,k} − p_ik)·g_i

where `u_i = (c_i − τ)/s`, `m* = argmax_m p_im`, `g_i = Σ_m (p̃_im − y_im) log p_im`, and
`p̃ = softmax(z/T)` is held constant. The gate-path term is `λ·...` and vanishes exactly at
`λ = 0`, so the degeneracy (== plain CE) holds bitwise. `--grad-mode detached` treats the
WHOLE target `t` as constant (`dL/dz = (p − t)/B` only); that is the standard self-distillation
convention the paper does NOT mark on `w`, and a variant under which Table 1 is reachable
(already at the default `s=0.15`); the literal gradient also reaches it for shallow gates
(`s ≥ ~2.0`) where its gate-path term (`∝ 1/s`) vanishes (§4 item 1).
Cross-checked by central finite differences of the loss with `p̃` frozen (w recomputed), on a
peaked net (`tests/test_invariants.py::test_gradient_matches_finite_differences`,
`stopgrad_grad_err ≈ 1.08e-3 < 5e-3`); the check discriminates both a no-stopgrad (through
`p̃`) and a detached (no gate path) implementation (M6/M7 mutations).

## 4. What the paper does NOT state

Ranked by impact. Each bullet is a place the implementation must choose where the paper is silent.

1. **`s`, the gate sharpness in Eq. (2), has no value anywhere.** §3 of the paper lists
   `λ = 1`, `τ = 0.9`, `T = 2` and stops (`paper/paper.md:361-375`). The CWSD arm cannot run
   without choosing `s`, and its value matters — it is **outcome-determinative for the
   headline under the paper-LITERAL gradient**.
   → **Choice: `s = 0.15` (CLI default, exposed as `--s`).** Rationale: `0.15` is the
   sharp-gate regime the paper's prose describes ("`s` controls how sharply the gate opens
   around `τ`", `paper/paper.md:164` — small `s` = sharp gate), and it is **provably
   non-tuning**: at `s = 0.15` the paper-LITERAL CWSD arm lands ≈ baseline, so the headline
   FAILS there — `s = 0.15` was not chosen to make the headline pass. (A prior pass
   calibrated `s = 0.15` against Table 1 under the DETACHED gradient; that calibration is
   moot under the literal default and is disclosed for honesty, not relied on.)
   → **The CWSD headline is `s`-DEPENDENT under the literal gradient, NOT a universal.**
   The gate-path gradient term is `∝ 1/s` (see `_gate_path_grad`), so the literal gradient
   converges to the detached one as `s` grows. The full s-sweep (`sweep_s.py` ->
   `s_sweep.json`, re-run 2026-08-05) across seeds 0/1/2 in BOTH grad modes:

   | `s` | literal CWSD (s0/s1/s2) | literal gaps (s0/s1/s2) | ord>0 all | value ≤0.015 all | mag ≤0.02 all | detached CWSD (s0/s1/s2) |
   |---|---|---|---|---|---|---|
   | 0.05 | 0.9537/0.9315/0.9426 | +0.0167/−0.0092/+0.0111 | no | no | no | 0.9519/0.9481/0.9444 |
   | 0.10 | 0.9370/0.9389/0.9333 | +0.0000/−0.0018/+0.0018 | no | no | no | 0.9574/0.9481/0.9444 |
   | 0.15 | 0.9407/0.9296/0.9333 | +0.0037/−0.0111/+0.0018 | no | no | no | 0.9611/0.9481/0.9556 |
   | 0.20 | 0.9352/0.9352/0.9370 | −0.0018/−0.0055/+0.0055 | no | no | no | 0.9630/0.9500/0.9556 |
   | 0.30 | 0.9426/0.9389/0.9370 | +0.0056/−0.0018/+0.0055 | no | no | no | 0.9630/0.9519/0.9611 |
   | 0.50 | 0.9519/0.9407/0.9444 | +0.0149/+0.0000/+0.0129 | no | no | no | 0.9648/0.9481/0.9593 |
   | 0.70 | 0.9556/0.9444/0.9500 | +0.0186/+0.0037/+0.0185 | **yes** | no | no | 0.9611/0.9519/0.9611 |
   | 1.00 | 0.9611/0.9444/0.9537 | +0.0241/+0.0037/+0.0222 | **yes** | no | no | 0.9667/0.9481/0.9630 |
   | 2.00 | 0.9648/0.9481/0.9611 | +0.0278/+0.0074/+0.0296 | **yes** | **yes** | **yes** | 0.9630/0.9500/0.9611 |
   | 5.00 | 0.9648/0.9500/0.9611 | +0.0278/+0.0093/+0.0296 | **yes** | **yes** | **yes** | 0.9667/0.9519/0.9611 |

   (baseline = 0.9370/0.9407/0.9315, grad-mode- and s-independent at λ=0.) **Crossover under
   the literal gradient: ordering `cwsd > baseline` at every seed from `s ≈ 0.7`; value
   `|cwsd − 0.962| ≤ 0.015` and magnitude `|gap − 0.025| ≤ 0.02` at every seed from
   `s ≈ 2.0`.** Under DETACHED the headline reproduces already at the default `s = 0.15`.
   The prior pass's sweep stopped at `s = 0.30` and concluded the literal gradient "does NOT
   reproduce at any `s`" — the sweep above, extended to `s = 5.0`, falsifies that universal:
   Table 1 reproduces under the literal equations for shallow gates (where literal →
   detached) and under the detached convention at the default `s`. Because the paper states
   neither `s` nor the stop-grad scope on `w`, the headline is **under-specified**: reachable
   under the paper's equations for a range of `(s, grad-mode)`, but NOT at the prose-aligned
   sharp-gate default `s = 0.15` under the literal gradient. The gated cwsd arm therefore pins
   `s = 0.15` (the disclosed non-headline rationale above): the value/magnitude claims are
   REFUTED at the gated `s`, and the ordering is UNTESTED (within noise, flips at seed 1). The
   `λ = 0` arm is bitwise insensitive to `s` (and to grad-mode), so the degeneracy check is
   not touched by this choice.
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
5. **Scope of the stop-gradient (a finding of the adversarial review).** Eq. (3)
   annotates `stopgrad` ONLY on `p̃` (`paper/paper.md:198`) and says only "the latter [= p̃]
   treated as a constant with respect to θ" (`:171-174`). The gate weight `w = λσ((c−τ)/s)` is
   a function of `θ` through `c = max_k p_k` and is NOT marked stopped; "Gradients are those of
   Eq. (4)" (`:359`) under the literal reading therefore includes the `L → t → w → c → z` path.
   → **Choice (DEFAULT, paper-faithful): `--grad-mode literal` — stopgrad ONLY on `p̃`; `w` is
   differentiable and the gate path is included** (the literal gradient derived in §3). This is
   what Eqs. (2)–(4) literally state. Whether the Table-1 CWSD number (0.9620) and the
   `cwsd > baseline` ordering reproduce under it is `s`-DEPENDENT (the paper never states `s`):
   at the sharp-gate default `s = 0.15` the headline does NOT reproduce (CWSD ≈ baseline; the
   ordering flips at seed 1), but for shallow gates it DOES — ordering from `s ≈ 0.7`, value
   and magnitude from `s ≈ 2.0` (§4 item 1) — because the gate-path term is `∝ 1/s` and the
   literal gradient converges to the detached one as `s` grows.
   → **Counterfactual (NOT the default, NOT the gated arm): `--grad-mode detached` — the WHOLE
   target `t` is constant (stopgrad on `p̃` AND `w`)**, so `dL/dz = (p − t)/B` only. This is the
   standard self-distillation convention the paper does NOT mark on `w`; it reproduces Table 1
   already at the default `s = 0.15` (0.9611/0.9481/0.9556), reported in `selfcheck.json` /
   REPRODUCTION.md. A prior adversarial review found an earlier run used the detached gradient
   as the primary arm while marking the headline "reproduced" — outcome-determinative and not
   what the paper states — so the literal gradient is the default and the detached variant is
   reported honestly as a counterfactual. At `λ = 0` both modes are bit-identical (the gate-path
   term is `λ·... = 0`), so the paper's degeneracy gate holds under either.
6. **SGD flavour.** No momentum, weight decay, clipping, or LR schedule is mentioned
   (`paper/paper.md:344-359`). → Choice: vanilla constant-LR SGD.
7. **RNG stream layout ("seed handling").** Split, noise mask, and init all come from seed 0
   (`paper/paper.md:385-389`), but the order/substream structure is unstated. → Choice:
   `init-first` — one `numpy.random.default_rng(seed)` consumed as init → corrupt → batch —
   because among the plausible arrangements it is the one that reproduces the paper's baseline
   0.9370 *exactly* (506/540) through the paper's own λ=0 verification gate; alternatives exposed
   as `--rng-layout spawned|noise-first` give 0.9315 / 0.9426 (re-measured 2026-08-04) and
   falsify themselves.
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
`stopgrad_grad_err < 5e-3` accommodates float32 central-difference noise on a peaked net
(~1.16e-3 measured, the frozen-target FD against the analytic `(p-t)/B` stop-grad); the
degeneracy `< 1e-12` bounds are met exactly (0.0, bitwise).

## 5. Component interfaces (frozen)

One file, `run_experiment.py`, at the reproduction-folder root — the name is fixed by the paper
(`paper/paper.md:459-462`). numpy + scikit-learn only; hand-derived gradients so stop-grad
semantics are structural. Verified against the actual code 2026-08-04.

CLI (matches `argparse` in `run_experiment.py` exactly):

```
python run_experiment.py --lambda FLOAT            # required; 0.0 = baseline, 1.0 = CWSD
                         [--grad-mode literal|detached]   # default literal (paper-faithful)
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
_gate_path_grad(p, log_p, Y, p_tilde, lam, tau, s) -> dz_gate[B,10]  # literal-mode gate-path term
loss_and_grads(params, X, Y, lam, tau, s, T, grad_mode="literal") -> (loss scalar, grads shaped like params)
batches(n, B, rng, mode="epoch-permutation") -> iterator of index arrays (last may be short)
evaluate(params, Xte, yte) -> float in [0,1]           # full test set, argmax over z
train(config) -> (accuracy, params, Xtr, Ytr_onehot)    # exactly --steps SGD updates; the
                                                         # data arrays are returned so main()
                                                         # can compute structural_metrics on
                                                         # a fixed batch for measured.json
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
| `baseline` ("Cross-entropy (baseline)") | 0.0 | `python run_experiment.py --lambda 0.0 --grad-mode literal --seed {seed} --metrics-out <tmp>` | Paper-stated: lr 0.1, batch 64, 4000 steps, 20% symmetric noise, seed 0. `τ/s/T/grad-mode` are **inert** at λ=0 (`w ≡ 0` and the gate-path term is `λ·...=0`, `:253-280`); the baseline is bit-identical under literal or detached. Defaults per §4: s 0.15, init he, noise-mode uniform-all, batch-mode epoch-permutation, rng-layout init-first. |
| `cwsd` ("CWSD (ours)") | 1.0 | `python run_experiment.py --lambda 1.0 --grad-mode literal --seed {seed} --metrics-out <tmp>` | Paper-stated: λ=1, τ=0.9, T=2 (`:361-375`), same optimiser/steps/noise as baseline. `--grad-mode literal` is the paper-faithful gradient (stopgrad ONLY on `p̃`, gate path active; §4 item 5). `s=0.15` unstated; pinned to the sharp-gate default with a disclosed non-headline rationale (provably non-tuning — the headline FAILS at `s=0.15` under literal; §4 item 1). The CWSD result is `s`-DEPENDENT under literal: at the gated `s=0.15` the headline +2.5 and the value/magnitude do NOT reproduce and the ordering is within noise (UNTESTED, flips at seed 1); the s-sweep (`sweep_s.py` -> `s_sweep.json`, §4 item 1) shows the headline REPRODUCES under literal for `s ≥ ~0.7` (ordering) / `s ≥ ~2.0` (value, magnitude) and under the DETACHED counterfactual at `s=0.15`. The DETACHED counterfactual (`--grad-mode detached`, same config) reproduces Table 1 and is reported in `selfcheck.json` / REPRODUCTION.md. |

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

| id | kind | compute_invariance | settles | literal-arm verdict (gated `s=0.15`) |
|---|---|---|---|---|
| `cwsd-improves-over-baseline` | ordering | **high** | `measured.cwsd.accuracy - measured.baseline.accuracy > 0` at every seed (spread-heuristic: within-noise → `untested`) | UNTESTED (within noise: +0.0037/−0.0111/+0.0019; mean −0.0019 ≤ spread 0.0148; flips at seed 1) |
| `baseline-accuracy-value` | value | low | `abs(measured.baseline.accuracy − 0.9370) ≤ 0.01` at every seed | PASS (0.9370/0.9407/0.9315) |
| `cwsd-accuracy-value` | value | low | `abs(measured.cwsd.accuracy − 0.9620) ≤ 0.015` at every seed | FAIL (0.9407/0.9296/0.9333; dev 0.021–0.032) |
| `improvement-magnitude-2p5-points` | value | low | `abs((cwsd−baseline) − 0.025) ≤ 0.02` at every seed | FAIL (gap ≈0; dev 0.021–0.036) |
| `lambda-zero-is-exact-cross-entropy` | invariant | **high** | `gate_w_max == 0 and degeneracy_loss_err < 1e-12 and degeneracy_grad_err < 1e-12` | PASS (holds under literal too) |
| `gate-weight-bounded-by-lambda` | invariant | **high** | `gate_w_min > 0 and gate_w_max < 1` | PASS |
| `target-is-convex-combination` | invariant | **high** | `target_min >= 0 and target_sum_err < 1e-6` | PASS |
| `stop-gradient-holds-target-constant` | invariant | **high** | `stopgrad_grad_err < 5e-3` (stopgrad on `p̃` only) | PASS (1.08e-3) |
| `single-network-no-extra-parameters` | existence | **high** | `param_count == 4` (COMPUTED) | PASS |

Compute-invariance rationale: the ordering claim is sign-only and the five invariant/existence
claims are structural, so all six are high; the three value claims are exact magnitudes from
single seed-0 runs and are low. Under the paper-LITERAL gradient at the gated sharp-gate
`s=0.15`, the 5 high structural claims + the baseline value PASS, the two CWSD value claims
FAIL (refuted at the gated `s`), and the central ordering is UNTESTED (within noise,
flips at seed 1) under the declared spread heuristic. These are verdicts about the gated
`s=0.15`, NOT universals: the CWSD headline is `s`-DEPENDENT under the literal gradient
(`sweep_s.py` -> `s_sweep.json`, §4 item 1) — the ordering holds at every seed for `s ≥ ~0.7`
and the value/magnitude for `s ≥ ~2.0` (e.g. `s=2.0`: 0.9648/0.9481/0.9611, gaps
+0.0278/+0.0074/+0.0296), because the gate-path term (`∝ 1/s`) vanishes and literal → detached.
The DETACHED counterfactual (`--grad-mode detached`, not the gated arm) reproduces Table 1
already at the default `s=0.15` (0.9611/0.9481/0.9556, ordering +0.0241/+0.0074/+0.0241); it is
the standard self-distillation convention the paper does NOT mark on `w`. Because the paper
states neither `s` nor the stop-grad scope on `w`, the headline is under-specified: reachable
under the paper's equations for a range of `(s, grad-mode)`, but not at the prose-aligned
sharp-gate default under the literal gradient. Measured evidence re-derived this pass
(2026-08-05, seeds 0/1/2): baseline 0.9370/0.9407/0.9315, CWSD-LITERAL (gated s=0.15)
0.9407/0.9296/0.9333; CWSD-DETACHED (counterfactual) 0.9611/0.9481/0.9556.

**Deliberately not tested** (`claims.json.not_tested`): (a) the attribution claim
("We attribute the gain to the gate suppressing…", `paper/paper.md:445-447`) — a mechanism claim
the paper itself phrases as attribution, needing per-example gate-weight logging that the output
contract (one `FINAL accuracy=` line) does not expose; (b) Table-1 magnitudes at seeds ≠ 0
(`paper/paper.md:385`) — at the gated `s=0.15` under the literal gradient the value claims fail
at every seed (including 0), so the widened tolerances are not the reason they fail here (the
gated `s` is); the CWSD headline IS reachable under the literal gradient for shallow gates
(`s ≥ ~0.7` ordering, `s ≥ ~2.0` value/magnitude) and under the DETACHED variant at `s=0.15`
(`sweep_s.py` -> `s_sweep.json`, §4 item 1); (c) the DETACHED counterfactual — the standard
self-distillation convention (whole target constant) the paper does NOT mark on `w`,
implemented as `--grad-mode detached` and reported in `selfcheck.json` / REPRODUCTION.md as ONE
configuration under which Table 1 is reachable (the literal gradient ALSO reaches it for
shallow gates), but NOT the gated arm (the gated cwsd arm is the paper-literal gradient at the
sharp-gate default `s=0.15`, faithful to Eq. 3 as written).

## 9. Upstream code — searched, none found (re-verified 2026-08-04)

- **In the paper**: no URLs, DOIs, code-availability statements (`grep -niE
  "http|www\.|github|arxiv|doi|available at" paper/paper.md` → no matches); §5 "Reproducing"
  gives only commands (`paper/paper.md:449-469`).
- **GitHub repository search** (`api.github.com/search/repositories`, re-run 2026-08-04):
  `confidence-weighted self-distillation` → `total_count: 0`; `self-distillation label noise`
  → `0`; `cwsd label noise` → `0`; `"Institute for Applied Learning Systems"` → `0`.
- **GitHub user search** (re-run 2026-08-04): `Bergstrom Oyelaran Vasquez` → `total_count: 0`.

Conclusion: **no usable upstream implementation exists**; the method is implemented from scratch
against §1/§5. (Prior runs reported the same.)

## 10. Validation targets

| Arm | λ | Paper (Table 1, `paper/paper.md:400-414`) | This pass 2026-08-05 (seed 0, LITERAL, gated `s=0.15`) | seed 0, LITERAL `s=2.0` | seed 0, DETACHED `s=0.15` (counterfactual) |
|---|---|---|---|---|---|
| Cross-entropy baseline | 0.0 | 0.9370 | 0.9370 (exact, 506/540) | 0.9370 (same; λ=0 is s- and grad-mode-independent) | 0.9370 (same) |
| CWSD | 1.0 | 0.9620 | 0.9407 (does NOT reproduce at gated `s=0.15`; +0.0037 over baseline) | 0.9648 (reproduces; +0.0278) | 0.9611 (reproduces; +0.0241) |

The paper's Table-1 CWSD number (0.9620) is **`s`-DEPENDENT under the paper-LITERAL gradient**
(stopgrad only on `p̃`, as Eq. 3 marks it), NOT a universal: at the gated sharp-gate default
`s=0.15` it does NOT reproduce at any seed (CWSD ≈ baseline); but because the gate-path term is
`∝ 1/s`, the literal gradient converges to the DETACHED one as `s` grows, and the headline
REPRODUCES under the literal gradient for shallow gates (`s ≥ ~0.7` ordering, `s ≥ ~2.0`
value/magnitude across seeds 0/1/2; e.g. `s=2.0` → 0.9648/0.9481/0.9611). It also reproduces
under the DETACHED variant (whole target constant) already at `s=0.15`. The paper states
neither `s` nor the stop-grad scope on `w`, so the headline is under-specified: reachable
under the paper's equations for a range of `(s, grad-mode)` but not at the prose-aligned
sharp-gate default under the literal gradient. The faithful gated arm pins `s=0.15` (a
disclosed, provably non-tuning choice — the headline fails there) and reports the headline
as NOT reproduced AT THE GATED `s`, with the full `s`-sweep (`sweep_s.py` -> `s_sweep.json`,
§4 item 1) and the detached counterfactual documented.

Structural gates (all implemented in `tests/`, 53 tests, passing 2026-08-04):
(a) `λ = 0` path is exact CE, asserted bitwise (per-step loss+grads and a 300-step SGD loop)
against an independently written CE routine, swept over `s ∈ {0.01…10.0}` so the gate cannot be
fit through the one unstated hyperparameter — and the degeneracy holds under BOTH grad modes
(the gate-path term is `λ·...=0` at λ=0); (b) `t = y` at `w = 0`; (c) Eq. (4) at `t = p̃`
equals a direct `CE(p̃, p)`; (d) the LITERAL `∂L/∂z = (p−t)/B + gate-path` vs finite differences
with `p̃` frozen (w recomputed) on all four parameters, on a peaked net where a no-stopgrad
(through `p̃`) and a detached (no gate path) implementation both diverge; (e) the loop performs
exactly `--steps` updates; (f) data loader fingerprinted against a re-derivation
(`tests/test_instruments.py`). Sensitivity over the §4 choices (`s`, init, noise-mode,
batch-mode, rng-layout, grad-mode) is what it is — the paper cannot adjudicate it; reported in
REPRODUCTION.md.

## 11. Constructed truth

Which standard constructed-truth strategies this reproduction uses, and where
each lives. The headline accuracy numbers (Table 1) are single seed-0 runs of a
noisy, seed-sensitive pipeline and so are NOT the evidence the gate leans on;
the gate settles on the high (structural / sign-only) claims below, each backed
by a constructed truth that does not depend on the 4000-step budget.

- **Degeneracy (the method's limiting case).** YES — the paper's own
  verification gate. At `λ = 0`, `w ≡ 0`, `t = y`, and Eq. (4) is plain
  cross-entropy (`paper/paper.md:253-280`). `tests/test_degeneracy.py` asserts
  this at three levels: `t == y` and `w == 0` element-wise; per-step loss +
  all four parameter gradients bitwise-equal an independently written CE
  routine; and a full SGD loop at `λ = 0` is bit-identical (params and accuracy)
  to an independent CE loop on the same RNG stream. The check is independent
  of the unstated `s`, so it cannot be fit through that hyperparameter; it is
  swept over `s ∈ {0.01…10.0}` to prove so.
- **The same quantity derived two ways.** YES — the degeneracy equivalence
  above IS this: `loss_and_grads(lam=0)` vs an independently written
  `ce_loss_and_grads_independent` / `_independent_ce_loss_and_grads` (different
  surface forms, no shared mutation anchors), required bitwise equal.
  `tests/test_instruments.py::test_degeneracy_equivalence_positive/negative`
  exercises the oracle on a known-degenerate (positive) and a
  known-non-degenerate (`λ = 1`, negative) input.
- **Naive implementation agreeing with the fast one.** YES — the hand-derived
  analytic literal gradient `∂L/∂z = (p − t)/B + gate-path` (with `p̃` held
  constant, the Eq. (3) stop-grad made structural) is checked against a central
  finite-difference of the loss with `p̃` **frozen at the unperturbed params**
  (and `w` recomputed — the literal gradient), on a **peaked** network (W2
  scaled 8×, so `p` is non-uniform and both the gate-path term and the
  `d p̃/dz` chain are non-negligible) over all four parameters
  (`tests/test_invariants.py::test_gradient_matches_finite_differences`,
  `stopgrad_grad_err ≈ 1.08e-3 < 5e-3`). The "slow" reference is finite
  differences; the "fast" one is the analytic backprop. Their agreement is the
  evidence that the stop-grad is on `p̃` only and the gate path is included.
  **Non-vacuity is proven two ways**: `tests/test_structural_metrics.py::test_stopgrad_grad_err_is_nonvacuous`
  shows a NO-stopgrad FD (recomputing `p̃`, i.e. the chain through `p̃`) diverges
  to ~2.1 and a DETACHED analytic (dropping the gate path) diverges to ~4.0 on
  the same peaked net; and two mutations are caught by the same check — M6
  (detached, drop the gate-path term — the exact divergence an adversarial review
  rejected) and M7 (no stop-grad on `p̃`, the trivial-solution hazard the paper
  warns about).
- **The method's limiting cases.** YES — the `λ = 0` degeneracy (above) is the
  method's limiting case, AND the gradient-mode contrast is itself a limiting-
  case argument: the LITERAL gradient (stopgrad on `p̃` only, the paper's
  letter) vs the DETACHED gradient (whole target constant, the standard
  convention) are two limiting readings of the under-specified stop-grad. The
  literal gradient's gate-path term is `∝ 1/s`, so the two readings CONVERGE as
  `s → ∞`: the headline reproduces under detached already at `s = 0.15` and
  under literal for shallow gates (`s ≥ ~0.7` ordering, `s ≥ ~2.0` value/
  magnitude). The `s`-sweep (`sweep_s.py` -> `s_sweep.json`, §4 item 1) over
  `s ∈ {0.05…5.0}` records whether each gated claim's verdict survives across
  the unstated `s` — it does NOT survive at the sharp-gate default `s = 0.15`
  under literal but DOES for `s ≥ ~0.7–2.0`. Because the paper states neither
  `s` nor the stop-grad scope on `w`, the headline's reachability is
  `s`-dependent (under-specified), not uniformly refuted or reproduced — this
  `s`-dependence IS the reproduction's central finding about the paper.
- **The paper's standard baseline as a common-knowledge oracle.** YES — the
  baseline `0.9370` (Table 1) is common knowledge from the paper, and the
  `init-first` RNG layout (§4 item 7) was selected precisely because it is the
  one arrangement under which the `λ = 0` arm reproduces `0.9370` *exactly*
  (506/540) through the paper's own degeneracy gate; the alternatives
  (`spawned`, `noise-first`) give 0.9315 / 0.9426 and falsify themselves
  against the oracle. The baseline is therefore an oracle the reproduction
  already possesses, not a number it tuned to.
- **Brute force at toy scale against a closed-form max/min/worst case.**
  NOT APPLICABLE — the paper makes no closed-form maximum, minimum, or
  worst-case claim; Table 1 is an empirical accuracy comparison only, so
  there is no closed form to brute-force against.
- **Planting a known structure in synthetic input and requiring recovery.**
  PARTIAL / not used as gate evidence — `tests/test_instruments.py`'s
  accuracy-scorer positive test plants a known classifier (one-hot `W2`
  columns over an identity hidden layer) so `argmax(z[i])` is forced, proving
  the scorer reports exactly 1.0 / 0.0 / 1⁄3 on known inputs. This exercises
  the scorer as an instrument (positive + negative), but it is not a
  "recover a planted structure from the method" test; the method itself is
  not validated this way because the paper asserts no recoverable-structure
  claim.
- **A slow exact or convex reference solver.** NOT APPLICABLE — CWSD is a
  training procedure, not an optimisation with an exact/convex reference; the
  finite-difference gradient check above is the closest analogue and is
  already listed under "naive agrees with fast".

Net: five of the eight candidate strategies apply (degeneracy, two-ways,
naive-vs-fast, limiting-cases, baseline-as-oracle); two are honestly N/A
(closed-form brute force, exact/convex reference); one is used only to exercise
an instrument, not as gate evidence (planted structure). The reproduction's
evidence is the structural / sign-only high claims, each tied to one of the
applying strategies, not the seed-0 magnitudes. The literal-vs-detached
limiting-case contrast — combined with the `s`-sweep over the unstated gate
sharpness — is itself the reproduction's central finding: the paper's headline
is `s`-dependent under the literal reading Eq. (3) marks (reproduces for
shallow gates `s ≥ ~0.7–2.0` where literal → detached, and under the detached
reading at the default `s=0.15`, but NOT at the prose-aligned sharp-gate
default `s=0.15` under literal), so its reachability is under-specified by a
paper that states neither `s` nor the stop-grad scope on `w`.
