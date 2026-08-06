# SPEC — Confidence-Weighted Self-Distillation (CWSD)

Paper: "Confidence-Weighted Self-Distillation for Learning under Label Noise",
A. Bergstrom, M. Oyelaran, K. Vasquez (Institute for Applied Learning Systems;
year unknown, arxiv_id unknown). Workflow re-run dated **2026-08-06**.

**Authoritative source: `paper/paper.md`** (471-line PDF extraction). arxiv_id is
unknown, so no LaTeX source exists; maths is reconstructed from the PDF token stream and
cross-checked against the surrounding prose. Three paper-internal checks anchor the
reconstruction: Eq. (2) must be a weight in `[0, λ]` ("maximum influence", :124); Eq. (3)
must be a convex combination (:171); Eq. (4) must reduce to plain cross-entropy at `λ = 0`
(:253–280). All anchors below were **re-executed 2026-08-06** and resolve (a checker script
verified 24 anchors, the 12 `claims.json` quotes, figure/URL absence, and the bare-`s`
token inventory).

**Provenance and trust policy.** A prior run of this same workflow left a complete
reproduction (code, tests, claims, measured numbers) in this folder; the re-run's ingest
step reset only `REPRODUCTION.md`. None of it is trusted without re-verification. This pass
re-derived the SPEC from `paper/paper.md` and independently re-verified, on a freshly built
pinned environment (Python 3.12.13, numpy 2.5.1, scikit-learn 1.9.0, pytest 9.1.1 — same
pinned deps as the prior run; the prior run's 3.13.5 host Python differs but the pipeline
uses numpy/sklearn RNG only):

- `python -m pytest -q tests` → **53 passed** (2026-08-06).
- Seed-0 spot checks re-executed 2026-08-06: baseline (`--lambda 0.0`) → **0.9370** (exact
  Table 1); CWSD literal `s=0.15` → **0.9407**; literal `s=2.0` → **0.9648**; detached
  `s=0.15` → **0.9611** — all matching `measured.json` / `s_sweep.json`, and the full
  sweep JSON equals the table in §4 item 1.
- All 24 `paper/paper.md` anchors cited below re-resolved to the stated lines; all 12
  `claims.json` quotes verified verbatim under the §8 normalization (12/12 PASS); the file's
  only two bare `s` tokens resolve to `paper/paper.md:139` (inside Eq. (2)) and `:162`
  (prose) — **no assignment anywhere**, so Eq. (2)'s `s` is genuinely valueless.
- `grep -niE "figure|fig\.|curve|plot" paper/paper.md` → 0 matches: no figures (§7).
- GitHub search re-run 2026-08-06: no upstream implementation (§9).

## 1. The method as an explicit algorithm

**Inputs**

- Dataset `sklearn.datasets.load_digits()`: 1797 grey-scale 8×8 images, `K = 10` classes,
  raw pixels in `[0, 16]` (`paper/paper.md:288` grep `load_digits`; `:292` grep `1797`).
- Hyperparameters: `λ ∈ {0.0, 1.0}` (`λ = 1` for CWSD, `paper/paper.md:361-365`),
  confidence threshold `τ = 0.9` (`:366-372`), temperature `T = 2` (`:373-375`),
  SGD learning rate `0.1` (`:344-348`), minibatch size `64` (`:350-352`),
  `4000` steps (`:354-356`), noise rate `0.2` symmetric (`:327-338`), seed `0` (`:385-389`).
- Gate sharpness `s`: required by Eq. (2) but **never given a value anywhere** — §4 item 1.

**One-time setup**

1. Scale pixels: `X ← X / 16` so `X ∈ [0, 1]` (`paper/paper.md:309-316`, grep `dividing by` → :316).
2. Class-stratified split, 30% held out as test, `random_state = seed` (`:319-325`,
   grep `class-stratified split at seed` → :324). sklearn 1.9.0 gives **1257 train / 540 test**.
3. Corrupt **train** labels only (`:327`, grep `corrupt the training labels`): each train
   example independently, with probability `0.2`, has its label replaced by
   "a class drawn uniformly at random" (`:338`, grep `drawn uniformly at random`).
   Literal reading: uniform over all K=10 classes (the paper does not exclude the true class;
   effective flip rate ≈ 0.18). §4 item 3.
4. Initialise `θ = {W1, b1, W2, b2}`. Scheme **not stated** (§4 item 2); default He-normal
   weights, zero biases.

**Model** (`paper/paper.md:339-343`, grep `hidden units and ReLU` → :343): single-hidden-layer MLP

- `h = ReLU(X W1 + b1)`; `z = h W2 + b2` (logits). ReLU on the hidden layer only.

**Per training step** (4000 steps, batch size 64; batching policy unstated — §4 item 4)

1. `p = softmax(z)` — definition of `p`, `paper/paper.md:70-77` ("p = softmax(f_θ(x))").
2. `c = max_k p_k` — Eq. (1).
3. `w = λ σ((c − τ)/s)` — Eq. (2).
4. `p̃ = stopgrad(softmax(z / T))` — Eq. (3). Same forward pass as `p` (one extra softmax,
   no second network, no extra parameters; `paper/paper.md:21-23`, grep `no second network` → :23).
5. `t = (1 − w) y + w p̃`, `w` broadcast per example across classes — Eq. (3). The
   stop-gradient on `p̃` is essential, else the objective admits a trivial solution
   (`paper/paper.md:213-214`, grep `trivial solution` → :214).
6. `L = −(1/B) Σ_i Σ_k t_ik log p_ik` — Eq. (4). Asymmetry note: the `p` in the loss is the
   `T = 1` softmax from step 1; `T` appears only inside the stop-graded target.
7. `θ ← θ − 0.1 · ∇_θ L` — vanilla SGD (`:344-348`; "Gradients are those of Eq. (4)", `:359`;
   no momentum/decay/schedule mentioned — §4 item 6). **`∇_θ L` is the paper-LITERAL
   gradient** (default `--grad-mode literal`, §3/§4 item 5): stopgrad ONLY on `p̃`, so the
   gate weight `w` is differentiable in `z` and `dL/dz = (p − t)/B + gate-path term`. The
   `--grad-mode detached` variant (whole target constant) drops the gate-path term.

**Evaluation**: single test accuracy after the final step on the clean 540-example test set,
printed as exactly one line `FINAL accuracy=<float>` (`paper/paper.md:466-469`,
grep `FINAL accuracy=` → :468). Intermediate eval/checkpointing never mentioned.

**Degeneracy gate supplied by the paper** (`paper/paper.md:253-280`, grep
`strict generalisation` → :273, `reproduces the baseline result exactly` → :280): at
`λ = 0`, `w = 0`, `t = y`, and Eq. (4) is plain cross-entropy. Both arms are the same
program under different `--lambda` (`:449-462`, grep `run_experiment.py` → :459, :462).
Target numbers (Table 1, tokens at `:400-406` and `:408-414`; prose at `:428-445`):
baseline **0.9370**, CWSD **0.9620**, gap **+2.5 points**.

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
| `p = softmax(z)` | `[B, K]` | predictive distribution at `T = 1` (softmax rows over class axis `K`) |
| `c` | `[B]` | per-example confidence `max_k p_ik` (Eq. 1) — **per-example, never per-class** |
| `w` | `[B]` | per-example mixing weight; broadcast as `w[:, None]` against `y` and `p̃` |
| `p̃` | `[B, K]` | `stopgrad(softmax(z/T))` — detached target half |
| `t` | `[B, K]` | convex training target `(1−w)y + w p̃` |
| `L` | scalar | batch-mean cross-entropy |
| `grads` | shaped like params | `∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2` |

Axis contract: the outer sum in Eq. (4) runs over the batch axis `i = 1..B`, the inner over
the class axis `k = 1..K`; the mean is over examples (divide by `B`, NOT `B·K`). `c` and `w`
are `[B]` vectors, one value per example.

## 3. Equations to implement, with citations

| Eq | Statement | Citation (grep anchor → resolving line, re-executed 2026-08-06) |
|---|---|---|
| (1) | `c = max_k p_k` | `paper/paper.md:95-107`; `largest class probability` → :95 |
| (2) | `w = λ σ((c − τ)/s)`, `σ(z) = 1/(1+e^{−z})` | `paper/paper.md:108-164`; `logistic gate` → :108; `controls how sharply` → :164; token layout: `(` :132, `c` :134, `−` :136, `τ` :138, **s** :139, `)` :141 — the stacked `τ`-over-`s` tokens inside the parentheses fix the fractional reading `(c − τ)/s` |
| (3) | `t = (1−w) y + w p̃`, `p̃ = stopgrad(softmax(f_θ(x)/T))` | `paper/paper.md:171-212`; `convex combination` → :171; `stopgrad` → :198; `T` token → :206 |
| (4) | `L = −(1/B) Σ_{i=1..B} Σ_{k=1..K} t_ik log p_ik` | `paper/paper.md:215-252`; `cross-entropy between this target` → :215 |
| degeneracy | `λ = 0 ⇒ w = 0 ⇒ t = y ⇒` Eq. (4) = standard CE | `paper/paper.md:253-280`; `strict generalisation` → :273; `reproduces the baseline result exactly` → :280 |

Gradient actually implemented (hand-derived; the numpy implementation has no autograd).
The DEFAULT `--grad-mode literal` is the paper-faithful gradient: stopgrad ONLY on `p̃`
(Eq. 3 as written, `paper/paper.md:198, :171-174`); the gate weight `w = λσ((c−τ)/s)` is
differentiable in `z` through `c = max_k p_k`, so the gradient INCLUDES the `L→t→w→c→z`
gate path:

    dL/dz_ik = (p_ik − t_ik)/B − (1/B)·λσ(u_i)(1−σ(u_i))·(1/s)·p_im*·(δ_{m*,k} − p_ik)·g_i

where `u_i = (c_i − τ)/s`, `m* = argmax_m p_im`, `g_i = Σ_m (p̃_im − y_im) log p_im`, and
`p̃ = softmax(z/T)` is held constant. The gate-path term is proportional to `λ` and to `1/s`;
it vanishes exactly at `λ = 0`, so the degeneracy (== plain CE) holds bitwise in both grad
modes. `--grad-mode detached` treats the WHOLE target `t` as constant (`dL/dz = (p − t)/B`
only) — the standard self-distillation convention, which the paper does NOT mark on `w`.
Cross-checked by central finite differences of the loss with `p̃` frozen (w recomputed), on a
peaked net (`tests/test_invariants.py::test_gradient_matches_finite_differences`,
`stopgrad_grad_err ≈ 1.08e-3 < 5e-3`); the check discriminates both a no-stopgrad (through
`p̃`) and a detached (no gate path) implementation (mutations M6/M7).

## 4. What the paper does NOT state — permitted readings, and the weakest choice

Ranked by impact. Each item: the gap; the readings the text actually permits; the chosen
reading and why it is the WEAKEST (commits to the least beyond what is written, so admits
the most behaviours consistent with the text — Bennett's razor, arXiv:2301.12987; weakness
is the cardinality of a reading's extension, not its brevity). The anti-pattern being avoided
is fitting an unstated constant to the paper's own numbers: that is the strongest hypothesis
of all, because its extension is nearly empty.

1. **`s`, the gate sharpness in Eq. (2), has no value anywhere.** §3 lists `λ = 1`,
   `τ = 0.9`, `T = 2` and stops (`paper/paper.md:361-375`); the file's only two bare `s`
   tokens are `:139` (inside Eq. (2)) and `:162` (prose) — no assignment (re-verified
   2026-08-06). `s` is **outcome-determinative for the headline under the paper-LITERAL
   gradient**.
   *Permitted readings:* any `s > 0`. Small `s` = sharp gate near a hard threshold
   (prose-aligned: "`s` controls how sharply the gate opens", :164); large `s` = shallow,
   near-linear gate. A value that makes Table 1 come out is also "permitted" by silence,
   but selecting it for that reason is exactly the fitted hypothesis the razor forbids.
   *Choice (weakest admissible):* **`s = 0.15`** as the disclosed gate default, selected on
   non-headline grounds (sharp-gate regime described by the prose) and **provably
   non-tuning: at `s = 0.15` under the literal gradient the CWSD arm lands ≈ baseline, so
   the headline FAILS there** — `s = 0.15` was not chosen to make any claim pass.
   (Disclosed for honesty: a prior pass had calibrated `s = 0.15` against Table 1 under the
   DETACHED gradient; that calibration is moot under the literal default and is not relied on.)
   Because one scalar choice cannot speak for a valueless symbol, the full neighbourhood is
   measured: `sweep_s.py` → `s_sweep.json`, grid `s ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50,
   0.70, 1.00, 2.00, 5.00}` × seeds `{0,1,2}` × both grad modes (re-verified 2026-08-06
   against the JSON on disk; spot checks re-executed):

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

   (baseline = 0.9370/0.9407/0.9315, grad-mode- and s-independent at λ=0.) **The headline is
   `s`-DEPENDENT under the literal gradient, NOT a universal:** the gate-path gradient term
   is `∝ 1/s`, so the literal gradient converges to the detached one as `s` grows. Ordering
   `cwsd > baseline` holds at every seed from `s ≈ 0.7`; value `|cwsd − 0.962| ≤ 0.015` and
   magnitude `|gap − 0.025| ≤ 0.02` at every seed from `s ≈ 2.0`. Under DETACHED the headline
   reproduces already at the default `s = 0.15`. A prior pass truncated this sweep at
   `s = 0.30` and wrongly concluded the literal gradient never reproduces — the extended
   sweep falsifies that universal. Because the paper states neither `s` nor the stop-grad
   scope on `w` (item 5), the headline is **under-specified**: reachable under the paper's
   equations for a range of `(s, grad-mode)`, but NOT at the prose-aligned sharp-gate
   default under the literal gradient. The `λ = 0` arm is bitwise insensitive to `s` and to
   grad-mode (s=0.01/s=10.0, literal/detached all → 0.9370), so the degeneracy check is not
   touched by this choice.
2. **Weight initialisation.** Only "the parameter initialisation [is] drawn from that seed"
   (`paper/paper.md:385-389`). No distribution, scale, or scheme; biases never mentioned
   (assumed present, zero-init; the architecture sentence `:339-343` doesn't say biases exist
   at all). *Permitted readings:* any scheme/scale. *Choice:* He-normal weights, zero biases
   (`--init he`, `xavier` exposed) — a conventional but content-light choice; nothing in the
   text favours one scheme, and the degeneracy gate + baseline value check constrain the
   effort only through the RNG layout (item 7).
3. **Replacement pool for symmetric noise** (`paper/paper.md:331-338`): "replaced by a class
   drawn uniformly at random" does not say whether the true class is excluded.
   *Permitted readings:* uniform over all K (effective flip rate ≈ 0.18) or over K−1 (exactly
   0.20). The K−1 reading adds an exclusion the sentence never states — strictly stronger.
   *Choice (weakest = literal):* all K (`--noise-mode uniform-all`; `uniform-other` exposed).
4. **Minibatch sampling.** 4000 steps × 64 over 1257 examples ≈ 203.7 epochs;
   with-replacement vs shuffled-epoch cycling and the short final batch (41 examples) are all
   unstated. *Choice:* reshuffled permutation per pass, short final batch kept
   (`--batch-mode epoch-permutation`).
5. **Scope of the stop-gradient** (the finding that survives adversarial review). Eq. (3)
   annotates `stopgrad` ONLY on `p̃` (`paper/paper.md:198`) and says only "the latter [= p̃]
   treated as a constant with respect to θ" (`:171-174`). `w = λσ((c−τ)/s)` is a function of
   `θ` through `c = max_k p_k` and is NOT marked stopped; "Gradients are those of Eq. (4)"
   (`:359`) under the literal reading includes the `L → t → w → c → z` path.
   *Permitted readings:* (a) literal — stopgrad exactly where marked (on `p̃`): `w` remains
   differentiable, gate-path gradient active; (b) detached — the whole target `t` constant,
   additionally stopping the un-marked `w`: the standard self-distillation convention.
   Reading (b) commits to an extra stop-gradient the text never writes — strictly stronger,
   and the cautionary shape this step exists to avoid ("stopgrad applied to the whole target
   where the paper marked it on one term").
   *Choice (weakest = literal):* **DEFAULT `--grad-mode literal`** — Eqs. (2)–(4) as written.
   `--grad-mode detached` kept as a documented counterfactual, exercised by tests and reported
   in `selfcheck.json`/`REPRODUCTION.md`, never the gated arm. At `λ = 0` both modes are
   bit-identical (gate-path term `λ·...=0`), so the paper's degeneracy gate holds either way.
6. **SGD flavour.** No momentum, weight decay, clipping, or LR schedule mentioned
   (`paper/paper.md:344-359`). *Choice:* vanilla constant-LR SGD.
7. **RNG stream layout ("seed handling").** Split, noise mask, and init all come from seed 0
   (`paper/paper.md:385-389`), but the order/substream structure is unstated.
   *Permitted readings:* any arrangement. *Choice:* `init-first` — one
   `numpy.random.default_rng(seed)` consumed as init → corrupt → batch — because among the
   plausible arrangements it is the one that reproduces the paper's baseline 0.9370 *exactly*
   (506/540) through the paper's own λ=0 verification gate; alternatives exposed as
   `--rng-layout spawned|noise-first` give 0.9315 / 0.9426 and falsify themselves against the
   oracle. Presented as the evidence-backed choice among permitted readings, not certainty
   about the authors' stream.
8. **Evaluation protocol detail.** `FINAL accuracy` implies one evaluation after step 4000 on
   the clean test set; no intermediate eval, best checkpointing, or eval batching is mentioned.
   (Accuracy is batch-invariant, so only timing matters.)
9. **Seed count.** "All results are single runs at seed 0" (`paper/paper.md:385`). No
   variance, no error bars; Table 1's magnitudes are one seed's output.
10. **Numeric details.** dtype (float32 vs float64), log-softmax vs softmax-then-log, and the
    framework are never stated. *Choice:* float32, log-softmax for stability, numpy.
11. **Test-time label corruption.** Negatively pinned: only *training* labels are corrupted
    (`paper/paper.md:327`); the test set stays clean.

Implementation-side tolerances the paper also cannot state (recorded for honesty):
`stopgrad_grad_err < 5e-3` accommodates float32 central-difference noise on a peaked net
(~1.08e-3 measured); the degeneracy `< 1e-12` bounds are met exactly (0.0, bitwise).

## 5. Component interfaces (frozen)

One file, `run_experiment.py`, at the reproduction-folder root — the name is fixed by the
paper (`paper/paper.md:459-462`). numpy + scikit-learn only; hand-derived gradients so
stop-grad semantics are structural. Verified against the code 2026-08-06 (`grep
add_argument` output matches).

CLI:

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

Output contract: exactly **one** stdout line at completion, `FINAL accuracy=<float>`
formatted `%.4f` (`paper/paper.md:466-469`); all diagnostics on stderr; optional
`--metrics-out` writes a JSON of structural metrics (stdout contract unchanged;
`tests/test_cli.py` pins it).

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
train(config) -> (accuracy, params, Xtr, Ytr_onehot)   # exactly --steps SGD updates; the
                                                       # data arrays are returned so main()
                                                       # can compute structural_metrics on
                                                       # a fixed batch for measured.json
```

RNG discipline: default `init-first` (one `default_rng(seed)`: init params → corrupt labels →
batching), the arrangement that passes the paper's own degeneracy gate exactly (§4 item 7).

## 6. Arms — configurations, and per-arm restriction check

The paper compares exactly **one method against one baseline** (Table 1) — two arms, the
same program under a different `--lambda` (`paper/paper.md:449-462`).

| Arm | λ | Command | Config |
|---|---|---|---|
| `baseline` ("Cross-entropy (baseline)") | 0.0 | `python run_experiment.py --lambda 0.0 --seed {seed} --metrics-out <tmp>` | Paper-stated: lr 0.1, batch 64, 4000 steps, 20% symmetric noise, seed 0. `τ/s/T/grad-mode` are **inert** at λ=0 (`w ≡ 0` and the gate-path term is `λ·...=0`, `:253-280`); bit-identical under literal or detached. Defaults per §4: s 0.15, init he, noise-mode uniform-all, batch-mode epoch-permutation, rng-layout init-first. |
| `cwsd` ("CWSD (ours)") | 1.0 | `python run_experiment.py --lambda 1.0 --grad-mode literal --seed {seed} --metrics-out <tmp>` | Paper-stated: λ=1, τ=0.9, T=2 (`:361-375`), same optimiser/steps/noise as baseline. `--grad-mode literal` is the paper-faithful gradient (stopgrad ONLY on `p̃`, gate path active; §4 item 5). `s` unstated; pinned to the sharp-gate default `s=0.15` with a disclosed, provably non-tuning rationale (§4 item 1). CWSD results are `s`-DEPENDENT under literal; the sweep reports the whole neighbourhood instead of generalising from one point. |

Metric (both arms): held-out test accuracy from the single `FINAL accuracy=<float>` line,
plus structural-invariant metrics via `--metrics-out` (`param_count`, `gate_w_min/max`,
`target_min`, `target_sum_err`, `stopgrad_grad_err` for `cwsd`; `degeneracy_loss_err`,
`degeneracy_grad_err` for `baseline`), collected by `run_all_arms.sh` into `measured.json`.
Structural metrics are cheap (one 128-example batch with trained params; `stopgrad_grad_err`
on a fixed tiny network at seed 123) and independent of the 4000-step budget.

**Restriction check per arm** (is our setup a child of the paper's in Bennett's sense —
definition 6: evaluated situations may be a subset, but what counts as CORRECT must not
change — and which claims it still supports):

- **Both arms — situations:** same dataset in full (all 1797 examples, same 70/30 stratified
  split), same noise rate, same model, same optimiser, **same 4000-step budget as the paper
  (no horizon restriction)**, same seeds the paper used plus two more (`{0,1,2}` vs `{0}`).
  The two added seeds expand the situation set along the seed axis with the correctness
  criterion unchanged; every claim is quantified "at every seed", which is strictly more
  demanding than the paper's single-seed statement — an expansion, never a relaxation.
- **Both arms — correctness criterion:** unchanged. The metric is the paper's: accuracy on
  the clean, class-stratified 540-example test set, computed identically in both arms;
  thresholds are the paper's own numbers (0.9370 / 0.9620 / +2.5 pts) with declared
  tolerances; the degeneracy predicate is the paper's own verification gate. We do not
  redefine accuracy, do not early-stop on the test set, do not select checkpoints, and do
  not measure a proxy under the paper's name.
- **`cwsd` arm — the genuine restriction:** the paper leaves `s` and the stop-grad scope
  open; we instantiate ONE point of the paper's permitted `(s, grad-mode)` region
  (`s=0.15`, literal). That narrows which permitted reading is measured — a restriction of
  situations along an axis the paper left free — while the correctness criterion is
  untouched. Compensation: `sweep_s.py` → `s_sweep.json` reports the neighbourhood (10 `s`
  values × 3 seeds × 2 grad modes) so no verdict over-generalises from the point.
- **Claims each arm still supports:** the structural invariant/existence claims and the
  ordering claim are budget-independent or sign-only, so the reduced compute (none — full
  budget) and seed choices do not weaken them: supported as gated. The two CWSD value claims
  are supported only AT the gated `s` (they adjudicate the gated point, and the sweep
  reports where else they hold); the baseline value claim is grad-mode- and `s`-insensitive,
  supported at all three seeds. No claim is gated on a horizon too short to separate the
  arms: the budget equals the paper's; where the arms are NOT separated (ordering at the
  gated `s=0.15`), the declared spread heuristic rules the verdict `untested` rather than
  reporting a within-noise number as a separation.

## 7. Figures

**The paper has no figures** — only Table 1. Verified 2026-08-06: `grep -niE
"figure|fig\.|curve|plot" paper/paper.md` matches nothing; `paper/` contains only `paper.md`
(arxiv_id unknown ⇒ no LaTeX/figure assets, nothing for `read-figure` to read). Therefore
there are no `curve` claims and `claims.json` carries no `figures` key.

## 8. `claims.json`

Written to `claims.json` at the reproduction-folder root (what the numbers gate settles).
Re-verified 2026-08-06: all 12 quotes verbatim against `paper/paper.md` under the
normalization below; seeds `[0, 1, 2]` (the paper uses only seed 0, `:385`; seeds 1/2 are
the seed-robustness expansion of §6); two arms per §6 with per-arm `config` and `metrics`;
a top-level machine-readable **`restrictions` map** keyed by arm —
`{"<arm>": {"kind": "none"|"narrows_situations"|"changes_correctness", "detail": ...}}`
(`baseline` → `none`: the unstated knobs are inert at `λ=0`, and the extra seeds expand
situations without touching correctness; `cwsd` → `narrows_situations`: one point of the
paper's permitted `(s, grad-mode)` region is gated, the criterion untouched, the
neighbourhood reported by the sweep) — mirroring §6's prose restriction check;
an `evaluation` block documenting resolution semantics and verdict rules; 9 claims;
`not_tested`.

**Quote policy** (every `quote` field): verbatim from `paper/paper.md` after (a) joining
end-of-line hyphens with the hyphen kept and (b) collapsing whitespace to single spaces;
PDF token spacing is preserved (`2 . 5`, `[0 , 1]`, `0 . 9370`). The verifier one-liner is
embedded in `claims.json.evaluation.quote_policy`; all 12 quotes PASS (2026-08-06).

| id | kind | compute_invariance | settles |
|---|---|---|---|
| `cwsd-improves-over-baseline` | ordering | **high** | `measured.cwsd.accuracy - measured.baseline.accuracy > 0` at every seed; declared spread heuristic: if |mean gap| ≤ cross-seed spread the arms are not separated → `untested`, not `refuted` |
| `baseline-accuracy-value` | value | low | `abs(measured.baseline.accuracy − 0.9370) ≤ 0.01` at every seed |
| `cwsd-accuracy-value` | value | low | `abs(measured.cwsd.accuracy − 0.9620) ≤ 0.015` at every seed (adjudicates the gated `s=0.15`; see §4 item 1 and §6's restriction check) |
| `improvement-magnitude-2p5-points` | value | low | `abs((cwsd−baseline) − 0.025) ≤ 0.02` at every seed (same scope) |
| `lambda-zero-is-exact-cross-entropy` | invariant | **high** | `gate_w_max == 0 and degeneracy_loss_err < 1e-12 and degeneracy_grad_err < 1e-12` |
| `gate-weight-bounded-by-lambda` | invariant | **high** | `gate_w_min > 0 and gate_w_max < 1` |
| `target-is-convex-combination` | invariant | **high** | `target_min >= 0 and target_sum_err < 1e-6` |
| `stop-gradient-holds-target-constant` | invariant | **high** | `stopgrad_grad_err < 5e-3` (stopgrad on `p̃` only, gate path active) |
| `single-network-no-extra-parameters` | existence | **high** | `param_count == 4` (COMPUTED via `len(params)`) |

Compute-invariance rationale: the ordering claim is sign-only and the five
invariant/existence claims are structural (independent of the 4000-step budget) — all six
high; the three value claims are exact magnitudes the paper reports for single seed-0 runs —
low. The gate gates on the high ones, so this reproduction's verdict does not reduce to
"nothing was affordable". Expected adjudication from this pass's verified measurements:
5 high structural claims + baseline value PASS; the two CWSD value claims REFUTED at the
gated `s=0.15`; the ordering UNTESTED at the gated `s` (within noise, flips at seed 1) —
with the §4-item-1 sweep recording that all three headline claims hold under the literal
gradient for `s ≥ ~0.7`–`2.0` and under the detached counterfactual at `s=0.15`.

**Deliberately not tested** (`claims.json.not_tested`): (a) the attribution claim
("We attribute the gain to the gate suppressing…", `paper/paper.md:445-447`) — a mechanism
claim the paper itself phrases as attribution, needing per-example gate-weight logging the
one-`FINAL accuracy=`-line output contract does not expose; (b) Table-1 magnitudes at
seeds ≠ 0 (`paper/paper.md:385`) — value claims at seeds 1/2 are out-of-distribution for the
paper's numbers (and moot at the gated `s`, where they fail at every seed including 0); (c)
the DETACHED counterfactual — implemented and reported, but not the gated arm: it is the
stronger, conventional stop-grad the paper never writes (§4 item 5).

## 9. Upstream code — searched, none found (re-verified 2026-08-06)

- **In the paper**: no URLs, DOIs, or code-availability statements (`grep -niE
  "http|www\.|github|arxiv|doi|available at" paper/paper.md` → 0 matches); §5 "Reproducing"
  gives only commands (`paper/paper.md:449-469`).
- **GitHub repository search** (api.github.com/search/repositories, re-run 2026-08-06):
  `confidence-weighted self-distillation` → `total_count: 0`; `self-distillation label noise`
  → `0`; `cwsd label noise` → `0`; `Institute for Applied Learning Systems` → 6 hits, all
  inspected and unrelated (course repos, theses, unrelated ML repos — no CWSD code).
- **GitHub user search** (re-run 2026-08-06): `Bergstrom Oyelaran Vasquez` → `total_count: 0`.

Conclusion: **no usable upstream implementation exists**; the method is implemented from
scratch against §1/§5.

## 10. Validation targets

| Arm | λ | Paper (Table 1, `paper/paper.md:400-414`) | This pass 2026-08-06 (seed 0, LITERAL, gated `s=0.15`) | seed 0, LITERAL `s=2.0` | seed 0, DETACHED `s=0.15` (counterfactual) |
|---|---|---|---|---|---|
| Cross-entropy baseline | 0.0 | 0.9370 | **0.9370** (exact, 506/540; re-executed 2026-08-06) | 0.9370 (same; λ=0 is s- and grad-mode-independent) | 0.9370 (same) |
| CWSD | 1.0 | 0.9620 | **0.9407** (does NOT reproduce at gated `s=0.15`; +0.0037 over baseline; re-executed 2026-08-06) | **0.9648** (reproduces; +0.0278; re-executed 2026-08-06) | **0.9611** (reproduces; +0.0241; re-executed 2026-08-06) |

The paper's Table-1 CWSD number (0.9620) is **`s`-DEPENDENT under the paper-LITERAL
gradient** (stopgrad only on `p̃`, as Eq. 3 marks it), NOT a universal: at the gated
sharp-gate default `s=0.15` it does NOT reproduce at any seed (CWSD ≈ baseline); but because
the gate-path term is `∝ 1/s`, the literal gradient converges to the detached one as `s`
grows, and the headline REPRODUCES under the literal gradient for shallow gates
(`s ≥ ~0.7` ordering, `s ≥ ~2.0` value/magnitude across seeds 0/1/2). It also reproduces
under the DETACHED variant already at `s=0.15`. The paper states neither `s` nor the
stop-grad scope on `w`, so the headline is under-specified: reachable under the paper's
equations for a range of `(s, grad-mode)` but not at the prose-aligned sharp-gate default
under the literal gradient. That `s`-dependence is the reproduction's central finding.

Structural gates (all implemented in `tests/`, 53 tests, passing 2026-08-06):
(a) `λ = 0` path is exact CE, asserted bitwise (per-step loss+grads and a 300-step SGD loop)
against an independently written CE routine, swept over `s ∈ {0.01…10.0}` so the gate cannot
be fit through the unstated sharpness — holds under BOTH grad modes; (b) `t = y` at `w = 0`;
(c) Eq. (4) at `t = p̃` equals a direct `CE(p̃, p)`; (d) the LITERAL `∂L/∂z = (p−t)/B +
gate-path` vs finite differences with `p̃` frozen (w recomputed), on a peaked net where a
no-stopgrad and a detached implementation both diverge; (e) the loop performs exactly
`--steps` updates; (f) the data loader is fingerprinted against a re-derivation
(`tests/test_instruments.py`). Sensitivity over the §4 choices (`s`, init, noise-mode,
batch-mode, rng-layout, grad-mode) is reported, not adjudicated — the paper cannot
adjudicate it; see REPRODUCTION.md and `s_sweep.json`.
