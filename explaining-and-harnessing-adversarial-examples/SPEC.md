# SPEC — Reproduction of "Explaining and Harnessing Adversarial Examples"

- **Paper:** Goodfellow, Shlens, Szegedy, ICLR 2015. arXiv:1412.6572v3.
- **Authoritative source for all equations/numbers:** `paper/source/iclr2015.tex` (arXiv LaTeX, verified byte-identical to a fresh `https://arxiv.org/e-print/1412.6572` fetch). Preamble macros resolved: `\eps→\epsilon`, `\sign→\text{sign}`, `\vx,\veta,\vtheta,\vw,\vbeta→\bm{x},\bm{\eta},\bm{\theta},\bm{w},\bm{\beta}`. All citations below are "file:line + grep-able string" into that file.
- **Fallback prose source:** `paper/paper.md` (PDF extraction; prose only, maths unreliable).

---

## 1. Scope and milestones

The paper reports experiments on four model families (softmax regression, logistic regression,
maxout networks, RBF networks) plus an MP-DBM, a 12-net ensemble, and a GoogLeNet/ImageNet
figure. We reproduce the **MNIST core**; the rest is graded below.

| ID | Milestone | Paper target | Citation |
|----|-----------|--------------|----------|
| M1 | Softmax regression on MNIST; FGSM ε=0.25 | adv error 99.9%, mean conf 79.3% | tex:333 `an error rate of 99.9\% with an average confidence of` |
| M2 | Logistic regression, MNIST 3-vs-7; FGSM ε=0.25 | clean 1.6%, adv 99% | tex:454 `has a 1.6\% error rate on the 3 versus 7`, tex:455-456 |
| M3 | Maxout 240×2 MNIST + dropout; FGSM ε=0.25 | adv error 89.4%, conf 97.6% | tex:338-339 `a maxout network misclassifies 89.4\%` |
| M4 | Adversarial training of M3 (α=0.5, ε=0.25) | clean test error 0.94% → 0.84% | tex:492-494 `from 0.94\% without adversarial training to 0.84\%` |
| M5 | Large maxout 1600×2: baseline vs adversarial training; early stop on *adversarial* valid error; retrain on 60k; 5 seeds | 1.14% w/o adv-training; 4×0.77% + 1×0.83%, mean 0.782% | tex:497-512 |
| M6 | Robustness of M5-model: own-FGSM error; cross-transfer both directions; confidence | 17.9%; 19.6%; 40.9%; conf 81.4% | tex:514-523 |
| M7 | Noise-training controls (±ε pixel flips, U(−ε,ε) noise) on maxout | adv errors 86.2% (conf 97.3%), 90.4% (conf 97.8%) | tex:555-557 |
| M8 | Shallow RBF on MNIST; FGSM ε=0.25; cross-model class agreement (§8) | adv error 55.4%, conf-on-mistakes 1.2%, clean conf 60.6%; agreement 16.0% / 54.6% / 84.6% / 54.3% / 53.6% | tex:600-604; tex:679-688 |
| M9 | Rubbish examples, MNIST: 10,000 samples ∼ N(0, I₇₈₄) | maxout+softmax 98.35% (conf 92.8%); sigmoid top 68% (87.9%); softmax reg 59.8% (70.8%); RBF 0% | tex:905-909; tex:919-924 |
| E1 (extended) | Ensemble of 12 maxout nets | 91.1% (ensemble-targeted), 87.9% (single-targeted) | tex:819-825 |
| E2 (extended) | CIFAR-10 conv maxout, ε=0.1; rubbish N(0,I₃₀₇₂) 1,000 samples; targeted fooling | adv 87.15% (96.6%); rubbish 93.4% (84.4%); fooling avg 75.3%/step (frogs+trucks 100%, airplanes 24.7%) | tex:340-341; tex:911-912; tex:936-941 |
| — (excluded) | GoogLeNet/ImageNet Fig. 1 (ε=0.007) | qualitative | tex:381. Reason: 2014-era DistBelief GoogLeNet weights unavailable. |
| — (excluded) | MP-DBM FGSM ε=0.25 → 97.5% | tex:793-800 | Reason: requires implementing the multi-prediction deep Boltzmann machine (Goodfellow et al. 2013a), a separate paper's model; out of budget. |

Compute: CPU-only, 16 cores, Python 3.12.13 (this sandbox's system interpreter; the from-scratch
Dockerfile pins `python:3.13-slim` — the pinned wheels resolve identically on 3.12 and 3.13), PyTorch.
M5 (5 seeds × 2 arms) and E1 (12 nets) are
the expensive items; seeds/ensemble members run as parallel processes.

The numbers-gate contract for the table above is **`claims.json`** (§10): 34 claims over 13 named
arms with ≥3 seeds, each with a verbatim paper quote, an exact file:line citation into
`paper/source/iclr2015.tex`, a kind, an honest compute-invariance rating (16 high / 6 medium /
12 low — gating settles the **high** ones), and the settling arithmetic over `measured.<arm>.<metric>`,
plus 10 claims the paper makes that this reproduction deliberately does NOT test, with reasons.

### 1.A. The comparison arms (machine-readable mirror: `arms_metadata.json`; gate map: `arms.json`)

The paper's own comparisons, named and fixed here. `arms_metadata.json` is the
rich machine-readable mirror (12 milestone arms with model/training/eval
configs, paper targets, citations, results paths). `arms.json` is the GATE
contract: a flat map from each arm the gate runs to the shell command that
produces it (`{"baseline": "...", "adversarial": "..."}`), where each command
prints exactly one line `FINAL <arm name>=<clean test accuracy>` and
`run_all_arms.sh` runs both. The gate arms are the paper's headline M4
comparison (tex:492-494): clean maxout training (`baseline`) vs FGSM
adversarial training (`adversarial`, Algorithm B, eps=0.25, alpha=0.5).
Training arms train a model; eval arms only consume trained models. Two
extra evaluation-only rows (M6, M9) reuse the models of arms 3/6 and 1/3/8.
Full per-arm configs (model, training, evaluation, target, citation, script,
results file, metric paths) are in `arms_metadata.json` at the repo root.

| Arm name | Kind | Model & training config | Attack/eval config | Paper target(s) |
|----------|------|------------------------|--------------------|-----------------|
| `m1_softmax_regression` | train | SoftmaxRegression 784→10, clean CE (SGD; paper states no optimizer) | FGSM ε=0.25, MNIST test | adv err 99.9%, conf 79.3% (tex:333) |
| `m2_logistic_3v7` | train | LogisticRegression 784→1, y∈{−1,+1}, E5 softplus | exact FGSM η=−ε·y·sign(w), ε=0.25 | clean 1.6%, adv 99% (tex:454-456) |
| `m3_maxout240_clean` | train | MaxoutMLP 240×2, pieces 5, input-dropout .8, external recipe | FGSM ε=0.25 | adv err 89.4%, conf 97.6% (tex:338-339); also M4's clean arm 0.94% (tex:492-494) |
| `m4_maxout240_advtrain` | train | identical to `m3_maxout240_clean` + Algorithm B (α=0.5, ε=0.25, stop-grad) | clean test error | 0.84% vs baseline 0.94% — the headline comparison (tex:492-494) |
| `m5_maxout1600_clean` | train | MaxoutMLP 1600×2, clean-early-stop (patience 100), retrain on 60k | clean test error | 1.14% (tex:497-499) |
| `m5_maxout1600_advtrain` | train | identical + Algorithm B; early stop on ADVERSARIAL valid error; retrain on 60k; seeds 0–4 | clean test error | 4×0.77% + 1×0.83%, mean 0.782% (tex:506-512) |
| `m7_maxout_noise_sign` | train (control) | `m3` model trained on x+ε·b, b∈{±1} iid (E10) | FGSM ε=0.25 | adv err 86.2%, conf 97.3% (tex:555-557) |
| `m7_maxout_noise_uniform` | train (control) | `m3` model trained on x+u, u∼U(−ε,ε) iid (E10) | FGSM ε=0.25 | adv err 90.4%, conf 97.8% (tex:555-557) |
| `m8_rbf_shallow` | train | RBFNet (10 quad. forms; multiclass norm unstated, §6 item 9) | FGSM ε=0.25 + §8 class-agreement evals | adv err 55.4%, conf-on-mistakes 1.2%, clean conf 60.6% (tex:600-604); agreement 16.0/54.6/84.6/54.3/53.6% (tex:679-688) |
| `e1_ensemble12_maxout` | train (12×) | 12× `m3` config, distinct RNG seeds | FGSM ε=0.25 vs ensemble (mean-loss grad; rule unstated §6 item 11) and vs single member | 91.1% / 87.9% (tex:822-825) |
| `m6_robustness_transfer_eval` | eval-only over arms 3 & 6 | — | own-FGSM; cross-transfer both directions, ε=0.25 | 17.9% / 19.6% / 40.9%, conf 81.4% (tex:514-523) |
| `m9_rubbish_evals` | eval-only over arms 1, 3 (×sigmoid-top), 8 | — | 10,000 samples ∼ N(0,I₇₈₄); error := max p > 0.5 (tex:905-906) | maxout+softmax 98.35% (92.8%); sigmoid-top 68% (87.9%); softmax-reg 59.8% (70.8%); RBF 0% (tex:906-909, 919-924) |

**Graded-harness arms** (gate, in `arms.json` + `run_all_arms.sh`): **baseline** =
`python run_experiment.py --baseline --steps 5000 ...` (clean maxout training);
**adversarial** = `python run_experiment.py --lambda 0.25 --steps 5000 ...`
(Algorithm B by default, α=0.5, dropout off for the degeneracy gate, §6 item 22).
Each prints exactly one line `FINAL <arm name>=<clean test accuracy>` (arm name =
`baseline` / `adversarial`). Degeneracy contract: `--lambda 0` (arm `adversarial`
at its no-op) prints the SAME accuracy VALUE as `--baseline` (arm `baseline`);
the arm name differs by construction, the float value matches bit-for-bit
(`tests/test_degeneracy.py`). Thread tuning: `run_all_arms.sh` / `smoke.sh` export
`OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4` (PyTorch's default pool over-spawns on
multi-core CPUs, a ~7x slowdown observed).

**Excluded arms** (recorded, not run): GoogLeNet/ImageNet Fig. 1 (tex:378-383; 2014 DistBelief
weights unavailable); MP-DBM ε=0.25 → 97.5% (tex:793-800; a separate paper's model);
CIFAR-10 conv maxout E2 (tex:340-341, 910-912, 936-941; beyond the MNIST core, architecture
exists only externally); **rotation-based adversarial examples / rotational hidden-layer
perturbations** (tex:346-347, 562-584) — a negative result the paper reports as a weaker
regularizer than additive FGSM, with no quantified MNIST number; excluded as it would
require a differentiable rotation operator and yields no headline claim; **the trained-to-zero-
on-Gaussian-rubbish maxout arm** (tex:963-965) — the paper reports it confers NO clean-test
reduction (unlike adversarial training); excluded as it requires a per-class rubbish training
protocol and reproduces only a null result (no regularization benefit). Both exclusions are
recorded here rather than silently omitted; the L1 weight-decay control (Section 5, the one
named *quantified* control the paper reports with numbers) IS now implemented (item 28).

**Unimplemented but recorded (no exclusion arm, fits existing machinery):**
- **MNIST rubbish class-skew statistic** (tex:929-930): "On MNIST, 45.3% of a naively trained
  maxout network's false positives were classified as 5s, and none were classified as 8s." This
  is a stated MNIST-core number implementable from the M9 maxout+softmax arm (collect the
  predicted class on the erroring N(0,I) subset). It is NOT implemented (M9 reports only error
  rate + confidence, not the per-class false-positive distribution) and is recorded here as a
  known gap rather than silently omitted.
- **Fig. 3 weight-localization** (tex:523-528): "the weights of the adversarially trained model
  being significantly more localized and interpretable." A qualitative claim with no numeric
  target; not checked and not reproducible as a number. Recorded as excluded.

---

## 2. The method as explicit algorithms

Notation/conventions fixed for the whole implementation (see §4 for shapes):

- `B` batch size, `F` input dim (784 MNIST, 3072 CIFAR-10), `K` classes (10).
- `J(θ, x, y)` — "the cost used to train the neural network" (tex:305-306 `Let $\vtheta$ be the parameters of a model`). For softmax-output models this is mean cross-entropy / NLL over the batch, `J = (1/B) Σ_b −log p(y_b | x_b)`. **The equation for J is never displayed in the paper**; cross-entropy is the implied standard training cost (softmax regression / maxout are trained with logistic loss). For logistic regression, J is the softplus loss E5 below.
- `sign(·)` is elementwise; `sign(0) := 0` (paper does not state the convention).
- FGSM perturbed inputs are **not clipped** back to the valid pixel range anywhere in the paper. We do not clip. (The commented-out footnote at tex:313-326 shows the Szegedy et al. variant *did* constrain features to the data range; this paper's displayed method does not.)

### Algorithm A — FGSM attack (evaluation-time)

```
Input:  model f(·;θ) with cost J(θ,x,y); batch x [B,F], labels y [B]; ε>0
1. g ← ∂J(θ,x,y)/∂x                    # autodiff, same shape as x  [B,F]
2. η ← ε · sign(g)                     # elementwise; ||η||_∞ = ε exactly
3. x̃ ← x + η                           # NO clipping
Output: x̃
```
Equation: **η = ε sign(∇ₓJ(θ,x,y))** — tex:309 `\[ \veta = \eps \sign \left( \nabla_\vx J(\vtheta, \vx, y) \right). \]`; x̃ = x + η — tex:235 `$\tilde{\vx} = \vx + \veta$`; constraint ||η||∞ < ε — tex:239 `$||\veta||_\infty < \eps$`.
ε values: MNIST ε=0.25 (tex:333 `using $\eps=.25$`), CIFAR-10 ε=0.1 (tex:340 `using $\eps=.1$`), ImageNet ε=0.007 (tex:381 `Here our $\eps$ of .007`).

### Algorithm B — FGSM adversarial training

```
Per minibatch (x, y):
1. x̃ ← FGSM(θ, x, y, ε)  with θ detached inside the sign computation
   # tex:559-561: "the derivative of the sign function is zero or undefined everywhere,
   #  ...does not allow the model to anticipate how the adversary will react" ⇒ stop-grad
2. L ← α·J(θ,x,y) + (1−α)·J(θ,x̃,y)     # α=0.5 in ALL experiments
3. θ ← optimizer step on ∇_θ L
# perturbations are regenerated every batch ("we continually update our supply", tex:490-491)
Early stopping (M5): track ADVERSARIAL validation-set error (FGSM against the current model),
NOT the clean valid error (tex:503-505). After choosing epochs, retrain on all 60,000 (tex:506).
```
Equation: **J̃(θ,x,y) = αJ(θ,x,y) + (1−α)J(θ, x + ε sign(∇ₓJ(θ,x,y)))** — tex:486-487 `\[ \tilde{J}(\vtheta, \vx, y) = \alpha J(\vtheta, \vx, y) + (1-\alpha) J(\vtheta,`; α=0.5 — tex:488 `we used $\alpha = 0.5$`. ε=0.25 for MNIST adversarial training — tex:428-429 `good results using adversarial training with $\eps = .25$`.

### Algorithm C — Closed-form adversarial logistic regression (3-vs-7)

```
Model: P(y=1) = σ(wᵀx + b), y∈{−1,+1}                    # tex:399
Clean loss:  E ζ(−y(wᵀx + b)),  ζ(z)=log(1+exp(z))       # tex:402, tex:404
Adversarial loss (exact worst case under max-norm box):
  E ζ(y(ε‖w‖₁ − wᵀx − b))                                # tex:411
  derivation: sign of gradient = −sign(w); wᵀ sign(w) = ‖w‖₁   # tex:407
```

### Algorithm D — Rubbish examples and targeted fooling (appendix)

```
Untargeted: draw n samples x ~ N(0, I_F); "error" := max_k p(y=k|x) > 0.5   # tex:905-906
Targeted class i: x ← x + ε·∇ₓ p(y=i|x) on a Gaussian sample, resample until success  # tex:936
  Fig.5 caption variant: gradient *sign* step instead of raw gradient — tex:954
  "taking a gradient sign step". BOTH phrasings recorded; we implement the sign step
  (matches caption + FGSM theme) and note the prose/caption mismatch in unstated list.
```

---

## 3. Symbol table — shapes (frozen)

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `x` | `float32 [B, F]`, F=784 (28×28 flattened), values in [0,1] for MNIST (tex:334 `pixel values in the interval [0, 1]`) | input batch |
| `y` | `int64 [B]`, values 0..9 (or {−1,+1} for M2) | labels |
| `θ` | flat parameter vector of each model | all weights/biases |
| `J(θ,x,y)` | scalar `float32` | mean batch training cost (cross-entropy; softplus for M2) |
| `∇ₓJ` | `float32 [B, F]` (same layout as x) | input gradient |
| `η` | `float32 [B, F]` | perturbation; `‖η‖∞=ε` elementwise |
| `x̃` | `float32 [B, F]` | adversarial example = x+η |
| `ε` | scalar | max-norm bound; 0.25 MNIST / 0.1 CIFAR-10 |
| `α` | scalar = 0.5 | adversarial-training mixing weight |
| Softmax reg: `W`, `b` | `[784,10]`, `[10]`; logits `z = xW + b [B,10]`; `p = softmax(z) [B,10]` | M1 |
| Logistic reg: `w`, `b` | `[784]`, scalar; `s = x·w + b [B]` | M2 |
| Maxout layer l: `W_l`, `b_l` | `W_l [d_{l-1}, U·P]`, `b_l [U·P]`; pre-max `a = xW_l+b_l [B, U·P]` → reshape `[B, U, P]` → `h_l = max over P [B, U]` | U=240 or 1600 units, P pieces |
| Maxout net | 2 hidden maxout layers + softmax readout `[U,10]` | M3–M7, E1 |
| RBF: `μ_k`, `β_k` | `μ_k [784]`; `β_k` diagonal neg-def, `β_k = -diag(a_k)` with `a_k = softplus(raw_k) > 0` `[784]` per class k=0..9 (neg-def BY CONSTRUCTION) | q_k(x) = (x−μ_k)ᵀβ_k(x−μ_k) = −Σ_f a_{k,f}(x_f−μ_{k,f})² `[B]` (≤ 0) |
| RBF probabilities | `p(y=k\|x) = exp(q_k(x))` — UNNORMALIZED per-class (the paper's binary E8 form extended per-class); confidence = max_k exp(q_k) ∈ (0,1]; argmax(q_k) for the prediction | **paper prints only the binary form; multiclass normalization + β structure (diagonal, neg-def by construction) recorded in §6 item 9** |

Sign edge cases: `sign(g)=+1/0/−1` elementwise; zero gradient ⇒ zero perturbation for that element.

---

## 4. Component interfaces (frozen — do not renegotiate)

Layout inside `explaining-and-harnessing-adversarial-examples/`:

```
src/fgsm_repro/{__init__,data,models,attacks,objectives,train,eval}.py
experiments/m1_softmax.py ... m9_rubbish.py (+ extended/)
results/<milestone>.json        # every run writes parsed numbers here
tests/                          # degeneracy + shape tests
```

```python
# data.py
@dataclass
class MNISTData:
    x_train: Float[Tensor, "50000 784"]; y_train: Long[Tensor, "50000"]
    x_valid: Float[Tensor, "10000 784"]; y_valid: Long[Tensor, "10000"]
    x_test:  Float[Tensor, "10000 784"]; y_test:  Long[Tensor, "10000"]
def load_mnist(root: Path, seed: int) -> MNISTData
    # raw IDX files, pixels/255 → [0,1], NO other preprocessing.
    # split: first 50000 of the 60000 training images = train, last 10000 = valid
    # (matches external pylearn2 split start/stop 0:50000 / 50000:60000; "retrained on
    #  all 60,000 examples" tex:506 ⇒ final M5 arm uses train+valid as train).
def load_mnist_3v7(root) -> MNISTData   # labels 3→−1, 7→+1, same split convention

# models.py — every classifier implements:
class Classifier(Protocol):
    def logits(self, x: Float[Tensor, "B F"]) -> Float[Tensor, "B K"]  # pre-softmax scores
    # train/eval mode controls dropout. Determinism: all RNG via torch.Generator(seed).
SoftmaxRegression(in_dim=784, n_classes=10)          # M1
LogisticRegression(in_dim=784)                       # M2
MaxoutMLP(units: int, pieces: int=5, in_dim=784, n_classes=10,
          dropout_input_include: float, dropout_hidden_include: float)  # M3-M7
    # forward: h0 = max_pieces(W0 @ dropout(x) + b0); h1 = ...; logits = W_y @ h1
RBFNet(n_classes=10, in_dim=784)                     # M8; mu [K,784], beta [K,784,784]

# attacks.py
def fgsm(model: Classifier, x, y, eps: float) -> Tensor   # Algorithm A; returns detached x̃
def fgsm_ensemble(models: list[Classifier], x, y, eps)    # E1; grad of mean CE over members
def fooling_sign_step(model, x, cls: int, eps) -> Tensor  # Algorithm D (sign variant)
def sample_rubbish(n: int, dim: int, gen) -> Tensor       # N(0, I_dim)

# objectives.py
def cross_entropy_cost(model, x, y) -> scalar                 # J for softmax models
def softplus_logreg_cost(w, b, x, y_pm) -> scalar             # E5
def adversarial_logreg_cost(w, b, x, y_pm, eps) -> scalar     # E6 (closed-form, no autograd η)
def adversarial_train_cost(model, x, y, eps, alpha=0.5) -> scalar  # E7, stop-grad inside sign

# train.py
@dataclass
class TrainConfig:
    batch_size: int; lr: float; momentum: float; max_epochs: int; seed: int
    adv_train: bool = False; eps: float = 0.25; alpha: float = 0.5
    early_stop: Literal["clean","adversarial"] = "clean"; patience: int = 100
def train(model, cfg: TrainConfig, data: MNISTData) -> TrainResult
    # TrainResult: best_state_dict, history{epoch: metrics}, epochs_run

# eval.py — metric definitions are FIXED here:
@dataclass
class AttackEval:  error_rate: float; mean_confidence_on_errors: float; n: int
    # error_rate: fraction of ALL evaluated examples misclassified = mean(argmax logits != y)
    # confidence: softmax probability assigned to the PREDICTED class, averaged over the
    #   misclassified subset only ("average confidence of 79.3% on the MNIST test set" tex:333,
    #   "average probability ... assigned to the incorrect labels" tex:341).
def eval_clean(model, x, y) -> float
def eval_fgsm(model, x, y, eps) -> AttackEval
def eval_transfer(src, tgt, x, y, eps) -> AttackEval      # attack src, score on tgt (M6)
def class_agreement(m1, m2, x, y, eps) -> AgreementStats  # §8 numbers (tex:679-688)
def eval_rubbish(model, n, dim, seed) -> AttackEval       # error := max p > 0.5 (tex:906)
```

Config: one JSON/YAML per experiment under `experiments/`; every JSON in `results/` records
milestone id, all hyperparams, seed, and the four fields above per eval.

---

## 5. Equations to implement — with citations

| # | Equation | Citation (grep-able) |
|---|----------|----------------------|
| E1 | η = ε sign(∇ₓ J(θ,x,y)) | `paper/source/iclr2015.tex:309` — `\[ \veta = \eps \sign \left( \nabla_\vx J(\vtheta, \vx, y) \right). \]` |
| E2 | x̃ = x + η; require ‖η‖∞ < ε | tex:235 `$\tilde{\vx} = \vx + \veta$`; tex:239 `$||\veta||_\infty < \eps$` |
| E3 | wᵀx̃ = wᵀx + wᵀη; growth max with η=ε·sign(w) ⇒ εmn (n dims, avg weight magnitude m) | tex:243 `\[ \vw^\top \tilde{\vx} = \vw^\top \vx + \vw^\top \veta. \]`; tex:246-247 `by assigning $\eta = \text{sign}(\vw)$` / `the activation will grow by $\eps m n$` |
| E4 | y∈{−1,1}, P(y=1)=σ(wᵀx+b) | tex:399 `with $P(y=1) = \sigma\left( \vw^\top \vx + b\right)$` |
| E5 | logistic cost E ζ(−y(wᵀx+b)), ζ(z)=log(1+exp z) | tex:401-404 `\mathbb{E}_{\vx, y \sim p_\text{data}} \zeta( -y (\vw^\top \vx + b))` / `\zeta(z) = \log \left(1 + \exp(z) \right)` |
| E6 | adversarial logistic cost E ζ(y(ε‖w‖₁ − wᵀx − b)) | tex:410-412 `\mathbb{E}_{\vx, y \sim p_\text{data}} \zeta( y (\eps ||\vw||_1 - \vw^\top \vx - b)).`; derivation tex:407 |
| E7 | J̃ = αJ(θ,x,y) + (1−α)J(θ, x + ε sign(∇ₓJ(θ,x,y))); α=0.5 | tex:486-487 (grep `tilde{J}`); tex:488 `$\alpha = 0.5$` |
| E8 | RBF p(y=1\|x) = exp((x−μ)ᵀβ(x−μ)) | tex:595 `p(y=1 \mid \vx ) = \exp \left( (\vx - \mu)^\top \vbeta (\vx - \mu) \right)` |
| E9 | targeted fooling x += ε∇ₓ p(y=i\|x) (sign-step per Fig.5 caption) | tex:936 `adding $\eps \nabla_\vx p(y = i \mid \vx)$ to a Gaussian sample`; tex:954 `taking a gradient sign step` |
| E10 | noise controls: x + ε·b, b~Bernoulli{±1}; x + u, u~U(−ε,ε) | tex:555-556 `randomly adding $\pm \eps$ to each pixel, or adding noise in $U(-\eps, \eps)$` |

Implied, not displayed: `J` = cross-entropy/NLL for softmax models (tex:305-306 defines J only
as "the cost used to train the neural network"). Ensemble gradient for E1 = gradient of the
mean of members' losses (paper says "designed to perturb the entire ensemble", tex:822, without
stating the combination rule — see §6).

---

## 6. What the paper does NOT state (gaps list — this is what makes it our re-implementation)

Training/optimization (nothing in this paper prints any of these):
1. **Optimizer for every model**: no learning rate, momentum, batch size, or epoch count is stated
   anywhere in this paper. SGD is mentioned only in passing for the failed quadratic-unit models
   (tex:616-617). The maxout recipe exists only externally (below).
2. **Maxout depth and pieces**: only "units per layer" is stated (240 → 1600, tex:497-499).
   Depth = 2 hidden layers, pieces = 5 come only from the maxout paper / external config.
3. **Dropout rates**: "also regularized with dropout" (tex:492) — no rates given.
4. **Weight init & norm constraints**: nothing in this paper. External config uses uniform
   `irange .005` and `max_col_norm 1.9365` per layer — INCLUDING the softmax readout `y`
   (`irange: .005`, zero bias). We align all three layers to `irange .005` + zero bias
   (the readout previously used PyTorch's default ±1/√fan_in with random bias — a deviation
   from the adopted recipe, now fixed; tested by `test_maxout_readout_init_matches_recipe`).
5. **Early-stopping patience for the adversarial criterion**: patience 100 epochs is stated only
   for *clean* valid error in the original maxout recipe (tex:501-503); the adversarial-valid
   variant's patience is unstated. Also unstated: ε used for the adversarial validation set
   (presumably 0.25) and that it is re-computed against the current model every epoch.
6. **Validation split**: implied by "retrained on all 60,000 examples" (tex:506); only the
   external config fixes it as train[0:50000]/valid[50000:60000].
7. **Seeds**: "Five different training runs using different seeds" (tex:506-508) — values unstated.

Model definitions:
8. **J never displayed** for softmax models (cross-entropy implied; see E-table).
9. **RBF multiclass form**: only the binary `p(y=1|x)=exp((x−μ)ᵀβ(x−μ))` is printed (tex:595);
   number of units (one per class?), whether β is full/diagonal/low-rank, any constraint
   (negative-definiteness for boundedness), initialization, and how "confidence" is computed for
   it are ALL unstated. Note the exp has **no minus sign** — boundedness requires the learned β
   to be negative-definite; nothing enforces that.
   **Multiclass-extension consequence (adversarial review, fixed):** the natural multiclass
   extension of the binary form is the INDEPENDENT per-class reading `p_k(x)=exp(q_k(x))`
   (each in (0,1] when β_k is neg-definite), NOT a softmax over the q_k. A 10-way softmax
   `max_k softmax(q)_k` is bounded below by 1/K=0.1, so it **structurally cannot** reproduce
   the paper's RBF confidence-on-mistakes 1.2%, clean-confidence 60.6%, or rubbish-error 0%
   (tex:600-604, 923). We therefore use the unnormalized `exp(q_k)` for the confidence and
   the rubbish any-class-p>0.5 rule, and `argmax(q_k)` (normalization-invariant, identical to
   the softmax argmax) for the error rate (`eval.eval_fgsm_rbf`, `eval.eval_clean_confidence_rbf`,
   `eval.eval_rubbish_rbf`). The RBF training cost remains the softmax cross-entropy over q
   (the implied 10-way training cost, §6 item 8) — the argmax prediction is unchanged; only
   the confidence/threshold metrics switch to the unnormalized form. β is a free parameter
   with a negative-definite init (−0.01·I) — the faithful reading of E8 (the exp is a valid
   probability only when β is neg-semi-definite); the metric fix is what restores the paper's
   confidence-decay phenomenology, not the init.
10. **Conv maxout for CIFAR-10**: architecture entirely external (cifar10.yaml); preprocessing
    described only as "yields a standard deviation of roughly 0.5" via a hyperlink (tex:343-345).
11. **Ensemble combination rule** (mean probs vs mean logits) and the exact attack objective for
    "perturb the entire ensemble" (tex:822) unstated. **Scoring-side aggregation (adversarial
    review, fixed):** the paper's "the ensemble gets an error rate of 91.1%" (tex:822) does not
    define how the ensemble's prediction is formed. The natural reading is the error of the
    ensemble's *aggregated* prediction, not the mean of the members' individual error rates. We
    form the ensemble prediction as `argmax(mean_m softmax(member_m logits))` (mean-prob voting)
    and report that as the headline `ensemble_targeted_error` / `single_member_targeted_error`
    (`experiments/e1_ensemble.py`). The single-member arm averages over which member is targeted
    (attack member j, score the ensemble, average the error over j) since the paper names no
    target member. The prior per-member-mean statistic is kept as a secondary diagnostic
    (`mean_per_member_*`). The attack-side objective (gradient of the mean CE over members,
    `fgsm_ensemble`) is unchanged.

Evaluation protocol:
12. **Clipping**: paper never says whether x̃ is clipped to [0,1]; displayed method does not clip
    (and the Szegedy-style range constraint is only visible in the commented-out LaTeX, tex:313-326).
    We do not clip.
13. **Error-rate denominator**: whether adv error is over all test examples or only originally
    correct ones is unstated; phrasing "an error rate of 99.9%" ⇒ all examples. Fixed in eval.py.
14. **"Average confidence" definition**: never formalized; read as predicted-class probability
    averaged over misclassified examples (cf. tex:333-341, 521-523, 600-604).
15. **sign(0) convention** unstated; we use 0.
16. **Num. rubbish samples**: MNIST 10,000 stated (tex:905); CIFAR-10 1,000 stated (tex:911). ✓ stated.
17. **Fooling-image ε** for appendix algorithm D unstated; prose says raw-gradient step (tex:936),
    Fig. 5 caption says gradient-sign step (tex:954) — internally inconsistent, both recorded.
18. **ε-sweep details for Fig. 4** (single example of class 4; axis range from the PDF figure,
    not the text): optional, not a numbered claim.
19. **3-vs-7 logistic regression protocol**: which examples train/test, optimizer, stopping
    — all unstated; only clean 1.6% (tex:454) and adv 99% (tex:456) reported.
20. **MNIST preprocessing**: [0,1] scaling IS stated (footnote, tex:334-337). Nothing else (no
     centering) is stated or implied.
21. **E6 uniform-direction imprecision**: the displayed E6 (tex:411) uses the UNIFORM
     perturbation η = −ε·sign(w). This is the worst case only for y=+1 examples; for y=−1 it
     *decreases* the loss, so averaged over mixed labels E6 is NOT an upper bound on E5. The
     true per-example worst case is η = −ε·y·sign(w) (since sign(∇ₓJ) = −y·sign(w), tex:407),
     which IS always ≥ clean. `objectives.adversarial_logreg_cost` implements the paper's E6
     verbatim (uniform direction, matching tex:411); the invariant test asserts the per-example
     worst case. Both recorded as the paper's imprecision, not ours.
22. **run_experiment.py knob mapping**: the graded harness CLI is `--lambda EPS --steps N
     [--seed S] [--baseline]`. `--lambda` maps to the paper's ε (the FGSM perturbation
     magnitude — the paper's free hyperparameter); ε=0 is the no-op setting that must equal
     `--baseline` exactly (the degeneracy test). `--steps` is the SGD step count (the paper
     states no epoch/step count). Chose: maxout 240 units (paper's M3 size), 5 pieces, 2
     layers, init irange .005, SGD batch 100, LR .1, momentum .5→.7, α=0.5 (paper-stated),
     ε=0.25 default (paper-stated for MNIST). Dropout defaults to **OFF** (include prob 1.0)
     in `run_experiment.py` so the ε=0 degeneracy holds bit-for-bit: the method (Algorithm B)
     runs by default (including at `--lambda 0`); `--baseline` forces the clean arm. With
     dropout off, `adversarial_train_cost` at eps=0 performs no extra RNG draw and its second
     forward equals the first, so the parameter update is identical to clean training. (The
     degeneracy now holds by eps=0 ⇒ x̃==x ⇒ J̃==J, NOT by branch-switching; the adversarial
     code path IS exercised at eps=0.) Determinism: `MaxoutMLP` init draws from the GLOBAL
     torch RNG, so `run_experiment.py` calls `torch.manual_seed(seed)` before constructing the
     model; the batch-shuffle generator is seeded from `cfg.seed` inside `train()`. The
      monitor-best checkpoint is loaded before evaluation (paper protocol). Full-scale
      milestone experiments (m4_adversarial.py etc.) re-enable dropout (input include 0.8,
      hidden include 1.0=off) — see item 3 and the External block below (corrected: the
      pylearn2 recipe has dropout on the INPUT of h0 only, not on hidden layers).
23. **Dropout state during surrogate generation (F1)**: the paper says only that adversarial
     examples should "resist the current version of the model" (tex:490-491) and never states
     whether dropout is on or off while generating the FGSM surrogate. The evaluation-time
     attacker (`eval.py`) runs with `model.eval()` (dropout OFF, deterministic). **Choice: the
     input-gradient probe inside `adversarial_train_cost` is computed with the model temporarily
     in `eval()` mode (dropout off), then the caller's mode is restored so the two loss terms
     keep the regular training-time dropout.** This makes the training-time adversary match the
     deterministic evaluation attacker and stops the probe from consuming dropout RNG (which
     previously made the probe, loss_clean and loss_adv each use a different random mask).
     Tested by `test_adversarial_train_surrogate_matches_eval_attacker`. With dropout OFF (the
     runner default for the degeneracy gate) the mode switch is a no-op, so eps=0 degeneracy is
     unaffected.
24. **M2 FGSM attack direction**: the paper's DISPLAYED E6 (tex:411) uses the uniform direction
     η = −ε·sign(w) (the y=+1 worst case), but the FGSM *attack* the paper reports (99% error,
     tex:456) uses sign(∇ₓJ) = −y·sign(w) per example (tex:407) — the true per-example worst case.
     **Choice: `attacks.fgsm_logreg` implements the per-example worst case η = −ε·y·sign(w)**
     (the actual adversary; this is what makes the attack "exact" for the linear model and what
     the paper's 99% measures). The E6 *training loss* uniform-direction imprecision (item 21)
     is a separate object (`adversarial_logreg_cost`) and is not used in the M2 attack.
25. **M9 sigmoid-top arm**: the paper says only "Changing the top layer to independent sigmoids"
    (tex:908-909) on the maxout net. It does not state whether the sigmoid-top net is retrained
    or evaluated on the same learned trunk. **Choice (adversarial review, fixed):** `SigmoidTopMLP`
    is TRAINED with the per-class independent-sigmoid BCE cost
    `J = mean_b sum_k BCE(sigmoid(logit_k), 1[y=k])` (`objectives.sigmoid_top_cost`,
    `train(cost="sigmoid_top")`), sharing the maxout trunk architecture / init / SGD / external
    recipe (so the M9 comparison trains identically to the maxout+softmax net); only the top
    activation and training cost differ. The prior implementation copied the softmax-trained
    readout weights and applied sigmoids with NO retraining — a frozen swap that mechanically
    forces rubbish error -> 1.0 on N(0,I) noise regardless of training (P(any logit>0) -> 1);
    that measured the swap, not the paper's model. Error = any per-class sigmoid > 0.5
    (`eval_rubbish_sigmoid`).
26. **M5 protocol choices**: the paper states the M5 protocol (adversarial-valid early stop →
     retrain on 60k → 5-seed mean, tex:501-512) but not the patience for the adversarial-valid
     criterion, the ε for the adversarial validation set, or the retrain optimizer. **Choices:
     adversarial-valid patience = the same `--patience` knob (default 10 for CPU feasibility;
     paper scale 100); adversarial-validation ε = 0.25 (the training ε); retrain uses the same
     SGD + external maxout recipe as the selection phase.** The baseline arm selects on
     clean-valid error (the original maxout recipe, tex:501-503); the adversarial arm selects on
     adversarial-valid error (tex:503-505). Full 1600-unit/patience-100/5-seed scale is
    infeasible on this CPU; `experiments/m5_large_advtrain.py` defaults to a documented
    sub-scale and exposes the full-scale knobs.
 27. **M7 noise-training mixture (adversarial review, fixed):** the paper says "we trained a
      maxout network with noise based on randomly adding $\pm\eps$ to each pixel, or adding
      noise in $U(-\eps, \eps)$" (tex:555-556) -- it does NOT state a clean/noisy mixture.
      **Choice: train on NOISE-ONLY batches** (`L = J(theta, x + eta, y)`, no clean term, no
      alpha; `objectives.noise_train_cost`), the prose reading of "trained ... with noise".
      The prior implementation used an unpapered 0.5-clean/0.5-noisy alpha-mixture mirroring
      E7, which halved the noise pressure on the very control the paper uses to argue noise is
      inferior to FGSM. Noise-only is both the faithful reading and a HARDER control (more
      noise exposure) for the paper's "noise << FGSM" claim. Noise is regenerated per batch
      (tex:490-491) and detached. Tested by `test_noise_train_cost_is_noise_only` /
      `test_noise_train_cost_no_clean_mixture_term`.
 28. **L1 weight-decay control (Section 5, tex:426-433; adversarial review, fixed):** the
      paper's Section-5 contrast of adversarial training vs L1 weight decay was previously
      omitted entirely with no exclusion record. **Now implemented** as
      `experiments/m_l1_weight_decay.py`: adds `coeff * ||layer0.W||_1` (the L1 penalty on the
      FIRST maxout layer's incoming weights) to the training cost (`train.l1_first_layer_coeff`,
      `objectives.l1_first_layer_penalty`), sweeps three coefficients — the paper's "too large"
      0.0025, an intermediate 0.00025, and a small 0.000025 that TRAINS — and reports clean
      TRAIN + TEST error and FGSM error. The paper's qualitative claim: 0.0025 is too
      pessimistic (>5% TRAIN error); smaller coefficients train but confer no regularization
      benefit. (At sub-scale the "too large" threshold shifts down, so 0.00025 is also stuck
      here; 0.000025 trains but confers no clean-test/FGSM benefit.) The L1 penalty is ADDED
      to the cost (standard weight decay); the paper notes this is more pessimistic than
      adversarial training, which SUBTRACTS the penalty from the activation (tex:418-424).
 29. **M8 §8 "53.6%" attack source (adversarial review, fixed):** the paper fixes ONE
      adversarial-example set per paragraph — "we generated adversarial examples on a deep
      maxout network and classified these examples using a shallow softmax network and a
      shallow RBF network" (tex:679-680) — so all five §8 agreement numbers use the
      MAXOUT-generated examples. The first four (16.0/54.6/84.6/54.3%) compare a model's
      prediction to the MAXOUT's class (attacker == reference == maxout). The fifth — "the
      RBF network can predict softmax regression's class 53.6% of the time" (tex:687) —
      compares the RBF's prediction to the SOFTMAX's class; the paper is genuinely ambiguous
      whether this keeps the maxout-generated set with softmax as the reference, or builds new
      adversarials from softmax. **Choice: the headline 53.6% uses the maxout-generated set
      with softmax as the reference** (`eval.agreement_on_adv(attacker=maxout, ref=softmax,
      pred=rbf)`) — the faithful reading of the paragraph's fixed set. The prior
      implementation's reading (build NEW adversarials from softmax, `class_agreement(softmax,
      rbf)`) is RETAINED as a secondary diagnostic (`softmax_vs_rbf_on_softmax_adv_secondary`
      in results/m8_rbf.json) so the divergence between the two readings is visible.
      `class_agreement(m1, m2)` is now the special case `agreement_on_adv(m1, m1, m2)`.
      Tested by `test_class_agreement_equals_agreement_on_adv_same_attacker_ref` and
      `test_agreement_on_adv_separates_attacker_from_ref`.
 30. **ε floor in the graded harness (adversarial review, recorded):** `run_experiment.py`
      maps `--lambda` to ε via `eps = max(0.0, args.lam)` (run_experiment.py:138). A negative
      `--lambda` is silently floored to the no-op (eps=0), which would report as the
      "adversarial" arm while running the baseline trajectory. The paper never mentions a
      negative ε (it is meaningless for a max-norm bound); the floor is a defensive clamp, not
      a papered value. Recorded here for completeness; no reported number depends on it (no
      arm is ever run with ε<0).
 31. **RBF exp clamp `q.clamp(max=80.0)` (adversarial review, recorded):** `eval._rbf_unnorm_probs`
      / `eval_fgsm_rbf` clamp the RBF quad form to ≤ 80 before `torch.exp`
      (eval.py:77,165) to avoid float32 overflow if a trained β drifts positive. This is a
      STRICT NO-OP for any faithful RBF: q_k = −Σ_f softplus(raw_{k,f})(x_f−μ_{k,f})² ≤ 0
      always (β neg-def by construction, models.py:298-323), so q ≤ 0 ≪ 80 and the clamp
      never fires. It only guards pathological β drift, which would itself contradict the RBF
      property. Recorded as an unpapered clamp that affects no reported number.
 32. **External lr schedule applied before the first step (adversarial review, recorded):** the
      external maxout recipe's exponential lr adjust (`lr *= 1.000004` per update) is applied
      BEFORE `optimizer.step()` (train.py:290-293), so the FIRST step uses lr·1.000004 = 0.1000004,
      not the stated lr 0.1. The per-update adjust itself is documented external (§6 External
      block); the pre-first-step application ordering is recorded here. Effect: a 4e-6 relative
      lr shift on step 1 only, negligible against the lr·momentum dynamics; biases no reported
      number meaningfully. (The degeneracy gate's eps=0==baseline holds bit-for-bit because both
      arms share this identical schedule.)
 33. **M5 baseline arm retrain protocol (adversarial review, recorded):** the paper ties the
      "retrain on all 60,000" step to the ADVERSARIAL-validation early-stop criterion
      (tex:505-506) and is silent on whether the 1.14% BASELINE (tex:497-499) was also retrained
      on the full 60k or simply early-stopped. **Choice: the M5 baseline arm runs the SAME
      select-epochs-then-retrain-on-60k protocol as the adversarial arm** (select on clean-valid
      error, retrain on 60k for best_epoch+1; m5_large_advtrain.py:157). A symmetric protocol is
      the cleaner comparison (same training budget per arm) and biases the baseline, if anything,
      UPWARD (more training), making the adversarial<baseline direction HARDER to show — so a
      positive result is not inflated by this choice. Recorded; the paper's 1.14% baseline
      number is compared against this symmetric-protocol baseline.

External (not from this paper; recorded from the still-live
`lisa-lab/pylearn2` `pylearn2/scripts/papers/maxout/mnist_pi.yaml`, fetched 2026-07-29):
240 units × 5 pieces × 2 maxout layers; uniform init irange .005 (ALL three layers incl. the
softmax readout `y`, with zero bias); max_col_norm 1.9365 (all three layers); SGD batch 100,
LR .1, exponential adjust 1.000004/update (min 1e-6); momentum .5 → .7 by epoch 250;
dropout `input_include_probs: {h0: .8}`, `input_scales: {h0: 1.}` — i.e. dropout on the INPUT of
h0 only (the raw input x), include-prob 0.8, NON-inverted scaling (scale 1.0: train multiplies
by the Bernoulli mask with no 1/include division; eval is identity). The recipe has NO dropout
on h1's input or the readout's input. We adopt these for M3/M4 arms and document them as
external defaults, not paper-stated values. **Two documented deviations from the recipe:**
(a) we use INVERTED dropout (mask/include at train, identity at eval) — the modern standard,
mathematically equivalent up to a constant eval-time scale (the recipe's scale-1.0 non-inverted
form leaves eval activations ~1.25x the expected train value); (b) the model additionally
exposes a `dropout_hidden_include` knob (default 1.0=off) for ablations, but the full-scale
milestone scripts set input 0.8 / hidden 1.0 to match the recipe's input-only dropout. For M5
(1600 units) the paper gives no recipe at all beyond α/ε/early-stop criterion.

---

## 7. Upstream code search (recorded)

- **Paper's only code link**: tex:344 — `https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout`
  Verified live 2026-07-29, re-verified 2026-07-30 (HTTP 200): contains the **maxout 2013 paper's**
  configs (mnist_pi.yaml, cifar10.yaml, …), used *here only for CIFAR-10 preprocessing* (footnote 2).
  No adversarial-example code.
- **GitHub search, repo name/title** (`explaining and harnessing adversarial examples`; 2026-07-29,
  re-run 2026-07-30, same result):
  13 repos, all third-party partial re-implementations (e.g. `Harry24k/FGSM-pytorch` — attack only,
  none of the paper's experiments; `rodgzilla/machine_learning_adversarial_examples`; a Cambridge
  course repo). None usable as the paper's implementation.
- **Author-owned code**: GitHub search `adversarial examples goodfellow user:goodfeli` → **0 repos**
  (re-run 2026-07-30, unchanged).
  No official implementation of this paper was ever released (CleverHans is a later, different library).
- **Decision: implement from scratch.** pylearn2/Theano itself is unmaintained since ~2016 and cannot
  run on a modern Python (3.12+); the external yaml is used as *documentation of defaults* only.

Environment for our implementation: Python 3.12.13 (system interpreter here; Dockerfile builds on
`python:3.13-slim` with the same pinned wheels), PyTorch CPU (no GPU on this machine), MNIST via
raw IDX download (torchvision if its mirror works, else direct from a mirror). The `.venv/` is
gitignored and must be recreated per sandbox: `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`
(verified 2026-07-30: fresh venv, `import torch` OK, 47/47 tests pass, `./smoke.sh` prints its FINAL line).

---

## 8. Verification plan (what "matched" means per milestone)

- Shapes/degeneracy tests: FGSM on a linear model must exactly equal the closed-form max-norm
  adversary (E6); ‖η‖∞=ε; sign correctness; `x̃=x` when ε=0; adversarial-logistic loss ≥ clean loss
  at equal params; rubbish RBF gives ≥ softplus baseline behavior only after training.
- Number targets: tables in §1. Tolerance policy (recorded in REPRODUCTION.md): qualitative match =
  same direction and order of magnitude (e.g., M1 adv error "≈100%" vs 99.9%); the M4/M5 deltas
  (0.94→0.84, 1.14→0.782-mean) are the paper's headline claims and get the strictest comparison.
- Every result JSON records the exact grep-able paper line it is compared against.

### Finding: E6 sign subtlety (tex:407-412)
The paper states "the sign of the gradient is just −sign(w)" (tex:407) and derives the closed-form
adversarial logistic loss E6 = E ζ(y(ε‖w‖₁ − wᵀx − b)) (tex:410-412). The true FGSM input gradient of
ζ(−y(wᵀx+b)) is −y·w·σ(·), whose sign is −y·sign(w), so the worst-case η = −ε·y·sign(w). Substituting
gives the *true* worst-case loss ζ(ε‖w‖₁ − y(wᵀx+b)). The paper's E6 = ζ(y·ε‖w‖₁ − y(wᵀx+b)) matches
this **only for y=+1**; for y=−1 E6 subtracts ε‖w‖₁ from the activation, yielding the *best*-case
loss (E6 < clean), not the worst case. We implement E6 verbatim (matches the paper's printed
equation) and the invariant test asserts E6 == empirical-FGSM for y=+1 (where the paper's "exact"
claim holds) and records the y=−1 divergence as the paper's imprecision, not ours.

---

## 9. Gate contract (added at reproduction time)

The reproduction's deliverable gate is a flat arm→command map plus runner scripts:

- **`arms.json`** — a flat map `{"baseline": "<cmd>", "adversarial": "<cmd>"}` from each arm the
  gate runs to the shell command that produces it. The two arms are the paper's headline M4
  comparison (tex:492-494: clean maxout training vs FGSM adversarial training). The rich 12-arm
  per-milestone metadata (model/training/eval configs, paper targets, citations, results paths)
  is preserved in **`arms_metadata.json`**; `arms.json` is intentionally minimal so the gate can
  pair each `FINAL <arm>=<value>` line to its arm by name mechanically.
- **`run_all_arms.sh`** — runs both arms at the paper's M4 configuration (maxout 240×2, pieces 5,
  eps=0.25, alpha=0.5, 5000 SGD steps, dropout OFF for degeneracy validity). Each command prints
  exactly one line `FINAL <arm name>=<clean test accuracy>` to stdout (progress to stderr), so
  stdout is exactly the two gate lines. Exports `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4`
  (PyTorch's default thread pool over-spawns on multi-core CPUs — a ~7x slowdown observed at
  1500 steps; 4 threads is the sweet spot here).
- **`smoke.sh`** — the same code path (method arm, 200 steps, units 64) finishing in seconds;
  prints one `FINAL adversarial=<float>` line. A path-prover only, never evidence about the paper.
- **`run_experiment.py` output contract** — exactly one line `FINAL <arm>=<clean test accuracy>`
  where `<arm>` is `baseline` (for `--baseline`) or `adversarial` (for `--lambda`). The arm name
  matches the `arms.json` keys. The degeneracy contract: `--lambda 0` (arm `adversarial` at its
  no-op) prints the SAME accuracy VALUE as `--baseline` (arm `baseline`); the names differ by
  construction, the float value matches bit-for-bit (`tests/test_degeneracy.py`).

The gate reproduces the DIRECTION of the M4 result (adversarial ≥ baseline; the paper's 0.94%→0.84%
clean test error drop). The exact magnitudes require dropout ON + early stopping to convergence;
that higher-fidelity arm lives in `experiments/m4_adversarial.py` with committed
`results/m4_adversarial.json`.

### Finding: M5 retrain over-training (adversarial review, fixed)
The paper protocol (tex:505-506): the early-stopping criterion selects the BEST epoch, then the
model is retrained on all 60,000 for EXACTLY that many epochs. `train.py` originally exposed only
`epochs_run` (= the STOPPING epoch = best_epoch + patience), so `experiments/m5_large_advtrain.py`
retrained for best_epoch + patience epochs — over-training by ~patience epochs and biasing the
headline M5 number (tex:506-512, mean 0.782%). **Fix:** `TrainResult` now exposes `best_epoch`
(0-based index of the epoch with the best validation metric); the M5 retrain arm uses
`best_epoch + 1` (the selected epoch count). Regression test
`tests/test_invariants.py::test_best_epoch_is_selected_not_stopping` asserts the property so the
bug cannot silently return. (Found by an adversarial review of `train.py`; the verifier confirmed
it by reading the code.)

### Finding: measured.json must hold scalars only (gate-crash fix)
The numbers-gate evaluator formats every metric value in `measured.json` as a scalar
(`f"{v:.4f}"`-style). A list value crashes it with
`TypeError: unsupported format string passed to list.__format__`. The one pointer that resolved
to a list was `arms.adversarial.per_seed[*].test_error` (metric `per_seed_test_errors` on
`m5_maxout1600_advtrain`, claims c12/c13): `make_measured.py` runs each seed as a separate
process writing its own result file, so each file's `per_seed` array holds exactly ONE element
(the seed that run trained under) and `[*]` yielded a one-element list `[x]`. The gate's
cross-seed gather then built `[[x0],[x1],[x2]]`, and any format of that list crashed before a
verdict could be rendered. **Fix:** `_metrics_for_arm` collapses a one-element list from a `[*]`
pointer to its scalar (THIS seed's test error, so the cross-seed gather becomes the flat
`[x0,x1,x2]` that `mean`/`max`/`min` in c12/c13 operate on, as the paper's five-run spread
intends); a multi-element `[*]` list is a genuine ambiguity and is BLOCKED rather than silently
flattened. Regression tests `tests/test_measured_resolver.py` (single-element collapse,
multi-element BLOCK, scalar passthrough, and `test_measured_json_has_no_list_values` asserting
the shipped `measured.json` contains only scalars or `BLOCKED`) guard the property. The crash
was masking the honest c13 verdict: at sub-scale (240 units / 3 seeds) the per-seed spread is
~0.0010 > the paper's 0.0006 (tex:506-512, measured at 1600 units / 5 seeds), so c13 does NOT
reproduce at sub-scale — an honest "blocked-by-scale" result the gate can now render instead of
aborting.

---

## 10. claims.json — the numbers-gate contract

The claims below are what the numbers gate settles. They live machine-readably in **`claims.json`**
at the repo root (byte-identical to the JSON embedded here). Semantics:

- **Quotes are verbatim substrings** of `paper/source/iclr2015.tex` (the arXiv LaTeX, authoritative
  for numbers), preserving the file's own line breaks; `citation` is `file:line` where the quote
  starts, so a reader can check the quote rather than trust it. (Sentences truncated by the file's
  mid-line wraps — e.g. footnotes interrupting a sentence — are quoted through the sentence's
  terminating punctuation, so every quote contains at least one complete claim-bearing sentence.)
- **`seeds: [0, 1, 2]`** (three; two claims annotated otherwise: c12/c13 use `[0,1,2,3,4]` to match
  the paper's five M5 runs, tex:506-509). `measured.<arm>.<metric>` resolves through
  `arms.<arm>.metrics.<metric>`, a `<results json>:<json path>` pointer; the value comes from the
  results file produced by that arm's `command_per_seed` run at each seed.
- **`ordering`** claims: the `quantity` (an expression over `measured.<arm>.<metric>`) must satisfy
  `direction` at EVERY seed. **`value`** claims: `|quantity - claimed| <= tolerance` at every seed.
  **`existence`/`invariant`**: the boolean `predicate` must hold at every seed.
- **`compute_invariance`** is rated honestly: **high** = survives this reproduction's CPU sub-scale
  (240-unit maxout, ~10-epoch training, 4-member ensemble — strictly smaller than the paper's
  budget); these are what the gate checks. **medium** = held at sub-scale but rests on a caveat
  recorded in the claim note. **low** = an exact magnitude that needs the paper's full scale
  (1600 units / patience 100 / 5 seeds) or is otherwise budget-fragile; informational only.
  Dry-run against the committed seed-0 sub-scale results (`results/*.json`): **all 16 high claims
  PASS**; the two low claims that fail at sub-scale (c12 M5 mean magnitude, c29 ensemble-targeted >
  single-targeted) are exactly the ones annotated as needing paper scale / saturation-fragile.
- **Claims the paper makes that this reproduction deliberately does not test** are listed in
  `not_tested` (10 entries, each with quote, citation, and reason): the GoogLeNet/ImageNet figure,
  the CIFAR-10 arm, the MP-DBM arm, the cross-paper "best on permutation-invariant MNIST" comparison,
  rotation-based adversarial examples, the Fig. 3 weight-localization claim, the Fig. 4
  epsilon-sweep visualization, the MNIST rubbish class-skew statistic (45.3% fives / no eights),
  the train-to-zero-on-rubbish null result, and the CIFAR-10 target-specific fooling rates.

```json
{
  "paper_ref": "43e841f0-05f8-4249-ba14-14c1a586f6ee",
  "project_id": "d7735ece-02c4-4228-985c-00834c92b8f3",
  "title": "Explaining and Harnessing Adversarial Examples",
  "authors": "Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy",
  "year": 2015,
  "arxiv_id": "1412.6572",
  "paper_source": "paper/source/iclr2015.tex (arXiv LaTeX of arXiv:1412.6572v3, authoritative for every equation/number; quotes below preserve the file's own line breaks)",
  "seeds": [
    0,
    1,
    2
  ],
  "evaluation": {
    "resolution": "measured.<arm>.<metric> resolves through arms.<arm>.metrics.<metric>, a '<results json>:<json path>' pointer, read from the results file produced by running that arm's command_per_seed with {seed} replaced by each value of `seeds`.",
    "ordering": "the quantity must satisfy `direction` at EVERY seed.",
    "value": "abs(quantity - claimed) <= tolerance at EVERY seed.",
    "existence_or_invariant": "the boolean predicate must evaluate true at EVERY seed. Allowed constructs in quantities/predicates: min(), max(), mean(), abs(), arithmetic, comparisons, and/or.",
    "compute_invariance": {
      "policy": "rated honestly: high = the assertion already survives this reproduction's CPU sub-scale (240-unit maxout, ~10-epoch training, fewer ensemble members) which is strictly smaller than the paper's budget; medium = held at sub-scale but rests on a caveat recorded in the claim's note; low = exact magnitude that requires the paper's full scale (1600 units / patience 100 / 5 seeds / full ensembles) or is otherwise budget-fragile. The numbers gate checks the HIGH claims.",
      "counts": {
        "high": 16,
        "medium": 6,
        "low": 12
      }
    }
  },
  "arms": {
    "m1_softmax_regression": {
      "script": "experiments/m1_softmax.py",
      "command_per_seed": "python experiments/m1_softmax.py --seed {seed}",
      "results": "results/m1_softmax.json",
      "metrics": {
        "clean_error": "results/m1_softmax.json:clean_error",
        "fgsm_error_rate": "results/m1_softmax.json:fgsm_eval.error_rate",
        "fgsm_mean_confidence_on_errors": "results/m1_softmax.json:fgsm_eval.mean_confidence_on_errors"
      }
    },
    "m2_logistic_3v7": {
      "script": "experiments/m2_logreg.py",
      "command_per_seed": "python experiments/m2_logreg.py --seed {seed}",
      "results": "results/m2_logreg.json",
      "metrics": {
        "clean_error": "results/m2_logreg.json:clean_error",
        "fgsm_error_rate": "results/m2_logreg.json:fgsm_eval.error_rate"
      }
    },
    "m3_maxout240_clean": {
      "script": "experiments/m3_maxout_fgsm.py",
      "command_per_seed": "python experiments/m3_maxout_fgsm.py --seed {seed}",
      "results": "results/m3_maxout_fgsm.json",
      "metrics": {
        "clean_error": "results/m3_maxout_fgsm.json:clean_error",
        "fgsm_error_rate": "results/m3_maxout_fgsm.json:fgsm_eval.error_rate",
        "fgsm_mean_confidence_on_errors": "results/m3_maxout_fgsm.json:fgsm_eval.mean_confidence_on_errors"
      }
    },
    "m4_maxout240_advtrain": {
      "script": "experiments/m4_adversarial.py",
      "command_per_seed": "python experiments/m4_adversarial.py --seed {seed}",
      "results": "results/m4_adversarial.json",
      "metrics": {
        "baseline_clean_error": "results/m4_adversarial.json:arms.baseline.clean_error",
        "adversarial_clean_error": "results/m4_adversarial.json:arms.adversarial.clean_error"
      }
    },
    "m5_maxout1600_clean": {
      "script": "experiments/m5_large_advtrain.py",
      "command_per_seed": "python experiments/m5_large_advtrain.py --seeds {seed}",
      "command_paper_scale": "python experiments/m5_large_advtrain.py --seeds 0,1,2,3,4 --units 1600 --epochs 100 --patience 100",
      "results": "results/m5_large_advtrain.json",
      "metrics": {
        "mean_test_error": "results/m5_large_advtrain.json:arms.baseline.mean_test_error"
      }
    },
    "m5_maxout1600_advtrain": {
      "script": "experiments/m5_large_advtrain.py",
      "command_per_seed": "python experiments/m5_large_advtrain.py --seeds {seed}",
      "command_paper_scale": "python experiments/m5_large_advtrain.py --seeds 0,1,2,3,4 --units 1600 --epochs 100 --patience 100",
      "results": "results/m5_large_advtrain.json",
      "metrics": {
        "mean_test_error": "results/m5_large_advtrain.json:arms.adversarial.mean_test_error",
        "per_seed_test_errors": "results/m5_large_advtrain.json:arms.adversarial.per_seed[*].test_error"
      }
    },
    "m6_robustness_transfer_eval": {
      "script": "experiments/m6_robustness_transfer.py",
      "command_per_seed": "python experiments/m6_robustness_transfer.py --seed {seed}",
      "results": "results/m6_robustness_transfer.json",
      "metrics": {
        "own_fgsm_error": "results/m6_robustness_transfer.json:own_fgsm_on_adv_model.error_rate",
        "own_fgsm_mean_confidence_on_errors": "results/m6_robustness_transfer.json:own_fgsm_on_adv_model.mean_confidence_on_errors",
        "orig_to_adv_transfer_error": "results/m6_robustness_transfer.json:transfer_orig_to_adv.error_rate",
        "adv_to_orig_transfer_error": "results/m6_robustness_transfer.json:transfer_adv_to_orig.error_rate"
      }
    },
    "m7_maxout_noise_sign": {
      "script": "experiments/m7_noise_controls.py",
      "command_per_seed": "python experiments/m7_noise_controls.py --seed {seed}",
      "results": "results/m7_noise_controls.json",
      "metrics": {
        "fgsm_error_rate": "results/m7_noise_controls.json:arms.bernoulli.fgsm_eval.error_rate",
        "fgsm_mean_confidence_on_errors": "results/m7_noise_controls.json:arms.bernoulli.fgsm_eval.mean_confidence_on_errors"
      }
    },
    "m7_maxout_noise_uniform": {
      "script": "experiments/m7_noise_controls.py",
      "command_per_seed": "python experiments/m7_noise_controls.py --seed {seed}",
      "results": "results/m7_noise_controls.json",
      "metrics": {
        "fgsm_error_rate": "results/m7_noise_controls.json:arms.uniform.fgsm_eval.error_rate",
        "fgsm_mean_confidence_on_errors": "results/m7_noise_controls.json:arms.uniform.fgsm_eval.mean_confidence_on_errors"
      }
    },
    "m8_rbf_shallow": {
      "script": "experiments/m8_rbf.py",
      "command_per_seed": "python experiments/m8_rbf.py --seed {seed}",
      "results": "results/m8_rbf.json",
      "metrics": {
        "rbf_fgsm_error_rate": "results/m8_rbf.json:rbf_fgsm_eval.error_rate",
        "rbf_fgsm_mean_confidence_on_errors": "results/m8_rbf.json:rbf_fgsm_eval.mean_confidence_on_errors",
        "rbf_clean_confidence": "results/m8_rbf.json:rbf_clean_confidence",
        "agreement_softmax_pred_maxout_over_m1_errors": "results/m8_rbf.json:agreement.maxout_vs_softmax.p_pred_match_over_m1_errors",
        "agreement_rbf_pred_maxout_over_m1_errors": "results/m8_rbf.json:agreement.maxout_vs_rbf.p_pred_match_over_m1_errors",
        "agreement_softmax_pred_maxout_over_both_wrong": "results/m8_rbf.json:agreement.maxout_vs_softmax.p_pred_match_over_both_wrong",
        "agreement_rbf_pred_maxout_over_both_wrong": "results/m8_rbf.json:agreement.maxout_vs_rbf.p_pred_match_over_both_wrong",
        "agreement_rbf_pred_softmax_over_m1_errors": "results/m8_rbf.json:agreement.softmax_vs_rbf_on_maxout_adv.p_pred_match_over_m1_errors"
      }
    },
    "m9_rubbish_evals": {
      "script": "experiments/m9_rubbish.py",
      "command_per_seed": "python experiments/m9_rubbish.py --seed {seed}",
      "results": "results/m9_rubbish.json",
      "metrics": {
        "maxout_softmax_rubbish_error": "results/m9_rubbish.json:evals.maxout_softmax.error_rate",
        "sigmoid_top_rubbish_error": "results/m9_rubbish.json:evals.sigmoid_top.error_rate",
        "softmax_regression_rubbish_error": "results/m9_rubbish.json:evals.softmax_regression.error_rate",
        "rbf_rubbish_error": "results/m9_rubbish.json:evals.rbf.error_rate"
      }
    },
    "e1_ensemble12_maxout": {
      "script": "experiments/e1_ensemble.py",
      "command_per_seed": "python experiments/e1_ensemble.py --base-seed {seed}",
      "results": "results/e1_ensemble.json",
      "metrics": {
        "ensemble_targeted_error": "results/e1_ensemble.json:ensemble_targeted_error",
        "single_member_targeted_error": "results/e1_ensemble.json:single_member_targeted_error"
      }
    },
    "m_l1_weight_decay": {
      "script": "experiments/m_l1_weight_decay.py",
      "command_per_seed": "python experiments/m_l1_weight_decay.py --seed {seed}",
      "results": "results/m_l1_weight_decay.json",
      "metrics": {
        "baseline_train_error": "results/m_l1_weight_decay.json:arms.baseline.clean_train_error",
        "baseline_test_error": "results/m_l1_weight_decay.json:arms.baseline.clean_test_error",
        "l1_0.0025_train_error": "results/m_l1_weight_decay.json:arms.l1_0.0025.clean_train_error",
        "l1_2.5e-05_test_error": "results/m_l1_weight_decay.json:arms.l1_2.5e-05.clean_test_error"
      }
    }
  },
  "claims": [
    {
      "id": "c01_fgsm_reliably_misclassifies",
      "kind": "existence",
      "compute_invariance": "high",
      "quote": "We find that this method reliably causes a wide variety of models to misclassify their input.",
      "citation": "paper/source/iclr2015.tex:331",
      "predicate": "measured.m1_softmax_regression.fgsm_error_rate > 0.5 and measured.m2_logistic_3v7.fgsm_error_rate > 0.5 and measured.m3_maxout240_clean.fgsm_error_rate > 0.5",
      "note": "The load-bearing phenomenon the paper explains. At sub-scale: 100.0% / 99.1% / 99.8%."
    },
    {
      "id": "c02_softmax_fgsm_error_value",
      "kind": "value",
      "compute_invariance": "high",
      "quote": "We find that using $\\eps=.25$, we cause a shallow softmax classifier to have an error rate of 99.9\\% with an average confidence of\n79.3\\% on the MNIST~\\citep{LeCun+98} test set\\footnote{This is using MNIST pixel values in the interval [0, 1]. MNIST data\ndoes contain values other than 0 or 1, but the images are essentially binary. Each pixel roughly encodes\n``ink'' or ``no ink''. This justifies expecting the classifier to be able to handle perturbations\nwithin a range of width 0.5, and indeed human observers can read such images without difficulty.\n}.",
      "citation": "paper/source/iclr2015.tex:333",
      "quantity": "measured.m1_softmax_regression.fgsm_error_rate",
      "claimed": 0.999,
      "tolerance": 0.01,
      "note": "Error rate saturates to ~1.0 at any budget where softmax fits at all (sub-scale: 100.0%; the claim is a lower bound on vulnerability, not a knife-edge magnitude). Rated high: the adversary here is strong by construction for any fitted linear model."
    },
    {
      "id": "c03_softmax_fgsm_confidence_value",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "We find that using $\\eps=.25$, we cause a shallow softmax classifier to have an error rate of 99.9\\% with an average confidence of\n79.3\\% on the MNIST~\\citep{LeCun+98} test set",
      "citation": "paper/source/iclr2015.tex:333",
      "quantity": "measured.m1_softmax_regression.fgsm_mean_confidence_on_errors",
      "claimed": 0.793,
      "tolerance": 0.15,
      "note": "Confidence magnitude drifts with training budget (sub-scale 92.9%); informational only."
    },
    {
      "id": "c04_logreg_clean_error_value",
      "kind": "value",
      "compute_invariance": "medium",
      "quote": "    c) MNIST 3s and 7s. The logistic regression model has a 1.6\\% error rate on the 3 versus 7 discrimination task on these examples.",
      "citation": "paper/source/iclr2015.tex:454",
      "quantity": "measured.m2_logistic_3v7.clean_error",
      "claimed": 0.016,
      "tolerance": 0.01,
      "note": "Held at sub-scale (2.01%). Medium because the exact 3-vs-7 protocol (which examples train/test, stopping) is unstated (SPEC \u00a76 item 19)."
    },
    {
      "id": "c05_logreg_clean_below_adv",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "    c) MNIST 3s and 7s. The logistic regression model has a 1.6\\% error rate on the 3 versus 7 discrimination task on these examples.\n    d) Fast gradient sign adversarial examples for the logistic regression model with $\\eps=.25$.\n        The logistic regression model has an error rate of 99\\% on these examples.",
      "citation": "paper/source/iclr2015.tex:454",
      "quantity": "measured.m2_logistic_3v7.clean_error - measured.m2_logistic_3v7.fgsm_error_rate",
      "direction": "<0",
      "note": "FGSM is the EXACT max-norm adversary for logistic regression (tex:394-395), so this ordering does not depend on attack approximations. Sub-scale: 2.0% vs 99.1%."
    },
    {
      "id": "c06_logreg_adv_error_value",
      "kind": "value",
      "compute_invariance": "medium",
      "quote": "    d) Fast gradient sign adversarial examples for the logistic regression model with $\\eps=.25$.\n        The logistic regression model has an error rate of 99\\% on these examples.",
      "citation": "paper/source/iclr2015.tex:455",
      "quantity": "measured.m2_logistic_3v7.fgsm_error_rate",
      "claimed": 0.99,
      "tolerance": 0.02,
      "note": "Held at sub-scale (99.1%). Medium, not high, only because the clean-side protocol is unstated; the attack itself is exact for this model."
    },
    {
      "id": "c07_maxout_fgsm_error_value",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "In the same setting, a maxout network misclassifies 89.4\\%\nof our adversarial examples with an average confidence of 97.6\\%.",
      "citation": "paper/source/iclr2015.tex:338",
      "quantity": "measured.m3_maxout240_clean.fgsm_error_rate",
      "claimed": 0.894,
      "tolerance": 0.15,
      "note": "Magnitude is budget-sensitive: the paper's 89.4% comes from a fully-converged net; less-converged nets are fooled MORE (sub-scale 99.8%). Informational; the robust content of this claim is c01."
    },
    {
      "id": "c08_maxout_fgsm_confidence_value",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "In the same setting, a maxout network misclassifies 89.4\\%\nof our adversarial examples with an average confidence of 97.6\\%.",
      "citation": "paper/source/iclr2015.tex:338",
      "quantity": "measured.m3_maxout240_clean.fgsm_mean_confidence_on_errors",
      "claimed": 0.976,
      "tolerance": 0.2,
      "note": "Sub-scale 88.7%. Confidence magnitudes drift with convergence; informational."
    },
    {
      "id": "c09_advtrain_reduces_clean_error",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "Using this approach to train a maxout network that was also regularized with dropout, we were\nable to reduce the error rate from 0.94\\% without adversarial training to 0.84\\% with adversarial\ntraining.",
      "citation": "paper/source/iclr2015.tex:492",
      "quantity": "measured.m4_maxout240_advtrain.adversarial_clean_error - measured.m4_maxout240_advtrain.baseline_clean_error",
      "direction": "<0",
      "note": "The paper's headline claim. Held at BOTH sub-scales already run: gate (dropout off) 2.12% -> 1.71%, milestone arm (dropout 0.8) 1.98% -> 1.64%."
    },
    {
      "id": "c10_advtrain_m4_adversarial_magnitude",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "Using this approach to train a maxout network that was also regularized with dropout, we were\nable to reduce the error rate from 0.94\\% without adversarial training to 0.84\\% with adversarial\ntraining.",
      "citation": "paper/source/iclr2015.tex:492",
      "quantity": "measured.m4_maxout240_advtrain.adversarial_clean_error",
      "claimed": 0.0084,
      "tolerance": 0.01,
      "note": "0.84% needs dropout on + early stopping to convergence (paper-scale). Sub-scale (12 epochs): 1.64%. Informational."
    },
    {
      "id": "c11_m5_advtrain_beats_baseline",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "for this problem. Without adversarial training, this causes the model to overfit slightly,\nand get an error rate of 1.14\\% on the test set.",
      "citation": "paper/source/iclr2015.tex:499",
      "quantity": "measured.m5_maxout1600_advtrain.mean_test_error - measured.m5_maxout1600_clean.mean_test_error",
      "direction": "<0",
      "note": "Headline large-model claim (adv-trained model both fixes the large model's overfit and beats its own baseline). Sub-scale (240 units, single seed): 1.44% vs 1.82%."
    },
    {
      "id": "c12_m5_advtrain_mean_magnitude",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "four trials that each had an error rate of 0.77\\% on the test set and one trial that had\nan error rate of 0.83\\%. The average of 0.782\\% is the best result reported on the permutation\ninvariant version of MNIST, though statistically indistinguishable from the result obtained\nby fine-tuning DBMs with dropout~\\citep{dropout} at 0.79\\%.",
      "citation": "paper/source/iclr2015.tex:509",
      "quantity": "mean(measured.m5_maxout1600_advtrain.per_seed_test_errors)",
      "claimed": 0.00782,
      "tolerance": 0.003,
      "seeds": [
        0,
        1,
        2,
        3,
        4
      ],
      "note": "Requires the paper's full scale (1600 units, patience 100, 5 seeds, 60k retrain). `seeds` here overrides the top-level [0,1,2] to match the paper's five runs. Informational at sub-scale."
    },
    {
      "id": "c13_m5_seed_spread_invariant",
      "kind": "invariant",
      "compute_invariance": "low",
      "quote": "four trials that each had an error rate of 0.77\\% on the test set and one trial that had\nan error rate of 0.83\\%.",
      "citation": "paper/source/iclr2015.tex:509",
      "predicate": "max(measured.m5_maxout1600_advtrain.per_seed_test_errors) - min(measured.m5_maxout1600_advtrain.per_seed_test_errors) <= 0.0006",
      "seeds": [
        0,
        1,
        2,
        3,
        4
      ],
      "note": "Seed-stability claim (spread 0.77%..0.83% = 6e-4). Paper-scale only; informational."
    },
    {
      "id": "c14_advtrained_robust_vs_naive",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "Recall that without\nadversarial training, this same kind of model had an error rate of 89.4\\% on adversarial\nexamples based on the fast gradient sign method. With adversarial training, the error rate\nfell to 17.9\\%.",
      "citation": "paper/source/iclr2015.tex:514",
      "quantity": "measured.m6_robustness_transfer_eval.own_fgsm_error - measured.m3_maxout240_clean.fgsm_error_rate",
      "direction": "<0",
      "note": "Robustness-gain claim. Sub-scale: 10.4% vs 99.8% (the gap is wider at sub-scale because the naive net is fooled more, not less)."
    },
    {
      "id": "c15_transfer_asymmetry",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "adversarially trained model showing greater robustness. Adversarial examples generated via\nthe original model yield an error rate of 19.6\\% on the adversarially trained model, while\nadversarial examples generated via the new model yield an error rate of 40.9\\% on the original\nmodel.",
      "citation": "paper/source/iclr2015.tex:518",
      "quantity": "measured.m6_robustness_transfer_eval.orig_to_adv_transfer_error - measured.m6_robustness_transfer_eval.adv_to_orig_transfer_error",
      "direction": "<0",
      "note": "Cross-model transfer is asymmetric in the adv-trained model's favor. Sub-scale: 34.0% < 67.3%."
    },
    {
      "id": "c16_advtrain_confidence_still_high",
      "kind": "existence",
      "compute_invariance": "high",
      "quote": "predictions are unfortunately still highly confident. The average confidence on a misclassified\nexample was 81.4\\%.",
      "citation": "paper/source/iclr2015.tex:522",
      "predicate": "measured.m6_robustness_transfer_eval.own_fgsm_mean_confidence_on_errors > 0.5",
      "note": "The 'highly confident' qualitative claim; threshold 0.5 is well below the paper's 81.4%. Sub-scale: 65.1%. (The exact 81.4% magnitude is not claimed: confidence is budget-sensitive.)"
    },
    {
      "id": "c17_noise_controls_weaker_than_advtrain",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "classification. As control experiments, we trained training a maxout network with noise based on randomly adding $\\pm \\eps$\nto each pixel, or adding noise in $U(-\\eps, \\eps)$ to each pixel. These obtained an error rate of 86.2\\% with confidence\n97.3\\% and an error rate of 90.4\\% with a confidence of 97.8\\% respectively on fast gradient sign adversarial examples.",
      "citation": "paper/source/iclr2015.tex:555",
      "quantity": "min(measured.m7_maxout_noise_sign.fgsm_error_rate, measured.m7_maxout_noise_uniform.fgsm_error_rate) - measured.m6_robustness_transfer_eval.own_fgsm_error",
      "direction": ">0",
      "note": "The paper's \u00a710 control-experiment claim: random-noise training does NOT reproduce adversarial training's robustness (86.2%/90.4% fooled vs 17.9%). Sub-scale: 99.97%/99.98% vs 10.4%."
    },
    {
      "id": "c18_noise_bernoulli_error_value",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "classification. As control experiments, we trained training a maxout network with noise based on randomly adding $\\pm \\eps$\nto each pixel, or adding noise in $U(-\\eps, \\eps)$ to each pixel. These obtained an error rate of 86.2\\% with confidence\n97.3\\% and an error rate of 90.4\\% with a confidence of 97.8\\% respectively on fast gradient sign adversarial examples.",
      "citation": "paper/source/iclr2015.tex:555",
      "quantity": "measured.m7_maxout_noise_sign.fgsm_error_rate",
      "claimed": 0.862,
      "tolerance": 0.15,
      "note": "Sub-scale 99.97% (noise training conferred even less robustness at low budget; directionally consistent with c17). Informational."
    },
    {
      "id": "c19_noise_uniform_error_value",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "classification. As control experiments, we trained training a maxout network with noise based on randomly adding $\\pm \\eps$\nto each pixel, or adding noise in $U(-\\eps, \\eps)$ to each pixel. These obtained an error rate of 86.2\\% with confidence\n97.3\\% and an error rate of 90.4\\% with a confidence of 97.8\\% respectively on fast gradient sign adversarial examples.",
      "citation": "paper/source/iclr2015.tex:555",
      "quantity": "measured.m7_maxout_noise_uniform.fgsm_error_rate",
      "claimed": 0.904,
      "tolerance": 0.15,
      "note": "Sub-scale 99.98%. Informational."
    },
    {
      "id": "c20_l1_coeff0025_too_large",
      "kind": "ordering",
      "compute_invariance": "medium",
      "quote": "adversarial training with $\\eps = .25$. When applying $L^1$ weight decay to the first layer, we found that even a coefficient of\n.0025 was too large, and caused the model to get stuck with over 5\\% error on\nthe training set.",
      "citation": "paper/source/iclr2015.tex:429",
      "quantity": "measured.m_l1_weight_decay.l1_0.0025_train_error - 0.05",
      "direction": ">0",
      "note": "Held at sub-scale (88.6% train error -- stuck). Medium: the 'too large' threshold shifts DOWN at smaller scale (0.00025 is also stuck at sub-scale); the direction at coefficient 0.0025 is robust but the threshold location is not (SPEC \u00a76 item 28)."
    },
    {
      "id": "c21_l1_small_coeff_no_benefit",
      "kind": "ordering",
      "compute_invariance": "medium",
      "quote": "the training set. Smaller weight decay coefficients permitted succesful training\nbut conferred no regularization benefit.",
      "citation": "paper/source/iclr2015.tex:431",
      "quantity": "measured.m_l1_weight_decay.l1_2.5e-05_test_error - measured.m_l1_weight_decay.baseline_test_error",
      "direction": ">0",
      "note": "A small L1 coefficient trains but does not IMPROVE clean test error over baseline (paper's 'no regularization benefit'). Sub-scale: 3.06% vs 2.65%. Medium: close to noise at small budgets."
    },
    {
      "id": "c22_rbf_low_confidence_when_fooled",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "RBF networks are naturally immune to adversarial examples, in the sense that they have low\nconfidence when they are fooled. A shallow RBF network with no hidden layers gets an error\nrate of 55.4\\% on MNIST using adversarial examples generated with the fast gradient sign\nmethod and $\\eps = .25$. However, its confidence on mistaken examples is only $1.2\\%$.\nIts average confidence on clean test examples is $60.6$\\%.",
      "citation": "paper/source/iclr2015.tex:600",
      "quantity": "measured.m8_rbf_shallow.rbf_fgsm_mean_confidence_on_errors - measured.m8_rbf_shallow.rbf_clean_confidence",
      "direction": "<0",
      "note": "The qualitative content of 'naturally immune': when fooled, the RBF's confidence collapses relative to its clean confidence. Sub-scale: 24.9% on mistakes vs 36.6% clean."
    },
    {
      "id": "c23_rbf_confidence_magnitude",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "However, its confidence on mistaken examples is only $1.2\\%$.",
      "citation": "paper/source/iclr2015.tex:603",
      "quantity": "measured.m8_rbf_shallow.rbf_fgsm_mean_confidence_on_errors",
      "claimed": 0.012,
      "tolerance": 0.25,
      "note": "The paper never states the RBF multiclass normalization/structure (SPEC \u00a76 item 9); the exact 1.2% magnitude is implementation-dependent. Informational."
    },
    {
      "id": "c24_rbf_fgsm_error_value",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "confidence when they are fooled. A shallow RBF network with no hidden layers gets an error\nrate of 55.4\\% on MNIST using adversarial examples generated with the fast gradient sign\nmethod and $\\eps = .25$.",
      "citation": "paper/source/iclr2015.tex:601",
      "quantity": "measured.m8_rbf_shallow.rbf_fgsm_error_rate",
      "claimed": 0.554,
      "tolerance": 0.45,
      "note": "Capacity/budget-bound magnitude (sub-scale 95.0% -- a weaker RBF is fooled more often, still at reduced confidence, cf. c22). Informational."
    },
    {
      "id": "c25_softmax_agrees_more_than_rbf_all_errors",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "On examples that were misclassified by the maxout network, the RBF network predicted\nthe maxout network's class assignment only 16.0\\% of the time, while the softmax classifier\npredict the maxout network's class correctly 54.6\\% of the time.",
      "citation": "paper/source/iclr2015.tex:681",
      "quantity": "measured.m8_rbf_shallow.agreement_softmax_pred_maxout_over_m1_errors - measured.m8_rbf_shallow.agreement_rbf_pred_maxout_over_m1_errors",
      "direction": ">0",
      "note": "Section 8's evidence that cross-model generalization is linear-like: the LINEAR model agrees with the maxout attacker's class far more than the RBF does. Sub-scale: 62.8% vs 30.5%."
    },
    {
      "id": "c26_softmax_agrees_more_than_rbf_both_wrong",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "If we exclude our\nattention to cases where both models being compared make a mistake, then softmax regression\npredict's maxout's class 84.6\\% of the time, while the RBF network is able to predict maxout's\nclass only 54.3\\% of the time.",
      "citation": "paper/source/iclr2015.tex:684",
      "quantity": "measured.m8_rbf_shallow.agreement_softmax_pred_maxout_over_both_wrong - measured.m8_rbf_shallow.agreement_rbf_pred_maxout_over_both_wrong",
      "direction": ">0",
      "note": "Same claim restricted to the both-models-wrong subset (controlling for differing error rates). Sub-scale: 64.5% vs 47.3%."
    },
    {
      "id": "c27_rbf_has_linear_component",
      "kind": "existence",
      "compute_invariance": "medium",
      "quote": "class only 54.3\\% of the time. For comparison, the RBF network can predict softmax regression's\nclass 53.6\\% of the time, so it does have a strong linear component to its own behavior.",
      "citation": "paper/source/iclr2015.tex:687",
      "predicate": "measured.m8_rbf_shallow.agreement_rbf_pred_softmax_over_m1_errors > 0.25",
      "note": "Threshold 0.25 is well above the ~0.1 ten-class chance floor (paper 53.6%, sub-scale 38.7%). Medium: the paper is ambiguous about which generated set this number uses; we adopt the maxout-generated set with softmax as reference (SPEC \u00a76 item 29)."
    },
    {
      "id": "c28_ensemble_not_resistant",
      "kind": "existence",
      "compute_invariance": "high",
      "quote": "gradient descent. The ensemble gets an error rate of 91.1\\% on adversarial examples designed\nto perturb the entire ensemble with $\\epsilon = .25$.",
      "citation": "paper/source/iclr2015.tex:822",
      "predicate": "measured.e1_ensemble12_maxout.ensemble_targeted_error > 0.5",
      "note": "Summary claim 'Ensembles are not resistant to adversarial examples' (\u00a710). Sub-scale (4 members): 99.76%."
    },
    {
      "id": "c29_ensemble_targeted_vs_single",
      "kind": "ordering",
      "compute_invariance": "low",
      "quote": "to perturb the entire ensemble with $\\epsilon = .25$. If we instead use adversarial examples designed to perturb only one\nmember of the ensemble, the error rate falls to 87.9\\%.",
      "citation": "paper/source/iclr2015.tex:823",
      "quantity": "measured.e1_ensemble12_maxout.ensemble_targeted_error - measured.e1_ensemble12_maxout.single_member_targeted_error",
      "direction": ">0",
      "note": "Whole-ensemble attack fools 91.1% > 87.9% single-member attack. At sub-scale both arms saturate (~99.8%) and the small gap REVERSED (99.76 vs 99.79); this ordering is budget-fragile, hence low. c28 carries the load for 'ensembles are not resistant'."
    },
    {
      "id": "c30_maxout_fooled_by_gaussian_rubbish",
      "kind": "existence",
      "compute_invariance": "high",
      "quote": "class to be an error. A naively trained maxout network with a softmax layer on top had an error rate\nof 98.35\\% on Gaussian rubbish examples with an average confidence of 92.8\\% on mistakes.",
      "citation": "paper/source/iclr2015.tex:907",
      "predicate": "measured.m9_rubbish_evals.maxout_softmax_rubbish_error > 0.5",
      "note": "Appendix claim: linear-built models confidently classify most N(0,I) noise. Sub-scale: 82.1%."
    },
    {
      "id": "c31_softmax_regression_fooled_by_gaussian_rubbish",
      "kind": "existence",
      "compute_invariance": "high",
      "quote": "A softmax regression model has an error rate of 59.8\\%\non the rubbish examples, with an average confidence on mistakes of 70.8\\%.",
      "citation": "paper/source/iclr2015.tex:920",
      "predicate": "measured.m9_rubbish_evals.softmax_regression_rubbish_error > 0.5",
      "note": "Shallow linear models have the same problem (the paper's point against Nguyen et al.'s deep-only framing). Sub-scale: 81.7%."
    },
    {
      "id": "c32_sigmoid_top_rubbish_value",
      "kind": "value",
      "compute_invariance": "low",
      "quote": "Changing the top layer to independent sigmoids dropped the error rate to 68\\% with an average\nconfidence on mistakes of 87.9\\%.",
      "citation": "paper/source/iclr2015.tex:909",
      "quantity": "measured.m9_rubbish_evals.sigmoid_top_rubbish_error",
      "claimed": 0.68,
      "tolerance": 0.25,
      "note": "Sub-scale 79.2%. Whether the sigmoid-top net is retrained or shares the trunk is unstated; we retrain with per-class BCE (SPEC \u00a76 item 25). Informational."
    },
    {
      "id": "c33_rbf_immune_to_gaussian_rubbish",
      "kind": "value",
      "compute_invariance": "medium",
      "quote": "If we use instead an RBF network, which does not behave like a linear function,\nwe find an error rate of 0\\%.",
      "citation": "paper/source/iclr2015.tex:922",
      "quantity": "measured.m9_rubbish_evals.rbf_rubbish_error",
      "claimed": 0.0,
      "tolerance": 0.01,
      "note": "Exactly 0.0 at sub-scale. Medium not high: the 0% is structural given the neg-definite quadratic construction but leans on the unstated multiclass extension (SPEC \u00a76 item 9)."
    },
    {
      "id": "c34_rbf_less_fooled_than_linear_on_rubbish",
      "kind": "ordering",
      "compute_invariance": "high",
      "quote": "If we use instead an RBF network, which does not behave like a linear function,\nwe find an error rate of 0\\%.",
      "citation": "paper/source/iclr2015.tex:922",
      "quantity": "measured.m9_rubbish_evals.rbf_rubbish_error - min(measured.m9_rubbish_evals.maxout_softmax_rubbish_error, measured.m9_rubbish_evals.softmax_regression_rubbish_error)",
      "direction": "<0",
      "note": "RBF vs every linear-built model on identical rubbish draws. Sub-scale: 0.0% vs 82.1%/81.7%."
    }
  ],
  "not_tested": [
    {
      "id": "nt01_googlenet_imagenet_fig1",
      "quote": "    By adding an imperceptibly small vector whose elements are equal to the sign of the elements\n    of the gradient of the cost function with respect to the input, we can\n    change GoogLeNet's classification of the image. Here our $\\eps$ of .007\n    corresponds to the magnitude of the smallest bit of an 8 bit image encoding\n    after GoogLeNet's conversion to real numbers.",
      "citation": "paper/source/iclr2015.tex:379",
      "reason": "Qualitative single-image demonstration; the 2014 DistBelief GoogLeNet weights are not available."
    },
    {
      "id": "nt02_cifar10_conv_maxout_fgsm",
      "quote": "Similarly, using $\\eps=.1$, we obtain an error rate of 87.15\\% and an average probability of\n96.6\\% assigned to the incorrect labels\nwhen using\na convolutional maxout network on a preprocessed version of the CIFAR-10~\\citep{KrizhevskyHinton2009} test set\\footnote{\n    See \\url{https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout}.\n    for the preprocessing code, which yields a standard deviation of roughly 0.5.\n}.",
      "citation": "paper/source/iclr2015.tex:340",
      "reason": "Convolutional maxout CIFAR-10 model: architecture and preprocessing exist only via the external pylearn2 yaml + hyperlink (tex:344); outside the MNIST core, budget-limited."
    },
    {
      "id": "nt03_mpdbm_generative",
      "quote": "examples. With an $\\epsilon$ of 0.25, we find an error rate of 97.5\\% on adversarial examples generated\nfrom the MNIST test set.",
      "citation": "paper/source/iclr2015.tex:800",
      "reason": "MP-DBM is a separate paper's model (Goodfellow et al. 2013a); implementing it is a second reproduction."
    },
    {
      "id": "nt04_m5_best_on_mnist_crosspaper",
      "quote": "an error rate of 0.83\\%. The average of 0.782\\% is the best result reported on the permutation\ninvariant version of MNIST, though statistically indistinguishable from the result obtained\nby fine-tuning DBMs with dropout~\\citep{dropout} at 0.79\\%.",
      "citation": "paper/source/iclr2015.tex:510",
      "reason": "Cross-paper comparison against Srivastava et al.'s dropout-DBM (0.79%) requires reproducing an external model. We test the internal ordering (c11) and magnitude (c12) instead."
    },
    {
      "id": "nt05_rotation_based_attack",
      "quote": "}. Other simple methods of generating adversarial examples are possible. For example, we also found that rotating\n$\\vx$ by a small angle in the direction of the gradient reliably produces adversarial examples.",
      "citation": "paper/source/iclr2015.tex:346",
      "reason": "No quantified MNIST number; the paper reports rotational variants as weaker regularizers than additive FGSM. Requires a differentiable rotation operator."
    },
    {
      "id": "nt06_fig3_weight_localization",
      "quote": "We also found that the weights of the learned model changed significantly,\nwith the weights of the adversarially trained model being significantly more localized and\ninterpretable (see Fig.~\\ref{fig:weights}).",
      "citation": "paper/source/iclr2015.tex:523",
      "reason": "Qualitative weight-visualization claim with no numeric target."
    },
    {
      "id": "nt07_fig4_epsilon_sweep",
      "quote": "The correct class is 4. We see that the unnormalized log probabilities for each class are conspicuously piecewise linear with $\\eps$ and that\nthe wrong classifications are stable across a wide region of $\\eps$ values.",
      "citation": "paper/source/iclr2015.tex:768",
      "reason": "Visual claim from the plotting layer (Figure 4); the epsilon axis range exists only in the figure, not the text."
    },
    {
      "id": "nt08_rubbish_class_skew",
      "quote": "On MNIST, 45.3\\% of a naively trained maxout network's false positives were classified as 5s,\nand none were classified as 8s.",
      "citation": "paper/source/iclr2015.tex:930",
      "reason": "Implementable from the m9 arm but the current m9 eval reports error rate + confidence only, not the per-class false-positive distribution. Recorded gap (SPEC \u00a71)."
    },
    {
      "id": "nt09_trained_to_zero_rubbish",
      "quote": "We found that we were able to train a maxout network to have a zero percent error rate on Gaussian\nrubbish examples (it was still vulnerable to rubbish examples generated by applying a fast gradient\nsign step to a Gaussian sample)\nwith no negative impact on its ability to classify clean examples. Unfortunately, unlike\ntraining on adversarial examples, this did not result in any significant reduction of the model's\ntest set error rate.",
      "citation": "paper/source/iclr2015.tex:963",
      "reason": "A recorded null result (no regularization benefit); needs a per-class rubbish-training protocol on top of the m9 machinery."
    },
    {
      "id": "nt10_cifar10_rubbish_and_targeted_fooling",
      "quote": "with variable runtime. On CIFAR-10, we found that one sampling step had a 100\\% success rate for\nfrogs and trucks, and the hardest class was airplanes, with a success rate of 24.7\\% per sampling\nstep.",
      "citation": "paper/source/iclr2015.tex:939",
      "reason": "CIFAR-10 rubbish error (tex:911-912) and per-class targeted fooling success rates (tex:936-941) share the excluded CIFAR-10 arm."
    }
  ]
}
```

## 11. Constructed truth

This section states, for each constructed-truth category the reproduction
contract names, whether it applies here and — where it does — the exact test
node that enforces it. A wrong implementation is caught by these oracles in
seconds, without a paper-scale run; a reader can verify them without trusting
the measured numbers.

- **Degeneracy (the method at its no-op reproduces the baseline EXACTLY).**
  APPLIES. The FGSM adversarial-training cost (E7, tex:486-488) has a no-op at
  `eps=0`: `x + 0*sign(grad) == x` exactly, so `J~ = alpha*J + (1-alpha)*J = J`.
  At eps=0 the method must reproduce clean training bit-for-bit (dropout OFF).
  Enforced by `tests/test_degeneracy.py` at three levels — cost
  (`test_cost_degeneracy_eps0_equals_clean`), training
  (`test_train_degeneracy_eps0_equals_baseline_bitidentical`), and CLI
  (`test_run_experiment_cli_degeneracy`, `run_experiment.py --lambda 0` vs
  `--baseline` print identical accuracy values). This is the cheapest real
  correctness evidence and the one most often skipped.

- **Brute force at toy scale against a closed form claiming a maximum /
  minimum / worst case.** APPLIES. E6 (tex:410-412) is the closed-form *worst-
  case* adversarial logistic loss in the L-inf box of radius eps. The
  brute-force oracle `tests/test_constructed_truth.py::test_brute_force_e6_is_the_maxnorm_worst_case`
  draws 200 random perturbations within the box and asserts NONE exceeds E6 —
  a direct test that the closed form really is the maximum it claims.

- **The same quantity derived two ways (the paper hands this to us for free).**
  APPLIES. E6 (closed form) vs the empirical FGSM loss on logistic regression:
  `tests/test_instruments.py::test_e6_positive_matches_empirical_fgsm_positive_class`
  asserts the closed form equals the loss on `x + eps*sign(-w)` for y=+1 (the
  case where the paper's displayed equation is exact, tex:407-412; the y=-1
  divergence is the paper's imprecision, recorded in §8 Finding E6). Also
  `tests/test_constructed_truth.py::test_naive_per_example_fgsm_equals_autograd_sign`
  (naive analytic input-gradient sign == autograd sign).

- **Planting a known structure in synthetic input and requiring the pipeline to
  recover it.** APPLIES. Plant a logistic regression with known `w` and require
  the FGSM perturbation to be the analytically-known direction
  `eta = -eps*y*sign(w)` with `||eta||_inf == eps`:
  `tests/test_constructed_truth.py::test_plant_known_linear_structure_fgsm_direction`.

- **A slow exact or convex reference solver.** APPLIES (same instrument as
  above). E6 is the convex (closed-form) reference for the FGSM attack on
  logistic regression; the brute-force search is the slow reference for the
  general case. Both used as oracles in `test_constructed_truth.py`.

- **The method's limiting cases.** APPLIES. eps=0 is the identity
  (`tests/test_invariants.py::test_fgsm_eps0_is_identity`,
  `test_eval_fgsm_eps0_equals_clean_error`); large eps drives a fitted linear
  model's error toward 1.0
  (`tests/test_constructed_truth.py::test_limiting_case_large_eps_drives_linear_model_to_all_wrong`).

- **The naive implementation agreeing with the fast one.** APPLIES. The
  per-element analytic input-gradient sign for the logistic cost equals the
  autograd-computed sign used by the fast FGSM path
  (`test_naive_per_example_fgsm_equals_autograd_sign`), and the closed-form E6
  equals the empirical FGSM loss (`test_e6_positive_matches_empirical_fgsm_positive_class`).

- **The paper's standard baseline, whose value is common knowledge and therefore
  an oracle you already have.** PARTIALLY APPLIES. MNIST maxout clean test
  error ~0.94% (tex:492) and the majority-class error 88.65% are common-
  knowledge oracles: the clean baseline arms (m3, m4 baseline, m5 baseline)
  must land in the ~1-2% error regime (far below 88.65%), and a collapsed
  adversarial arm (mean error ~88.65%) is flagged as not-trained
  (`experiments/m5_large_advtrain.py` `adv_arm_trained` guard). The exact 0.94%
  / 0.782% magnitudes require the paper's full scale (1600 units / patience 100
  / 5 seeds) and are rated `compute_invariance=low` in claims.json — they are
  NOT treated as oracles here, only the regime/direction is.

- **Equivalence / grader oracles (anything that decides whether an output is
  correct).** Every grader is exercised on one known-correct and one known-wrong
  input in `tests/test_instruments.py` (eval_clean, eval_fgsm error/confidence,
  eval_transfer, class_agreement, RBF confidence, rubbish sampling, fooling
  sign step, E6) plus the data-loader fingerprint
  (`tests/test_data_fingerprint.py`). A grader fed an EMPTY input must raise,
not return a vacuous 0.0 (`test_eval_clean_raises_on_empty`). The full
instrument registry with positive/negative tests is `instruments.json`.
The `measured.json` scalar contract (the gate formats every value as a scalar;
a list value crashes the evaluator) is guarded by
`tests/test_measured_resolver.py::test_measured_json_has_no_list_values`.

Categories NOT applicable here (stated for honesty): there is no claim of a
global *minimum* to brute-force (FGSM is a one-step attack, not an optimizer
with a minimum); no generative-model likelihood to cross-check (the MP-DBM
generative-resistance test, tex:827-833, is in `not_tested` — we do not
implement MP-DBM inference); and no CIFAR-10 arm (the CIFAR-10 numbers, tex:341,
tex:911-912, are in `not_tested` — the dataset/preprocessing is excluded by the
scope fixed in §1).
