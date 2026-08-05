# SPEC — Reproduction of "Explaining and Harnessing Adversarial Examples"

- **Paper:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy, *Explaining and Harnessing
  Adversarial Examples*, ICLR 2015 (arXiv:1412.6572v3, 20 Mar 2015).
- **Authoritative source on disk:** `paper/source/iclr2015.tex` (arXiv LaTeX, 975 lines). Every
  citation in this SPEC is `<file>:<line>` into that file and was grep-verified by script in this
  run (all 70 claim quotes resolve verbatim at their cited lines). Math macros in the preamble
  resolve as `\eps`→ε, `\sign`→sign, `\veta`→**η**, `\vtheta`→**θ**, `\vx`→**x**, `\vw`→**w**
  (`iclr2015.tex:14-55`).
- **Convenience copy:** `paper/paper.md` (PDF extraction; prose reliable, maths not — never cited).
- **Percent convention:** every error / confidence / agreement metric is a **percent** in
  [0, 100] so `measured` values compare directly against the paper's printed numbers.
- **This SPEC is authored from the LaTeX source in this run**; where a numeric value could only
  come from a figure, it was read with `read-figure` and the exchange is committed at
  `figures/read-figure.jsonl` (see §7).

## 0. Upstream code status

Checked in this run (2026-08-04):

- **The paper's only code link** is the CIFAR-10 preprocessing footnote:
  `https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout`
  (`iclr2015.tex:343-345`). Fetched: the repo exists but that directory's README recreates the
  **maxout paper** (Goodfellow et al. 2013c) experiments ("The files in this directory recreate
  some of the experiments reported in the paper Maxout Networks"), not this paper's. The stack is
  Theano/Pylearn2 (Python 2, archived; Theano unmaintained) and cannot execute today. No FGSM or
  adversarial-training code from this paper ships anywhere in it.
- **`github.com/goodfeli/adversarial`** (lead author's account) is the GAN paper's code
  ("Generative Adversarial Nets"), not this paper's. Not usable here.
- **GitHub search on the exact title** (`"explaining and harnessing adversarial examples"`,
  repositories): 13 hits, all third-party partial reimplementations of the *attack only*
  (e.g. Harry24k/FGSM-pytorch, 70 stars, Jupyter notebook; rodgzilla/
  machine_learning_adversarial_examples, 55 stars). None implements FGSM adversarial training,
  none of the paper's arms (maxout adversarial training, agreement/transfer/rubbish/ensemble
  experiments), and none carries the paper's configs.
- **`github.com/cleverhans-lab/cleverhans`** (later library co-authored by the lead author)
  contains `cleverhans/torch/attacks/fast_gradient_method.py` — verified by fetch: its docstring
  cites `https://arxiv.org/abs/1412.6572` and it implements the single-step sign attack with
  optional clipping. It is a generic library primitive; it contains none of this paper's models,
  training procedures, or numbers. Not upstream code for the experiments.
- **Decision:** no usable upstream code exists for this paper's experiments. We implement from
  this SPEC.
- **Reference-only material:** the maxout MLP training details this paper defers to ("the 240
  used by the original maxout network", `iclr2015.tex:498`) live in the maxout paper's pylearn2
  YAML configs. Where this paper is silent (§4), those configs may be consulted as *reference
  defaults*; every such fill-in is logged as **our choice, not the paper's**.

## 1. The method as an explicit algorithm

### 1.1 FGSM — fast gradient sign method (the attack)

```
Input:  model with parameters θ and cost J(θ, x, y); batch x [B,D]; labels y [B]; ε > 0.
 1. g ← ∇_x J(θ, x, y)                 # [B,D], by backprop; θ held fixed
 2. η ← ε · sign(g)                     # [B,D], sign elementwise; sign(0) := 0
 3. x̃ ← x + η                           # [B,D]; ‖η‖_∞ = ε exactly
Output: adversarial examples x̃.
```

This is "an optimal max-norm constrained perturbation" under the linearization of J in x
(`iclr2015.tex:305-311`). **No clipping** of x̃ back into [0,1] is stated anywhere in the paper;
the default here is **no clipping** (§4.9). The gradient is w.r.t. x only; θ is constant during
attack generation.

### 1.2 FGSM adversarial training (the harness)

```
Input: training set, model f_θ, mixing constant α, perturbation size ε.
Repeat per minibatch (x, y):
  1. x̃ ← FGSM(f_θ, x, y, ε) using the CURRENT θ ("continually update our supply",
     iclr2015.tex:490-491). sign() has zero/undefined derivative, so no gradient flows from
     x̃ back through the perturbation step w.r.t. θ (iclr2015.tex:559-561).
     The FGSM direction is computed with f_θ in EVAL mode (dropout OFF) — the paper's
     FGSM (tex:309, Fig.1) is defined on the deterministic network; computing it under an
     active dropout mask zeroes the input gradient on masked pixels and yields a weaker/
     different perturbation. The adversarial-half LOSS J(θ, x̃, y) is then evaluated in
     TRAIN mode (with dropout), as part of training.
  2. L ← α · mean_i J(θ, x_i, y_i) + (1−α) · mean_i J(θ, x̃_i, y_i)
  3. θ ← θ − lr · ∇_θ L
```


α = 0.5 in **all** the paper's experiments (`iclr2015.tex:488`). ε for adversarial training is
stated only for MNIST maxout: ε = 0.25 (`iclr2015.tex:428-429`). Both halves reuse the **same**
labels y (the printed equation drops the y in the second term — typo, see §3 eq. 6).

### 1.3 Adversarial logistic regression (analytic special case, §5)

Binary logistic regression, y ∈ {−1, +1}, P(y=1) = σ(wᵀx + b), trained by gradient descent on
E ζ(−y(wᵀx + b)) with ζ(z) = log(1 + e^z) (`iclr2015.tex:399-404`). The sign of the input
gradient is −sign(w) (times y) and wᵀsign(w) = ‖w‖₁ (`iclr2015.tex:407`), so FGSM is **exact**
and adversarial training has the closed form

```
minimize  E_{x,y} ζ( y · ( ε‖w‖₁ − wᵀx − b ) )          (iclr2015.tex:411)
```

i.e. the ‖w‖₁ penalty is subtracted from the **activation** (margin) during training, not added
to the cost as in L¹ weight decay (`iclr2015.tex:413-419`).

**Sign-slip caveat (c07).** `tex:407` states "the sign of the gradient is just
−sign(w)", dropping the per-example label factor: the true input-gradient sign for
logistic regression is **−y·sign(w)** (since `∇ₓ ζ(−y(wᵀx+b)) ∝ −y·w`), so the
worst-case perturbation is `x_adv_i = x_i − ε·y_i·sign(w)`, giving loss
`ζ(ε‖w‖₁ − y_i(wᵀx_i+b))`. The paper's `tex:411` closed form
`ζ(y·(ε‖w‖₁ − wᵀx − b))` equals this **only for y=+1**; for y=−1 it gives
`ζ(m − ε‖w‖₁)`, which *decreases* the loss (not the worst case). c07 therefore
checks the identity against the **correct** worst-case closed form
`ζ(ε‖w‖₁ − y(wᵀx+b))` using the real gradient-based `attack.fgsm` (per-example,
mixed labels), not the paper's `tex:411` form. We implement both the FGSM form
and the correct closed form and assert their numerical equality (claim c07, §8).

### 1.4 Rubbish-class and targeted-fooling protocols (Appendix)

- **Rubbish eval (MNIST):** draw 10,000 samples x ~ N(0, I₇₈₄) (`iclr2015.tex:905`). "Error" :=
  the model assigns probability > 0.5 to **any** class (`iclr2015.tex:906-907`): max_k p_k > 0.5
  for softmax models; any p_k > 0.5 for the independent-sigmoid top.
- **Rubbish eval (CIFAR-10):** same, with 1,000 samples x ~ N(0, I₃₀₇₂) (`iclr2015.tex:911`).
- **Targeted fooling (CIFAR-10):** to fool class i, take one **gradient-sign step** from a
  Gaussian sample toward increasing p(y = i | x). The Fig. 5 caption states the sign step
  (`iclr2015.tex:953-955`); the body text writes the step without `sign` (`iclr2015.tex:936`) —
  we implement the caption's sign-step version (§4). Success := p(y = i | x̃) > 0.5
  (`iclr2015.tex:955-956`). Per-class per-step success rate over ≥ 1,000 fresh samples.

## 2. Symbols with shapes

B = minibatch size, D = input dim (784 for MNIST, 3072 = 32·32·3 flattened CIFAR-10), K = 10
classes. All arrays float32 unless noted.

| symbol | shape | meaning |
|---|---|---|
| x | `[B, D]` f32 | input batch; MNIST pixels in **[0,1]** (footnote 1, `iclr2015.tex:334-337`); CIFAR-10 GCN-preprocessed to std ≈ 0.5 (footnote 2, `iclr2015.tex:343-345`) |
| y | `[B]` int64 | class indices {0..9}; for 3-vs-7 logistic regression, y ∈ {−1,+1} (`iclr2015.tex:399`) |
| θ | parameter pytree | all model parameters (`iclr2015.tex:305`) |
| J(θ, x, y) | scalar f32 | mean cost over the batch: softmax NLL for K-class models; mean ζ(−y(wᵀx+b)) for logistic regression (`iclr2015.tex:306, 401-404`) |
| ∇_x J | `[B, D]` f32 | gradient of the scalar batch cost w.r.t. the input |
| sign(∇_x J) | `[B, D]` in {−1,0,1} | elementwise; sign(0) := 0 (§4.22) |
| η | `[B, D]` f32 | ε·sign(∇_x J); ‖η‖_∞ = ε by construction |
| x̃ = x + η | `[B, D]` f32 | adversarial example (`\tilde{\vx} = \vx + \veta`, `iclr2015.tex:235`) |
| ε | scalar f32 | max-norm ball radius; 0.25 MNIST, 0.1 CIFAR-10, 0.007 ImageNet demo (`iclr2015.tex:333, 340, 381`) |
| w, b (logreg) | `[D]` f32, scalar | logistic-regression weight vector and bias (`iclr2015.tex:399`) |
| σ(z), ζ(z) | scalar→scalar | logistic sigmoid; softplus log(1+e^z) (`iclr2015.tex:399-404`) |
| logits a | `[B, K]` f32 | "argument to softmax" — the y-axis of Fig. 4 (`iclr2015.tex:767`) |
| p = softmax(a) | `[B, K]` f32 | class probabilities; **confidence** of one example := max_k p_k |
| α | scalar | adversarial-training mixing constant, 0.5 (`iclr2015.tex:488`) |
| maxout W^(l) | `[fan_in, units, pieces]` | per-unit max over `pieces` affine slices (`iclr2015.tex:298` cites the maxout paper for the unit def); units/layers per arm (§6) |
| μ_k, β_k (RBF) | `[D]`, `[D, D]` | class template and quadratic form for class k; **β_k negative-semidefinite** (§4.4; printed eq. `iclr2015.tex:595` has no minus sign) |
| noise (controls) | `[B, D]` f32 | per-pixel ±ε (Rademacher) or U(−ε, ε), resampled per minibatch (`iclr2015.tex:555-557`) |
| ε grid (Fig. 4) | `[21]` | ε ∈ {−10, …, +10} for the logit trace arm (§5, §7) |

## 3. Equations to implement, each with a citation that resolves

1. `wᵀx̃ = wᵀx + wᵀη`; optimal constrained η = ε·sign(w); activation grows by ε·m·n for n-dim w
   with average |w_i| = m.
   Cites: `paper/source/iclr2015.tex:243` (equation), `:246-247` (η = sign(w), growth εmn).
2. **FGSM:** `η = ε sign(∇_x J(θ, x, y))` — `paper/source/iclr2015.tex:309`.
3. **Logistic-regression training:** minimize `E ζ(−y(wᵀx + b))`, ζ softplus —
   `paper/source/iclr2015.tex:402` (loss), `:404` (ζ def), `:399` (P(y=1)=σ(wᵀx+b), y∈{−1,1}).
4. **FGSM exact for logreg:** sign(∇_x J) = −y·sign(w), wᵀsign(w) = ‖w‖₁ —
   `paper/source/iclr2015.tex:407`.
5. **Adversarial logistic regression:** minimize `E ζ(y(ε‖w‖₁ − wᵀx − b))` —
   `paper/source/iclr2015.tex:411`.
6. **Adversarial training of deep nets:**
   `J̃(θ,x,y) = αJ(θ,x,y) + (1−α)J(θ, x + ε sign(∇_x J(θ,x,y)), y)` —
   `paper/source/iclr2015.tex:486-487` (**as printed, the second J is missing its `y` argument
   and its closing paren**; the intended reading is unambiguous from the definition of J at
   `iclr2015.tex:306`). α = 0.5 — `:488`.
7. **Shallow RBF:** `p(y=1|x) = exp((x − μ)ᵀ β (x − μ))` — `paper/source/iclr2015.tex:595`.
   As printed this only *decays* away from μ if β is negative (semi)definite; the paper never
   says so. We parametrize β_k = −ψ_k ψ_kᵀ − ν·I, ν ≥ 0 (§4.4).
8. **Targeted fooling:** x̃ = x + ε·sign(∇_x p(y = i | x)) with x ~ N(0, I_D) — body text
   without sign: `paper/source/iclr2015.tex:936`; Fig. 5 caption with sign step:
   `paper/source/iclr2015.tex:953-955`. We implement the caption's sign-step version.
9. **Rubbish error rule:** error ⇔ any class prob > 0.5 on x ~ N(0, I₇₈₄) (10,000 samples,
   MNIST) or x ~ N(0, I₃₀₇₂) (1,000 samples, CIFAR-10) —
   `paper/source/iclr2015.tex:905-907, 911`.

## 4. What the paper does NOT state

This list is the gap between the paper and any implementation. Unless marked RESOLVED (with the
resolution stated), each item must be chosen by the implementation and logged in README.md as
**our choice, not the paper's**.

**Training protocol (all arms)**
1. Optimizer, learning rate, momentum, LR schedule, batch size, number of epochs: **unstated
   everywhere** in this paper (`iclr2015.tex:498` only defers to "the original maxout network").
2. Maxout architecture detail: number of linear **pieces per maxout unit**, number of hidden
   layers ("240/1600 units per layer", `:498`, never says which layers), dropout rates
   (input/hidden), weight initialization scheme — all unstated here.
3. Softmax-regression and logistic-regression training hyperparameters: entirely unstated.
4. RBF network: training procedure, class bias/prior term, parametrization/sign of β (eq. 7 is
   printed without the needed negative sign), and how the 10 binary class-conditionals yield one
   predicted class — all unstated. RESOLVED: 10 units p_k = exp((x−μ_k)ᵀβ_k(x−μ_k)), β_k
   negative-semidefinite, predict argmax_k p_k, confidence = max_k p_k. This is the only reading
   consistent with "confidence on mistaken examples is only 1.2%" (`:603`): under a 10-way
   softmax the minimum possible confidence is 10%.
5. Whether RBF "average confidence on clean test examples is 60.6%" (`:604`) is mean max-prob —
   assumed yes.
6. RBF FGSM cost: which J is differentiated to attack the RBF — unstated. RESOLVED: −log p_y.
7. Ensemble training: members differ only by RNG seed (stated, `:819-821`); nothing else stated.
8. L¹ weight-decay control: "to the first layer" is stated (`:429`); optimizer/epochs unstated;
   smaller coefficients "permitted successful training but conferred no regularization benefit"
   (`:431-432`) with no numbers.

**Attack / evaluation detail**
9. **Clipping** of x̃ to [0,1] (MNIST) or the valid preprocessed range (CIFAR): never mentioned
   anywhere. RESOLVED default: **no clipping**; a clipped variant may be run as sensitivity
   analysis but the paper-comparable numbers are unclipped.
10. "Average confidence" denominators: for softmax 79.3% (`:333-334`) and maxout 97.6% (`:339`)
    it is unstated whether the mean is over all adversarial examples or mistakes only; for
    CIFAR 96.6% "assigned to the incorrect labels" (`:340-341`) implies mistakes-only; RBF gives
    both explicitly (`:603-604`). RESOLVED: record **both** `adv_conf_all` and
    `adv_conf_mistakes`; value claims c02/c10 use `_all` (with 99.9%/89.4% error the two differ
    little).
11. Test-set coverage: FGSM metrics are "on the MNIST test set" (`:334`) — we use all 10,000
    test examples; CIFAR-10 likewise (`:343`).
12. Ensemble attack objective: "designed to perturb the entire ensemble" (`:822-823`) — what is
    differentiated is unstated. RESOLVED: FGSM on the **mean-logits** ensemble NLL —
    `cross_entropy(mean_member logits, y)`, a differentiable "perturb the whole ensemble"
    objective (the paper is silent on the form; mean-probability NLL is the alternative; we
    use mean-logits CE, which the code implements and which sends gradients to all members);
    single-member attacks use member 0's gradient (`:823-824`). Prediction combines mean
    PROBABILITIES (argmax of `mean_member softmax`), as stated.
13. Ensemble decision rule: mean-probability argmax — unstated, assumed.
14. Targeted-fooling ε (`:936`) and per-class sample count: unstated. RESOLVED: implementation
    picks and logs ε; ≥ 1,000 samples per class.
15. Fig. 4 trace: which test example (caption only says "The correct class is 4", `:768`), which
    ε grid, and — implicitly but never stated — that the FGSM direction is computed **once at
    ε = 0** and held fixed while ε sweeps (only this reading makes the logit curves exactly
    piecewise linear). RESOLVED: Figure 4 reports ONE illustrative example, not a population
    claim, so the arm uses a **deterministic** fixed example — the **first (lowest-index)
    class-4 test example that ALL seed models classify correctly** — with **NO selection on the
    thin-manifold predicates** (margin>0 at ε=0, <0 at the tails, monotone increase). The example
    is chosen purely by index, so the curve claims (c65–c70, rated `low` — single-example
    illustration, not a population invariant) can genuinely FAIL if this example does not
    reproduce the figure shape. The FGSM direction is computed once at ε=0 and held fixed. Fall
    back to the first correct class-4 example of seed 0 if no class-4 example is correct for all
    seeds. (An earlier version selected the example by the very predicates the claims evaluate —
    "pass by construction"; that selection is removed.)
16. Transfer pair (19.6% / 40.9%, `:519-520`): which architectures "the original model" and
    "the new model" are is not explicit (§6 discusses both the 240-unit 0.94→0.84 result and the
    1600-unit 0.782 result). RESOLVED: primary pair = (maxout_large_naive, maxout_large_adv);
    the small pair may be recorded as secondary.
17. Agreement experiment (§8 of the paper): which maxout architecture generated the examples is
    unstated; and the conditioning of "the RBF network can predict softmax regression's class
    53.6% of the time" (`:687-688`) is unstated — read as both-models-wrong conditioned,
    mirroring the preceding sentence.

**Stopping / seed protocol**
18. Validation split: not stated (50k train / 10k val inferred from "retrained on all 60,000
    examples", `:505-506`). RESOLVED: standard last-10k-of-60k split.
19. Early stopping: patience 100 epochs on validation error for the original maxout result
    (`:501-503`); **adversarial validation set error** for the large adversarially trained model
    (`:504-505`) — its ε (assumed 0.25) and refresh frequency (assumed every epoch) are unstated.
    NOTE (sub-scale override): the paper's patience-100 budget is infeasible on CPU; the run caps
    patience at **8** (MNIST arms) / **5** (CIFAR arm) with the same early-stopping monitor. This
    is a CPU-horizon cap, not a method change; it is recorded in `run_all_arms.py` and does not
    affect any HIGH-invariance claim. `train.py`'s default patience remains 100.
20. Retrain-on-60k uses the early-stopped epoch count (`:505-506`) — that rule is stated; the
    epoch counts themselves are not.
21. Seeds: only "different seeds for the random number generators used to select minibatches of
    training examples, initialize model weights, and generate dropout masks" (`:506-508`). No
    seed values, no counts except the five runs of the large model (`:508-510`).

**Numerical / kernel minutiae**
22. sign(0) convention: unstated. RESOLVED: 0.
23. Numerical stabilizers (softmax max-subtraction, softplus overflow): unstated.
24. CIFAR conv-maxout architecture (layers/channels/kernels) and its **clean test error**: both
    unstated; preprocessing only referenced to the pylearn2 maxout scripts yielding std ≈ 0.5
    (`:343-345`). RESOLVED: GCN variant chosen to give global std ≈ 0.5; exact recipe logged as
    ours. The conv-maxout stages use **NO post-ReLU** (maxout is itself the nonlinearity, per
    Goodfellow et al. 2013c; an earlier version inserted `F.relu` after each stage, an extra
    nonlinearity the paper never describes — removed). The conv net uses **NO dropout** (the
    paper's maxout networks were dropout-regularized, but the paper does not describe the CIFAR
    conv arch; dropping dropout is an arch-ours choice for the CPU sub-scale net, logged here).
    Targeted-fooling uses **1,000 samples per class** (§4.14); the run previously used 200
    (underpowered) and now matches §4.14.
25. Whether control-noise is resampled per minibatch/epoch (`:555-557`): unstated. RESOLVED:
    resampled at every minibatch presentation.
26. Adversarial-training minibatch composition: whether the α-weighted halves share one batch —
    RESOLVED yes (single-batch reading of eq. 6) — and both halves receive gradient every step.
26b. Adversarial-training FGSM mode: the paper defines FGSM on the cost J of the (deterministic)
    network (`tex:309`, Fig.1) but does not state whether the in-training attack runs under the
    model's active dropout mask. RESOLVED: the perturbation is computed with the model in **eval
    mode** (dropout OFF); the adversarial-half loss is then evaluated in train mode (with
    dropout). Computing the attack under an active mask zeroes ∇ₓ on masked pixels and yields a
    materially weaker/different perturbation (the train-mode FGSM sign is exactly 0 on ~20% of
    pixels); eval-mode attacks are the faithful reading. The clean half always runs in train mode.
26c. RBF network: the printed equation (`tex:595`) has no per-class temperature and no loss clamp.
    RESOLVED: no `log_temp` parameter (an earlier inert `log_temp` slot is removed) and no loss
    clamp (RBF logits ≤ 0 ⇒ NLL ≥ 0 by construction, so a clamp is dead). Training procedure
    (β/μ parametrization, ν floor, mean init) is paper-silent and logged as ours (§4.4).
27. MP-DBM arm: all detail is external to this paper (`:793-800`); **not built** (§9).
28. Fig. 1 ImageNet demo (ε = .007, `:381-383`): GoogLeNet weights/preprocessing not given;
    **not built** (§9).

## 5. Component interfaces (fixed now; parallel agents build against these)

Framework: a single array-programming stack picked once at implementation time and recorded in
README (CPU is sufficient at this paper's scale; FGSM needs the input gradient, i.e. autograd
w.r.t. x). **One arm = one entry per seed** containing exactly the metric keys listed below;
claims resolve `measured.<arm>.<key>` against them.

```
data.py
  load_mnist(seed:int)          -> dict(x_train[50000,784], y_train[50000],
                                        x_val[10000,784],   y_val[10000],
                                        x_test[10000,784],  y_test[10000])   pixels in [0,1]
  load_mnist_full(seed:int)     -> same but train covers all 60,000 (for the retrain arm)
  load_mnist_3v7(seed:int)      -> subset of labels 3 and 7, y in {+1, -1};
                                   CONVENTION (fixed here): y=+1 ≡ digit 3, y=-1 ≡ digit 7
  load_cifar10(seed:int)        -> x_*[N,3072] GCN-preprocessed (global std ≈ 0.5), y_* int64
  rubbish(dim:int, n:int, seed:int) -> [n,dim] f32 ~ N(0, I_dim)

models.py  — every model implements:
  .logits(x[B,D]) -> [B,K] f32          (RBF/ensemble return K=10 score functions;
                                         logreg_3v7 implements .margin(x[B,784]) -> [B])
  .prob(x[B,D])   -> [B,K] f32          (RBF: raw exp-quadratic per-class probabilities, rows
                                         need NOT sum to 1 — §4.4)
  .loss(x[B,D], y[B]) -> scalar
  .predict(x) -> [B] int64 ;  .confidence(x) -> [B] f32 = max_k prob
  SoftmaxRegression(784,10) ; LogisticRegression3v7(784)
  MaxoutMLP(units_per_layer:int, layers:int, pieces:int, dropout:dict)   # 240/1600 variants
  MaxoutSigmoid(...)          # maxout backbone + 10 independent sigmoid heads
  RBFNet(K=10, D=784)         # §4.4 parametrization
  Ensemble(members:list)      # mean-prob combine
  ConvMaxoutCIFAR(...)        # arch logged; paper silent

attack.py
  fgsm(model, x[B,D], y[B], eps) -> x_adv[B,D]                    # no clipping (§4.9)
  fgsm_logits_trace(model, x[D], y, eps_grid[21]) -> logits[21,K] # direction fixed at eps=0
  targeted_fool_step(model, x0[B,D], target:int, eps) -> x1[B,D]  # sign step (eq. 8)

train.py
  train(model, data, cfg) -> history
    cfg fields: adversarial: None | {alpha:0.5, eps:0.25};  noise: None | {type, eps};
    l1_first_layer: None | 0.0025;
    early_stop: None | {monitor: "val_err"|"adv_val_err", patience:100};
    retrain_full_60k: bool.
    history logs per epoch: epoch, train_err, val_err, adv_val_err.

eval.py
  error(model, x, y) -> percent
  confidence_stats(model, x, y) -> {"conf_all": percent, "conf_mistakes": percent}
  rubbish_eval(model, dim, n, seed) -> {"rubbish_err", "rubbish_conf_mistakes",
        "rubbish_class_shares": {0..9: percent of mistakes predicted as class k}}
  agreement(modelA, modelB, x_adv, y) -> {"agree_all", "agree_cond"}
        # label agreement on A's mistakes; _cond conditioned on both models wrong
```

**Metric keys per arm** (emitted per seed):

- `softmax_reg`: clean_err, adv_err, adv_conf_all, adv_conf_mistakes, rubbish_err,
  rubbish_conf_mistakes, rubbish_class_shares
- `logreg_3v7`: clean_err, adv_err, train_err, **analytic_equiv_max_absdiff** (c07 check)
- `maxout_naive`: clean_err, train_err, adv_err, adv_conf_all, adv_conf_mistakes,
  rubbish_err, rubbish_conf_mistakes, rubbish_class_shares
- `maxout_sigmoid`: clean_err, rubbish_err, rubbish_conf_mistakes
- `maxout_adv` (240u), `maxout_large_naive` (1600u): clean_err, adv_err, adv_conf_all,
  adv_conf_mistakes
- `maxout_large_adv` (1600u, **5 seeds**): clean_err, adv_err, adv_conf_all,
  adv_conf_mistakes, epochs_run
- `noise_rademacher`, `noise_uniform`: clean_err, adv_err, adv_conf_all, adv_conf_mistakes
- `rbf_shallow`: clean_err, clean_conf_all, adv_err, adv_conf_mistakes, rubbish_err
- `ensemble12`: clean_err, adv_err_ensemble_crafted, adv_err_single_crafted
- `agreement_mnist`: agree_softmax_all, agree_softmax_cond, agree_rbf_all, agree_rbf_cond,
  agree_rbf_on_softmax
- `transfer_mnist`: err_orig_on_advfromnew, err_new_on_advfromorig
  (orig = maxout_large_naive, new = maxout_large_adv, ε = 0.25)
- `eps_trace`: eps_grid [−10..10], logit_correct_seq[21], logit_maxwrong_seq[21],
  margin_seq[21] (= correct − maxwrong), example_index, example_true_label
- `cifar_conv_maxout`: clean_err, adv_err (ε=0.1), adv_conf_mistakes, rubbish_err,
  rubbish_conf_mistakes, rubbish_class_shares, fool_success {class 0..9: percent},
  fool_success_avg
- `l1_maxout`: train_err

## 6. Arms this paper compares

The comparison the paper itself makes: shallow softmax regression, logistic regression (3-vs-7),
naive maxout, adversarially trained maxout (two sizes), two noise controls, L¹ weight decay,
shallow RBF, sigmoid-top maxout, a 12-member ensemble, a conv maxout on CIFAR-10, plus an
MP-DBM (§9, not built) and the GoogLeNet/ImageNet Fig. 1 demo (§9, not built).

| arm | dataset | arch/config | training | eval |
|---|---|---|---|---|
| `softmax_reg` | MNIST [0,1] | linear 784→10 softmax | NLL to convergence | clean; FGSM ε=0.25; rubbish N(0,I₇₈₄) |
| `logreg_3v7` | MNIST 3v7 | linear sigmoid, y∈{±1} | softplus margin (eq. 3) | clean 1.6%; FGSM ε=0.25 → 99%; analytic check |
| `maxout_naive` | MNIST | maxout MLP, 240 units/layer, dropout | early stop on val err, patience 100 (run cap 8) | clean 0.94%; FGSM ε=0.25 → 89.4%/97.6%; rubbish |
| `maxout_adv` | MNIST | same + adv training α=0.5, ε=0.25 (FGSM in eval mode) | same | clean 0.84%; FGSM |
| `maxout_large_naive` | MNIST | 1600 units/layer, no adv training | early stop val err (run cap 8) | clean 1.14%; FGSM (recalled 89.4%) |
| `maxout_large_adv` | MNIST | 1600 units/layer + adv training | early stop on **adv val err** (run cap 8); retrain on all 60,000 from scratch; **5 seeds** | clean: 4×0.77, 1×0.83, mean 0.782; FGSM 17.9%, conf-mistakes 81.4% |
| `maxout_sigmoid` | MNIST | maxout backbone + 10 independent sigmoid outputs, BCE top | as naive | rubbish 68%/87.9% |
| `noise_rademacher` | MNIST | maxout_naive config | + per-pixel ±0.25 noise, resampled per minibatch | FGSM 86.2%/97.3% |
| `noise_uniform` | MNIST | maxout_naive config | + per-pixel U(−0.25,0.25) noise | FGSM 90.4%/97.8% |
| `l1_maxout` | MNIST | maxout_naive config | + L¹ coef 0.0025 on **first layer** | train err > 5% (negative control) |
| `rbf_shallow` | MNIST | 10 RBF units, no hidden layer (§4.4) | NLL | FGSM 55.4%/conf-mistakes 1.2%, clean conf 60.6%, rubbish 0% |
| `ensemble12` | MNIST | 12 × maxout_naive, distinct seeds | per-member as naive | FGSM(ε=.25): 91.1% ensemble-crafted, 87.9% single-crafted |
| `agreement_mnist` | MNIST | reuses maxout_naive + softmax_reg + rbf_shallow | — | 54.6 / 16.0 / 84.6 / 54.3 / 53.6% |
| `transfer_mnist` | MNIST | reuses maxout_large_naive ↔ maxout_large_adv | — | 19.6% / 40.9% |
| `eps_trace` | MNIST | reuses maxout_naive (per seed); fixed first-common-correct class-4 example | — | Fig. 4 curves, ε ∈ [−10,10] |
| `cifar_conv_maxout` | CIFAR-10 GCN | conv maxout, no post-ReLU (arch ours, logged); fooling 1,000/class | to convergence (run cap patience 5) | FGSM ε=0.1 → 87.15%/96.6%; rubbish 93.4%/84.4%; fooling 75.3% avg |
| ~~`mp_dbm`~~ | MNIST | MP-DBM | — | **not built** (§9) |
| ~~`googlenet_imagenet`~~ | ImageNet | GoogLeNet | — | **not built** (§9) |

## 7. Figure readings

Tool: `read-figure <png> "<question>"`, transcript appended to `figures/read-figure.jsonl`
(committed). `eps_curve.pdf` was rasterized at 4× with pymupdf before reading (the reader takes
PNGs). Questions were phrased for terse answers ("answer with only the number", "exactly one
word"); **none of this run's 26 exchanges is flagged as deliberation**. Empty answers
(the model declines some counting questions) are recorded as-is. The 11 earlier entries in the
same transcript — from the prior run, several flagged `looks_like_reasoning` by inspection —
are retained for history but are not used as evidence below.

**Figure 4 left (`eps_curve.pdf`)** — the one figure carrying quantities the prose does not.
Reads this run:

- x-axis tick range: **−15 to 15**; y-axis ticks top→bottom: **1000, 500, 0, −500, −1000,
  −1500, −2000** ("argument to softmax").
- Line shape: **"piecewise-linear"** (one word) — matches the caption's "conspicuously piecewise
  linear with ε" (`iclr2015.tex:768`).
- "Between x=0 and x=15, does any other curve end up above the curve labeled 4?" → **yes** —
  matches "wrong classifications are stable across a wide region of ε values" (`:769`).
- At ε = +10: highest curve ≈ **+400**; curve labeled 4 ≈ **−400**.
- At ε = −10: highest curve ≈ **+800**; curve labeled 4 ≈ **+200** — i.e. the correct-class
  curve is below the top wrong-class curve at **both** tails: the thin-manifold claim
  ("Correct classifications occur only on a thin manifold where x occurs in the data", `:764`).
- At x=0 the curves are bunched and the color→legend mapping could not be read reliably
  (one read answered "no" to "is the curve labeled 4 topmost", follow-ups returned empty) —
  weak evidence, not used: the corresponding claim (c65) is now rated `low` and evaluated on a
  **deterministic** first-common-correct class-4 example (no predicate selection), so it can
  genuinely fail and does not depend on reading the paper's exact example.

These enter curve claims c65–c70; the `matches` anchors c69 (+400 at ε=+10) and c70 (−400 at
ε=+10) are this run's own reads, rated low compute-invariance with tolerances covering reading
error and reproduction spread.

**Figure 4 right (`eps_curve_inputs.png`)** — thumbnail montage of x + ε·sign(∇J) (yellow box
= still correctly classified, `:771-772`). Reads: grid **10×10**; "are the yellow-boxed
thumbnails clustered near the middle of the grid rather than at the top or bottom edges?" →
**yes** (row-major ε from most negative top-left to most positive bottom-right, so correct
classification survives only near ε ≈ 0 — the thin-manifold claim again). Count of yellow boxes
itself: the reader returned empty twice; not used. This claim is subsumed by the (now `low`)
curve claims c65–c67 and is not separately gated.

**Figure 5 (`airplane.png`)** — targeted "airplane" fooling montage. Reads: grid **10×10**
(100 samples); "do some thumbnails have a yellow border and others not?" → **yes**, consistent
with a per-step success rate below 100% for the hardest class (24.7%, `:940`); "do the
thumbnails look like clearly recognizable photographs of airplanes?" → **no** — consistent with
the caption's claim that these are fooling images, not rendered real airplanes (`:943-945,
956-958`). The gated content is c61–c63 on the CIFAR arm; the montage itself adds no number.

**Figure 2 (`logreg_clean.png`, `logreg_adv.png`, `logreg_weights_sign.png`)** — "do these
images look like readable handwritten digits?" on panel (d) (FGSM ε=.25 adversarial examples) →
**yes** — the caption's claim that humans read such images without difficulty (`:336-337`,
panel c/d). "Does `sign(w)` show a clearly recognizable handwritten digit shape?" → **no** —
the caption's "not readily recognizable as having anything to do with the relationship between
3s and 7s" (`:450-453`). Qualitative; not gated (§9), recorded as evidence.

**Figure 3 (`naive_weights.png`, `adv_weights.png`)** — "do the filter rows look mostly like
localized pen-stroke fragments of handwritten digits?": naive → **no**, adversarially trained →
**yes** — the paper's "significantly more localized and interpretable" (`:523-524`).
Qualitative; not gated (§9), recorded as evidence.

## 8. claims.json

The gate: claims with `compute_invariance: "high"` are the gating set — directions and orderings
with wide margins, curve shapes, and the §1.3 algebraic invariant; they survive a budget smaller
than the paper's (smaller epoch counts, CPU). Tight value claims (clean 0.94%, 0.782%) need the
paper's full GPU budget and are rated low/medium. Seeds: top-level `[0, 1, 2]` is the default;
`maxout_large_adv` claims (c14, c18–c20) override to five seeds `[0..4]`, matching the paper's
five runs (`iclr2015.tex:508-510`); per seed, three RNG streams — weight init, minibatch order,
dropout masks (`iclr2015.tex:506-508`). `quantity` expressions over `measured.<arm>.<metric>`
are evaluated at **every seed in the claim**; value claims compare the **mean over seeds**
against `claimed` within `tolerance`; ordering/invariant/existence claims must hold at every
seed; curve comparisons sample `quantity` — and for `above`/`below`/`crosses` also the curve
named by `against` — at `x`, with diffs = quantity − against: `above`/`below` = all sampled
diffs > 0 / < 0; `crosses` = first sampled diff > 0 and last < 0; `increasing` = last > first
with ≥ 80% of consecutive diffs ≥ 0; `matches` = elementwise within `tolerance`.
`claims.json` at the repo root is byte-identical to this block (verified by script).

```json
{
  "paper": "Goodfellow, Shlens & Szegedy, Explaining and Harnessing Adversarial Examples, ICLR 2015 (arXiv:1412.6572v3)",
  "source_of_truth": "paper/source/iclr2015.tex; every citation is grep-verified <file>:<line>",
  "metric_convention": "all errors/confidences/agreements are percents in [0,100], matching the paper's printed numbers",
  "seeds": [
    0,
    1,
    2
  ],
  "seed_protocol": {
    "seeds_default": [
      0,
      1,
      2
    ],
    "maxout_large_adv": [
      0,
      1,
      2,
      3,
      4
    ],
    "rngs_per_seed": [
      "weight_init",
      "minibatch_order",
      "dropout_masks"
    ],
    "rngs_citation": "paper/source/iclr2015.tex:506-508"
  },
  "evaluation_rules": {
    "value": "mean of measured.<arm>.<metric> over the claim's seeds must satisfy |mean - claimed| <= tolerance",
    "ordering": "the quantity expression must satisfy direction at EVERY seed of the claim",
    "invariant": "the boolean predicate must be true at every seed",
    "existence": "the boolean predicate must be true (at every seed unless the predicate says otherwise)",
    "curve": "quantity is a sequence stored by the arm; sample it at x; for above/below/crosses the field against names the other curve, sampled at the same x, and diffs = quantity - against: above = all sampled diffs >0; below = all sampled diffs <0; crosses = first sampled diff >0 and last sampled diff <0; increasing = last>first with >=80% of consecutive diffs >=0; matches = elementwise |sampled-claimed| <= tolerance",
    "gate": "only claims with compute_invariance == \"high\" gate the reproduction; medium/low are reported"
  },
  "arms": {
    "softmax_reg": {
      "dataset": "mnist [0,1]",
      "arch": "linear 784->10 softmax",
      "train": "NLL to convergence",
      "eval": "clean; FGSM eps=0.25; rubbish N(0,I784)"
    },
    "logreg_3v7": {
      "dataset": "mnist 3-vs-7 subset, y in {-1,+1}, +1=digit 3",
      "arch": "linear sigmoid",
      "train": "mean softplus margin (tex:402)",
      "eval": "clean; FGSM eps=0.25; analytic-equivalence check"
    },
    "maxout_naive": {
      "dataset": "mnist [0,1]",
      "arch": "maxout MLP 240 units/layer, dropout (pieces/rates: paper silent, see SPEC 4)",
      "train": "early stop on val_err, patience 100 (tex:501-503)",
      "eval": "clean; FGSM eps=0.25; rubbish"
    },
    "maxout_adv": {
      "dataset": "mnist",
      "arch": "maxout_naive config",
      "train": "adversarial training alpha=0.5 eps=0.25 (tex:486-488)",
      "eval": "clean; FGSM eps=0.25"
    },
    "maxout_large_naive": {
      "dataset": "mnist",
      "arch": "maxout MLP 1600 units/layer (tex:498)",
      "train": "early stop val_err",
      "eval": "clean (claimed 1.14, tex:500); FGSM"
    },
    "maxout_large_adv": {
      "dataset": "mnist",
      "arch": "1600 units/layer + adversarial training",
      "train": "early stop on adversarial validation error (tex:504-505); retrain on all 60,000 (tex:505-506)",
      "eval": "clean 5 seeds; FGSM eps=0.25"
    },
    "maxout_sigmoid": {
      "dataset": "mnist",
      "arch": "maxout_naive backbone + 10 independent sigmoid outputs, BCE top (tex:909)",
      "train": "as maxout_naive",
      "eval": "rubbish"
    },
    "noise_rademacher": {
      "dataset": "mnist",
      "arch": "maxout_naive config",
      "train": "+ per-pixel eps*{-1,+1} noise, resampled per minibatch (tex:555-556)",
      "eval": "FGSM eps=0.25"
    },
    "noise_uniform": {
      "dataset": "mnist",
      "arch": "maxout_naive config",
      "train": "+ per-pixel U(-0.25,0.25) noise (tex:556)",
      "eval": "FGSM eps=0.25"
    },
    "l1_maxout": {
      "dataset": "mnist",
      "arch": "maxout_naive config",
      "train": "+ L1 coef 0.0025 on FIRST layer (tex:429-430)",
      "eval": "train_err"
    },
    "rbf_shallow": {
      "dataset": "mnist [0,1]",
      "arch": "10 independent RBF units exp((x-mu)'beta(x-mu)), beta negative-semidefinite (SPEC 4.4; tex:595)",
      "train": "NLL",
      "eval": "clean conf; FGSM eps=0.25; rubbish"
    },
    "ensemble12": {
      "dataset": "mnist",
      "arch": "12 x maxout_naive, distinct seeds (tex:819-821)",
      "combine": "mean probability",
      "eval": "FGSM eps=0.25 vs (a) full-ensemble gradient (b) member-0 gradient"
    },
    "agreement_mnist": {
      "dataset": "mnist",
      "reuses": [
        "maxout_naive",
        "softmax_reg",
        "rbf_shallow"
      ],
      "eval": "label agreement on FGSM (eps=0.25) examples misclassified by maxout_naive; *_cond conditioned on both models wrong"
    },
    "transfer_mnist": {
      "dataset": "mnist",
      "pair": [
        "maxout_large_naive (orig)",
        "maxout_large_adv (new)"
      ],
      "eval": "FGSM eps=0.25 generated on one model, error measured on the other (tex:518-520)"
    },
    "eps_trace": {
      "dataset": "mnist test",
      "reuses": "maxout_naive of same seed",
      "example": "first class-4 test example correctly classified by ALL seed models (deterministic; no selection on the thin-manifold predicates)",
      "protocol": "train each seed's maxout network; take the lowest-index class-4 test example all seeds classify correctly; FGSM direction computed once at eps=0; sweep eps=-10..10 step 1; record the [21,10] logits (tex:762-770)"
    },
    "cifar_conv_maxout": {
      "dataset": "cifar-10, GCN preprocessed to std~0.5 (tex:343-345)",
      "arch": "conv maxout (arch ours, paper silent)",
      "train": "to convergence",
      "eval": "FGSM eps=0.1; rubbish N(0,I3072) 1000 samples; targeted fooling sign-step (tex:953-955)"
    },
    "mp_dbm": {
      "status": "NOT BUILT - see not_tested"
    },
    "googlenet_imagenet": {
      "status": "NOT BUILT - see not_tested"
    }
  },
  "claims": [
    {
      "id": "c01",
      "kind": "value",
      "arm": "softmax_reg",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.softmax_reg.adv_err",
      "claimed": 99.9,
      "tolerance": 2.0,
      "quote": "a shallow softmax classifier to have an error rate of 99.9\\% with an average confidence of",
      "citation": "paper/source/iclr2015.tex:333"
    },
    {
      "id": "c02",
      "kind": "value",
      "arm": "softmax_reg",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.softmax_reg.adv_conf_all",
      "claimed": 79.3,
      "tolerance": 8.0,
      "note": "denominator all vs mistakes unstated (SPEC 4.10); with 99.9% error the two nearly coincide",
      "quote": "79.3\\% on the MNIST",
      "citation": "paper/source/iclr2015.tex:334"
    },
    {
      "id": "c03",
      "kind": "ordering",
      "arm": "softmax_reg",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.softmax_reg.adv_err - measured.softmax_reg.clean_err",
      "direction": ">0",
      "quote": "We find that this method reliably causes a wide variety of models to misclassify their input.",
      "citation": "paper/source/iclr2015.tex:331"
    },
    {
      "id": "c04",
      "kind": "value",
      "arm": "logreg_3v7",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.logreg_3v7.clean_err",
      "claimed": 1.6,
      "tolerance": 0.8,
      "quote": "c) MNIST 3s and 7s. The logistic regression model has a 1.6\\% error rate on the 3 versus 7 discrimination task on these examples.",
      "citation": "paper/source/iclr2015.tex:454"
    },
    {
      "id": "c05",
      "kind": "value",
      "arm": "logreg_3v7",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.logreg_3v7.adv_err",
      "claimed": 99.0,
      "tolerance": 2.0,
      "quote": "The logistic regression model has an error rate of 99\\% on these examples.",
      "citation": "paper/source/iclr2015.tex:456"
    },
    {
      "id": "c06",
      "kind": "ordering",
      "arm": "logreg_3v7",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.logreg_3v7.adv_err - measured.logreg_3v7.clean_err",
      "direction": ">0",
      "quote": "The logistic regression model has an error rate of 99\\% on these examples.",
      "citation": "paper/source/iclr2015.tex:456"
    },
    {
      "id": "c07",
      "kind": "invariant",
      "arm": "logreg_3v7",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "predicate": "measured.logreg_3v7.analytic_equiv_max_absdiff < 1e-5",
      "note": "| mean_i zeta(-y_i*(w.(x_adv_i) + b)) - mean_i zeta(eps*||w||_1 - y_i*(w.x_i + b)) | on a fixed batch, where x_adv = attack.fgsm(m,x,y,eps) is the REAL gradient-based per-example FGSM (sign(grad_x J) = -y*sign(w) for logreg, so x_adv_i = x_i - eps*y_i*sign(w)). FGSM is exact for logreg. NOTE on the paper sign slip: tex:407 states the gradient sign is -sign(w) (dropping the y factor) and tex:411 gives the closed form zeta(y*(eps*||w||_1 - w.x - b)); that holds only for y=+1. The true worst-case closed form is zeta(eps*||w||_1 - y*(w.x+b)), which the per-example FGSM realizes exactly; we check against THIS form (not the paper tex:411 form, which would fail for y=-1). The invariant exercises the real attack.fgsm, so a sign bug in either the attack or the closed form breaks it (discriminating).",
      "quote": "Note that the sign of the gradient is just $- \\sign(\\vw)$, and that $\\vw^\\top \\sign(\\vw) = ||\\vw||_1$.",
      "citation": "paper/source/iclr2015.tex:407"
    },
    {
      "id": "c08",
      "kind": "value",
      "arm": "maxout_naive",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_naive.clean_err",
      "claimed": 0.94,
      "tolerance": 0.3,
      "quote": "able to reduce the error rate from 0.94\\% without adversarial training to 0.84\\% with adversarial",
      "citation": "paper/source/iclr2015.tex:493"
    },
    {
      "id": "c09",
      "kind": "value",
      "arm": "maxout_naive",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.maxout_naive.adv_err",
      "claimed": 89.4,
      "tolerance": 8.0,
      "quote": "}. In the same setting, a maxout network misclassifies 89.4\\%",
      "citation": "paper/source/iclr2015.tex:338"
    },
    {
      "id": "c10",
      "kind": "value",
      "arm": "maxout_naive",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_naive.adv_conf_all",
      "claimed": 97.6,
      "tolerance": 8.0,
      "quote": "of our adversarial examples with an average confidence of 97.6\\%.",
      "citation": "paper/source/iclr2015.tex:339"
    },
    {
      "id": "c11",
      "kind": "value",
      "arm": "maxout_adv",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_adv.clean_err",
      "claimed": 0.84,
      "tolerance": 0.25,
      "quote": "able to reduce the error rate from 0.94\\% without adversarial training to 0.84\\% with adversarial",
      "citation": "paper/source/iclr2015.tex:493"
    },
    {
      "id": "c12",
      "kind": "ordering",
      "arm": "maxout_adv",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.maxout_naive.clean_err - measured.maxout_adv.clean_err",
      "direction": ">0",
      "note": "margin in the paper is only 0.1pp, so rated medium not high",
      "quote": "able to reduce the error rate from 0.94\\% without adversarial training to 0.84\\% with adversarial",
      "citation": "paper/source/iclr2015.tex:493"
    },
    {
      "id": "c13",
      "kind": "ordering",
      "arm": "maxout_adv",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.maxout_naive.adv_err - measured.maxout_adv.adv_err",
      "direction": ">0",
      "note": "same-architecture version of 89.4% -> 17.9%; the paper's own numbers cross architectures",
      "quote": "examples based on the fast gradient sign method. With adversarial training, the error rate",
      "citation": "paper/source/iclr2015.tex:516"
    },
    {
      "id": "c14",
      "kind": "value",
      "arm": "maxout_large_adv",
      "seeds": [
        0,
        1,
        2,
        3,
        4
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_large_adv.adv_err",
      "claimed": 17.9,
      "tolerance": 8.0,
      "quote": "fell to 17.9\\%. Adversarial examples are transferable between the two models but with the",
      "citation": "paper/source/iclr2015.tex:517"
    },
    {
      "id": "c15",
      "kind": "value",
      "arm": "maxout_large_naive",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.maxout_large_naive.adv_err",
      "claimed": 89.4,
      "tolerance": 10.0,
      "note": "paper recalls the 89.4% figure for 'this same kind of model' (the 1600-unit one)",
      "quote": "adversarial training, this same kind of model had an error rate of 89.4\\% on adversarial",
      "citation": "paper/source/iclr2015.tex:515"
    },
    {
      "id": "c16",
      "kind": "ordering",
      "arm": "maxout_large_adv",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.maxout_large_naive.adv_err - measured.maxout_large_adv.adv_err",
      "direction": ">0",
      "quote": "fell to 17.9\\%. Adversarial examples are transferable between the two models but with the",
      "citation": "paper/source/iclr2015.tex:517"
    },
    {
      "id": "c17",
      "kind": "value",
      "arm": "maxout_large_naive",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_large_naive.clean_err",
      "claimed": 1.14,
      "tolerance": 0.3,
      "quote": "and get an error rate of 1.14\\% on the test set. With adversarial training, we found that",
      "citation": "paper/source/iclr2015.tex:500"
    },
    {
      "id": "c18",
      "kind": "existence",
      "arm": "maxout_large_adv",
      "seeds": [
        0,
        1,
        2,
        3,
        4
      ],
      "compute_invariance": "low",
      "predicate": "measured.maxout_large_adv.clean_err <= 1.1",
      "quote": "four trials that each had an error rate of 0.77\\% on the test set and one trial that had",
      "citation": "paper/source/iclr2015.tex:509",
      "note": "reformulated from the paper's \"4 trials at 0.77%, 1 at 0.83%\" (which used disallowed count/mean) to the evaluable core: every trial's clean error <= 1.1% (all five paper trials are 0.77-0.83%). The predicate is per-seed (existence = holds at every seed = every trial's clean err <= 1.1%); an earlier `max(measured...clean_err)` wrapper made the gate resolve a single float per seed and fail with `TypeError: 'float' object is not iterable` (gate verdict unevaluable). Our sub-scale CPU 1600-unit adv model reaches ~1.8% so this is refuted, documenting the training-budget gap rather than gating."
    },
    {
      "id": "c19",
      "kind": "value",
      "arm": "maxout_large_adv",
      "seeds": [
        0,
        1,
        2,
        3,
        4
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_large_adv.clean_err",
      "claimed": 0.782,
      "tolerance": 0.15,
      "quote": "an error rate of 0.83\\%. The average of 0.782\\% is the best result reported on the permutation",
      "citation": "paper/source/iclr2015.tex:510"
    },
    {
      "id": "c20",
      "kind": "value",
      "arm": "maxout_large_adv",
      "seeds": [
        0,
        1,
        2,
        3,
        4
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_large_adv.adv_conf_mistakes",
      "claimed": 81.4,
      "tolerance": 12.0,
      "quote": "example was 81.4\\%. We also found that the weights of the learned model changed significantly,",
      "citation": "paper/source/iclr2015.tex:523"
    },
    {
      "id": "c21",
      "kind": "value",
      "arm": "transfer_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.transfer_mnist.err_orig_on_advfromnew",
      "claimed": 40.9,
      "tolerance": 10.0,
      "quote": "adversarial examples generated via the new model yield an error rate of 40.9\\% on the original",
      "citation": "paper/source/iclr2015.tex:520"
    },
    {
      "id": "c22",
      "kind": "value",
      "arm": "transfer_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.transfer_mnist.err_new_on_advfromorig",
      "claimed": 19.6,
      "tolerance": 8.0,
      "quote": "the original model yield an error rate of 19.6\\% on the adversarially trained model, while",
      "citation": "paper/source/iclr2015.tex:519"
    },
    {
      "id": "c23",
      "kind": "ordering",
      "arm": "transfer_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.transfer_mnist.err_orig_on_advfromnew - measured.transfer_mnist.err_new_on_advfromorig",
      "direction": ">0",
      "quote": "adversarially trained model showing greater robustness. Adversarial examples generated via",
      "citation": "paper/source/iclr2015.tex:518"
    },
    {
      "id": "c24",
      "kind": "value",
      "arm": "noise_rademacher",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.noise_rademacher.adv_err",
      "claimed": 86.2,
      "tolerance": 8.0,
      "quote": "to each pixel, or adding noise in $U(-\\eps, \\eps)$ to each pixel. These obtained an error rate of 86.2\\% with confidence",
      "citation": "paper/source/iclr2015.tex:556"
    },
    {
      "id": "c25",
      "kind": "value",
      "arm": "noise_rademacher",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.noise_rademacher.adv_conf_all",
      "claimed": 97.3,
      "tolerance": 8.0,
      "quote": "97.3\\% and an error rate of 90.4\\% with a confidence of 97.8\\% respectively on fast gradient sign adversarial examples.",
      "citation": "paper/source/iclr2015.tex:557"
    },
    {
      "id": "c26",
      "kind": "value",
      "arm": "noise_uniform",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.noise_uniform.adv_err",
      "claimed": 90.4,
      "tolerance": 8.0,
      "quote": "97.3\\% and an error rate of 90.4\\% with a confidence of 97.8\\% respectively on fast gradient sign adversarial examples.",
      "citation": "paper/source/iclr2015.tex:557"
    },
    {
      "id": "c27",
      "kind": "value",
      "arm": "noise_uniform",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.noise_uniform.adv_conf_all",
      "claimed": 97.8,
      "tolerance": 8.0,
      "quote": "97.3\\% and an error rate of 90.4\\% with a confidence of 97.8\\% respectively on fast gradient sign adversarial examples.",
      "citation": "paper/source/iclr2015.tex:557"
    },
    {
      "id": "c28",
      "kind": "ordering",
      "arm": "noise_uniform",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "min(measured.noise_rademacher.adv_err, measured.noise_uniform.adv_err) - measured.maxout_large_adv.adv_err",
      "direction": ">0",
      "quote": "zero mean and zero covariance is very inefficient at preventing adversarial examples. The expected dot product",
      "citation": "paper/source/iclr2015.tex:550"
    },
    {
      "id": "c29",
      "kind": "value",
      "arm": "rbf_shallow",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.rbf_shallow.adv_err",
      "claimed": 55.4,
      "tolerance": 15.0,
      "quote": "rate of 55.4\\% on MNIST using adversarial examples generated with the fast gradient sign",
      "citation": "paper/source/iclr2015.tex:602"
    },
    {
      "id": "c30",
      "kind": "value",
      "arm": "rbf_shallow",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.rbf_shallow.adv_conf_mistakes",
      "claimed": 1.2,
      "tolerance": 2.0,
      "quote": "method and $\\eps = .25$. However, its confidence on mistaken examples is only $1.2\\%$.",
      "citation": "paper/source/iclr2015.tex:603"
    },
    {
      "id": "c31",
      "kind": "value",
      "arm": "rbf_shallow",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.rbf_shallow.clean_conf_all",
      "claimed": 60.6,
      "tolerance": 15.0,
      "quote": "Its average confidence on clean test examples is $60.6$\\%.",
      "citation": "paper/source/iclr2015.tex:604"
    },
    {
      "id": "c32",
      "kind": "ordering",
      "arm": "rbf_shallow",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.maxout_naive.adv_conf_mistakes - measured.rbf_shallow.adv_conf_mistakes",
      "direction": ">0",
      "quote": "RBF networks are naturally immune to adversarial examples, in the sense that they have low",
      "citation": "paper/source/iclr2015.tex:600"
    },
    {
      "id": "c33",
      "kind": "ordering",
      "arm": "rbf_shallow",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.rbf_shallow.clean_conf_all - measured.rbf_shallow.adv_conf_mistakes",
      "direction": ">0",
      "quote": "method and $\\eps = .25$. However, its confidence on mistaken examples is only $1.2\\%$.",
      "citation": "paper/source/iclr2015.tex:603"
    },
    {
      "id": "c34",
      "kind": "value",
      "arm": "ensemble12",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.ensemble12.adv_err_ensemble_crafted",
      "claimed": 91.1,
      "tolerance": 8.0,
      "quote": "gradient descent. The ensemble gets an error rate of 91.1\\% on adversarial examples designed",
      "citation": "paper/source/iclr2015.tex:822"
    },
    {
      "id": "c35",
      "kind": "value",
      "arm": "ensemble12",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.ensemble12.adv_err_single_crafted",
      "claimed": 87.9,
      "tolerance": 8.0,
      "quote": "member of the ensemble, the error rate falls to 87.9\\%. Ensembling provides only",
      "citation": "paper/source/iclr2015.tex:824"
    },
    {
      "id": "c36",
      "kind": "ordering",
      "arm": "ensemble12",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.ensemble12.adv_err_ensemble_crafted - measured.ensemble12.clean_err",
      "direction": ">0",
      "quote": "gradient descent. The ensemble gets an error rate of 91.1\\% on adversarial examples designed",
      "citation": "paper/source/iclr2015.tex:822"
    },
    {
      "id": "c37",
      "kind": "ordering",
      "arm": "ensemble12",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.ensemble12.adv_err_ensemble_crafted - measured.ensemble12.adv_err_single_crafted",
      "direction": ">0",
      "note": "paper margin only 3.2pp, so medium",
      "quote": "member of the ensemble, the error rate falls to 87.9\\%. Ensembling provides only",
      "citation": "paper/source/iclr2015.tex:824"
    },
    {
      "id": "c38",
      "kind": "value",
      "arm": "agreement_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.agreement_mnist.agree_softmax_all",
      "claimed": 54.6,
      "tolerance": 10.0,
      "quote": "predict the maxout network's class correctly 54.6\\% of the time. These numbers are largely",
      "citation": "paper/source/iclr2015.tex:683"
    },
    {
      "id": "c39",
      "kind": "value",
      "arm": "agreement_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.agreement_mnist.agree_rbf_all",
      "claimed": 16.0,
      "tolerance": 8.0,
      "quote": "the maxout network's class assignment only 16.0\\% of the time, while the softmax classifier",
      "citation": "paper/source/iclr2015.tex:682"
    },
    {
      "id": "c40",
      "kind": "value",
      "arm": "agreement_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.agreement_mnist.agree_softmax_cond",
      "claimed": 84.6,
      "tolerance": 10.0,
      "quote": "predict's maxout's class 84.6\\% of the time, while the RBF network is able to predict maxout's",
      "citation": "paper/source/iclr2015.tex:686"
    },
    {
      "id": "c41",
      "kind": "value",
      "arm": "agreement_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.agreement_mnist.agree_rbf_cond",
      "claimed": 54.3,
      "tolerance": 12.0,
      "quote": "class only 54.3\\% of the time. For comparison, the RBF network can predict softmax regression's",
      "citation": "paper/source/iclr2015.tex:687"
    },
    {
      "id": "c42",
      "kind": "value",
      "arm": "agreement_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.agreement_mnist.agree_rbf_on_softmax",
      "claimed": 53.6,
      "tolerance": 12.0,
      "note": "conditioning ambiguous (SPEC 4.17); read as both-wrong conditioned",
      "quote": "class 53.6\\% of the time, so it does have a strong linear component to its own behavior.",
      "citation": "paper/source/iclr2015.tex:688"
    },
    {
      "id": "c43",
      "kind": "ordering",
      "arm": "agreement_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.agreement_mnist.agree_softmax_all - measured.agreement_mnist.agree_rbf_all",
      "direction": ">0",
      "quote": "predict the maxout network's class correctly 54.6\\% of the time. These numbers are largely",
      "citation": "paper/source/iclr2015.tex:683"
    },
    {
      "id": "c44",
      "kind": "ordering",
      "arm": "agreement_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.agreement_mnist.agree_softmax_cond - measured.agreement_mnist.agree_rbf_cond",
      "direction": ">0",
      "quote": "predict's maxout's class 84.6\\% of the time, while the RBF network is able to predict maxout's",
      "citation": "paper/source/iclr2015.tex:686"
    },
    {
      "id": "c45",
      "kind": "existence",
      "arm": "agreement_mnist",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "predicate": "measured.agreement_mnist.agree_softmax_cond > 40",
      "note": "'a significant proportion ... consistent with linear behavior'; 40 is far above 10-class chance (~10) and far below the paper's 84.6",
      "quote": "predict's maxout's class 84.6\\% of the time, while the RBF network is able to predict maxout's",
      "citation": "paper/source/iclr2015.tex:686"
    },
    {
      "id": "c46",
      "kind": "value",
      "arm": "maxout_naive",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.maxout_naive.rubbish_err",
      "claimed": 98.35,
      "tolerance": 5.0,
      "quote": "of 98.35\\% on Gaussian rubbish examples with an average confidence of 92.8\\% on mistakes.",
      "citation": "paper/source/iclr2015.tex:908"
    },
    {
      "id": "c47",
      "kind": "value",
      "arm": "maxout_naive",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_naive.rubbish_conf_mistakes",
      "claimed": 92.8,
      "tolerance": 10.0,
      "quote": "of 98.35\\% on Gaussian rubbish examples with an average confidence of 92.8\\% on mistakes.",
      "citation": "paper/source/iclr2015.tex:908"
    },
    {
      "id": "c48",
      "kind": "value",
      "arm": "maxout_sigmoid",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_sigmoid.rubbish_err",
      "claimed": 68.0,
      "tolerance": 15.0,
      "quote": "Changing the top layer to independent sigmoids dropped the error rate to 68\\% with an average",
      "citation": "paper/source/iclr2015.tex:909"
    },
    {
      "id": "c49",
      "kind": "value",
      "arm": "maxout_sigmoid",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_sigmoid.rubbish_conf_mistakes",
      "claimed": 87.9,
      "tolerance": 10.0,
      "quote": "confidence on mistakes of 87.9\\%.",
      "citation": "paper/source/iclr2015.tex:910"
    },
    {
      "id": "c50",
      "kind": "value",
      "arm": "softmax_reg",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.softmax_reg.rubbish_err",
      "claimed": 59.8,
      "tolerance": 15.0,
      "quote": "A softmax regression model has an error rate of 59.8\\%",
      "citation": "paper/source/iclr2015.tex:920"
    },
    {
      "id": "c51",
      "kind": "value",
      "arm": "softmax_reg",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.softmax_reg.rubbish_conf_mistakes",
      "claimed": 70.8,
      "tolerance": 15.0,
      "quote": "on the rubbish examples, with an average confidence on mistakes of 70.8\\%.",
      "citation": "paper/source/iclr2015.tex:921"
    },
    {
      "id": "c52",
      "kind": "value",
      "arm": "rbf_shallow",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.rbf_shallow.rubbish_err",
      "claimed": 0.0,
      "tolerance": 1.0,
      "quote": "we find an error rate of 0\\%. Note that when the error rate is zero the average confidence on a mistake",
      "citation": "paper/source/iclr2015.tex:923"
    },
    {
      "id": "c53",
      "kind": "ordering",
      "arm": "rbf_shallow",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "high",
      "quantity": "measured.maxout_naive.rubbish_err - measured.rbf_shallow.rubbish_err",
      "direction": ">0",
      "quote": "far from the training data, are not fooled by this phenomenon.",
      "citation": "paper/source/iclr2015.tex:903"
    },
    {
      "id": "c54",
      "kind": "existence",
      "arm": "maxout_naive",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "predicate": "measured.maxout_naive.rubbish_share_8 < 1.0",
      "note": "share of rubbish false positives predicted as 8; paper observed exactly none",
      "quote": "and none were classified as 8s. Likewise, on CIFAR-10, 49.7\\% of the convolutional network's",
      "citation": "paper/source/iclr2015.tex:931"
    },
    {
      "id": "c55",
      "kind": "value",
      "arm": "maxout_naive",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.maxout_naive.rubbish_share_5",
      "claimed": 45.3,
      "tolerance": 25.0,
      "quote": "On MNIST, 45.3\\% of a naively trained maxout network's false positives were classified as 5s,",
      "citation": "paper/source/iclr2015.tex:930"
    },
    {
      "id": "c56",
      "kind": "value",
      "arm": "cifar_conv_maxout",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.cifar_conv_maxout.adv_err",
      "claimed": 87.15,
      "tolerance": 10.0,
      "quote": "Similarly, using $\\eps=.1$, we obtain an error rate of 87.15\\% and an average probability of",
      "citation": "paper/source/iclr2015.tex:340"
    },
    {
      "id": "c57",
      "kind": "value",
      "arm": "cifar_conv_maxout",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.cifar_conv_maxout.adv_conf_mistakes",
      "claimed": 96.6,
      "tolerance": 8.0,
      "quote": "96.6\\% assigned to the incorrect labels",
      "citation": "paper/source/iclr2015.tex:341"
    },
    {
      "id": "c58",
      "kind": "value",
      "arm": "cifar_conv_maxout",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "quantity": "measured.cifar_conv_maxout.rubbish_err",
      "claimed": 93.4,
      "tolerance": 8.0,
      "quote": "obtains an error rate of 93.4\\%, with an average confidence of 84.4\\%.",
      "citation": "paper/source/iclr2015.tex:912"
    },
    {
      "id": "c59",
      "kind": "value",
      "arm": "cifar_conv_maxout",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.cifar_conv_maxout.rubbish_conf_mistakes",
      "claimed": 84.4,
      "tolerance": 10.0,
      "quote": "obtains an error rate of 93.4\\%, with an average confidence of 84.4\\%.",
      "citation": "paper/source/iclr2015.tex:912"
    },
    {
      "id": "c60",
      "kind": "value",
      "arm": "cifar_conv_maxout",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.cifar_conv_maxout.fool_success_avg",
      "claimed": 75.3,
      "tolerance": 20.0,
      "quote": "step. Averaged over all ten classes, the method has an average per-step success rate of 75.3\\%.",
      "citation": "paper/source/iclr2015.tex:941"
    },
    {
      "id": "c61",
      "kind": "value",
      "arm": "cifar_conv_maxout",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.cifar_conv_maxout.fool_success_0",
      "claimed": 24.7,
      "tolerance": 15.0,
      "note": "class 0 = airplane in CIFAR-10 label order",
      "quote": "frogs and trucks, and the hardest class was airplanes, with a success rate of 24.7\\% per sampling",
      "citation": "paper/source/iclr2015.tex:940"
    },
    {
      "id": "c62",
      "kind": "existence",
      "arm": "cifar_conv_maxout",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "predicate": "measured.cifar_conv_maxout.fool_success_6 >= 99 and measured.cifar_conv_maxout.fool_success_9 >= 99",
      "note": "class 6 = frog, 9 = truck; paper: 100% per-step success for both",
      "quote": "with variable runtime. On CIFAR-10, we found that one sampling step had a 100\\% success rate for",
      "citation": "paper/source/iclr2015.tex:939"
    },
    {
      "id": "c63",
      "kind": "existence",
      "arm": "cifar_conv_maxout",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "predicate": "measured.cifar_conv_maxout.fool_success_0 <= min(measured.cifar_conv_maxout.fool_success_0, measured.cifar_conv_maxout.fool_success_1, measured.cifar_conv_maxout.fool_success_2, measured.cifar_conv_maxout.fool_success_3, measured.cifar_conv_maxout.fool_success_4, measured.cifar_conv_maxout.fool_success_5, measured.cifar_conv_maxout.fool_success_6, measured.cifar_conv_maxout.fool_success_7, measured.cifar_conv_maxout.fool_success_8, measured.cifar_conv_maxout.fool_success_9)",
      "note": "airplane is the hardest class: its success rate is the minimum over classes. RECLASSIFIED high->low after a 3-seed sweep: this is a single-run class-ordering observation, not a structural invariant. REFUTED -- dog (class 5) was consistently the hardest fooling class across the 3 seeds while airplane (class 0) varied widely (even at 1,000 samples/class the per-class success rate has high variance on the sub-scale conv net). The companion claim from the same sentence, c62 (frog & truck = 100% per-step success), is rated medium (a single-run observation on the paper's conv net, not a structural invariant) and at this run's sub-scale conv net does NOT hold at every seed (frog is 100% at all seeds but truck varies), so c62 fails honestly at sub-scale. See REPRODUCTION.md.",
      "quote": "frogs and trucks, and the hardest class was airplanes, with a success rate of 24.7\\% per sampling",
      "citation": "paper/source/iclr2015.tex:940"
    },
    {
      "id": "c64",
      "kind": "existence",
      "arm": "l1_maxout",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "medium",
      "predicate": "measured.l1_maxout.train_err > 5.0",
      "quote": ".0025 was too large, and caused the model to get stuck with over 5\\% error on",
      "citation": "paper/source/iclr2015.tex:430"
    },
    {
      "id": "c65",
      "kind": "curve",
      "arm": "eps_trace",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.eps_trace.logit_correct_e0",
      "against": "measured.eps_trace.logit_maxwrong_e0",
      "x": [
        0
      ],
      "comparison": "above",
      "note": "DEMOTE high->low: Figure 4 reports ONE illustrative class-4 example (tex:768 'The correct class is 4'), not a population claim that every class-4 example has this shape. The eps_trace arm now uses a DETERMINISTIC fixed example (first class-4 test example correctly classified by all seeds) with NO selection on the claim predicates, so this claim can genuinely fail if that example does not reproduce the figure shape; it is informational (low), not gating. at eps=0 the example is unperturbed and correctly classified as 4, so the correct-class curve sits above the max wrong-class curve (equivalently margin_seq > 0)",
      "quote": "the correct direction. Correct classifications occur only on a thin manifold where $\\vx$ occurs in the data.",
      "citation": "paper/source/iclr2015.tex:764"
    },
    {
      "id": "c66",
      "kind": "curve",
      "arm": "eps_trace",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.eps_trace.logit_correct_tails",
      "against": "measured.eps_trace.logit_maxwrong_tails",
      "x": [
        -10,
        10
      ],
      "comparison": "below",
      "note": "DEMOTE high->low: Figure 4 reports ONE illustrative class-4 example (tex:768 'The correct class is 4'), not a population claim that every class-4 example has this shape. The eps_trace arm now uses a DETERMINISTIC fixed example (first class-4 test example correctly classified by all seeds) with NO selection on the claim predicates, so this claim can genuinely fail if that example does not reproduce the figure shape; it is informational (low), not gating. at BOTH tails the correct-class curve sits under the max wrong-class curve - the thin-manifold claim; figure reads (figures/read-figure.jsonl): at eps=-10 class-4 ~ +200 vs top wrong-class ~ +800; at eps=+10 class-4 ~ -400 vs top ~ +400",
      "quote": "the correct direction. Correct classifications occur only on a thin manifold where $\\vx$ occurs in the data.",
      "citation": "paper/source/iclr2015.tex:764"
    },
    {
      "id": "c67",
      "kind": "curve",
      "arm": "eps_trace",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.eps_trace.logit_correct_pos",
      "against": "measured.eps_trace.logit_maxwrong_pos",
      "x": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "comparison": "crosses",
      "note": "DEMOTE high->low: Figure 4 reports ONE illustrative class-4 example (tex:768 'The correct class is 4'), not a population claim that every class-4 example has this shape. The eps_trace arm now uses a DETERMINISTIC fixed example (first class-4 test example correctly classified by all seeds) with NO selection on the claim predicates, so this claim can genuinely fail if that example does not reproduce the figure shape; it is informational (low), not gating. correct-class curve starts above the max wrong-class curve at eps=0 and ends below it at eps=10, crossing near eps of order 1 (vision read: at eps=+5 correct class already below top wrong class)",
      "quote": "the wrong classifications are stable across a wide region of $\\eps$ values. Moreover, the predictions become very extreme as we",
      "citation": "paper/source/iclr2015.tex:769"
    },
    {
      "id": "c68",
      "kind": "curve",
      "arm": "eps_trace",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.eps_trace.logit_maxwrong_pos",
      "x": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "comparison": "increasing",
      "note": "DEMOTE high->low: Figure 4 reports ONE illustrative class-4 example (tex:768 'The correct class is 4'), not a population claim that every class-4 example has this shape. The eps_trace arm now uses a DETERMINISTIC fixed example (first class-4 test example correctly classified by all seeds) with NO selection on the claim predicates, so this claim can genuinely fail if that example does not reproduce the figure shape; it is informational (low), not gating. predictions become very extreme moving into the rubbish regime",
      "quote": "the wrong classifications are stable across a wide region of $\\eps$ values. Moreover, the predictions become very extreme as we",
      "citation": "paper/source/iclr2015.tex:769"
    },
    {
      "id": "c69",
      "kind": "curve",
      "arm": "eps_trace",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.eps_trace.logit_maxwrong_e10",
      "x": [
        10
      ],
      "comparison": "matches",
      "claimed": [
        400
      ],
      "tolerance": 400,
      "note": "value read off Figure 4 left by the vision model in this run (top curve ~ +400 at eps=+10, figures/read-figure.jsonl); tolerance covers reading error and reproduction spread - anchor only",
      "quote": "the wrong classifications are stable across a wide region of $\\eps$ values. Moreover, the predictions become very extreme as we",
      "citation": "paper/source/iclr2015.tex:769"
    },
    {
      "id": "c70",
      "kind": "curve",
      "arm": "eps_trace",
      "seeds": [
        0,
        1,
        2
      ],
      "compute_invariance": "low",
      "quantity": "measured.eps_trace.logit_correct_e10",
      "x": [
        10
      ],
      "comparison": "matches",
      "claimed": [
        -400
      ],
      "tolerance": 400,
      "note": "class-4 logit read as ~ -400 at eps=+10 in this run (figures/read-figure.jsonl); anchor only",
      "quote": "The correct class is 4. We see that the unnormalized log probabilities for each class are conspicuously piecewise linear with $\\eps$ and that",
      "citation": "paper/source/iclr2015.tex:768"
    }
  ],
  "not_tested": [
    {
      "what": "MP-DBM is vulnerable to FGSM eps=0.25 (error 97.5%); its clean error is 0.88%",
      "citation": "paper/source/iclr2015.tex:794,800",
      "reason": "requires training a multi-prediction deep Boltzmann machine; outside compute scope"
    },
    {
      "what": "Fig.1 ImageNet demo: panda 57.7% -> gibbon 99.3% at eps=.007 on GoogLeNet",
      "citation": "paper/source/iclr2015.tex:364-383",
      "reason": "illustrative; needs pretrained GoogLeNet + ImageNet pipeline; not load-bearing"
    },
    {
      "what": "Fig.3 weight-localization claim (adversarially trained weights more localized/interpretable)",
      "citation": "paper/source/iclr2015.tex:523-525",
      "reason": "qualitative; paper defines no metric"
    },
    {
      "what": "rotational / scaled-gradient perturbation training weaker than FGSM adversarial training",
      "citation": "paper/source/iclr2015.tex:559-565",
      "reason": "no numbers reported"
    },
    {
      "what": "hidden-layer vs input perturbation comparison",
      "citation": "paper/source/iclr2015.tex:567-585",
      "reason": "qualitative only"
    },
    {
      "what": "maxout trained to 0% error on Gaussian rubbish with no clean-error benefit",
      "citation": "paper/source/iclr2015.tex:963-968",
      "reason": "tangential; stretch goal"
    },
    {
      "what": "46.2% error rate for direction transferred across clean examples",
      "citation": "paper/source/iclr2015.tex:776-779",
      "reason": "commented out (%) by the authors in v3 - not a claim of the published paper"
    },
    {
      "what": "quadratic/V1 model numbers (0.84% clean, 30.7% adversarial)",
      "citation": "paper/source/iclr2015.tex:620-643",
      "reason": "commented out (%) by the authors in v3"
    }
  ]
}
```

## 8.5. Constructed truth

Which sources of ground truth this reproduction uses to check itself, and where one does not
apply, why:

- **Degeneracy (the no-op setting):** FGSM adversarial training at ε=0, the Rademacher/uniform
  noise controls at ε=0, and L¹ weight decay at coef=0 each reduce to plain training. The
  implementation makes ε=0 / coef=0 a true no-op (no RNG consumed, no redundant forward) so the
  code path is bit-identical to the baseline; a degeneracy test asserts exact equality on
  weights. This is the cheapest real correctness evidence and ships in the repo.
- **The same quantity derived two ways (c07):** for logistic regression FGSM is exact, so the
  correct worst-case closed form `E ζ(ε‖w‖₁ − y(wᵀx+b))` equals the actual adversarial loss under
  the real gradient-based per-example FGSM `x_adv = x − ε·y·sign(w)` (sign(grad_x J) = −y·sign(w)).
  We assert `|FGSM-form − analytic-form| < 1e-5` on a fixed batch with mixed labels using the real
  `attack.fgsm` — so a sign bug in the attack OR the closed form breaks the invariant. The paper's
  own `tex:411` form `ζ(y(ε‖w‖₁ − wᵀx − b))` has a sign slip for y=−1 (it decreases, not maximizes,
  the loss); c07 checks the corrected form, not the paper's. The paper hands this identity to us
  for free modulo that sign slip.
- **Invariants from the maths:** `‖η‖_∞ == ε` exactly (tex:309); `sign(0) := 0`; no clipping of
  x̃ (§4.9); `wᵀsign(w) = ‖w‖₁` (tex:407); softmax rows sum to 1 while RBF rows need NOT
  (§4.4 — the latter is what makes "confidence on mistakes 1.2%" possible); a non-negative loss
  never goes negative; with the FGSM direction fixed at ε=0 the logits of a linear-activation
  network are exactly piecewise linear in ε (tex:762-770). All asserted in unit tests.
- **Planting a known structure in synthetic input (eps trace, c65–c70):** the Figure 4 curve
  claims use a **deterministic** fixed example (the first class-4 test example all seed models
  classify correctly — chosen by index, NOT by the thin-manifold predicates the claims evaluate)
  and check the thin-manifold shape point by point. These claims are rated `low`: Figure 4
  reports ONE illustrative example, not a population invariant, so a fail on this particular
  deterministic example is informational, not a gate failure. The direction is fixed at ε=0 so
  the curve is exactly piecewise linear. (An earlier version selected the example by the very
  predicates the claims evaluate — "pass by construction"; that selection is removed.)
- **The paper's standard baseline as an oracle:** the FGSM error rates the paper reports
  (softmax 99.9%, maxout 89.4%) are an oracle we already have; the high-invariance ordering
  claims (c03/c06/c13/c16/c23/c28/c32/c33/c36/c43/c44/c45/c53) check DIRECTIONS, which survive
  a CPU sub-scale budget where the tight value claims (0.94%, 0.782%) do not.
- **Brute force at toy scale against a closed form:** not used — the paper gives no worst-case
  bound to check against beyond the logistic closed form (c07, above).
- **Slow exact / convex reference solver:** not used; the L-BFGS adversarial-example finder of
  Szegedy et al. 2014b is the slow reference, but FGSM is the fast method this paper introduces
  and we test it directly.
- **Naive implementation agreeing with the fast one:** not separately needed — the FGSM
  implementation IS the naive implementation (a single sign step); the c07 closed-form
  equivalence plays this role for logistic regression.
- **Limiting cases:** ε=0 (degeneracy, above); ε→large moves into the rubbish regime (Fig. 4
  right tail), covered by c66/c68.
- **Protocol invariant (retrain-on-60k is from scratch):** the paper picks an epoch count on
  the val split then RETRAINS from scratch on all 60k (`tex:505-506`). `train.train` captures the
  initial weights at entry and Phase 2 reloads them + creates a FRESH optimizer (no carried
  momentum) and a FRESH RNG stream, then trains for the early-stopped epoch count.
  `tests/test_degeneracy.py::test_retrain_full_60k_is_from_scratch` asserts Phase 2 starts from
  the init weights (phase2_start_state == init_state — directly detects the old continuation
  bug where Phase 2 resumed from the Phase-1 state) AND that the final weights differ from a
  Phase-1-only run (Phase 2 actually retrains, not a no-op).
- **RBF autograd invariant (nu_trainable):** `tests/test_invariants.py::test_rbf_nu_trainable_
  receives_gradient` asserts a learnable `nu` receives a non-zero gradient on data near the means
  (the active, non-clamped loss region) — guarding the `float(nu)` detach regression that silently
  zeroed the nu gradient. The default `nu_trainable=False` buffer is the paper-silent choice
  (`tests/test_invariants.py::test_rbf_nu_buffer_default_is_not_learnable`).

## 9. Deliberately not tested (with reasons)

- **MP-DBM generative-model claim** (clean 0.88%, FGSM ε=.25 → 97.5%; `:794`, `:800`): requires
  training a multi-prediction deep Boltzmann machine — outside this reproduction's compute
  scope. Recorded, not gated.
- **Fig. 1 ImageNet demo** (panda 57.7% → gibbon 99.3% at ε=.007; `:364-383`): illustrative;
  needs pretrained GoogLeNet + ImageNet pipeline; not load-bearing for the paper's argument.
- **Fig. 3 weight localization** ("significantly more localized and interpretable", `:523-525`):
  qualitative; the paper defines no metric. Figure-read evidence recorded in §7.
- **Fig. 2 visual resemblance** of w vs sign(w) (`:447-453`): qualitative. §7 records the read.
- **Rotational / scaled-gradient perturbation training weaker than FGSM adversarial training**
  (`:559-565`): paper reports no numbers ("did not find nearly as powerful of a regularizing
  result").
- **Hidden-layer vs input perturbation comparisons** (`:567-585`): qualitative only.
- **Rubbish-training paragraph** (maxout trained to 0% error on Gaussian rubbish, with no
  clean-error benefit, `:963-968`): feasible but tangential; stretch goal.
- **Commented-out 46.2% cross-example direction-transfer number** (`:776-779`, a `%` comment in
  v3): the authors removed it from the published version — not a claim of the paper; not tested.
- **Commented-out quadratic/V1 model numbers** (0.84% clean, 30.7% adversarial; `:620-643`):
  same reason.
- **Class-skew numbers beyond the stated ones** (only 45.3%-as-5s / none-as-8s on MNIST and
  49.7%-frogs / five zero-classes on CIFAR are stated, `:930-933`): tested as c54/c55; CIFAR
  skew folded into the arm's `rubbish_class_shares` metric.
