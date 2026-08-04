# SPEC — Reproduction of "Explaining and Harnessing Adversarial Examples"

- **Paper:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy, *Explaining and Harnessing
  Adversarial Examples*, ICLR 2015 (arXiv:1412.6572v3, 20 Mar 2015).
- **Authoritative source on disk:** `paper/source/iclr2015.tex` (arXiv LaTeX, 975 lines). All
  citations below are `<file>:<line>` into this file and are grep-verified. Math macros in the
  preamble resolve as `\eps`→ε, `\sign`→sign, `\veta`→**η**, `\vtheta`→**θ**, `\vx`→**x**,
  `\vw`→**w** (`iclr2015.tex:14-55`).
- **Convenience copy:** `paper/paper.md` (PDF extraction; prose reliable, maths not — do not cite).
- **Percent convention:** every error / confidence / agreement metric is a **percent** in
  [0, 100] so `measured` values compare directly against the paper's numbers.

## 0. Upstream code status

- The paper itself links only one piece of code: `https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout` for the CIFAR-10 preprocessing (footnote 2, `iclr2015.tex:343-345`). Pylearn2 is archived and Theano-based (Python 2); the stack is dead and cannot execute today.
- `github.com/goodfeli/adversarial` (checked 2026-08-04) is the **GAN** paper's code, not this paper's: its README says "Code and hyperparameters for the paper 'Generative Adversarial Networks'". Not usable here.
- `github.com/tensorflow/cleverhans` (later library by the same lead author) implements FGSM and adversarial training generically but contains none of this paper's configs or hyperparameters; it is a different artifact.
- GitHub search on the exact title + authors returns no upstream reproduction of this paper's experiments (only a third-party genetic-programming variant, `washikia/dnn-fuzzer`).
- **Decision:** no usable upstream code exists. We implement from this SPEC.
- **Reference-only use of upstream material:** the maxout MLP hyperparameters this paper defers
  to ("the 240 used by the original maxout network", `iclr2015.tex:498`) live in the maxout paper
  (Goodfellow et al. 2013c) and its pylearn2 YAML. Where this paper is silent (§4 below), we may
  consult those configs as *reference defaults*; every such fill-in is logged as **our choice, not
  the paper's**.

## 1. The method as an explicit algorithm

### 1.1 FGSM — fast gradient sign method (the attack)

```
Input:  model with parameters θ and cost J(θ, x, y); batch x [B,D]; labels y [B]; ε > 0.
 1. g ← ∇_x J(θ, x, y)                 # [B,D], backprop, θ held fixed
 2. η ← ε · sign(g)                     # [B,D], sign elementwise; sign(0) := 0
 3. x̃ ← x + η                           # [B,D]; ‖η‖_∞ = ε exactly
Output: adversarial examples x̃.
```

This is "an optimal max-norm constrained perturbation" under the linearization of J in x
(`iclr2015.tex:305-311`). **No clipping** of x̃ back into [0,1] is stated anywhere; default is
**no clipping** (see §4). Gradient is w.r.t. x only; θ is constant during attack generation.

### 1.2 FGSM adversarial training (the harness)

```
Input: training set, model f_θ, mixing constant α, perturbation size ε.
Repeat per minibatch (x, y):
 1. x̃ ← FGSM(f_θ, x, y, ε) using the CURRENT θ ("continually update our supply",
    iclr2015.tex:490-491). sign() is non-differentiable, so gradients flow through x̃ as if
    x̃ were an input-dependent constant w.r.t. θ (iclr2015.tex:559-561).
 2. L ← α · mean_i J(θ, x_i, y_i) + (1−α) · mean_i J(θ, x̃_i, y_i)
 3. θ ← θ − lr · ∇_θ L
```

α = 0.5 in **all** the paper's experiments (`iclr2015.tex:488`). ε for training is stated only
for MNIST maxout: ε = 0.25 (`iclr2015.tex:429`). Both halves of the loss reuse the **same**
minibatch labels y (the paper's equation omits the y argument in the second term — a typo,
see §4/§8).

### 1.3 Adversarial logistic regression (analytic special case, §5)

Binary logistic regression, y ∈ {−1, +1}, P(y=1) = σ(wᵀx + b), loss
E ζ(−y(wᵀx + b)) with ζ(z) = log(1 + e^z) (`iclr2015.tex:399-404`). Since
sign(∇_x J) = −y·sign(w) and wᵀsign(w) = ‖w‖₁ (`iclr2015.tex:407`), exact FGSM-adversarial
training is minimization of

```
E_{x,y} ζ( y · ( ε‖w‖₁ − wᵀx − b ) )          (iclr2015.tex:411)
```

i.e. the ‖w‖₁ term is subtracted from the **activation** (margin), not added to the cost like
L¹ weight decay (`iclr2015.tex:413-419`). We implement both the FGSM form and this closed form
and assert their equality (claim c07).

### 1.4 Rubbish-class and targeted-fooling protocols (Appendix)

- **Rubbish eval (MNIST):** draw 10,000 samples x ~ N(0, I₇₈₄) (`iclr2015.tex:905`). "Error" :=
  the model assigns probability > 0.5 to **any** class (`iclr2015.tex:906-907`). For a softmax
  model that is max_k p_k > 0.5; for sigmoid-top, any p_k > 0.5.
- **Rubbish eval (CIFAR-10):** same but 1,000 samples x ~ N(0, I₃₀₇₂) (`iclr2015.tex:911`).
- **Targeted fooling (CIFAR-10):** to fool class i, take one **gradient-sign step** from a
  Gaussian sample in the direction that increases p(y = i | x) (Fig. 5 caption,
  `iclr2015.tex:953-955`; the body text, `iclr2015.tex:936`, writes the step without `sign` —
  we follow the caption's sign step, see §4). Success := p(y = i | x̃) > 0.5
  (`iclr2015.tex:955-956`). Per-class per-step success rate measured over ≥ 1,000 fresh samples.

## 2. Symbols with shapes

B = minibatch size, D = input dim (784 MNIST, 3072 CIFAR-10 as flattened 32·32·3), K = 10 classes.

| symbol | shape | meaning |
|---|---|---|
| x | `[B, D]` f32 | input batch; MNIST scaled to **[0,1]** (`iclr2015.tex:334`, footnote 1); CIFAR GCN-preprocessed to std ≈ 0.5 (footnote 2, `iclr2015.tex:343-345`) |
| y | `[B]` int64 | class indices {0..9}; for 3-vs-7 logistic regression, `{−1,+1}` (`iclr2015.tex:399`) |
| θ | parameter pytree | all model parameters (`iclr2015.tex:305`) |
| J(θ, x, y) | scalar f32 | mean training cost over the batch: softmax NLL for K-class models; mean softplus margin ζ(−y(wᵀx+b)) for logistic (`iclr2015.tex:306, 401-404`) |
| ∇_x J | `[B, D]` f32 | input gradient of the scalar batch cost |
| sign(∇_x J) | `[B, D]` in {−1,0,1} | elementwise; sign(0) := 0 |
| η | `[B, D]` f32 | ε·sign(∇_x J); ‖η‖_∞ = ε |
| x̃ = x + η | `[B, D]` f32 | adversarial example (`\tilde{\vx} = \vx + \veta`, `iclr2015.tex:235`) |
| ε | scalar f32 | max-norm constraint; 0.25 MNIST, 0.1 CIFAR, 0.007 ImageNet demo (`iclr2015.tex:333, 340, 381`) |
| w (logreg) | `[D]` f32 | weight vector; b scalar bias |
| σ(z), ζ(z) | scalar→scalar | logistic sigmoid, softplus log(1+e^z) (`iclr2015.tex:399-404`) |
| logits a | `[B, K]` f32 | "argument to softmax" (Fig. 4 y-axis) |
| p = softmax(a) | `[B, K]` f32 | class probabilities; **confidence** := max_k p_k per example |
| α | scalar | adversarial-training mix, 0.5 (`iclr2015.tex:488`) |
| maxout W^(l) | `[fan_in, units, pieces]` | per-unit max over `pieces` affine slices; layers/units per arm (§6) |
| μ_k, β_k (RBF) | `[D]`, `[D, D]` | class template and quadratic form; **β negative-semidefinite** — see §4 (printed eq. `iclr2015.tex:595` has no minus sign) |
| U(−ε, ε), ±ε noise | `[B, D]` f32 | training-time input noise for the two control arms (`iclr2015.tex:556-557`) |
| eps-grids (Fig. 4) | `[21]` | ε ∈ {−10, −9, …, 10} for the logit trace (axis read from figure, §7) |

## 3. Equations to implement, each with a citation that resolves

1. `wᵀ x̃ = wᵀx + wᵀη` — linear-model perturbation growth; optimal η = ε·sign(w), growth ε·m·n for
   n-dim w with average |w_i| = m.
   Cites: `paper/source/iclr2015.tex:243` (equation), `:246-247` (η = sign(w), growth εmn).
2. **FGSM:** `η = ε sign(∇_x J(θ, x, y))` — `paper/source/iclr2015.tex:309`.
3. **Logistic training:** minimize `E ζ(−y(wᵀx + b))`, ζ softplus —
   `paper/source/iclr2015.tex:402` (loss), `:404` (ζ def), `:399` (P(y=1)=σ(wᵀx+b), y∈{−1,1}).
4. **FGSM exact for logreg:** sign(∇) = −sign(w) [modulo the ±y factor, see 1.3],
   wᵀsign(w) = ‖w‖₁ — `paper/source/iclr2015.tex:407`.
5. **Adversarial logistic regression:** minimize `E ζ(y(ε‖w‖₁ − wᵀx − b))` —
   `paper/source/iclr2015.tex:411`.
6. **Adversarial training of deep nets:** `J̃(θ,x,y) = αJ(θ,x,y) + (1−α)J(θ, x + ε sign(∇_x J(θ,x,y)), y)` — `paper/source/iclr2015.tex:486-487` (equation **as printed drops the `y` argument and the closing paren** of the second J; the intended reading is unambiguous from `iclr2015.tex:306`), α=0.5 — `:488`.
7. **Shallow RBF:** `p(y=1|x) = exp((x − μ)ᵀ β (x − μ))` — `paper/source/iclr2015.tex:595`.
   **As printed, the form only decays away from μ if β is negative (semi)definite**; the paper
   never says so. We parametrize β_k = −ψ_k ψ_kᵀ − ν·I, ν ≥ 0. See §4.
8. **Targeted fooling:** x̃ = x + ε·sign(∇_x p(y = i | x)), x ~ N(0, I_D) —
   body (no sign) `paper/source/iclr2015.tex:936`; Fig. 5 caption (sign step)
   `paper/source/iclr2015.tex:953-955`. We implement the caption's sign-step version.
9. **Rubbish error rule:** error ⇔ any class prob > 0.5 on `x ~ N(0, I_784)` (10,000 samples) /
   `N(0, I_3072)` (1,000 samples) — `paper/source/iclr2015.tex:905-907, 911`.

## 4. What the paper does NOT state

This list is the gap between the paper and any implementation. Unless marked "RESOLVED", each
item must be chosen by the implementation and logged in `README.md` as **our choice**.

**Training protocol (all arms)**
1. Optimizer, learning rate, momentum, LR schedule, batch size, number of epochs: **unstated
   everywhere** in this paper (deferred implicitly to the maxout paper's MNIST config,
   `iclr2015.tex:498` says only "240 used by the original maxout network").
2. Maxout architecture detail: number of linear **pieces per maxout unit**, number of hidden
   layers ("240/1600 units per layer", `:498`, never says which layers), dropout rates
   (input/hidden), weight initialization scheme — all unstated here.
3. Softmax-regression and logistic-regression training hyperparameters: entirely unstated.
4. RBF network: training procedure, class bias/prior term, parametrization/sign of β
   (eq. 7 above is printed without the needed negative sign), and how the 10 binary
   class-conditionals yield a single predicted class — all unstated. RESOLVED: 10 units
   p_k = exp((x−μ_k)ᵀβ_k(x−μ_k)), β_k negative-semidefinite, predict argmax_k p_k,
   confidence = max_k p_k (this is the only reading consistent with "confidence on mistaken
   examples 1.2%", `:603`, which is impossible under a K-class softmax floor of 10%).
5. Whether RBF "confidence on clean test examples 60.6%" (`:604`) is mean max-prob — assumed.
6. RBF FGSM cost: what J is differentiated to attack the RBF — unstated. RESOLVED: −log p_{y}.
7. Ensemble training: members differ only by RNG seed (stated, `:819-821`); nothing else.
8. L¹ weight-decay control: which layers (first layer stated, `:429`), optimizer, epochs —
   unstated; smaller coefficients "permitted successful training but conferred no
   regularization benefit" (`:431-432`) without numbers.

**Attack / evaluation detail**
9. **Clipping** of x̃ to [0,1] (MNIST) or the valid preprocessed range (CIFAR): never mentioned.
   RESOLVED default: **no clipping**; record as sensitivity analysis if run.
10. "Average confidence" denominators: for softmax 79.3% (`:333-334`) and maxout 97.6% (`:339`)
    it is unstated whether the mean is over all adversarial examples or mistakes only; for
    CIFAR 96.6% (`:340-341`) "assigned to the incorrect labels" implies mistakes-only; RBF
    gives both explicitly (`:603-604`). RESOLVED: we record **both**
    `adv_conf_all` and `adv_conf_mistakes`; value claims c02/c10 use `_all` (with 99.9 %/89.4 %
    error the two differ little).
11. Test-set coverage: FGSM metrics are "on the MNIST test set" (`:334`) — we use all 10,000
    test examples; on CIFAR all 10,000 (`:343`).
12. Ensemble attack objective: "designed to perturb the entire ensemble" (`:822-823`) — unstated
    what is differentiated. RESOLVED: FGSM on mean-probability ensemble NLL; single-member
    attacks use member 0.
13. Ensemble decision rule: mean probability argmax — unstated, assumed.
14. Targeted-fooling ε (`:936`) and sample count per class: unstated. RESOLVED: implementation
    picks and logs ε; ≥ 1,000 samples/class.
15. Fig. 4 trace: which test example (only "The correct class is 4", `:768`), which ε grid, and
    (implicitly but not stated) that the FGSM direction is computed **once at ε = 0** and held
    fixed while ε sweeps — only this reading makes the logit lines exactly (piecewise) linear.
    RESOLVED: first class-4 test example correctly classified by that seed's model; direction
    fixed at ε = 0; grid −10…10 step 1 (axis from figure reading, §7).
16. Adversarial-examples transfer pair (19.6 % / 40.9 %, `:519-520`): which architectures are
    "the original model" / "the new model" is not explicit (§6 discusses both the 240-unit
    0.94→0.84 result and the 1600-unit 0.782 result). RESOLVED: primary pair =
    (maxout_large_naive, maxout_large_adv); the small pair is recorded as secondary.
17. Agreement experiment (§8): which maxout arch generated the examples, and the exact
    conditioning of "the RBF network can predict softmax regression's class 53.6 % of the
    time" (`:688-689`) — read as both-models-wrong conditioned, mirroring the previous sentence.

**Stopping / seed protocol**
18. Validation split: not stated (50k train / 10k val inferred from "retrained on all 60,000
    examples", `:505-506`). RESOLVED: standard last-10k-of-60k split.
19. Early stopping: patience 100 epochs on validation error for the original maxout result
    (`:501-503`); **adversarial validation set error** for the big adversarially trained model
    (`:504-505`) — its ε (assumed 0.25) and refresh frequency (assumed every epoch) are unstated.
20. Retrain-on-60k uses the early-stopped epoch count (`:505-506`) — stated; the epoch counts
    themselves are not.
21. Seeds: only "different seeds for the random number generators used to select minibatches of
    training examples, initialize model weights, and generate dropout masks" (`:506-508`).
    No seed values, no count except the 5 runs of the large model (`:508-510`).

**Numerical/kernel minutiae**
22. sign(0) convention: unstated. RESOLVED: 0.
23. Numerical stabilizers (ε in softmax, softplus overflow): unstated.
24. CIFAR conv-maxout architecture (layers/channels/kernels) and its **clean test error**:
    both unstated; preprocessing only referenced to the pylearn2 maxout scripts yielding
    std ≈ 0.5 (`:343-345`). RESOLVED: GCN with parameters chosen to produce global std ≈ 0.5;
    exact recipe logged as ours.
25. Whether noise-control noise is resampled per minibatch/epoch (`:555-557`): unstated.
    RESOLVED: resampled every minibatch presentation.
26. Adversarial-training minibatch composition: whether α-weighted halves share one batch
    (RESOLVED yes, from eq. 6) and whether clean+adversarial halves each see gradient each step.
27. MP-DBM arm: all detail external to this paper (`:793-800`); **not built** (§8, §9).
28. The paper's Fig. 1 ImageNet demo (ε = .007, `:381-383`): GoogLeNet weights/preprocessing —
    not built (§9).

## 5. Component interfaces (fixed now; parallel agents build against these)

Framework: a single array-programming stack (flax/jax or torch — picked once at implementation
time and recorded in README). All arrays float32 unless stated. **One arm = one entry in
`results/_per_seed/<arm>__seed<i>.json`** containing exactly the metric keys listed below.

```
data.py
  load_mnist(seed:int)          -> dict(x_train[50000,784], y_train[50000],
                                        x_val[10000,784],   y_val[10000],
                                        x_test[10000,784],  y_test[10000])   pixels in [0,1]
  load_mnist_full(seed:int)     -> same but x_train/y_train cover all 60,000 (for the retrain arm)
  load_mnist_3v7(seed:int)      -> subset of labels 3 and 7; y in {-1:+3? NO: +1 = class 3}, {-1 = class 7}
                                   (convention: y=+1 ≡ digit 3, y=-1 ≡ digit 7; fixed here)
  load_cifar10(seed:int)        -> x_*_[N,3072] GCN-preprocessed (global std ≈ 0.5), y_* int64
  rubbish(dim:int, n:int, seed:int) -> [n,dim] f32 ~ N(0, I_dim)

models.py  — every model implements:
  .logits(x[B,D]) -> [B,K] f32          (RBF/ensemble implement K=10 score functions;
                                         logreg_3v7 implements .margin(x[B,784]) -> [B])
  .prob(x[B,D])   -> [B,K] f32          (RBF: raw exp-quadratic per-class probabilities, rows
                                         need NOT sum to 1 — see §4.4)
  .loss(x[B,D], y[B]) -> scalar
  .predict(x) -> [B] int64 ;  .confidence(x) -> [B] f32 = max_k prob
  SoftmaxRegression(784,10) ; LogisticRegression3v7(784)
  MaxoutMLP(units_per_layer:int, layers:int, pieces:int, dropout:dict)   # 240/1600 variants
  RBFNet(K=10, D=784)         # §4.4 parametrization
  MaxoutSigmoid(...)          # maxout backbone + 10 independent sigmoid heads
  Ensemble(members:list)      # mean-prob combine
  ConvMaxoutCIFAR(...)        # arch logged; paper silent

attack.py
  fgsm(model, x[B,D], y[B], eps) -> x_adv[B,D]          # no clipping (§4.9)
  fgsm_logits_trace(model, x[D], y, eps_grid[21]) -> logits[21,K]   # direction fixed at eps=0
  targeted_fool_step(model, x0[B,D], target:int, eps) -> x1[B,D]    # sign step (eq. 8)

train.py
  train(model, data, cfg) -> history
    cfg fields: adversarial: None | {alpha:0.5, eps:0.25};  noise: None | {type, eps};
    l1_first_layer: None | 0.0025;
    early_stop: None | {monitor: "val_err"|"adv_val_err", patience:100};
    retrain_full_60k: bool.   history logs per epoch: epoch, train_err, val_err, adv_val_err.

eval.py
  error(model, x, y) -> percent ;  confidence_stats(model, x, y) ->
        {"conf_all": percent, "conf_mistakes": percent}
  rubbish_eval(model, dim, n, seed) -> {"rubbish_err", "rubbish_conf_mistakes",
        "rubbish_class_shares": {0..9: percent of mistakes predicted as k}}
  agreement(modelA, modelB, x_adv, y) -> {"agree_all", "agree_cond"}  # on A's mistakes;
        # _cond conditioned on both wrong
```

**Metric keys per arm** (emitted per seed; claims resolve `measured.<arm>.<key>` against these):

- `softmax_reg`: clean_err, adv_err, adv_conf_all, adv_conf_mistakes, rubbish_err,
  rubbish_conf_mistakes, rubbish_class_shares
- `logreg_3v7`: clean_err, adv_err, train_err, **analytic_equiv_max_absdiff** (c07 check)
- `maxout_naive`: clean_err, train_err, adv_err, adv_conf_all, adv_conf_mistakes,
  rubbish_err, rubbish_conf_mistakes, rubbish_class_shares
- `maxout_sigmoid`: clean_err, rubbish_err, rubbish_conf_mistakes
- `maxout_adv` (240u), `maxout_large_naive` (1600u): clean_err, adv_err, adv_conf_all,
  adv_conf_mistakes
- `maxout_large_adv` (1600u, 5 seeds): clean_err, adv_err, adv_conf_all, adv_conf_mistakes,
  epochs_run
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

Named arms and the configuration each needs. The paper compares: shallow softmax regression,
logistic regression (3-vs-7), naive maxout, adversarially trained maxout (2 sizes), two noise
controls, L¹ weight decay, shallow RBF, sigmoid-top maxout, a 12-member ensemble, a conv maxout
on CIFAR-10, an MP-DBM (§9, not built), GoogLeNet/ImageNet (Fig. 1, not built).

| arm | dataset | arch/config | training | eval |
|---|---|---|---|---|
| `softmax_reg` | MNIST [0,1] | linear 784→10 softmax | NLL, to convergence | clean; FGSM ε=0.25; rubbish |
| `logreg_3v7` | MNIST 3v7 | linear sigmoid, y∈{±1} | softplus margin (eq. 3) | clean 1.6 %; FGSM ε=0.25 → 99 %; analytic check |
| `maxout_naive` | MNIST | maxout MLP, 240 units/layer, dropout | early stop on val err (patience 100) | clean 0.94 %; FGSM ε=0.25 → 89.4 %/97.6 %; rubbish |
| `maxout_adv` | MNIST | same + adv training α=0.5, ε=0.25 | same | clean 0.84 %; FGSM |
| `maxout_large_naive` | MNIST | 1600 units/layer, no adv training | early stop val err | clean 1.14 %; FGSM (recalled 89.4 %) |
| `maxout_large_adv` | MNIST | 1600 units/layer + adv training | early stop on **adv val err**; retrain on 60k; **5 seeds** | clean: 4×0.77, 1×0.83, mean 0.782; FGSM 17.9 %, conf-mistakes 81.4 % |
| `maxout_sigmoid` | MNIST | maxout backbone + independent sigmoid top, BCE | same as naive | rubbish 68 %/87.9 % |
| `noise_rademacher` | MNIST | maxout_naive config | + per-pixel ±0.25 noise | FGSM 86.2 %/97.3 % |
| `noise_uniform` | MNIST | maxout_naive config | + per-pixel U(−0.25,0.25) | FGSM 90.4 %/97.8 % |
| `l1_maxout` | MNIST | maxout_naive config | + L¹ coef 0.0025 on first layer | train err > 5 % (negative control) |
| `rbf_shallow` | MNIST | 10 RBF units, no hidden layer (§4.4) | NLL | FGSM 55.4 %/1.2 %, clean conf 60.6 %, rubbish 0 % |
| `ensemble12` | MNIST | 12 × maxout_naive, distinct seeds | per-member as naive | FGSM(ε=.25): 91.1 % ensemble-crafted, 87.9 % single-crafted |
| `agreement_mnist` | MNIST | reuses maxout_naive + softmax_reg + rbf_shallow | — | 54.6/16.0/84.6/54.3/53.6 % |
| `transfer_mnist` | MNIST | reuses maxout_large_naive ↔ maxout_large_adv | — | 19.6 % / 40.9 % |
| `eps_trace` | MNIST | reuses maxout_naive (per seed) | — | Fig. 4 curves, ε ∈ [−10,10] |
| `cifar_conv_maxout` | CIFAR-10 GCN | conv maxout (arch ours, logged) | to convergence | FGSM ε=0.1 → 87.15 %/96.6 %; rubbish 93.4 %/84.4 %; fooling 75.3 % avg |
| ~~`mp_dbm`~~ | MNIST | MP-DBM | — | **not built** (§9) |
| ~~`googlenet_imagenet`~~ | ImageNet | GoogLeNet | — | **not built** (§9) |

## 7. Figure readings

`read-figure` transcript: `figures/read-figure.jsonl` (committed). `eps_curve.pdf` was rasterized
with pymupdf before reading (the reader takes PNGs).

**Figure 4 left (`eps_curve.pdf`)** — the one figure with quantitative content the prose lacks.
Consistent across repeated reads: x-axis ε ticks −15…15, but the curves are **drawn only for
ε ∈ [−10, 10]**; y-axis "argument to softmax" ticks −2000…1000 step 500; ten dashed class curves
plus a thick solid magenta curve (class 4, the correct class, matching `:768`); the nine
non-magenta curves are straight lines; at ε = +10 the top wrong-class curve (cyan, class 5) ends
at ≈ +400 to +430 while the class-4 curve ends at ≈ −350; at ε = −10 the class-4 curve is at
≈ +200 while several wrong-class curves are up at ≈ +800; at ε = +5 the correct class is already
**below** the top wrong-class curve, and at ε = −5 it is **not** above all wrong-class curves —
i.e. the correct class wins only in a thin band around ε ≈ 0. This matches the caption:
correct classifications "only on a thin manifold" (`:764`), logits "conspicuously piecewise
linear" (`:768`), "wrong classifications are stable across a wide region of ε values" and
"predictions become very extreme" (`:769-770`). These enter claims c65–c70.

**Figure 4 right (`eps_curve_inputs.png`)** — thumbnail montage (yellow = still correctly
classified). Two attempted reads exhausted the vision model's token budget before producing a
grid count (transcript retained); its claim — correctly classified inputs form a thin band near
ε ≈ 0 — is subsumed by curve claims c65–c67 and is not separately claimed.

**Other figures:** Fig. 1 (panda→gibbon), Fig. 2 (logreg panels), Fig. 3 (weight localization),
Fig. 5 (airplane montage) carry no quantities beyond their captions/prose; see §9 for what is
not tested.

## 8. claims.json

The gate: claims with `compute_invariance: "high"` are the gating set (they survive a budget
smaller than the paper's — directions, orderings with wide margins, curve shapes, and the §1.3
algebraic invariant). Seeds: the top-level `seeds: [0, 1, 2]` is the claims default; the
`maxout_large_adv` claims (c14, c18–c20) override to five seeds `[0..4]`, matching the paper's
five runs (`iclr2015.tex:506-510`); per seed, three RNG streams are used — weight init,
minibatch order, dropout masks (`iclr2015.tex:506-508`). `quantity` expressions over
`measured.<arm>.<metric>` are evaluated at
**every seed listed in the claim**; value claims compare the **mean over seeds** against
`claimed` within `tolerance`; ordering claims must hold at every seed; curve comparisons
sample `quantity` — and for `above`/`below`/`crosses` also the other curve named by
`against` — at `x`, with diffs = quantity − against: `above`/`below` = all sampled diffs
> 0 / < 0, `crosses` = first sampled diff > 0 and last < 0,
`increasing` = last > first with ≥80 % of consecutive diffs ≥ 0, `matches` = elementwise
within `tolerance`. `claims.json` at the repo root is byte-identical to this block.

```json
{
  "paper": "Goodfellow, Shlens & Szegedy, Explaining and Harnessing Adversarial Examples, ICLR 2015 (arXiv:1412.6572v3)",
  "source_of_truth": "paper/source/iclr2015.tex; every citation is grep-verified <file>:<line>",
  "metric_convention": "all errors/confidences/agreements are percents in [0,100], matching the paper's printed numbers",
  "seeds": [0, 1, 2],
  "seed_protocol": {
    "seeds_default": [0, 1, 2],
    "maxout_large_adv": [0, 1, 2, 3, 4],
    "rngs_per_seed": ["weight_init", "minibatch_order", "dropout_masks"],
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
    "softmax_reg": {"dataset": "mnist [0,1]", "arch": "linear 784->10 softmax", "train": "NLL to convergence", "eval": "clean; FGSM eps=0.25; rubbish N(0,I784)"},
    "logreg_3v7": {"dataset": "mnist 3-vs-7 subset, y in {-1,+1}, +1=digit 3", "arch": "linear sigmoid", "train": "mean softplus margin (tex:402)", "eval": "clean; FGSM eps=0.25; analytic-equivalence check"},
    "maxout_naive": {"dataset": "mnist [0,1]", "arch": "maxout MLP 240 units/layer, dropout (pieces/rates: paper silent, see SPEC 4)", "train": "early stop on val_err, patience 100 (tex:501-503)", "eval": "clean; FGSM eps=0.25; rubbish"},
    "maxout_adv": {"dataset": "mnist", "arch": "maxout_naive config", "train": "adversarial training alpha=0.5 eps=0.25 (tex:486-488)", "eval": "clean; FGSM eps=0.25"},
    "maxout_large_naive": {"dataset": "mnist", "arch": "maxout MLP 1600 units/layer (tex:498)", "train": "early stop val_err", "eval": "clean (claimed 1.14, tex:500); FGSM"},
    "maxout_large_adv": {"dataset": "mnist", "arch": "1600 units/layer + adversarial training", "train": "early stop on adversarial validation error (tex:504-505); retrain on all 60,000 (tex:505-506)", "eval": "clean 5 seeds; FGSM eps=0.25"},
    "maxout_sigmoid": {"dataset": "mnist", "arch": "maxout_naive backbone + 10 independent sigmoid outputs, BCE top (tex:909)", "train": "as maxout_naive", "eval": "rubbish"},
    "noise_rademacher": {"dataset": "mnist", "arch": "maxout_naive config", "train": "+ per-pixel eps*{-1,+1} noise, resampled per minibatch (tex:555-556)", "eval": "FGSM eps=0.25"},
    "noise_uniform": {"dataset": "mnist", "arch": "maxout_naive config", "train": "+ per-pixel U(-0.25,0.25) noise (tex:556)", "eval": "FGSM eps=0.25"},
    "l1_maxout": {"dataset": "mnist", "arch": "maxout_naive config", "train": "+ L1 coef 0.0025 on FIRST layer (tex:429-430)", "eval": "train_err"},
    "rbf_shallow": {"dataset": "mnist [0,1]", "arch": "10 independent RBF units exp((x-mu)'beta(x-mu)), beta negative-semidefinite (SPEC 4.4; tex:595)", "train": "NLL", "eval": "clean conf; FGSM eps=0.25; rubbish"},
    "ensemble12": {"dataset": "mnist", "arch": "12 x maxout_naive, distinct seeds (tex:819-821)", "combine": "mean probability", "eval": "FGSM eps=0.25 vs (a) full-ensemble gradient (b) member-0 gradient"},
    "agreement_mnist": {"dataset": "mnist", "reuses": ["maxout_naive", "softmax_reg", "rbf_shallow"], "eval": "label agreement on FGSM (eps=0.25) examples misclassified by maxout_naive; *_cond conditioned on both models wrong"},
    "transfer_mnist": {"dataset": "mnist", "pair": ["maxout_large_naive (orig)", "maxout_large_adv (new)"], "eval": "FGSM eps=0.25 generated on one model, error measured on the other (tex:518-520)"},
    "eps_trace": {"dataset": "mnist test", "reuses": "maxout_naive of same seed", "example": "first class-4 test example correctly classified by that seed's model", "protocol": "FGSM direction computed once at eps=0; sweep eps=-10..10 step 1; record the [21,10] logits (tex:762-770)"},
    "cifar_conv_maxout": {"dataset": "cifar-10, GCN preprocessed to std~0.5 (tex:343-345)", "arch": "conv maxout (arch ours, paper silent)", "train": "to convergence", "eval": "FGSM eps=0.1; rubbish N(0,I3072) 1000 samples; targeted fooling sign-step (tex:953-955)"},
    "mp_dbm": {"status": "NOT BUILT - see not_tested"},
    "googlenet_imagenet": {"status": "NOT BUILT - see not_tested"}
  },
  "claims": [
    {"id": "c01", "kind": "value", "arm": "softmax_reg", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.softmax_reg.adv_err", "claimed": 99.9, "tolerance": 2.0, "quote": "a shallow softmax classifier to have an error rate of 99.9\\% with an average confidence of", "citation": "paper/source/iclr2015.tex:333"},
    {"id": "c02", "kind": "value", "arm": "softmax_reg", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.softmax_reg.adv_conf_all", "claimed": 79.3, "tolerance": 8.0, "note": "denominator all vs mistakes unstated (SPEC 4.10); with 99.9% error the two nearly coincide", "quote": "79.3\\% on the MNIST", "citation": "paper/source/iclr2015.tex:334"},
    {"id": "c03", "kind": "ordering", "arm": "softmax_reg", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.softmax_reg.adv_err - measured.softmax_reg.clean_err", "direction": ">0", "quote": "We find that this method reliably causes a wide variety of models to misclassify their input.", "citation": "paper/source/iclr2015.tex:331"},
    {"id": "c04", "kind": "value", "arm": "logreg_3v7", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.logreg_3v7.clean_err", "claimed": 1.6, "tolerance": 0.8, "quote": "c) MNIST 3s and 7s. The logistic regression model has a 1.6\\% error rate on the 3 versus 7 discrimination task on these examples.", "citation": "paper/source/iclr2015.tex:454"},
    {"id": "c05", "kind": "value", "arm": "logreg_3v7", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.logreg_3v7.adv_err", "claimed": 99.0, "tolerance": 2.0, "quote": "The logistic regression model has an error rate of 99\\% on these examples.", "citation": "paper/source/iclr2015.tex:456"},
    {"id": "c06", "kind": "ordering", "arm": "logreg_3v7", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.logreg_3v7.adv_err - measured.logreg_3v7.clean_err", "direction": ">0", "quote": "The logistic regression model has an error rate of 99\\% on these examples.", "citation": "paper/source/iclr2015.tex:456"},
    {"id": "c07", "kind": "invariant", "arm": "logreg_3v7", "seeds": [0,1,2], "compute_invariance": "high", "predicate": "measured.logreg_3v7.analytic_equiv_max_absdiff < 1e-5", "note": "| mean_i zeta(-y_i(w.(x_i - eps*y_i*sign(w)) + b)) - mean_i zeta(y_i*(eps*||w||_1 - w.x_i - b)) | on a fixed batch; FGSM is exact for logreg", "quote": "Note that the sign of the gradient is just $- \\sign(\\vw)$, and that $\\vw^\\top \\sign(\\vw) = ||\\vw||_1$.", "citation": "paper/source/iclr2015.tex:407"},
    {"id": "c08", "kind": "value", "arm": "maxout_naive", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.maxout_naive.clean_err", "claimed": 0.94, "tolerance": 0.3, "quote": "able to reduce the error rate from 0.94\\% without adversarial training to 0.84\\% with adversarial", "citation": "paper/source/iclr2015.tex:493"},
    {"id": "c09", "kind": "value", "arm": "maxout_naive", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.maxout_naive.adv_err", "claimed": 89.4, "tolerance": 8.0, "quote": "}. In the same setting, a maxout network misclassifies 89.4\\%", "citation": "paper/source/iclr2015.tex:338"},
    {"id": "c10", "kind": "value", "arm": "maxout_naive", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.maxout_naive.adv_conf_all", "claimed": 97.6, "tolerance": 8.0, "quote": "of our adversarial examples with an average confidence of 97.6\\%.", "citation": "paper/source/iclr2015.tex:339"},
    {"id": "c11", "kind": "value", "arm": "maxout_adv", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.maxout_adv.clean_err", "claimed": 0.84, "tolerance": 0.25, "quote": "able to reduce the error rate from 0.94\\% without adversarial training to 0.84\\% with adversarial", "citation": "paper/source/iclr2015.tex:493"},
    {"id": "c12", "kind": "ordering", "arm": "maxout_adv", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.maxout_naive.clean_err - measured.maxout_adv.clean_err", "direction": ">0", "note": "margin in the paper is only 0.1pp, so rated medium not high", "quote": "able to reduce the error rate from 0.94\\% without adversarial training to 0.84\\% with adversarial", "citation": "paper/source/iclr2015.tex:493"},
    {"id": "c13", "kind": "ordering", "arm": "maxout_adv", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.maxout_naive.adv_err - measured.maxout_adv.adv_err", "direction": ">0", "note": "same-architecture version of 89.4% -> 17.9%; the paper's own numbers cross architectures", "quote": "examples based on the fast gradient sign method. With adversarial training, the error rate", "citation": "paper/source/iclr2015.tex:516"},
    {"id": "c14", "kind": "value", "arm": "maxout_large_adv", "seeds": [0,1,2,3,4], "compute_invariance": "low", "quantity": "measured.maxout_large_adv.adv_err", "claimed": 17.9, "tolerance": 8.0, "quote": "fell to 17.9\\%. Adversarial examples are transferable between the two models but with the", "citation": "paper/source/iclr2015.tex:517"},
    {"id": "c15", "kind": "value", "arm": "maxout_large_naive", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.maxout_large_naive.adv_err", "claimed": 89.4, "tolerance": 10.0, "note": "paper recalls the 89.4% figure for 'this same kind of model' (the 1600-unit one)", "quote": "adversarial training, this same kind of model had an error rate of 89.4\\% on adversarial", "citation": "paper/source/iclr2015.tex:515"},
    {"id": "c16", "kind": "ordering", "arm": "maxout_large_adv", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.maxout_large_naive.adv_err - measured.maxout_large_adv.adv_err", "direction": ">0", "quote": "fell to 17.9\\%. Adversarial examples are transferable between the two models but with the", "citation": "paper/source/iclr2015.tex:517"},
    {"id": "c17", "kind": "value", "arm": "maxout_large_naive", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.maxout_large_naive.clean_err", "claimed": 1.14, "tolerance": 0.3, "quote": "and get an error rate of 1.14\\% on the test set. With adversarial training, we found that", "citation": "paper/source/iclr2015.tex:500"},
    {"id": "c18", "kind": "existence", "arm": "maxout_large_adv", "seeds": [0,1,2,3,4], "compute_invariance": "low", "predicate": "count(measured.maxout_large_adv.clean_err <= 0.9) >= 4 and mean(measured.maxout_large_adv.clean_err) <= 0.98 and max(measured.maxout_large_adv.clean_err) <= 1.1", "quote": "four trials that each had an error rate of 0.77\\% on the test set and one trial that had", "citation": "paper/source/iclr2015.tex:509"},
    {"id": "c19", "kind": "value", "arm": "maxout_large_adv", "seeds": [0,1,2,3,4], "compute_invariance": "low", "quantity": "measured.maxout_large_adv.clean_err", "claimed": 0.782, "tolerance": 0.15, "quote": "an error rate of 0.83\\%. The average of 0.782\\% is the best result reported on the permutation", "citation": "paper/source/iclr2015.tex:510"},
    {"id": "c20", "kind": "value", "arm": "maxout_large_adv", "seeds": [0,1,2,3,4], "compute_invariance": "low", "quantity": "measured.maxout_large_adv.adv_conf_mistakes", "claimed": 81.4, "tolerance": 12.0, "quote": "example was 81.4\\%. We also found that the weights of the learned model changed significantly,", "citation": "paper/source/iclr2015.tex:523"},
    {"id": "c21", "kind": "value", "arm": "transfer_mnist", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.transfer_mnist.err_orig_on_advfromnew", "claimed": 40.9, "tolerance": 10.0, "quote": "adversarial examples generated via the new model yield an error rate of 40.9\\% on the original", "citation": "paper/source/iclr2015.tex:520"},
    {"id": "c22", "kind": "value", "arm": "transfer_mnist", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.transfer_mnist.err_new_on_advfromorig", "claimed": 19.6, "tolerance": 8.0, "quote": "the original model yield an error rate of 19.6\\% on the adversarially trained model, while", "citation": "paper/source/iclr2015.tex:519"},
    {"id": "c23", "kind": "ordering", "arm": "transfer_mnist", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.transfer_mnist.err_orig_on_advfromnew - measured.transfer_mnist.err_new_on_advfromorig", "direction": ">0", "quote": "adversarially trained model showing greater robustness. Adversarial examples generated via", "citation": "paper/source/iclr2015.tex:518"},
    {"id": "c24", "kind": "value", "arm": "noise_rademacher", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.noise_rademacher.adv_err", "claimed": 86.2, "tolerance": 8.0, "quote": "to each pixel, or adding noise in $U(-\\eps, \\eps)$ to each pixel. These obtained an error rate of 86.2\\% with confidence", "citation": "paper/source/iclr2015.tex:556"},
    {"id": "c25", "kind": "value", "arm": "noise_rademacher", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.noise_rademacher.adv_conf_all", "claimed": 97.3, "tolerance": 8.0, "quote": "97.3\\% and an error rate of 90.4\\% with a confidence of 97.8\\% respectively on fast gradient sign adversarial examples.", "citation": "paper/source/iclr2015.tex:557"},
    {"id": "c26", "kind": "value", "arm": "noise_uniform", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.noise_uniform.adv_err", "claimed": 90.4, "tolerance": 8.0, "quote": "97.3\\% and an error rate of 90.4\\% with a confidence of 97.8\\% respectively on fast gradient sign adversarial examples.", "citation": "paper/source/iclr2015.tex:557"},
    {"id": "c27", "kind": "value", "arm": "noise_uniform", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.noise_uniform.adv_conf_all", "claimed": 97.8, "tolerance": 8.0, "quote": "97.3\\% and an error rate of 90.4\\% with a confidence of 97.8\\% respectively on fast gradient sign adversarial examples.", "citation": "paper/source/iclr2015.tex:557"},
    {"id": "c28", "kind": "ordering", "arm": "noise_uniform", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "min(measured.noise_rademacher.adv_err, measured.noise_uniform.adv_err) - measured.maxout_large_adv.adv_err", "direction": ">0", "quote": "zero mean and zero covariance is very inefficient at preventing adversarial examples. The expected dot product", "citation": "paper/source/iclr2015.tex:550"},
    {"id": "c29", "kind": "value", "arm": "rbf_shallow", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.rbf_shallow.adv_err", "claimed": 55.4, "tolerance": 15.0, "quote": "rate of 55.4\\% on MNIST using adversarial examples generated with the fast gradient sign", "citation": "paper/source/iclr2015.tex:602"},
    {"id": "c30", "kind": "value", "arm": "rbf_shallow", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.rbf_shallow.adv_conf_mistakes", "claimed": 1.2, "tolerance": 2.0, "quote": "method and $\\eps = .25$. However, its confidence on mistaken examples is only $1.2\\%$.", "citation": "paper/source/iclr2015.tex:603"},
    {"id": "c31", "kind": "value", "arm": "rbf_shallow", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.rbf_shallow.clean_conf_all", "claimed": 60.6, "tolerance": 15.0, "quote": "Its average confidence on clean test examples is $60.6$\\%.", "citation": "paper/source/iclr2015.tex:604"},
    {"id": "c32", "kind": "ordering", "arm": "rbf_shallow", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.maxout_naive.adv_conf_mistakes - measured.rbf_shallow.adv_conf_mistakes", "direction": ">0", "quote": "RBF networks are naturally immune to adversarial examples, in the sense that they have low", "citation": "paper/source/iclr2015.tex:600"},
    {"id": "c33", "kind": "ordering", "arm": "rbf_shallow", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.rbf_shallow.clean_conf_all - measured.rbf_shallow.adv_conf_mistakes", "direction": ">0", "quote": "method and $\\eps = .25$. However, its confidence on mistaken examples is only $1.2\\%$.", "citation": "paper/source/iclr2015.tex:603"},
    {"id": "c34", "kind": "value", "arm": "ensemble12", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.ensemble12.adv_err_ensemble_crafted", "claimed": 91.1, "tolerance": 8.0, "quote": "gradient descent. The ensemble gets an error rate of 91.1\\% on adversarial examples designed", "citation": "paper/source/iclr2015.tex:822"},
    {"id": "c35", "kind": "value", "arm": "ensemble12", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.ensemble12.adv_err_single_crafted", "claimed": 87.9, "tolerance": 8.0, "quote": "member of the ensemble, the error rate falls to 87.9\\%. Ensembling provides only", "citation": "paper/source/iclr2015.tex:824"},
    {"id": "c36", "kind": "ordering", "arm": "ensemble12", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.ensemble12.adv_err_ensemble_crafted - measured.ensemble12.clean_err", "direction": ">0", "quote": "gradient descent. The ensemble gets an error rate of 91.1\\% on adversarial examples designed", "citation": "paper/source/iclr2015.tex:822"},
    {"id": "c37", "kind": "ordering", "arm": "ensemble12", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.ensemble12.adv_err_ensemble_crafted - measured.ensemble12.adv_err_single_crafted", "direction": ">0", "note": "paper margin only 3.2pp, so medium", "quote": "member of the ensemble, the error rate falls to 87.9\\%. Ensembling provides only", "citation": "paper/source/iclr2015.tex:824"},
    {"id": "c38", "kind": "value", "arm": "agreement_mnist", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.agreement_mnist.agree_softmax_all", "claimed": 54.6, "tolerance": 10.0, "quote": "predict the maxout network's class correctly 54.6\\% of the time. These numbers are largely", "citation": "paper/source/iclr2015.tex:683"},
    {"id": "c39", "kind": "value", "arm": "agreement_mnist", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.agreement_mnist.agree_rbf_all", "claimed": 16.0, "tolerance": 8.0, "quote": "the maxout network's class assignment only 16.0\\% of the time, while the softmax classifier", "citation": "paper/source/iclr2015.tex:682"},
    {"id": "c40", "kind": "value", "arm": "agreement_mnist", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.agreement_mnist.agree_softmax_cond", "claimed": 84.6, "tolerance": 10.0, "quote": "predict's maxout's class 84.6\\% of the time, while the RBF network is able to predict maxout's", "citation": "paper/source/iclr2015.tex:686"},
    {"id": "c41", "kind": "value", "arm": "agreement_mnist", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.agreement_mnist.agree_rbf_cond", "claimed": 54.3, "tolerance": 12.0, "quote": "class only 54.3\\% of the time. For comparison, the RBF network can predict softmax regression's", "citation": "paper/source/iclr2015.tex:687"},
    {"id": "c42", "kind": "value", "arm": "agreement_mnist", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.agreement_mnist.agree_rbf_on_softmax", "claimed": 53.6, "tolerance": 12.0, "note": "conditioning ambiguous (SPEC 4.17); read as both-wrong conditioned", "quote": "class 53.6\\% of the time, so it does have a strong linear component to its own behavior.", "citation": "paper/source/iclr2015.tex:688"},
    {"id": "c43", "kind": "ordering", "arm": "agreement_mnist", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.agreement_mnist.agree_softmax_all - measured.agreement_mnist.agree_rbf_all", "direction": ">0", "quote": "predict the maxout network's class correctly 54.6\\% of the time. These numbers are largely", "citation": "paper/source/iclr2015.tex:683"},
    {"id": "c44", "kind": "ordering", "arm": "agreement_mnist", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.agreement_mnist.agree_softmax_cond - measured.agreement_mnist.agree_rbf_cond", "direction": ">0", "quote": "predict's maxout's class 84.6\\% of the time, while the RBF network is able to predict maxout's", "citation": "paper/source/iclr2015.tex:686"},
    {"id": "c45", "kind": "existence", "arm": "agreement_mnist", "seeds": [0,1,2], "compute_invariance": "high", "predicate": "measured.agreement_mnist.agree_softmax_cond > 40", "note": "'a significant proportion ... consistent with linear behavior'; 40 is far above 10-class chance (~10) and far below the paper's 84.6", "quote": "predict's maxout's class 84.6\\% of the time, while the RBF network is able to predict maxout's", "citation": "paper/source/iclr2015.tex:686"},
    {"id": "c46", "kind": "value", "arm": "maxout_naive", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.maxout_naive.rubbish_err", "claimed": 98.35, "tolerance": 5.0, "quote": "of 98.35\\% on Gaussian rubbish examples with an average confidence of 92.8\\% on mistakes.", "citation": "paper/source/iclr2015.tex:908"},
    {"id": "c47", "kind": "value", "arm": "maxout_naive", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.maxout_naive.rubbish_conf_mistakes", "claimed": 92.8, "tolerance": 10.0, "quote": "of 98.35\\% on Gaussian rubbish examples with an average confidence of 92.8\\% on mistakes.", "citation": "paper/source/iclr2015.tex:908"},
    {"id": "c48", "kind": "value", "arm": "maxout_sigmoid", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.maxout_sigmoid.rubbish_err", "claimed": 68.0, "tolerance": 15.0, "quote": "Changing the top layer to independent sigmoids dropped the error rate to 68\\% with an average", "citation": "paper/source/iclr2015.tex:909"},
    {"id": "c49", "kind": "value", "arm": "maxout_sigmoid", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.maxout_sigmoid.rubbish_conf_mistakes", "claimed": 87.9, "tolerance": 10.0, "quote": "confidence on mistakes of 87.9\\%.", "citation": "paper/source/iclr2015.tex:910"},
    {"id": "c50", "kind": "value", "arm": "softmax_reg", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.softmax_reg.rubbish_err", "claimed": 59.8, "tolerance": 15.0, "quote": "A softmax regression model has an error rate of 59.8\\%", "citation": "paper/source/iclr2015.tex:920"},
    {"id": "c51", "kind": "value", "arm": "softmax_reg", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.softmax_reg.rubbish_conf_mistakes", "claimed": 70.8, "tolerance": 15.0, "quote": "on the rubbish examples, with an average confidence on mistakes of 70.8\\%.", "citation": "paper/source/iclr2015.tex:921"},
    {"id": "c52", "kind": "value", "arm": "rbf_shallow", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.rbf_shallow.rubbish_err", "claimed": 0.0, "tolerance": 1.0, "quote": "we find an error rate of 0\\%. Note that when the error rate is zero the average confidence on a mistake", "citation": "paper/source/iclr2015.tex:923"},
    {"id": "c53", "kind": "ordering", "arm": "rbf_shallow", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.maxout_naive.rubbish_err - measured.rbf_shallow.rubbish_err", "direction": ">0", "quote": "far from the training data, are not fooled by this phenomenon.", "citation": "paper/source/iclr2015.tex:903"},
    {"id": "c54", "kind": "existence", "arm": "maxout_naive", "seeds": [0,1,2], "compute_invariance": "medium", "predicate": "measured.maxout_naive.rubbish_class_shares['8'] < 1.0", "note": "share of rubbish false positives predicted as 8; paper observed exactly none", "quote": "and none were classified as 8s. Likewise, on CIFAR-10, 49.7\\% of the convolutional network's", "citation": "paper/source/iclr2015.tex:931"},
    {"id": "c55", "kind": "value", "arm": "maxout_naive", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.maxout_naive.rubbish_class_shares['5']", "claimed": 45.3, "tolerance": 25.0, "quote": "On MNIST, 45.3\\% of a naively trained maxout network's false positives were classified as 5s,", "citation": "paper/source/iclr2015.tex:930"},
    {"id": "c56", "kind": "value", "arm": "cifar_conv_maxout", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.cifar_conv_maxout.adv_err", "claimed": 87.15, "tolerance": 10.0, "quote": "Similarly, using $\\eps=.1$, we obtain an error rate of 87.15\\% and an average probability of", "citation": "paper/source/iclr2015.tex:340"},
    {"id": "c57", "kind": "value", "arm": "cifar_conv_maxout", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.cifar_conv_maxout.adv_conf_mistakes", "claimed": 96.6, "tolerance": 8.0, "quote": "96.6\\% assigned to the incorrect labels", "citation": "paper/source/iclr2015.tex:341"},
    {"id": "c58", "kind": "value", "arm": "cifar_conv_maxout", "seeds": [0,1,2], "compute_invariance": "medium", "quantity": "measured.cifar_conv_maxout.rubbish_err", "claimed": 93.4, "tolerance": 8.0, "quote": "obtains an error rate of 93.4\\%, with an average confidence of 84.4\\%.", "citation": "paper/source/iclr2015.tex:912"},
    {"id": "c59", "kind": "value", "arm": "cifar_conv_maxout", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.cifar_conv_maxout.rubbish_conf_mistakes", "claimed": 84.4, "tolerance": 10.0, "quote": "obtains an error rate of 93.4\\%, with an average confidence of 84.4\\%.", "citation": "paper/source/iclr2015.tex:912"},
    {"id": "c60", "kind": "value", "arm": "cifar_conv_maxout", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.cifar_conv_maxout.fool_success_avg", "claimed": 75.3, "tolerance": 20.0, "quote": "step. Averaged over all ten classes, the method has an average per-step success rate of 75.3\\%.", "citation": "paper/source/iclr2015.tex:941"},
    {"id": "c61", "kind": "value", "arm": "cifar_conv_maxout", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.cifar_conv_maxout.fool_success['0']", "claimed": 24.7, "tolerance": 15.0, "note": "class 0 = airplane in CIFAR-10 label order", "quote": "frogs and trucks, and the hardest class was airplanes, with a success rate of 24.7\\% per sampling", "citation": "paper/source/iclr2015.tex:940"},
    {"id": "c62", "kind": "existence", "arm": "cifar_conv_maxout", "seeds": [0,1,2], "compute_invariance": "medium", "predicate": "measured.cifar_conv_maxout.fool_success['6'] >= 99 and measured.cifar_conv_maxout.fool_success['9'] >= 99", "note": "class 6 = frog, 9 = truck; paper: 100% per-step success for both", "quote": "with variable runtime. On CIFAR-10, we found that one sampling step had a 100\\% success rate for", "citation": "paper/source/iclr2015.tex:939"},
    {"id": "c63", "kind": "existence", "arm": "cifar_conv_maxout", "seeds": [0,1,2], "compute_invariance": "high", "predicate": "measured.cifar_conv_maxout.fool_success['0'] <= min(measured.cifar_conv_maxout.fool_success[c] for c in 0..9)", "note": "airplane is the hardest class: its success rate is the minimum over classes", "quote": "frogs and trucks, and the hardest class was airplanes, with a success rate of 24.7\\% per sampling", "citation": "paper/source/iclr2015.tex:940"},
    {"id": "c64", "kind": "existence", "arm": "l1_maxout", "seeds": [0,1,2], "compute_invariance": "medium", "predicate": "measured.l1_maxout.train_err > 5.0", "quote": ".0025 was too large, and caused the model to get stuck with over 5\\% error on", "citation": "paper/source/iclr2015.tex:430"},
    {"id": "c65", "kind": "curve", "arm": "eps_trace", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.eps_trace.logit_correct_seq", "against": "measured.eps_trace.logit_maxwrong_seq", "x": [0], "comparison": "above", "note": "at eps=0 the example is unperturbed and correctly classified as 4, so the correct-class curve sits above the max wrong-class curve (equivalently margin_seq > 0)", "quote": "the correct direction. Correct classifications occur only on a thin manifold where $\\vx$ occurs in the data.", "citation": "paper/source/iclr2015.tex:764"},
    {"id": "c66", "kind": "curve", "arm": "eps_trace", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.eps_trace.logit_correct_seq", "against": "measured.eps_trace.logit_maxwrong_seq", "x": [-10, 10], "comparison": "below", "note": "at BOTH tails the correct-class curve sits under the max wrong-class curve - the thin-manifold claim; figure read: class-4 curve at ~+200 vs wrong classes up to ~+800 at eps=-10; ~-350 vs ~+420 at eps=+10 (figures/read-figure.jsonl)", "quote": "the correct direction. Correct classifications occur only on a thin manifold where $\\vx$ occurs in the data.", "citation": "paper/source/iclr2015.tex:764"},
    {"id": "c67", "kind": "curve", "arm": "eps_trace", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.eps_trace.logit_correct_seq", "against": "measured.eps_trace.logit_maxwrong_seq", "x": [0,1,2,3,4,5,6,7,8,9,10], "comparison": "crosses", "note": "correct-class curve starts above the max wrong-class curve at eps=0 and ends below it at eps=10, crossing near eps of order 1 (vision read: at eps=+5 correct class already below top wrong class)", "quote": "the wrong classifications are stable across a wide region of $\\eps$ values. Moreover, the predictions become very extreme as we", "citation": "paper/source/iclr2015.tex:769"},
    {"id": "c68", "kind": "curve", "arm": "eps_trace", "seeds": [0,1,2], "compute_invariance": "high", "quantity": "measured.eps_trace.logit_maxwrong_seq", "x": [0,1,2,3,4,5,6,7,8,9,10], "comparison": "increasing", "note": "predictions become very extreme moving into the rubbish regime", "quote": "the wrong classifications are stable across a wide region of $\\eps$ values. Moreover, the predictions become very extreme as we", "citation": "paper/source/iclr2015.tex:769"},
    {"id": "c69", "kind": "curve", "arm": "eps_trace", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.eps_trace.logit_maxwrong_seq", "x": [10], "comparison": "matches", "claimed": [420], "tolerance": 400, "note": "value read off Figure 4 left by the vision model (top wrong-class logit ~ +400..430 at eps=+10); tolerance covers reading error and reproduction spread - anchor only", "quote": "the wrong classifications are stable across a wide region of $\\eps$ values. Moreover, the predictions become very extreme as we", "citation": "paper/source/iclr2015.tex:769"},
    {"id": "c70", "kind": "curve", "arm": "eps_trace", "seeds": [0,1,2], "compute_invariance": "low", "quantity": "measured.eps_trace.logit_correct_seq", "x": [10], "comparison": "matches", "claimed": [-350], "tolerance": 350, "note": "class-4 logit read as ~ -350 at eps=+10 (figures/read-figure.jsonl); anchor only", "quote": "The correct class is 4. We see that the unnormalized log probabilities for each class are conspicuously piecewise linear with $\\eps$ and that", "citation": "paper/source/iclr2015.tex:768"}
  ],
  "not_tested": [
    {"what": "MP-DBM is vulnerable to FGSM eps=0.25 (error 97.5%); its clean error is 0.88%", "citation": "paper/source/iclr2015.tex:794,800", "reason": "requires training a multi-prediction deep Boltzmann machine; outside compute scope"},
    {"what": "Fig.1 ImageNet demo: panda 57.7% -> gibbon 99.3% at eps=.007 on GoogLeNet", "citation": "paper/source/iclr2015.tex:364-383", "reason": "illustrative; needs pretrained GoogLeNet + ImageNet pipeline; not load-bearing"},
    {"what": "Fig.3 weight-localization claim (adversarially trained weights more localized/interpretable)", "citation": "paper/source/iclr2015.tex:523-525", "reason": "qualitative; paper defines no metric"},
    {"what": "rotational / scaled-gradient perturbation training weaker than FGSM adversarial training", "citation": "paper/source/iclr2015.tex:559-565", "reason": "no numbers reported"},
    {"what": "hidden-layer vs input perturbation comparison", "citation": "paper/source/iclr2015.tex:567-585", "reason": "qualitative only"},
    {"what": "maxout trained to 0% error on Gaussian rubbish with no clean-error benefit", "citation": "paper/source/iclr2015.tex:963-968", "reason": "tangential; stretch goal"},
    {"what": "46.2% error rate for direction transferred across clean examples", "citation": "paper/source/iclr2015.tex:776-779", "reason": "commented out (%) by the authors in v3 - not a claim of the published paper"},
    {"what": "quadratic/V1 model numbers (0.84% clean, 30.7% adversarial)", "citation": "paper/source/iclr2015.tex:620-643", "reason": "commented out (%) by the authors in v3"}
  ]
}
```

## 9. Deliberately not tested (with reasons)

- **MP-DBM generative-model claim** (clean 0.88 %, FGSM 97.5 %; `:794`, `:800`): requires
  training a multi-prediction deep Boltzmann machine — outside this reproduction's compute
  scope. Recorded, not gated.
- **Fig. 1 ImageNet demo** (panda 57.7 % → gibbon 99.3 % at ε=.007; `:364-383`): illustrative,
  needs pretrained GoogLeNet + ImageNet pipeline; not load-bearing for the paper's argument.
- **Fig. 3 weight localization** ("significantly more localized and interpretable", `:523-525`):
  qualitative; the paper defines no metric.
- **Fig. 2 visual resemblance** of w vs sign(w) (`:447-452`): qualitative.
- **Rotational / scaled-gradient perturbation training** (`:559-565`): paper reports no numbers
  ("did not find nearly as powerful of a regularizing result").
- **Hidden-layer vs input perturbation comparisons** (`:567-585`): qualitative only.
- **Rubbish-training paragraph** (maxout trained to 0 % on Gaussian rubbish, no clean benefit,
  `:963-968`): feasible but tangential; listed as stretch goal.
- **Commented-out 46.2 % cross-example direction-transfer number** (`:776-779`, a `%` comment):
  the authors removed it from v3 — not a claim of the published paper; we do not test it.
- **Commented-out quadratic/V1 model numbers** (`:620-643`): same reason.
- **Class-skew numbers beyond the stated ones** (only 45.3 %-as-5s / none-as-8s for MNIST and
  49.7 %-frogs / five zero classes for CIFAR are stated, `:930-933`): tested as c54/c55 + CIFAR
  skew folded into §8 arm metrics.
