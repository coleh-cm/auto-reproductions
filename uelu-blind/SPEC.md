# SPEC — A Structural Interpretation of GELU and Threshold-Transmission Activations via the First-Order Loss Function

- **Paper**: Roberto Rossi (2026), University of Edinburgh. arXiv:2607.03664.
- **paper_ref**: 2607.03664 · **project_id**: blind-eval · **slug**: uelu-blind
- **Authoritative source on disk**: `paper/main.tex` (993 lines; md5 `b6da2a35a75369ec4ca338d0619c0e75`).
  All line citations below are into that file and are grep-able, e.g.
  `grep -n 'operatorname{GELU}(z)=z\\Phi(z)' paper/main.tex` → line 38.

---

## 0. Upstream code — FOUND

**The author's official code exists and is explicitly for this paper.**

- Repository: <https://github.com/gwr3n/uelu> — description:
  *"This folder contains the supporting code and data for arXiv:2607.03664"*.
  Author GitHub account `gwr3n` (name "Roberto Rossi", company "University of Edinburgh",
  bio matches, blog `gwr3n.github.io`). First commit 2026-07-03, last push 2026-07-27.
- Local snapshot of the search record (queries run 2026-07-29, all reproducible):
  - Paper text: the only URL in `main.tex` is the Tiny Shakespeare dataset footnote
    (`main.tex:543`); the code-availability paragraph is **redacted** in the supplied
    source (`main.tex:692` comment: `[REDACTED FOR THIS EVALUATION: the authors' code-availability paragraph]`).
  - arXiv listing `https://arxiv.org/abs/2607.03664` resolves (title/authors match,
    dated 2026/07/04); the abs page lists **no code/paper-with-code link**.
  - GitHub repo search by title fragments → 0 hits; user search
    `roberto rossi edinburgh` → exactly one user, `gwr3n`; their repo list contains `uelu`.
- Layout of upstream repo (verified via GitHub API tree):
  `run_computational_study.py` (orchestrator, 640 lines),
  `mixer/gelu_vs_uelu_mixer.py`, `vit_cifar100/gelu_vs_uelu_vit.py`,
  `tiny_gpt_text/gelu_vs_uelu_tiny_gpt.py`, `tiny_gpt_common.py`,
  `tiny_gpt_TinyStories/gelu_vs_uelu_tinystories_gpt.py`,
  `tiny_gpt_WikiText-2/gelu_vs_uelu_wikitext2_gpt.py`,
  `README.md`, `requirements.txt` (`torch, torchvision, numpy, matplotlib, tqdm,
  joblib, scikit-learn, timm, torchsummary, datasets, tiktoken, tokenizers` — all unpinned),
  and `study_outputs/` with the authors' own checkpoints/figures/CSVs.
- **Consequence for this reproduction**: per the workflow ("use what the authors shipped"),
  the implementation step should vendor/clone `gwr3n/uelu` and drive it, not re-derive code.
  Where the paper is silent the upstream defaults below are recorded; any deviation we make
  to make it run must be logged in REPRODUCTION.md.
- One upstream quirk already noted: the CLI default `--uelu-beta` is **1.25** (a truncation
  of the paper's √(π/2) ≈ 1.2533141373…). The paper's table heading also prints β = 1.25.
  We keep 1.25 to match the reported protocol, and record this as a deliberate choice.

---

## 1. Method as an explicit algorithm

The paper's contribution is an *activation-function family*, so "the method" = (a) closed-form
activation gates, (b) the five controlled training comparisons. All activations are
elementwise (`out[i] = f(in[i])`, same shape in and out).

### 1a. Activation definitions (the learnable/free parameters in brackets)

```
GELU(z)      = z·Φ(z),  Φ(z)=½(1+erf(z/√2))                                   [none]
SiLU(z)      = z·σ(z),  σ(z)=1/(1+e^{-z})                                      [none]
ReLU(z)      = z·𝟙{z≥0}                                                         [none]
DGELU(z,t)   = z·Φ(z) + γ(t)·(φ(z) − φ(0)),                                    [γ(t) annealed 1→0 linearly]
               φ(z)=(1/√2π)·e^{−z²/2},  φ(0)=1/√2π ≈ 0.3989422804
UELU_β(z)    = z·clip((z+β)/(2β), 0, 1)                                        [β fixed: 1.25 or 3.0]
             = { 0                       if z < −β
               { z(z+β)/(2β)             if −β ≤ z ≤ β
               { z                       if z > β
TUELU(z)     = UELU_{β(ρ)}(z)                                                  [ρ: single shared scalar nn.Parameter]
               β(ρ) = β_min + softplus(ρ),  β_min = 1e-3,
               ρ₀ = log(expm1(β₀ − β_min)),  β₀ = 1.25
```

### 1b. Training algorithm (per experiment E, per activation arm A, per seed s ∈ {1,2,3,4,5})

1. Seed Python/NumPy/PyTorch with `s`; set cuDNN deterministic; rebuild model from scratch
   (weight init *inside* the seeded region).
2. Vision split (CIFAR-100, Mixer & ViT): seeded permutation (`split-seed 123`) of the 50k
   train set → 45k train / 5k val; the official 10k test set untouched. Same split for all arms.
3. Optimisation:
   - Vision: AdamW(lr=3e−4, wd=0.05) on **all** parameters (incl. LayerNorm/bias/ρ);
     CE loss with label_smoothing=0.1; batch 128; 20 epochs;
     CosineAnnealingLR(T_max=20) (→0 at end).
   - GPT (all three): AdamW(lr=3e−4, wd=0.1); token CE (no smoothing); grad-norm clip 1.0;
     lr schedule = linear warmup iters 1..100 to 3e−4, then cosine decay to 3e−5 at iter 10,000;
     batch 64 (Tiny Shakespeare) / 32 (TinyStories, WikiText-2).
4. DGELU arm only: before each epoch/iter set γ: vision `progress=(epoch−1)/19`,
   GPT `progress=(iter−1)/9999`, γ = start + progress·(end−start) with start=1, end=0, linear.
5. Checkpoint the weights whenever the validation metric improves
   (vision: val accuracy; GPT: val loss estimated every 500 iters on 50 random batches/split).
6. After training: reload best checkpoint; vision → test loss & test accuracy;
   GPT → final val loss & perplexity = e^{val_loss} computed from the reloaded checkpoint.
7. Region occupancy (uniform-threshold arms only): run ≤20 held-out validation batches through
   the model with forward hooks on every UELU/TUELU module; with that module's current β count
   preactivations in closed {z<−β}, transition {−β≤z≤β}, open {z>β}; report fractions.
   For TUELU also record learned β = β_min + softplus(ρ) at the end.
8. Aggregate over the 5 seeds: mean ± std with **population std** (`np.std(ddof=0)`);
   perplexity column = mean over seeds of per-seed exp(val_loss).

### 1c. Loss functions

- Vision: multi-class cross-entropy on logits `[B,100]` vs labels `[B]`, label_smoothing 0.1
  (upstream uses the same criterion for train/val/test reporting — so "test loss" includes smoothing).
- GPT: token-level cross-entropy on logits `[B·T,V]` vs targets `[B·T]` (next-token).
- There is **no loss term tied to the first-order loss functions**; L/L̂ are interpretive
  scaffolding only. The only optimisation objectives are the two above.

### 1d. Update rule

Plain AdamW updates; the activation width ρ (TUELU) is an ordinary model parameter updated by
the same optimiser (no projection, no special lr; the positivity floor comes only from the
softplus parameterisation + β_min).

---

## 2. Symbols with shapes

`B` = batch size, `T` = token/patch sequence length, `E` = embedding dim, `V` = vocab size.

| Symbol | Shape | Meaning |
|---|---|---|
| `z` | scalar ℝ in equations; elementwise over any `FloatTensor[...]` at runtime | preactivation of the activation function |
| `Φ(z)`, `φ(z)` | same as `z` | Gaussian CDF (via `erf`); Gaussian density |
| `σ(z)` | same as `z` | logistic sigmoid |
| `β` (UELU) | scalar float | fixed uniform-threshold half-width (1.25 or 3.0) |
| `β(ρ)` (TUELU) | scalar tensor `[]` | learned half-width, **one shared value across every activation site in the whole model** |
| `ρ` | `nn.Parameter` shape `[]` | raw softplus pre-image of β |
| `β_min` | scalar float = 1e−3 | positivity floor |
| `γ(t)` | scalar buffer `[]` | DGELU correction weight, annealed 1→0 |
| Mixer `X` | `[B,3,32,32]` (B=128) | input image batch |
| Mixer patches `P` | `[B,64,48]` | 4×4 patches flattened (8×8=64 patches, 48 px-values) |
| Mixer `h` | `[B,64,192]` | token embeddings through 6 blocks |
| Mixer token-MLP activation input | `[B,192,96]` | after `Linear(64→96)` on transposed tokens |
| Mixer channel-MLP activation input | `[B,64,384]` | after `Linear(192→384)` |
| Mixer logits | `[B,100]` | LayerNorm → mean over 64 tokens → `Linear(192→100)` |
| ViT tokens | `[B,65,192]` | cls token prepended to 64 patches; pos-emb `[1,65,192]` |
| ViT FFN activation input | `[B,65,384]` | `Linear(192→384)` inside each of 6 pre-norm blocks |
| ViT logits | `[B,100]` | final LayerNorm on **cls row only** (`x[:,0]`) → Linear |
| GPT `idx`, `targets` | `[B,T]` int64 | token ids; targets = idx shifted by one inside dataset sampling |
| GPT FFN activation input | `[B,T,4E]` | `Linear(E→4E)` inside each block's FeedForward |
| GPT logits | `[B,T,V]` | tied head: `head.weight is token_embedding.weight`, `Linear(E→V, bias=False)` |
| char-GPT dims | B=64, T=128, E=128, V=65, 4 layers, 4 heads | Tiny Shakespeare |
| TinyStories dims | B=32, T=256, E=384, V=8000 (bpe8k), 6 layers, 6 heads | |
| WikiText-2 dims | B=32, T=256, E=256, V=8000 (bpe8k), 4 layers, 4 heads | |
| Region counts | 3 scalar ints per measurement | closed/transition/open preactivation counts |

Region boundary convention (frozen, matches paper text at `main.tex:457` and upstream hooks):
closed `z < −β`, transition `−β ≤ z ≤ β`, open `z > β`.

---

## 3. Equations to implement, each with a grep-able citation

Citations are `paper/main.tex:LINE` plus the exact LaTeX fragment to grep.

Implemented in the forward path:

1. **GELU** — `z·Φ(z)`. `main.tex:38` `\operatorname{GELU}(z)=z\Phi(z),`
2. **SiLU/Swish** — `z·σ(z)`. `main.tex:398` (`SiLU/Swish \cite{Elfwing2018}, $ z\sigma(z) $`);
   gate row `main.tex:388` `Logistic & $\sigma(z)$ & SiLU/Swish`.
3. **ReLU** — degenerate threshold. `main.tex:278` `\operatorname{ReLU}(z)=z\mathbf{1}_{\{0\leq z\}}.`
4. **UELU piecewise** — `main.tex:334-340` `\operatorname{UELU}_{\beta}(z)` …
   `\dfrac{z(z+\beta)}{2\beta}` … with the **clip form (the one to code):**
   `main.tex:443` `\operatorname{UELU}_{\beta}(z)=z\operatorname{clip}\left(\frac{z+\beta}{2\beta},0,1\right),`
5. **Hard swish = UELU(β=3)** — sanity identity. `main.tex:402`
   `z\operatorname{clip}\left(\frac{z+3}{6},0,1\right).`
6. **TUELU softplus parameterisation** — `main.tex:457`
   `a single positive $\beta$ is shared across all UELU layers and learned through a softplus parameterisation`
   … `initialised to the GELU-matched value $\sqrt{\pi/2}$`.
   (Exact map `β = β_min + softplus(log(expm1(β₀−β_min)))`, β_min=1e−3, is upstream, not paper;
   see §4 item U-A.)
7. **DGELU** — `main.tex:435`
   `\operatorname{DGELU}(z,t)=z\Phi(z)+\gamma(t)\bigl(\phi(z)-\phi(0)\bigr),`
   with `main.tex:437` `with $\gamma(t)$ annealed linearly from one to zero`.
8. **GELU-matched calibration β*=√(π/2)** — near-origin expansions `main.tex:447`
   `\operatorname{GELU}(z)=z\Phi(z)\approx \frac{z}{2}+\phi(0)z^2,` and `main.tex:451`
   `\operatorname{UELU}_{\beta}(z)=\frac{z}{2}+\frac{z^2}{2\beta}.`; coefficient match
   `main.tex:455` `1/(2\beta)=\phi(0)=1/\sqrt{2\pi}`; conclusion `main.tex:457` `$\beta=\sqrt{\pi/2}$`.
9. **Region occupancy** — `main.tex:457`
   `closed region $z<-\beta$, transition region $-\beta\le z\le\beta$, and open region $z>\beta$`;
   measured on `main.tex:675` `Percentages are computed from held-out validation batches.`
10. **UELU in-region derivative** (reference/tests) — `main.tex:344-346`
    `\operatorname{UELU}_{\beta}'(z) = \frac{2z+\beta}{2\beta},`
11. **GELU derivative** (reference/tests) — `main.tex:174-176`
    `\frac{d}{dz}\operatorname{GELU}(z) = \Phi(z)+z\phi(z),`

Interpretation / unit-test scaffolding (not in any forward path):

12. Gaussian complementary FOL identity — `main.tex:88`
    `L(z)=\phi(z)-z(1-\Phi(z)), \qquad \widehat{L}(z)=\phi(z)+z\Phi(z).` and `main.tex:115`
    `\widehat{L}(z)=z\Phi(z)+\phi(z).` ⇒ test: `GELU(z) + φ(z) == L̂(z)` (`main.tex:119`
    `\operatorname{GELU}(z)=\widehat{L}(z)-\phi(z).`)
13. Centred loss + plateau — `main.tex:147-151` `\widetilde{L}(z) = \widehat{L}(z)-\widehat{L}(0)
    = \phi(z)-\phi(0)+z\Phi(z)`; derivative `main.tex:155` `\widetilde{L}'(z)=…=\Phi(z).`;
    `main.tex:164` `maps strongly negative inputs to a negative plateau, $-\phi(0)=-1/\sqrt{2\pi}$`.
14. DGELU note: `DGELU(z,γ=1) ≡ L̃(z)` by construction (centred correction added back).
15. Logistic FOL = softplus — `main.tex:757-763` (integral) and `main.tex:773`
    `$$\widehat{L}_{\log}(z)=\underbrace{z\sigma(z)}_{…}+\quad\underbrace{\bigl(\log(1+e^z)-z\sigma(z)\bigr)}_{…}.$$`
16. Uniform FOL — `main.tex:843-845` `\widehat{L}_{U}(z) = \frac{(z+\beta)^2}{4\beta}.` (in-region);
    `main.tex:857-863` (`\widehat{L}_{U}(z) = … = z.` for `z>\beta`, region noted at line 863).
17. ExpELU (defined in paper, **not** among the 7 study arms; implement for completeness/tests) —
    `main.tex:323-328` `\operatorname{ExpELU}_{\lambda}(z)`, `main.tex:814` appendix restatement.
    `main.tex:321`: `an exponential threshold with rate $\lambda>0$`.

Preamble macro resolution (per workflow note): `\best{x}` = bold table marker (no math effect);
`normalcdf(x)` (declared `main.tex:13-15`) is a pgfmath Φ-approximation used **only** to plot Figure 1
(`main.tex:217-247`) — never implement it; use exact `erf`.

---

## 4. What the paper does NOT state (and whether upstream resolves it)

Format: **U-x**: gap → upstream resolution (if any). Items marked [paper-silent everywhere]
must be flagged again at review time; anything upstream resolves is still a
paper-silence — we are inheriting the author's choice, not the paper's.

- **U-A: TUELU softplus details.** The exact map, the positivity floor, the raw-parameter
  initialisation: absent (`main.tex:457` says only "softplus parameterisation", init √π/2).
  Upstream: `β = β_min + softplus(ρ)` with `β_min = 1e-3`, `ρ₀ = log(expm1(β₀−β_min))`,
  one `nn.Parameter` scalar per model (`class TrainableUELU`); ρ is inside
  `model.parameters()` → **it receives the same weight decay as every other parameter**
  (no parameter-group carve-out; paper silent on weight-decay scope in general).
- **U-B: β₀ value mismatch, cosmetic but real.** Paper says initialise at √(π/2)≈1.25331
  (`main.tex:445,457`) and prints β=1.25 in every table. Upstream uses exactly `--uelu-beta 1.25`.
  Resolution: follow the paper protocol *as printed* → 1.25; the difference is below noise.
- **U-C: DGELU annealing discretisation.** "annealed linearly from one to zero" (`main.tex:437`)
  does not say per-epoch vs per-iter, nor endpoint inclusion. Upstream: recomputed before each
  epoch (vision, `progress=(epoch−1)/(epochs−1)`) and each iteration (GPT,
  `progress=(iter−1)/(max_iters−1)`), so γ(1)=1 exactly and γ=0 at the final step.
  CLI also exposes cosine/constant schedules but default is linear.
- **U-D: Seed values.** "five seeds" (`main.tex:461` etc.) — values never stated.
  Upstream: `1,2,3,4,5` (runner default) and CIFAR-100 split fixed by a separate
  `--split-seed 123` randperm, shared across arms (paper says only "We hold out 5,000
  examples … as validation data", `main.tex:461`).
- **U-E: CIFAR-100 "standard statistics"** (`main.tex:461`) — values unstated.
  Upstream: mean (0.5071, 0.4867, 0.4408), std (0.2675, 0.2565, 0.2761). Random cropping
  padding unstated → upstream `RandomCrop(32, padding=4)`; horizontal flip as stated.
- **U-F: Vision LR-floor.** "cosine decay" (`main.tex:461`) gives no floor.
  Upstream: `CosineAnnealingLR(T_max=epochs)` → decays to 0. AdamW βs/ε unstated → PyTorch
  defaults (0.9, 0.999, 1e-8). EMA/AMP/dtype unstated → none used, fp32, no mixed precision.
- **U-G: Vision architecture details the paper omits.** Mixer: pre-norm vs post-norm,
  LayerNorm vs other, stem type, final pooling — upstream: pre-LayerNorm blocks, linear stem
  `Linear(48→192)`, final LayerNorm + mean-pool over 64 tokens. ViT: pre-norm blocks with
  `nn.MultiheadAttention` (6 heads), patch embed `Linear(48→192)`, cls row pooling,
  dropout 0.0 everywhere in vision (paper mentions dropout only for GPTs, so 0.0 is consistent).
  Vision weight init: upstream defaults except ViT pos-emb/cls/head `trunc_normal(std=0.02)`.
- **U-H: Best-checkpoint protocol.** Paper prints "Val. acc." + "Test acc." but never says
  test numbers come from the best-val epoch. Upstream: checkpoint on val improvement, reload
  at end, then evaluate test. Same for GPTs (best val loss on a light estimate; final numbers
  re-estimated from the reloaded checkpoint).
- **U-I: GPT weight tying, init, FFN width, attention details.** Paper gives block counts/heads/
  dims/dropout (`main.tex:543,584,627`) but not: FFN hidden width (upstream: 4·E), embedding/
  LM-head weight tying (upstream: tied), init (upstream: normal σ=0.02, zeros biases, head
  bias-free), attention scale 1/√(E/heads) with bool causal mask, dropout sites (upstream: on
  embeddings, attention weights, residual outputs, FFN).
- **U-J: GPT evaluation protocol.** Cadence and estimator unstated. Upstream: evaluate at
  iter 1, every 500 iters, and final; each estimate averages 50 random batches per split;
  final reported val loss re-estimated (50 batches) from the reloaded best checkpoint;
  perplexity = exp(min(20, loss)) per seed (the min(20,·) guard is numerically inert:
  all losses ≪ 20). Vision region occupancy reads ≤20 val batches, GPT 20 val batches.
- **U-K: TinyStories/WikiText-2 tokenisation.** Paper: byte-level BPE, 8000 tokens, trained on
  train split (`main.tex:584,627`); "up to 50,000 training stories and 5,000 validation stories"
  (`main.tex:584`) — "up to" implies truncation without stating order. Upstream: HuggingFace
  `datasets` load, byte-level BPE via `tokenizers`, cached to disk; stories taken in dataset
  order up to the cap; text joined with EOS between documents (verify at implementation time
  against `tiny_gpt_common.py:317` `load_or_train_bpe_tokenizer`). Char-GPT: 90/10 split stated
  (`main.tex:543`); charset = sorted unique chars (V=65), unstated.
- **U-L: Aggregation statistics.** "mean ± standard deviation over five seeds" never defines the
  std convention. Upstream: population std (`np.std(ddof=0)`). Region percentages in tables =
  mean over seeds of per-seed fractions (registry: `aggregate_final_metrics.csv`).
- **U-M: GELU numerics.** Exact erf form vs tanh approximation unstated. Upstream: `nn.GELU()`
  (= exact erf, PyTorch default `approximate='none'`) and explicit `erf` in DGELU. Frozen: exact erf.
- **U-N: Hardware/precision.** The hardware sentence is commented out of the paper
  (`main.tex:438`: Xeon w5-2465X, RTX 2000 Ada). Nothing about runtime budget. This workspace
  currently has **no GPU** and no torch installed — a scaling decision for later steps:
  upstream supports `--dry-run` and the runner has an `exploratory` profile (GPT 5000 iters
  instead of 10000); full 5×7×5 CPU replication of the paper protocol is expensive
  (author GPT checkpoints run 3.5–57 MB; the README's `--dry-run-modules` smoke path exists).
- **U-O: Code availability statement itself is redacted** (`main.tex:692`), so the paper on disk
  never *says* the upstream repo exists — the finding in §0 comes from the GitHub user search.
- **U-P: expelu lambda.** ExpELU is only a "contrast case" (`main.tex:330`) shown for λ=1 in
  Figure 2 (`main.tex:363`); it is not an experiment arm. Implement with default λ=1.
- **U-Q: `normalcdf` plotting macro** (`main.tex:12-14`) — a transcription of Φ for pgfplots;
  explicitly not part of the method (see §3 note). Do not implement.

---

## 5. Frozen component interfaces

All parallel builds code against these. Layout mirrors upstream so later steps can diff code
against `gwr3n/uelu` module by module.

```
uelu_repro/
  paper/main.tex                  # authoritative source (do not edit)
  upstream/                       # vendored clone of gwr3n/uelu @ pinned commit
  run_computational_study.py      # driver: --profile {exploratory,final} --experiments …
                                  #   --seeds 1 2 3 4 5 --output-root …
  activations protocol (duplicated per experiment module exactly as upstream):
    class UELU(beta: float)                      # forward: x*clamp((x+β)/(2β),0,1)
    class TrainableUELU(raw_beta: nn.Parameter, beta_min: float)
                                                 # β=β_min+softplus(raw_beta); x*clamp((x+β)/(2β),0,1)
    class DynamicGELU(gamma: float)              # zΦ(z)+γ(φ(z)−φ(0)); set_gamma()
    make_activation(name, uelu_beta=1.25, trainable_raw_beta=None, trainable_beta_min=1e-3)
                                                 # names: relu|silu|gelu|dgelu|uelu|tuelu|uelu3
    raw_beta_from_beta(beta, beta_min) -> float  # log(expm1(beta − beta_min));
                                                 #   raises if beta ≤ beta_min
    dynamic_gamma_for_progress(progress, args)   # linear default; gamma∈[0,1]
    class ActivationRegionTracker                # hooks UELU/TrainableUELU; per-module β;
                                                 # fractions() -> (closed, transition, open)
  models:
    SmallMixer(activation, hidden_dim=192, depth=6, tokens_mlp_dim=96,
               channels_mlp_dim=384, uelu_beta, trainable_beta_min, dropout=0.0)
        # forward: [B,3,32,32] -> logits [B,100]; attr raw_uelu_beta (Parameter or None)
    CompactViT(activation, dim=192, depth=6, heads=6, mlp_dim=384, ...)
        # forward: [B,3,32,32] -> logits [B,100]; cls-row pooling
    TokenGPT(vocab_size, block_size, activation, n_layer, n_head, n_embd, ...)
        # forward: idx [B,T] long, targets [B,T] long|None -> (logits [B,T,V], loss scalar|None)
        # head.weight tied to token_embedding.weight
  experiment entrypoints (each a CLI script, all supporting --activations --seeds --dry-run
  --output-dir; paper protocol = runner --profile final):
    mixer/gelu_vs_uelu_mixer.py            --epochs 20 --batch-size 128
    vit_cifar100/gelu_vs_uelu_vit.py       --epochs 20 --batch-size 128
    tiny_gpt_text/gelu_vs_uelu_tiny_gpt.py --max-iters 10000 --batch-size 64 --block-size 128
    tiny_gpt_TinyStories/...               --max-iters 10000 --batch-size 32 --block-size 256
    tiny_gpt_WikiText-2/...                --max-iters 10000 --batch-size 32 --block-size 256
  metrics files per module: epoch_metrics.csv|eval_metrics.csv, final_metrics.csv, summary.json
  aggregate: study_outputs/aggregate_final_metrics.csv + tables/*.tex + figures/*.png
```

Frozen decisions: exact-erf Φ; β₀=1.25 (as printed); β_min=1e−3; shared scalar ρ;
region boundaries `(<−β, ≤β, >β)` with per-module current β; population std; perplexity =
mean of per-seed exp(val loss); seeds {1,2,3,4,5}; vision split-seed 123.

---

## 6. Experiment matrix and paper target numbers (5 seeds, mean±std)

Arms: ReLU, SiLU, GELU, DGELU, Hard swish/UELU(3.0), UELU(1.25), TUELU(1.25→learned).

| # | Experiment | Paper table (`main.tex`) | Key target (TUELU) |
|---|---|---|---|
| 1 | MLP-Mixer/CIFAR-100 | `tab:mixer-results` lines 465-484 | test acc 51.29±0.45; learned β 1.248±0.009; regions 9.43/88.6/1.98 |
| 2 | Compact ViT/CIFAR-100 | `tab:vit-results` 506-525 | test acc 49.41±0.34; β 1.136±0.006; regions 12.6/84.3/3.15 |
| 3 | Tiny char-GPT/Shakespeare | `tab:gpt-results` 547-566 | val ppl 4.575±0.037; β 0.696±0.007; regions 10.9/87.9/1.18 |
| 4 | TinyStories GPT | `tab:tinystories-gpt-results` 590-609 | val ppl 6.863±0.048; β 1.021±0.001; regions 25.7/73.7/0.66 |
| 5 | WikiText-2 GPT | `tab:wikitext2-gpt-results` 631-650 | val ppl 82.816±1.195; β 0.887±0.003; regions 18.1/81.3/0.59 |

Full arm-by-arm numbers for comparison live in the tables at the cited lines; the reproduction
report should reproduce all 7 columns per table (β, learned β, metrics, regions).

Also to keep honest in the final report: the paper's claims are comparative
("consistently competitive", `main.tex:686`), so the primary check is the *ordering/delta*
pattern above, not absolute equality across hardware.

## 7. Environment

- Python 3.12; deps per upstream `requirements.txt` (unpinned — pin at vendoring time and
  record versions in REPRODUCTION.md).
- This sandbox: CPU-only (16 cores, 63 GB RAM), no torch installed. Network available
  (GitHub/arXiv/HF reachable). Upstream runs on CUDA/MPS/CPU; expect the `final` GPT profile
  to be slow on CPU — plan smoke (dry-run) first, then scaled verification, then full runs.
