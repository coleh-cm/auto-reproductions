# SPEC — LEACE: Perfect linear concept erasure in closed form

**Paper:** Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman. *LEACE: Perfect linear concept erasure in closed form.* NeurIPS 2023. arXiv:2306.03819.
**Sources on disk:** `paper/main.tex` (authoritative LaTeX; all citations below are `<file>:<line>` into it), `paper/rebuttal.tex`, `paper/paper.html` (reading surface), `paper/figures/*.pdf` (read via `read-figure`; transcript in `paper/figure-transcript.md`).

---

## 1. The method as an explicit algorithm

LEACE has **no training loop and no gradient-based loss**. The "loss" is the constrained objective it solves in closed form, and the "update rule" is the formula itself. Iteration exists only in (a) the sequential per-layer fitting of concept scrubbing and (b) the baseline arms (INLP, logistic probes).

### 1.1 Objective (what the method minimizes)

Given jointly distributed random vectors $\rv X \in \R^d$ (features) and $\rv Z \in \R^k$ (one-hot labels, or continuous), solve

$$\mathop{\mathrm{argmin}}_{\mathbf{P} \in \mathbb{R}^{d \times d},\; \mathbf b \in \R^d} \E \Big [ \big\| \mathbf{P}\rv X + \mathbf b - \rv X \big\|^2_{\mathbf M} \Big ]\quad \mathrm{subject\;to}\:\: \mathrm{Cov}(\mathbf{P}\rv X + \mathbf b, \rv Z) = \mathbf{0}$$

for **any** p.s.d. inner-product matrix $\mathbf M$ — the same minimizer is optimal for all $\mathbf M$ simultaneously (paper/main.tex:478-487, Euclidean and Mahalanobis named at main.tex:474). Feasibility is characterized by Theorem 4.1: $\mathbf P\sxz = \mathbf 0$, i.e. $\mathrm{colsp}(\sxz) \subseteq \ker(\mathbf P)$ (main.tex:457-467).

### 1.2 LEACE fit (closed form)

**Inputs:** design matrix `X: [n, d]` float64, labels `Z: [n]` class ids (or one-hot `[n, k]`), flags `affine`, `constrain_trace`, `shrinkage`, `svd_tol`.

1. $\mu_x = \E[\rv X]$ — `[d]`; $\mu_z = \E[\rv Z]$ — `[k]` (Welford online updates; streaming-safe per main.tex:633).
2. $\sxx = \mathrm{Cov}(\rv X)$ — `[d, d]`; $\sxz = \mathrm{Cov}(\rv X, \rv Z)$ — `[d, k]` (Bessel $1/(n-1)$; see unstated #1).
3. $\mathbf{W} = (\sxx^{1/2})^+$ — `[d, d]` symmetric p.s.d. whitening map; $\mathbf W^+ = (\sxx^{1/2})$ — `[d, d]` (main.tex:486).
4. $\pwxz = (\mathbf{W}\sxz)(\mathbf{W}\sxz)^+$ — `[d, d]` orthogonal projector onto $\mathrm{colsp}(\mathbf W\sxz)$, rank $r = \mathrm{rank}(\sxz) \le k-1$ for one-hot $\rv Z$ (main.tex:486).
5. $\mathbf{P^*} = \mathbf{I} - \mathbf{W}^+ \pwxz\mathbf{W}$ — `[d, d]` (main.tex:484).
6. $\mathbf{b^*} = \E[\rv X] - \mathbf{P^*}\E[\rv X]$ — `[d]` (main.tex:496).
7. Optional trace constraint (Appendix "Constraining Norm Growth", main.tex:970-972): if $\mathrm{tr}(\mathbf{P^*}\sxx\mathbf{P^*}^T) > \mathrm{tr}(\sxx)$, replace $\mathbf P^*$ by $\mathbf{P}' = \alpha \mathbf{P^*} + (1-\alpha)\mathbf{Q}$ with $\alpha$ solving $\mathrm{tr}(\mathbf{P}'\sxx\mathbf{P}'^T) = \mathrm{tr}(\sxx)$ (quadratic in $\alpha$), where $\mathbf Q$ is an orthogonal projection at least as disruptive — see unstated #3 and #4.

**Apply:** $r_{\mathrm{LEACE}}(\xx) = \xx - \mathbf{W}^+ \pwxz\mathbf{W}\,(\xx - \E[\rv X])$ — Eq. (1), main.tex:515-517. For a batch `X: [n, d]`: `X' = X - (X - mu_x) @ (Wp @ PWX @ W).T`.

### 1.3 Concept scrubbing (Algorithm 1, main.tex:609-626)

**Inputs:** model $f = f_\ell \circ \dots \circ f_1$ with $\ell$ layers; token design matrix `X: [n, d]`; label matrix `Z: [n, k]` (one POS tag per token).
For $l = 1 \dots \ell$: fit $(\mathbf P, \mathbf b)$ on current $\mathbf H_l$ and $\mathbf Z$; store; **immediately** scrub $\mathbf H_l \gets \mathbf P(\mathbf H_l - \mu) + \mu$ (line 622, equivalent to applying Eq. 1 with the fitted bias); then $\mathbf H_{l+1} \gets f_l(\mathbf H_l)$. Naive independent per-layer fitting is explicitly rejected as possibly failing to erase (main.tex:631-633). Hook point: the input of each transformer block, immediately after normalization (main.tex:666) — see unstated #6.

### 1.4 Equations implemented, with citations

| # | Equation | Citation |
|---|----------|----------|
| E1 | $\mathbf{P^*} = \mathbf{I} - \mathbf{W}^+ \pwxz\mathbf{W}$ | main.tex:484 |
| E2 | $\mathbf{W} = (\sxx^{1/2})^+$; $\pwxz = (\mathbf{W}\sxz)(\mathbf{W}\sxz)^+$ | main.tex:486 |
| E3 | $\mathbf{b^*} = \E[\rv X] - \mathbf{P^*}\E[\rv X]$ | main.tex:496 |
| E4 | $r_{\mathrm{LEACE}}(\xx) = \xx - \mathbf{W}^+ \pwxz\mathbf{W}(\xx - \E[\rv X])$ (Eq. 1) | main.tex:515-517 |
| E5 | Guardedness iff $\mathbf P\sxz = \mathbf 0$ | main.tex:457-467 (Thm 4.1), via main.tex:423-434 (Thm 3.5) |
| E6 | Oracle LEACE: $\rv X' = \rv X - \sxz\szz^+(\rv Z - \E[\rv Z])$ | main.tex:1001-1003 |
| E7 | Full-rank SAL: $\mathbf{Q}_{\mathrm{SAL}} = \mathbf{I} - \mathbf{U}_{:n}\mathbf{U}_{:n}^T$, $\mathbf U_{:n}$ top-$n$ left singular vectors of $\sxz$, $n=\mathrm{rank}(\sxz)$ | main.tex:851 |
| E8 | Trace constraint: $\mathbf{P}' = \alpha\mathbf{P} + (1-\alpha)\mathbf{Q}$, $\mathrm{tr}(\mathbf{P}'\sxx\mathbf{P}'^T) = \mathrm{tr}(\sxx)$ | main.tex:972 |
| E9 | Scrubbing update $\mathbf{H}_l \gets \mathbf P(\mathbf H_l - \mu_{\mathbf H_l}) + \mu_{\mathbf H_l}$ | main.tex:622 |
| E10 | $\mathrm{GAP}_{z}^{\mathrm{TPR},\mathrm{RMS}} = \sqrt{\frac{1}{|C|}\sum_{y \in C}(\mathrm{GAP}_{z,y}^{\mathrm{TPR}})^2}$ | main.tex:573-575; per-class gap definition from De-Arteaga et al. as cited at main.tex:571 |
| E11 | INLP (baseline): iterate $t=1..m$: fit linear probe on $(\mathbf X_t, \mathbf Z)$, project $\mathbf X_{t+1} = (\mathbf I - \mathbf N_t\mathbf N_t^T)\mathbf X_t$, $\mathbf N_t$ = orthonormal basis of the probe's nullspace complement | main.tex:597 (named), baseline in main.tex:599 and fig caption main.tex:549 |
| E12 | Bits per byte: bpb = (nats/token) × (tokens/bytes) / ln 2; paper's ratios in comment: LLaMA 0.38511430072768005, Pythia 0.3366084909549386 | main.tex:637-639, unit at main.tex:656 |

### 1.5 Symbol table (shapes pinned)

Implementation layout is **row-major examples**: `X[n, d]`, one example per row. The paper's random vectors are column vectors; matmuls below are transposed accordingly at the code boundary and nowhere else.

| Symbol | Shape | Meaning |
|---|---|---|
| `n, d, k` | scalars | examples, feature dim (768 bios / 768×160m-model dims scrub), concept classes (2 gender, 18 POS) |
| `X` | `[n, d]` | design matrix (bios: [CLS] last hidden layer; scrub: post-norm hidden states flattened over tokens) |
| `Z` | `[n]` int64 ids or `[n, k]` one-hot | concept labels; one-hot has rank ≤ k−1 cross-covariance |
| `mu_x, mu_z` | `[d]`, `[k]` | running means |
| `sxx` ($\sxx$) | `[d, d]` | covariance of X |
| `sxz` ($\sxz$) | `[d, k]` | cross-covariance |
| `szz` ($\szz$) | `[k, k]` | covariance of Z (oracle only) |
| `W` ($\mathbf W = (\sxx^{1/2})^+$) | `[d, d]` symmetric | whitener; `X @ W` whitens |
| `Wp` ($\mathbf W^+ = \sxx^{1/2}$) | `[d, d]` symmetric | unwhitener |
| `PWX` ($\pwxz$) | `[d, d]` | orthogonal projector, rank r ≤ k−1 |
| `P` ($\mathbf{P}^*$) | `[d, d]` | oblique erasure projection, I minus rank-r |
| `b` ($\mathbf{b}^*$) | `[d]` | bias |
| `Q` ($\mathbf{Q}_{\mathrm{SAL}}$) | `[d, d]` | orthogonal SAL projection (eigenvalues in {0,1}, main.tex:972) |
| `alpha` | scalar | trace-constraint mixture weight ∈ [0,1] |
| `H_l` | `[n_tokens, d_model]` | hidden states at layer l during scrub fitting |
| `eraser` factor | `proj_left: [d, r]`, `proj_right: [r, d]` | low-rank storage: P = I − proj_left @ proj_right (never materialize [d, d] inside model hooks) |
| probe coef | `[k, d]` (multinomial) / `[d]` (binary) | sklearn LogisticRegression weights |
| per-layer metrics | `[12]` | mlm_acc / mlm_loss per BERT layer |

---

## 2. What the paper does NOT state (and the reading taken)

For each gap: the readings the text permits, then the **weakest** one (the one committing to least beyond the text), which is what we implement unless marked otherwise.

1. **Covariance estimator.** The theorems are population-level; nothing says how $\sxx, \sxz$ are estimated from samples. Permitted: plain empirical covariance (1/n or 1/(n−1)), shrunk estimators, anything consistent. Upstream code defaults to RMT-based "optimal linear shrinkage" for $\sxx$ (`shrinkage=True`, concept_erasure/leace.py:121,136,269-270) and Bessel $1/(n-1)$ for $\sxz$. **Weakest choice: plain sample covariance, Bessel 1/(n−1) for both, shrinkage OFF.** Sensitivity sweep runs shrinkage {off, on} on claims C1, C16.
2. **Rank/singular-value truncation.** The paper uses exact pseudoinverses; the numerical tolerance is never stated. Upstream truncates singular values of $\mathbf W\sxz$ at an absolute `svd_tol=0.01` (leace.py:104,211) — an extra commitment that limits erased rank. **Weakest choice: numerical-rank tolerance only (σ > σ_max·d·ε), no absolute truncation.** Sweep {0, 0.01} on C1/C3/C20.
3. **The orthogonal projection Q inside the trace constraint.** main.tex:972 says "SAL uses an orthogonal projection Q" and the convex-combination feasibility argument (**"the set of matrices which ensure linear guardedness is convex"**, same line) requires $\mathbf Q\sxz = \mathbf 0$. Two readings: (a) full-rank SAL's Q from $\sxz$'s left singular vectors (E7 above — guaranteed feasible by Thm 4.1); (b) upstream's choice Q = I − u uᵀ with u the **truncated, whitened** left singular vectors of $\mathbf W\sxz$ (leace.py:228), which is *not* in the feasible set in general. **Weakest (and feasibility-preserving) choice: (a).** If (b) is what the paper ran, the divergence is upstream's, not ours. Trace constraint branches are logged whenever they fire.
4. **Which quadratic root α.** Neither root is named. Upstream picks "the positive root" clamped to [0,1] (leace.py:241-245). Readings: smaller/larger root; the larger α stays closer to $\mathbf P^*$ (smaller-MSE side of the line segment). **Choice: root in [0,1] closest to 1 (max α); if both in [0,1], max; if none, 1 (no constraint).** Admitted by the text; closest to $\mathbf P^*$, hence weakest extra damage commitment consistent with the constraint.
5. **Bios experiment protocol.** "We embed each biography with the [CLS] embedding in the last layer of BERT" (main.tex:541). Unstated: which BERT checkpoint (§5.3 uses `bert-base-uncased`, main.tex:597; we extend that reading to §5.1-5.2 — weakest shared reading), the train/test split ("train on train, evaluate on test" is the only defensible reading; we use the dataset's own splits, subsampled), probe/classifier solver and regularization (any converged linear predictor admitted; we take sklearn LogisticRegression multinomial, lbfgs, max_iter 1000, C=1.0), whether gender-probe accuracy is measured on train or test (we use test), and the MSE norm in Fig 1 (we read it as the objective's own M = I, Euclidean). Also the dashed "random accuracy" line at **0.52** comes only from the figure; our claims measure `majority_acc` from our data instead.
6. **Scrub hook granularity.** "the input of each transformer block, immediately after normalization is applied (LayerNorm or RMSNorm)" (main.tex:666). GPT-NeoX has two norms per block (attention input, MLP input). Readings: block-input norm only, or both sublayer norms. Upstream scrubs **both** by default (`--skip-sublayers` to disable, concept_scrubber.py + scrub.py:117; `is_norm_layer` post-hooks). **Choice: both (upstream), with `sublayers: false` (block-input only, the most literal reading) in the sensitivity sweep for C16.** Erasure applies to the full `[batch, seq, d]` tensor pointwise.
7. **SpaCy tagger.** "we use the model from the SpaCy library to automatically generate Universal Dependency tags" (main.tex:662) — unnamed. Upstream tagged-pile README pins `en_core_web_trf`. **Choice: `en_core_web_trf`; fallback `en_core_web_sm` if the transformer tagger is CPU-infeasible (logged as a divergence; the metric definition is unaffected).**
8. **Word-piece → tag alignment for scrubbing.** Not described in the paper for the scrubbing experiments (it *is* described for amnesic probing, main.tex:599: "map each word-piece to the POS tag of the word to which it belongs"). Upstream tagged-pile: BPE token takes the tag of the SpaCy token containing it; if split across SpaCy tokens, the first one. **Choice: upstream rule = the §5.3 rule; weakest consistent reading.**
9. **Random-erasure subspace rank.** §5.3: "whose null space has the same rank as the label space (18)" (main.tex:599). §6: "a random linear subspace of the same rank as the cross-covariance matrix" (main.tex:668). One-hot $\rv Z$ with k=18 gives $\mathrm{rank}(\sxz) \le 17$ (LEACE "decreases the rank just by 17", main.tex:599). Readings: k=18 or k−1=17. **Choice: literal per-experiment text (18 for amnesic; rank($\sxz$) computed per layer for scrubbing, expected 17). Sweep {17,18} on C15/C17/C18.**
10. **MLM masking protocol (amnesic).** Masking rate and scheme never stated; "masked language modeling accuracy" admits only standard BERT masking: 15%, 80/10/10. **Choice: 15%, 80/10/10, mask resampled per seed.** Statistics are fit on **unmasked** embeddings (main.tex:599 states this); intervention happens during the masked forward pass.
11. **Precision/dtype.** Never stated; upstream scrub.py uses `torch_dtype=float16`. **Choice: float64 statistics, float32 model compute (CPU); fp16 recorded as upstream's convention.**
12. **Token counts in scrub.py vs paper.** Paper: 2²² tokens fit + 2²² eval (main.tex:662). Current upstream scrub.py uses 2¹⁴ examples × 2048 tokens = 2²⁵. **Choice: paper text (restricted to 2²⁰ + 2²⁰ on CPU).**
13. **Seeds.** None anywhere (upstream scrub.py: `torch.manual_seed(42)`, split seed 42). **Choice: seeds {0,1,2}, recorded in claims.json.**
14. **"Random accuracy" reference.** The dashed 0.52 line (read from `paper/figures/acc-vs-mse.pdf`) is not a number printed in text; C3 uses data-measured `majority_acc`.
15. **INLP probe and bookkeeping.** Iterations stated (bios grid at rebuttal.tex:18: 1,4,8,16,32,50,64,100; amnesic 20, main.tex:599) but probe type/regularization not. **Choice: multinomial LogisticRegression, sklearn defaults; restricted grid {1,50,100} for bios.**
16. **Bias in SAL arm.** "full-rank SAL ... lacks a bias term and does not adjust for correlations between features" (main.tex:664) — stated for the scrubbing baseline: SAL = pure orthogonal projection, no bias. **Choice: no bias in SAL arms.**

---

## 3. Upstream code

**Exists, linked from the abstract** (main.tex:283): `https://github.com/EleutherAI/concept-erasure` (MIT, torch-only core, `LeaceFitter`/`LeaceEraser`, streaming Welford stats, `concept_erasure.scrubbing` for LLaMA/GPT-NeoX). GitHub search on title/authors confirms it as the authors' canonical repo; POS tagging pipeline: `https://github.com/EleutherAI/tagged-pile` (pinned commit 5719699; tags with `en_core_web_trf`, aligns BPE→tag, 2048-token chunks). Baselines: INLP (`shauli-ravfogel/nullspace_projection`), RLACE (`shauli-ravfogel/rlace`) — not used; RLACE arm deliberately excluded.

**Decision:** use the authors' package for the eraser core, pinned to commit `9f51753821316a1edacf78b52b464ab26d40e60a` (main @ 2024-10-15), with paper-literal flags: `shrinkage=False`, `svd_tol→0`, `affine=True`, `constrain_cov_trace=True`. The trace-constraint Q and α-root (unstated #3, #4) differ from upstream; our wrapper implements the paper-literal branch and logs firings. Experiment pipelines (bios, amnesic, scrub harness, metrics) are written here against the interfaces in §4.

**Divergences upstream vs paper text (all recorded in §2):** shrinkage default ON (#1), svd_tol = 0.01 (#2), whitened truncated Q (#3), positive-root rule (#4), fp16 (#11), 2²⁵ tokens (#12), seed 42 (#13).

## 4. Component interfaces (frozen)

```python
# erasers.py — all stats float64; X: F64Tensor[n, d]; Z: I64Tensor[n] | F64Tensor[n, k]
class Eraser(Protocol):
    P: F64Tensor          # [d, d] materialized on demand
    bias: F64Tensor | None  # [d]
    def __call__(self, x: Tensor) -> Tensor  # [..., d] -> [..., d], apply = x - (x-bias) @ (Wp@PWX@W).T
def fit_eraser(X, Z, *, method: Literal["leace","sal","oracle","inlp","random"],
               affine=True, shrinkage=False, svd_tol=0.0, constrain_trace=True,
               iterations=0, rng: torch.Generator|None=None,
               random_rank: int|Literal["sigma_xz"]="sigma_xz") -> Eraser
# method="sal" -> E7 full-rank, affine must be False (unstated #16)
# method="inlp" -> iterations>=1; binary uses probe coef direction; multiclass uses row-space basis
# method="random" -> stateless: draws fresh orthonormal U[d, rank] per call from rng (main.tex:668)

# stats.py — Welford accumulators mirroring upstream LeaceFitter semantics
class CovStats:
    n: int; mean_x: F64Tensor  # [d]
    mean_z: F64Tensor          # [k]
    sigma_xz_: F64Tensor       # [d, k] unnormalized
    sigma_xx_: F64Tensor       # [d, d] unnormalized
    def update(self, x: F64Tensor[m, d], z: F64Tensor[m, k]) -> None

# experiments — each returns MeasuredDict: {seed: {metric: float | list[float]}}
def run_synth(seed)                 # arms synth_*
def run_bios(seed, n_train=60000, n_test=20000)   # arms bios_*; embeds once, reuses cache
def run_amnesic(seed, n_fit=4096, n_eval=1024)    # arms amnesic_*; 12-layer sweeps
def run_scrub(seed, fit_tokens_exp=20, eval_tokens_exp=20)  # arms scrub_*; Algorithm 1
```

Metric definitions are fixed in `claims.json.metrics`; every arm of an experiment reports every metric of that experiment (NA where structurally undefined, e.g. `sig_after` outside synth).

## 5. Arms, restrictions, and supported claims

The paper compares **four method families**: LEACE, SAL (full-rank), INLP, RLACE, plus two control conditions (no intervention, random projection). RLACE is deliberately excluded (gradient-based, hyperparameters not restated in the paper; claims under test rest on LEACE vs INLP/SAL/random — see `not_tested`). Per arm, how we restrict and why each restriction is only a restriction (subset of situations, same correctness criterion):

| Experiment / arm | Restriction | Only restricts? | Claims still supported |
|---|---|---|---|
| synth_* | not a paper experiment; supports Thm-level sanity (C1, C2, C20). Gaussian synthetic data | n/a (added by us; does not stand in for any paper arm) | C1, C2, C20 |
| bios_* | per-seed subsample 60k/20k of 257k/99k bios; INLP grid {1,50,100} ⊂ paper's 8-value grid; bert-base-uncased explicit (§5.3's named model) | yes: fewer examples, fewer hyperparameter points, same dataset/split protocol/metrics | C3-C12 all ordering/invariant claims; C7-C11 value claims with wide tolerance (magnitudes do shift with subsampling → rated low) |
| amnesic_* | 4096 fit / 1024 eval sentences ⊂ full UD English EWT; same model, layers, k=18, iterations | yes: fewer sentences; identical intervention, same MLM metric definition | C13-C15 (curve shape + argmax claims are robust to subsampling; absolute accuracies are not claimed) |
| scrub_* | Pythia-160m only (1 of 7 models); 2^20+2^20 tokens (paper: 2^22); float32; `en_core_web_trf` tagger per upstream | yes: subset of models, subset of tokens, same unit/arms/algorithm. Model-size trends and LLaMA numbers are NOT supported — a 160m-only run cannot speak to them and does not try | C16-C18 (orderings, stated per-model in the paper → one member checkable); C19 magnitude (low, tolerance covers budget) |

## 6. Figure evidence (transcript: paper/figure-transcript.md)

- `acc-vs-mse.pdf`: dashed random-accuracy line at **0.52**; LEACE point ≈ **(2.2, 0.52)**; leftmost INLP point at MSE ≈ **1.6** (smaller edit than LEACE, but erasure fails there — consistent with main.tex:544's "with a small edit" qualifier); RLACE minimum accuracy ≈ **0.54** ("comes close").
- `bios_frozen_bert_layer_vs_drop_in_acc.pdf`: y-axis is "MLM Accuracy" (so "drop" = downward movement); No-Intervention line ≈ **0.89** (the on-disk figure is the *uncorrected* one; rebuttal.tex:23 documents 0.892→0.928); 'Ours' 0.87 (layer 0) → **0.82 (layer 11, maximum damage)**; INLP minimum at **layer 6**.
- `bios_frozen_bert_layer_vs_loss_inlp.pdf`: 3 curves; green (INLP) spikes at layer 6; blue (Ours) maximal at layer 11; peak any-curve value ≈ 5.1. Matches main.tex:601.
- `tpr_vs_bias_*.pdf`: no annotated R; R = 0.867 / 0.392 exist only in the caption (main.tex:566) — hence cited from text, not figure.
- `leace-plot.pdf`, `oleace.pdf`: schematics; no quantitative claims.
- All replies were forced to bare numbers/phrases; none flagged as deliberation (`looks_like_reasoning: false` throughout).

## 7. claims.json

The gate-settling artifact — arms, seeds, metrics, and 20 claims with verbatim quotes and `<file>:<line>` citations — lives in `claims.json` at the repo root and is embedded below verbatim. Gating concentrates on the `high` claims: C1-C5, C13-C18, C20. `low` claims (C7-C11, C19) are magnitude checks whose tolerances absorb our restrictions.

```json
{
  "paper": {
    "paper_ref": "2d009c8a-c609-4bbc-ac9c-81a5e7bf7e80",
    "project_id": "06910d54-1d99-4864-8bff-ab3007a0c70e",
    "arxiv_id": "2306.03819",
    "title": "LEACE: Perfect linear concept erasure in closed form",
    "source_dir": "paper/"
  },
  "seeds": [
    0,
    1,
    2
  ],
  "seed_role": "The paper fixes no RNG seeds anywhere in its text. These three seeds drive (a) dataset subsampling, (b) random-projection subspace draws, (c) MLM mask sampling. LEACE/SAL/INLP fits are closed-form or deterministic probes given the data, so seed variance measures data/mask/random-baseline variance only.",
  "metrics": {
    "gender_acc": "Test accuracy of a multinomial LogisticRegression gender probe fit on each arm's erased [CLS] embeddings (bert-base-uncased, last hidden layer), same split protocol in every arm.",
    "majority_acc": "Majority-class frequency of the probe's test labels (the 'random accuracy' reference; the paper's dashed line is at 0.52, read off paper/figures/acc-vs-mse.pdf).",
    "mse": "Mean over test examples of ||r(x) - x||_2^2, the mean squared edit distance (Euclidean; the paper's objective with M = I).",
    "prof_acc": "Test accuracy of the logistic-regression profession classifier.",
    "tpr_gap_rms": "RMS over professions y of GAP_female,y^TPR = P(Yhat=y | Y=y, female) - P(Yhat=y | Y=y, male), per paper formula at paper/main.tex:573-575.",
    "tpr_corr": "Pearson correlation over professions between GAP_female,y^TPR and the fraction of women in profession y (fraction computed once from the full bios train split, seed-independent).",
    "mlm_acc": "Sequence over 12 layers: top-1 accuracy over masked positions of bert-base-uncased MLM after a single-layer intervention at that layer (15% masking, 80/10/10, mask drawn per seed).",
    "mlm_loss": "Sequence over 12 layers: mean cross-entropy over masked positions, same protocol.",
    "mlm_drop": "Elementwise mlm_acc of amnesic_none minus mlm_acc of the arm (positive = damage).",
    "bpb": "Language-modeling cross-entropy converted to bits per UTF-8 byte: mean nats per token x (tokens / bytes) / ln(2), the paper's Table 1 unit (paper/main.tex:656 text; conversion-ratio comment at paper/main.tex:637-639).",
    "sig_after": "Max absolute entry of the empirical cross-covariance between erased features and one-hot labels on held-out synthetic data."
  },
  "arms": {
    "synth_none": {
      "description": "Synthetic Gaussian 3-class data (n=20000, d=32, class centers spread so a linear probe reaches >90% accuracy), no erasure. Restricts the population theory of paper Sec. 3-4 (paper/main.tex:330-531) to finite Gaussian samples; metric definitions unchanged.",
      "config": {
        "n": 20000,
        "d": 32,
        "k": 3,
        "fit_fraction": 0.5
      }
    },
    "synth_leace": {
      "description": "LEACE eraser fit on synth train split: affine, plain sample covariance (Bessel 1/(n-1)), numerical-rank pseudoinverse, trace constraint on. Core method arm.",
      "config": {
        "method": "leace",
        "affine": true,
        "shrinkage": false,
        "svd_tol": 0.0,
        "constrain_trace": true
      }
    },
    "synth_oracle": {
      "description": "Oracle LEACE (OLS-residual eraser, paper/main.tex:1001-1003) on the same synthetic data.",
      "config": {
        "method": "oracle",
        "shrinkage": false,
        "svd_tol": 0.0
      }
    },
    "synth_sal": {
      "description": "Full-rank SAL: Q_SAL = I - U U^T, U = left singular vectors of Sigma_XZ at numerical rank (paper/main.tex:851). No bias term (paper/main.tex:664).",
      "config": {
        "method": "sal_full_rank",
        "affine": false
      }
    },
    "synth_leace_toy": {
      "description": "LEACE on the exact 4-point distribution of the paper's oblique-optimality example (paper/main.tex:932-945). Deterministic; seeds nominal.",
      "config": {
        "method": "leace",
        "affine": true,
        "shrinkage": false,
        "svd_tol": 0.0
      }
    },
    "synth_sal_toy": {
      "description": "Full-rank SAL on the same 4-point toy distribution (equals P_ortho of paper/main.tex:935-939).",
      "config": {
        "method": "sal_full_rank",
        "affine": false
      }
    },
    "bios_none": {
      "description": "bias_in_bios biographies embedded as [CLS] of the last hidden layer of frozen bert-base-uncased; no erasure. Restricts the paper's bios experiment (paper/main.tex:541-542) only in sample size: per-seed subsample 60000 train / 20000 test bios (of 257540/99392). Metric definitions unchanged.",
      "config": {
        "model": "bert-base-uncased",
        "layer": "last",
        "pool": "cls",
        "n_train": 60000,
        "n_test": 20000
      }
    },
    "bios_leace": {
      "description": "LEACE (affine, plain covariance) erasing binary gender from the [CLS] embeddings. Rank 1 concept, matching rebuttal.tex:18 ('Our method is rank 1').",
      "config": {
        "method": "leace",
        "affine": true,
        "shrinkage": false,
        "svd_tol": 0.0,
        "constrain_trace": true,
        "k": 2
      }
    },
    "bios_inlp_r1": {
      "description": "INLP, 1 iteration (one binary logistic-regression nullspace direction). Corresponds to the leftmost INLP point of paper/figures/acc-vs-mse.pdf (rank grid from rebuttal.tex:18).",
      "config": {
        "method": "inlp",
        "iterations": 1,
        "probe": "multinomial_logreg"
      }
    },
    "bios_inlp_r50": {
      "description": "INLP, 50 iterations (restriction of the paper's rank grid 1,4,8,16,32,50,64,100 from rebuttal.tex:18 to {1,50,100}).",
      "config": {
        "method": "inlp",
        "iterations": 50,
        "probe": "multinomial_logreg"
      }
    },
    "bios_inlp_r100": {
      "description": "INLP, 100 iterations; the paper's most aggressive INLP setting on this dataset.",
      "config": {
        "method": "inlp",
        "iterations": 100,
        "probe": "multinomial_logreg"
      }
    },
    "bios_leace_refit": {
      "description": "Same LEACE erasure as bios_leace, but the profession classifier is RE-FIT on the projected embeddings (paper/main.tex:578).",
      "config": {
        "method": "leace",
        "affine": true,
        "shrinkage": false,
        "svd_tol": 0.0,
        "refit_main_classifier": true
      }
    },
    "amnesic_none": {
      "description": "bert-base-uncased MLM on UD English EWT sentences, no intervention. Restricts the amnesic-probing setup (paper/main.tex:597-599) only in corpus size: per-seed subsample 4096 sentences for fitting statistics, 1024 for evaluation. All 12 layers, 18-way coarse POS concept, metrics unchanged.",
      "config": {
        "model": "bert-base-uncased",
        "n_fit": 4096,
        "n_eval": 1024,
        "mask_rate": 0.15,
        "mask_scheme": "80/10/10"
      }
    },
    "amnesic_random": {
      "description": "Random orthogonal projection whose null space has the same rank as the label space (18 directions), fresh per eval batch (paper/main.tex:599; fresh-per-minibatch sampling per paper/main.tex:668).",
      "config": {
        "method": "random_projection",
        "rank": 18,
        "fresh_per_batch": true
      }
    },
    "amnesic_leace": {
      "description": "LEACE erasing the 18-way POS concept from a single layer, swept over all 12 layers. Removes rank(Sigma_XZ) <= k-1 = 17 dimensions (paper/main.tex:599).",
      "config": {
        "method": "leace",
        "affine": true,
        "shrinkage": false,
        "svd_tol": 0.0,
        "constrain_trace": true,
        "k": 18
      }
    },
    "amnesic_inlp": {
      "description": "INLP, 20 iterations with an 18-class probe per iteration; removes 20 x 18 = 360 dimensions (paper/main.tex:599).",
      "config": {
        "method": "inlp",
        "iterations": 20,
        "probe": "multinomial_logreg",
        "k": 18
      }
    },
    "scrub_none": {
      "description": "Pythia-160m (GPT-NeoX) language modeling on a Pile-validation slice with POS tags, no intervention. Restricts the concept-scrubbing experiment (paper/main.tex:660-670) to 1 of 7 models and 2^20 fit + 2^20 eval tokens (paper used 2^22 + 2^22); hook point, unit (bpb), arms and metrics unchanged.",
      "config": {
        "model": "EleutherAI/pythia-160m",
        "fit_tokens_exp": 20,
        "eval_tokens_exp": 20,
        "tagger": "en_core_web_trf"
      }
    },
    "scrub_random": {
      "description": "Random orthogonal projection erasing a random linear subspace of the same rank as the cross-covariance matrix, fresh subspace each minibatch, loss averaged over subspaces (paper/main.tex:668).",
      "config": {
        "method": "random_projection",
        "rank": "rank(sigma_xz) per layer in {k-1, k}; paper admits both",
        "fresh_per_batch": true
      }
    },
    "scrub_leace": {
      "description": "Concept scrubbing with LEACE at the input of each transformer block immediately after normalization (paper/main.tex:666), fit sequentially per Algorithm 1 (paper/main.tex:609-626), both sublayer norms scrubbed (upstream default).",
      "config": {
        "method": "leace",
        "affine": true,
        "shrinkage": false,
        "svd_tol": 0.0,
        "constrain_trace": true,
        "sublayers": true
      }
    },
    "scrub_sal": {
      "description": "Concept scrubbing with full-rank SAL: no bias term, no whitening adjustment for feature correlations (paper/main.tex:664).",
      "config": {
        "method": "sal_full_rank",
        "affine": false,
        "sublayers": true
      }
    }
  },
  "claims": [
    {
      "id": "C1",
      "quote": "Then given some affine function $r(\\xx) = \\mathbf{P}\\xx + \\mathbf{b}$, the modified random vector $r(\\rv X)$ linearly guards $\\rv Z$ if and only if the columns of the cross-covariance matrix $\\sxz$ are contained in the null space of $\\mathbf{P}$.",
      "citation": "paper/main.tex:458",
      "kind": "invariant",
      "compute_invariance": "high",
      "arm_role": "sanity (synthetic)",
      "predicate": "all_seeds: (synth_leace.sig_after <= 0.01) and ((synth_none.probe_acc - synth_leace.probe_acc) > 0.5) and (abs(synth_leace.probe_acc - synth_none.majority_acc) <= 0.02)",
      "settlement": "After LEACE, empirical cross-covariance is numerically zero and a linear probe falls from >90% to majority accuracy, on every seed.",
      "sensitivity": {
        "unstated": "svd_tol (rank truncation in the pseudoinverse/SVD steps)",
        "plausible": [
          0.0,
          0.01
        ],
        "survives": [
          0.0,
          0.01
        ],
        "note": "Paper never truncates beyond the pseudoinverse; upstream default is 0.01. Verdict expected to hold across the whole interval; check at both endpoints."
      }
    },
    {
      "id": "C2",
      "quote": "\\E \\|\\mathbf{P}_\\text{ortho}\\rv X - \\rv X\\|^2 = 2,\\quad\n\\E \\|\\mathbf{P}_\\text{oblique}\\rv X - \\rv X\\|^2 = 1",
      "citation": "paper/main.tex:948-949",
      "kind": "value",
      "compute_invariance": "high",
      "arm_role": "sanity (paper's own toy distribution, App. E)",
      "quantity": "measured.synth_leace_toy.mse - measured.synth_sal_toy.mse",
      "claimed": -1.0,
      "tolerance": 0.01,
      "settlement": "On the paper's 4-point example, LEACE reproduces the oblique edit MSE 1 and full-rank SAL (= P_ortho) gives 2, so the difference is exactly -1.",
      "sensitivity": {
        "fixed_by_paper": true,
        "citation": "paper/main.tex:932-949"
      }
    },
    {
      "id": "C3",
      "quote": "our method is the only to achieve random accuracy (perfect erasure) with a small edit",
      "citation": "paper/main.tex:544",
      "kind": "invariant",
      "compute_invariance": "high",
      "arm_role": "bios intrinsic",
      "predicate": "all_seeds: abs(bios_leace.gender_acc - bios_none.majority_acc) <= 0.03",
      "settlement": "Gender probe sits at majority accuracy after LEACE at every seed; 'random accuracy' is measured as our data's majority frequency (paper's dashed line read off the figure: 0.52).",
      "sensitivity": {
        "unstated": "svd_tol",
        "plausible": [
          0.0,
          0.01
        ],
        "survives": [
          0.0,
          0.01
        ],
        "note": "Gender is rank 1; the dominant singular value of W Sigma_XZ is orders above both endpoints, so truncation choice cannot matter."
      }
    },
    {
      "id": "C4",
      "quote": "our method is the only to achieve random accuracy (perfect erasure) with a small edit, although RLACE (but not INLP) comes close",
      "citation": "paper/main.tex:544",
      "kind": "ordering",
      "compute_invariance": "high",
      "arm_role": "bios intrinsic",
      "quantity": "measured.bios_inlp_r1.gender_acc - measured.bios_none.majority_acc",
      "direction": ">0",
      "tolerance": 0.03,
      "settlement": "At an edit size comparable to or smaller than LEACE's (INLP rank-1 MSE read off the figure as ~1.6 vs LEACE ~2.2), INLP still fails to erase: its residual probe accuracy exceeds majority by more than 0.03 at every seed.",
      "sensitivity": {
        "unstated": "inlp probe regularization C (sklearn default 1.0; paper silent)",
        "plausible": [
          0.1,
          10.0
        ],
        "survives": [
          0.5,
          10.0
        ],
        "note": "INLP's nullspace direction depends on the probe, and the paper never states its settings; the verdict leans on this choice."
      }
    },
    {
      "id": "C5",
      "quote": "At the same time, the TPR gap drops significantly from 0.198 to 0.084",
      "citation": "paper/main.tex:578",
      "kind": "ordering",
      "compute_invariance": "high",
      "arm_role": "bios fairness",
      "quantity": "measured.bios_leace.tpr_gap_rms - measured.bios_none.tpr_gap_rms",
      "direction": "<0",
      "settlement": "TPR-gap RMS of the gender-unaware (unrefit) profession classifier strictly decreases after LEACE erasure, at every seed.",
      "sensitivity": {
        "unstated": "train/test protocol of the profession classifier (paper never states splits; we use the dataset's own train/test)",
        "plausible": [
          0.5,
          0.9
        ],
        "survives": [
          0.5,
          0.9
        ],
        "note": "Expressed as the train fraction a reader could have assumed; the direction is robust to any held-out protocol because both arms share it."
      }
    },
    {
      "id": "C6",
      "quote": "this correlation plummets from 0.867 to 0.392 after erasure",
      "citation": "paper/main.tex:578",
      "kind": "ordering",
      "compute_invariance": "medium",
      "arm_role": "bios fairness",
      "quantity": "measured.bios_leace.tpr_corr - measured.bios_none.tpr_corr",
      "direction": "<0",
      "settlement": "Pearson correlation between per-profession TPR gap and per-profession % women decreases after erasure at every seed. Medium: the statistic pools only 28 profession-level points, so subsampling noise is larger than for C5.",
      "sensitivity": {
        "unstated": "profession roster passed to the RMS/correlation (paper cites the bios dataset without listing professions)",
        "plausible": [
          28,
          28
        ],
        "survives": [
          28,
          28
        ],
        "note": "All 28 professions of bias_in_bios kept; no restriction."
      }
    },
    {
      "id": "C7",
      "quote": "The main-task classifier achieves profession-prediction accuracy of 77.3\\% on the projected embeddings (compared with 79.3\\% over the original embeddings)",
      "citation": "paper/main.tex:578",
      "kind": "value",
      "compute_invariance": "low",
      "arm_role": "bios fairness",
      "quantity": "measured.bios_none.prof_acc",
      "claimed": 0.793,
      "tolerance": 0.02,
      "settlement": "Profession accuracy of the original classifier on original embeddings.",
      "note": "Low: requires the paper's (unstated) exact split/classifier settings and full-data statistics."
    },
    {
      "id": "C8",
      "quote": "The main-task classifier achieves profession-prediction accuracy of 77.3\\% on the projected embeddings (compared with 79.3\\% over the original embeddings)",
      "citation": "paper/main.tex:578",
      "kind": "value",
      "compute_invariance": "low",
      "arm_role": "bios fairness",
      "quantity": "measured.bios_leace.prof_acc",
      "claimed": 0.773,
      "tolerance": 0.02,
      "settlement": "Profession accuracy of the original (unrefit) classifier applied to projected embeddings."
    },
    {
      "id": "C9",
      "quote": "yields a slightly higher main-task accuracy of 78.1\\%, at the price of significantly increasing the TPR gap to 0.158.",
      "citation": "paper/main.tex:578",
      "kind": "value",
      "compute_invariance": "low",
      "arm_role": "bios fairness",
      "quantity": "measured.bios_leace_refit.prof_acc",
      "claimed": 0.781,
      "tolerance": 0.02,
      "settlement": "Profession accuracy after refitting the classifier on projected embeddings."
    },
    {
      "id": "C10",
      "quote": "At the same time, the TPR gap drops significantly from 0.198 to 0.084",
      "citation": "paper/main.tex:578",
      "kind": "value",
      "compute_invariance": "low",
      "arm_role": "bios fairness",
      "quantity": "measured.bios_none.tpr_gap_rms",
      "claimed": 0.198,
      "tolerance": 0.05,
      "settlement": "TPR-gap RMS before erasure (magnitude claim; pairs with C5 which carries the direction)."
    },
    {
      "id": "C11",
      "quote": "At the same time, the TPR gap drops significantly from 0.198 to 0.084",
      "citation": "paper/main.tex:578",
      "kind": "value",
      "compute_invariance": "low",
      "arm_role": "bios fairness",
      "quantity": "measured.bios_leace.tpr_gap_rms",
      "claimed": 0.084,
      "tolerance": 0.05,
      "settlement": "TPR-gap RMS after LEACE (magnitude claim)."
    },
    {
      "id": "C12",
      "quote": "yields a slightly higher main-task accuracy of 78.1\\%, at the price of significantly increasing the TPR gap to 0.158.",
      "citation": "paper/main.tex:578",
      "kind": "ordering",
      "compute_invariance": "medium",
      "arm_role": "bios fairness",
      "quantity": "measured.bios_leace_refit.tpr_gap_rms - measured.bios_leace.tpr_gap_rms",
      "direction": ">0",
      "settlement": "Refitting the main classifier on the projected embeddings re-inflates the TPR gap relative to the unrefit classifier, at every seed. Medium: depends on the unstated probe regularization.",
      "sensitivity": {
        "unstated": "profession classifier regularization C",
        "plausible": [
          0.1,
          10.0
        ],
        "survives": [
          0.5,
          2.0
        ]
      }
    },
    {
      "id": "C13",
      "quote": "Our intervention only mildly changes BERT LM accuracy and loss until layer 8, with the highest drop recorded in layer 11. INLP, in contrast, shows maximum effect at layer 6.",
      "citation": "paper/main.tex:601",
      "kind": "invariant",
      "compute_invariance": "high",
      "arm_role": "amnesic probing",
      "predicate": "all_seeds: argmax_l(amnesic_none.mlm_acc[l] - amnesic_leace.mlm_acc[l]) == 11",
      "settlement": "The LEACE single-layer intervention does its maximal MLM damage at layer 11 at every seed. (Figure on disk agrees: 'Ours' bottom is at layer 11 - figure-transcript.md.)",
      "sensitivity": {
        "unstated": "MLM masking protocol (rate/scheme; paper states only 'masked language modeling')",
        "plausible": [
          0.15,
          0.15
        ],
        "survives": [
          0.12,
          0.3
        ],
        "note": "Only 0.15 (80/10/10) is a defensible reading of standard BERT MLM; survives estimated on wider rates."
      }
    },
    {
      "id": "C14",
      "quote": "Our intervention only mildly changes BERT LM accuracy and loss until layer 8, with the highest drop recorded in layer 11. INLP, in contrast, shows maximum effect at layer 6.",
      "citation": "paper/main.tex:601",
      "kind": "invariant",
      "compute_invariance": "high",
      "arm_role": "amnesic probing",
      "predicate": "all_seeds: argmax_l(amnesic_none.mlm_loss[l] - amnesic_inlp.mlm_loss[l]) == 6",
      "settlement": "The INLP-20 intervention does its maximal MLM damage at layer 6 at every seed. (Figure on disk agrees: green INLP curve spikes at layer 6 - figure-transcript.md.)",
      "sensitivity": {
        "unstated": "MLM masking protocol",
        "plausible": [
          0.15,
          0.15
        ],
        "survives": [
          0.12,
          0.3
        ]
      }
    },
    {
      "id": "C15",
      "quote": "We compare with a baseline intervention of a random orthogonal projection whose null space has the same rank as the label space (18).",
      "citation": "paper/main.tex:599",
      "kind": "curve",
      "compute_invariance": "high",
      "arm_role": "amnesic probing (figure-derived; paper/figures/bios_frozen_bert_layer_vs_drop_in_acc.pdf shows 'Random' flat near 'No Intervention' with 'Ours' below it)",
      "quantity": "measured.amnesic_leace.mlm_drop - measured.amnesic_random.mlm_drop",
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
        10,
        11
      ],
      "comparison": "above",
      "tolerance": 0.01,
      "settlement": "The LEACE accuracy-drop curve stays above the random-projection drop curve at every layer (within 0.01), at every seed; i.e. the damage is attributable to the erased concept, not to deleting 17-18 dimensions.",
      "sensitivity": {
        "unstated": "random null-space rank (18 = k per main.tex:599 vs 17 = rank(Sigma_XZ) <= k-1; paper's amnesic text says 18)",
        "plausible": [
          17,
          18
        ],
        "survives": [
          17,
          18
        ]
      }
    },
    {
      "id": "C16",
      "quote": "The specific numbers, however, depend on the erasure method used: SAL induces significantly larger increases in perplexity for all models we tested.",
      "citation": "paper/main.tex:676",
      "kind": "ordering",
      "compute_invariance": "high",
      "arm_role": "concept scrubbing",
      "quantity": "measured.scrub_sal.bpb - measured.scrub_leace.bpb",
      "direction": ">0",
      "settlement": "On Pythia-160m, SAL concept-scrubbing perplexity strictly exceeds LEACE concept-scrubbing perplexity at every seed. Restriction: 1 of the paper's 7 models; the claim states 'for all models', so this arm supports one member of the universal.",
      "sensitivity": {
        "unstated": "covariance estimator (plain sample vs RMT-shrinkage; paper silent, upstream defaults to shrinkage)",
        "plausible": [
          0,
          1
        ],
        "survives": [
          0,
          1
        ],
        "note": "0 = plain sample covariance, 1 = optimal linear shrinkage; gap (~0.7 bpb in the paper) is large enough that either estimator preserves the ordering. Also checked across sublayers in {block-input-only, both-norms}."
      }
    },
    {
      "id": "C17",
      "quote": "While erasing a randomly selected subspace has little to no effect on language modeling performance, scrubbing away part-of-speech information induces a large increase in perplexity across all models",
      "citation": "paper/main.tex:674",
      "kind": "ordering",
      "compute_invariance": "high",
      "arm_role": "concept scrubbing",
      "quantity": "measured.scrub_leace.bpb - measured.scrub_none.bpb",
      "direction": ">0",
      "settlement": "LEACE scrubbing increases perplexity over no intervention at every seed (positive = POS damage).",
      "sensitivity": {
        "unstated": "random/null-space rank (k-1 = rank(Sigma_XZ) per main.tex:668 'same rank as the cross-covariance matrix' vs k = 18 upstream)",
        "plausible": [
          17,
          18
        ],
        "survives": [
          17,
          18
        ]
      }
    },
    {
      "id": "C18",
      "quote": "While erasing a randomly selected subspace has little to no effect on language modeling performance, scrubbing away part-of-speech information induces a large increase in perplexity across all models",
      "citation": "paper/main.tex:674",
      "kind": "ordering",
      "compute_invariance": "high",
      "arm_role": "concept scrubbing",
      "quantity": "(measured.scrub_leace.bpb - measured.scrub_none.bpb) - (measured.scrub_random.bpb - measured.scrub_none.bpb)",
      "direction": ">0",
      "settlement": "The LEACE perplexity increase strictly exceeds the random-subspace increase at every seed: the effect belongs to the POS concept, not to the rank of the edit. (Paper's Pythia-160M gap: 1.89 vs 0.09 bpb.)",
      "sensitivity": {
        "unstated": "random/null-space rank (k-1 vs k)",
        "plausible": [
          17,
          18
        ],
        "survives": [
          17,
          18
        ]
      }
    },
    {
      "id": "C19",
      "quote": "No intervention & 0.69           & 0.66       & 0.62          & 0.90          & 0.70            & 0.64          & 0.62           \\\\\nRandom erasure  & 0.69           & 0.66       & 0.62          & 0.99          & 0.72            & 0.66          & 0.63           \\\\\n\\midrule\nLEACE & 1.73           & 1.84       & 1.96          & 2.79          & 2.25            & 3.57          & 3.20              \\\\\nSAL   & 3.24           & 3.26       & 3.16          & 3.53          & 3.44            & 4.17          & 4.69             \\\\",
      "citation": "paper/main.tex:647-651",
      "kind": "value",
      "compute_invariance": "low",
      "arm_role": "concept scrubbing",
      "quantity": "measured.scrub_leace.bpb",
      "claimed": 2.79,
      "tolerance": 0.8,
      "settlement": "Pythia-160M LEACE-scrubbed perplexity in bits per byte. Low: restricted token budget (2^20 vs 2^22), fp32 vs unstated precision, and the unstated SpaCy tagger all move this magnitude; tolerance set wide enough to reflect that.",
      "note": "Companion magnitudes: scrub_none 0.90, scrub_random 0.99, scrub_sal 3.53 (same table cells); evaluated at the same tolerance 0.8 (0.15 for scrub_none)."
    },
    {
      "id": "C20",
      "quote": "\\rv X'_{\\mathrm{LEACE}} = \\rv X - \\sxz \\szz^+ \\big( \\rv Z - \\E[\\rv Z] \\big).",
      "citation": "paper/main.tex:1002",
      "kind": "ordering",
      "compute_invariance": "high",
      "arm_role": "sanity (synthetic)",
      "quantity": "measured.synth_oracle.mse - measured.synth_leace.mse",
      "direction": "<0",
      "settlement": "Oracle LEACE, which sees the labels at erasure time, makes a strictly smaller edit than label-blind LEACE on synthetic Gaussian data ('we can achieve an even more surgical edit', paper/main.tex:988), at every seed.",
      "sensitivity": {
        "unstated": "svd_tol",
        "plausible": [
          0.0,
          0.01
        ],
        "survives": [
          0.0,
          0.01
        ]
      }
    }
  ],
  "not_tested": [
    {
      "claim": "RLACE 'comes close' to random accuracy on bios (figure-only; min RLACE accuracy read as 0.54)",
      "citation": "paper/main.tex:544",
      "reason": "RLACE requires reimplementing Ravfogel et al.'s gradient-based adversarial game, whose hyperparameters the paper does not restate; the claims this reproduction tests rest on LEACE vs INLP/SAL/random, not on RLACE. Deliberately excluded."
    },
    {
      "claim": "'our method is around 2 orders of magnitude faster, and does not require gradient-based optimization'",
      "citation": "paper/main.tex:544",
      "reason": "Wall-clock claim; environment-dependent, no stated protocol. Untestable as a reproduction verdict here."
    },
    {
      "claim": "LLaMA 7B/13B/30B and Pythia 1.4B/6.9B/12B scrubbing magnitudes and trends",
      "citation": "paper/main.tex:645-653",
      "reason": "No GPU in this environment and LLaMA weights are license-gated; restricted to Pythia-160m as documented per-arm."
    },
    {
      "claim": "Sequential fitting (Algorithm 1) vs naive independent per-layer fitting: naive 'may fail to fully erase'",
      "citation": "paper/main.tex:633",
      "reason": "Hedged ('may'); no quantitative statement to settle."
    },
    {
      "claim": "Stacked-classifier leak footnote (softmax probabilities of the refit classifier leak the concept to a second stacked classifier)",
      "citation": "paper/main.tex:578",
      "reason": "Footnote aside citing ravfogel2022linear; nonlinear stacking is outside the linear-guardedness scope and not part of the paper's main comparisons."
    },
    {
      "claim": "Unigram-entropy baseline rows (2.90 / 2.66 bpb)",
      "citation": "paper/main.tex:653",
      "reason": "Contextual baseline, not a property of the method; not gated."
    },
    {
      "claim": "Norm-growth divergence observation ('we find this solves the divergence issue in practice')",
      "citation": "paper/main.tex:972",
      "reason": "No NaN events expected at Pythia-160m scale in 2^20-token eval; the trace-constraint branch may never fire. If it fires we log it; we do not gate on divergence."
    }
  ]
}
```
