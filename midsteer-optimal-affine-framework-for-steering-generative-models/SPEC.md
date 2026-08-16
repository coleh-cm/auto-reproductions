# SPEC — MidSteer: Optimal Affine Framework for Steering Generative Models

Paper: Gaintseva, Stepanov, Liu, Benning, Slabaugh, Deng, Elezi (ICML 2026), arXiv:2605.05220v3.
paper_ref: `0e6756c9-d827-4749-a8d7-3140a8c98a25` · project_id: `06910d54-1d99-4864-8bff-ab3007a0c70e`.
Authoritative source on disk: `paper/main.tex`, `paper/content/*.tex`, `paper/flipping_main.tex`,
`paper/artefacts/**/*.tex` (arXiv LaTeX v3). `paper/paper.txt` is a convenience copy and is NOT
authoritative for maths. All citations below are `<file>:<line>` into the on-disk LaTeX.

---

## 1. Upstream code check

- The paper links code: `paper/main.tex:160` → https://github.com/Atmyre/MidSteer .
- Verified 2026-08-16: repo exists, public, `HEAD = 0f3b31e15cdda6ad0d46167e10319e896d6f1541`.
- Layout: `core/{controller,math,llm_steering,diffusion_steering,vector_dump,prompts,eval/*}.py`,
  `scripts/{llm,diffusion}/*.py`, `requirements/{base,linux,darwin}.txt`, `README.md`,
  `.env.example` (needs `HF_TOKEN`; `OPENAI_API_KEY` only for the GPT-4o-mini cross-validation),
  imagenet_classes.txt, helpers.
- **Decision: adopt upstream code** as the base implementation (the paper's own code ships the
  method), running its entrypoints; record every change made to make it run.
- Divergences between upstream code and the paper's literal text found on first inspection
  (to be verified, not assumed, during adoption):
  1. `core/controller.py` applies `intermediate_clipping=True` (projection scores clipped to ≥ 0,
     lines 216–217 and 260–261) in all steer paths. **Not in the paper.** Default ON. This is a
     candidate cause of a paper-vs-code mismatch; the paper's affine map is linear in (X−μ).
  2. `core/controller.py:157-176` uses class-conditional *mean differences* relative to a neutral
     mean (`mu_neutral`) instead of the literal `Cov(X, Z_i)` (which includes the class-prior
     factor `P(Z_i=1)`; see gap G3).
  3. `core/math.py:13` pseudoinverse/square-root eigen-tolerance `τ = λ_max·d·ε` (torch.pinverse
     convention). Not stated in the paper (gap G7).
  4. Statistics are per attention head (`core/vector_dump.py`, `Welford` per head) — consistent
     with `paper/content/suppl.tex:7,14` (gap G5).
  5. Steering is applied via forward hooks at runtime, not folded into weights (folding is the
     paper's deployment option, `paper/content/suppl.tex:77-83`); mathematically equivalent
     (gap G17).
  6. Upstream indexes steering statistics by diffusion step (`use_first_diffusion_step`); the
     paper does not state which denoising steps are used (gap G6).

---

## 2. The method as an explicit algorithm

This is a **closed-form, training-free** intervention. There is no optimisation loop, no
initialisation, no schedule. The "loss" is the minimal-disturbance quadratic objective below, and
the "update rule" is its analytic solution. Per generative model, per concept pair (c_s → c_t):

**Inputs**
- A generative model G (LLM or diffusion) with attention blocks whose last sublayer is a linear
  out-projection (`paper/content/suppl.tex:58-61`).
- A source concept c_s and a target concept c_t (or "constant c_t" for erasure,
  `paper/main.tex:461`).
- A steering strength β ∈ ℝ (`paper/main.tex:463-482`).
- Concept prompt sets P_s, P_t of size N = 1000 (`paper/content/experiments.tex:63`,
  `paper/content/suppl.tex:374-407`) and a broad prompt set P_bg of size M = 50 000
  (`paper/content/experiments.tex:63`) unrelated to the steering concepts
  (`paper/content/suppl.tex:7`).

**Step 1 — collect activations.** For every self-attention layer (LLM,
`paper/content/experiments.tex:89`) or every cross-attention layer (diffusion,
`paper/content/experiments.tex:101`), run P_s, P_t, P_bg and record the activations of the SA/CA
block output, retaining the head split (gap G5). For LLMs use the last prompt token's activation
(`paper/content/experiments.tex:89`, `paper/content/suppl.tex:377`); for diffusion, all image
patches (`paper/content/experiments.tex:101`).

**Step 2 — estimate statistics (Welford, Algorithm 1, `paper/content/suppl.tex:11-52`).**
Per head h:
- μ ∈ ℝ^{H×d}, Σ_XX ∈ ℝ^{H×d×d} from P_bg activations;
- XZ statistics for Z_1 (source) and Z_2 (target) from P_s, P_t relative to the background
  (gap G3/G19) giving Σ_XZ1, Σ_XZ2 ∈ ℝ^{H×d×1} (l = 1 in all experiments, `paper/main.tex:457`).

**Step 3 — whiten.** W = (Σ_XX^{1/2})^+ ∈ ℝ^{H×d×d}; Σ_XX is symmetric PSD so
W^+ = Σ_XX^{1/2} (`paper/content/guardedness.tex:85,88-90`).

**Step 4 — closed-form affine map.** Per head, for the chosen method (see §4):
  - LEACE (erasure / LEACE-Switch with β): compute Σ_WX,Z = W Σ_XZ (ℝ^{H×d×1}), then
    Â = I − β·W^+ (Σ_WX,Z)(Σ_WX,Z)^+ W   (Eq. 22, `paper/main.tex:471-474`)
  - MidSteer: Σ_WX,Zi = W Σ_XZi, then
    Â = I + β·W^+ (Σ_WX,Z2 − Σ_WX,Z1) (Σ_WX,Z1)^+ W   (Eq. 23, `paper/main.tex:478-481`)
  - Vanilla: s = μ_s − μ_t (steering vector definition Eq. 1 specialised to the pair, gap G3),
    unit-normalised (gap G8), Â = I − β·s sᵀ (Eqs. 24/25/21, `paper/content/suppl.tex:63-73`,
    `paper/main.tex:466-469`).
  - b̂ = E[X] − Â E[X] = μ − Â μ (`paper/main.tex:366-367` / `:448` / `paper/content/guardedness.tex:80-83`).

**Step 5 — apply.** h′ = Â h + b̂ at every SA/CA layer output (LLM: every token, gap G17; diffusion:
every patch). Optionally fold into the out-projection weights:
W_s = Â W_proj_out, b_s = Â b + b̂ (Eq. 26, `paper/content/suppl.tex:78-80`) → zero inference
overhead (`paper/main.tex:372`).

**Objective ("loss") and guarantee.** Among affine maps satisfying the constraint, Â minimises
E‖AX + b − X‖₂²:
- LEACE: constraint Cov(AX+b, Z) = 0 (`paper/content/guardedness.tex:62-72`).
- LEACE-Switch: constraint Cov(AX+b, Z) = −Cov(X, Z) (`paper/main.tex:359-360`).
- MidSteer: constraint Cov(AX+b, Z_1) = Cov(X, Z_2) (`paper/main.tex:440-441`).
Uniqueness/optimality rest on Im(Σ_XZ) ⊆ Im(Σ_XX) and, for MidSteer, rk(Σ_XZ1) = l
(`paper/main.tex:425`); with l = 1 this is Σ_XZ1 ≠ 0 (`paper/main.tex:456-457`).

---

## 3. Symbols and shapes

Conventions: H = number of attention heads at the hooked layer; d = per-head hidden dim
(e.g., 128 for Llama-2-7B's 32 heads × 4096; SDXL CA head_dim 64/128 depending on block);
B = batch size; T = token positions; P = image patch positions.

| Symbol | Shape | Meaning |
|---|---|---|
| X, h | ℝ^d per sample; batched LLM activations `[B, T]` → per head ℝ^{n×d} | activation of one layer (this is what the affine map acts on) |
| C, Z | scalar ∈ {0,1} (k = 1) in all experiments | concept indicator |
| Z_1, Z_2 | scalar ∈ {0,1} (l = 1; theorem allows ℝ^l) | source / target concept indicators (`paper/main.tex:412`) |
| k, l | k = 2l; experiments l = 1, k = 2 (`paper/main.tex:421,457`) | number of concept dims |
| s_c | ℝ^d; optionally unit-norm (`paper/main.tex:226`) | steering vector, Eq. 1 |
| α, β | ℝ scalars | vanilla strength (`paper/main.tex:228`); affine strength (`paper/main.tex:463-482`) |
| μ | ℝ^{H×d} | background mean activation (per head), Welford est. |
| Σ_XX | ℝ^{H×d×d}, symmetric PSD | activation self-covariance, est. on M = 50 000 (`paper/content/experiments.tex:63`) |
| Σ_XZ, Σ_XZ1, Σ_XZ2 | ℝ^{H×d×1} (l=1) | cross-covariances, est. on N = 1000 per concept (`paper/content/experiments.tex:63`) |
| W | ℝ^{H×d×d}, symmetric PSD | whitening = (Σ_XX^{1/2})^+ (`paper/content/guardedness.tex:85`) |
| W^+ | ℝ^{H×d×d} | = Σ_XX^{1/2} on the support of X |
| Σ_WX,Zi | ℝ^{H×d×1} | W Σ_XZi (`paper/main.tex:429`) |
| (Σ_WX,Z1)^+ | ℝ^{H×1×d} | Moore–Penrose pseudoinverse; for a column vector u: uᵀ/‖u‖² (`paper/content/suppl.tex:271-272`) |
| Â | ℝ^{H×d×d} | affine map matrix (block-diagonal across heads, gap G5) |
| b̂ | ℝ^{H×d} | affine map bias, μ − Âμ |
| f(·) | ℝ^d → ℝ^d | intervention function |
| Welford state (n, M, S) | scalar n; M ∈ ℝ^{H×d}; S ∈ ℝ^{H×d×d} | `paper/content/suppl.tex:14-20` |
| Λ (proof only) | ℝ^{d×l} | Lagrange multipliers; not implemented |
| CS | ℝ (0–10 LLM judge; CLIP×100 images) | concept score (`paper/content/experiments.tex:94,103`) |

Batched-hook layout (frozen, matches upstream): LLM hidden chunks reshaped to `[B, T, H, d]`;
diffusion CA outputs `[B, P, H, d]`; all affine math broadcast per head: `h′[b,t,h,:] = Â[h] h[b,t,h,:] + b̂[h]`.

---

## 4. Equations to implement (with citations)

Frozen equation set. Numbers are the paper's printed equation numbers; citations are on-disk lines.

### 4.1 Vanilla steering baselines

1. **Steering vector**: s_c = E[h|C=1] − E[h|C=0]. `paper/main.tex:222-225` (Eq. 1).
   Optional unit-norm post-processing `paper/main.tex:226`; for the matrix/folded form unit norm is
   **assumed**: `paper/content/suppl.tex:63`. For pair switching we take s = μ(source) − μ(target)
   (gap G3 — the paper never defines s for the two-concept case; this is the weakest reading
   under which the Householder map points source → target).
2. **Erasure**: f_delete(h,s) = h − ⟨h,s⟩s. `paper/main.tex:244-247` (Eq. 3).
3. **Switch (Householder)**: f_switch(h,s) = h − 2⟨h,s⟩s. `paper/main.tex:250-253` (Eq. 4).
4. **Vanilla with strength**: f(h,s) = h − β⟨h,s⟩s. `paper/main.tex:466-469` (Eq. 21).
   Matrix forms f = (I − ssᵀ)h and (I − 2ssᵀ)h: `paper/content/suppl.tex:64-73` (Eqs. 24–25).
5. α-form f(h,s) = h + αs (`paper/main.tex:229-232`, Eq. 2) — NOT used for the paper's
   experiments' arms; the erasure/switch arms use Eq. 21. Recorded because it is the "general"
   form the paper opens with.

### 4.2 Estimation

6. **Welford online mean/covariance** (Algorithm 1): `paper/content/suppl.tex:11-52`.
   Input batches X_k ∈ ℝ^{h×m_k×d} (`paper/content/suppl.tex:14`); finalisation
   Σ = (S + Sᴴ)/(2(n−1)) (`paper/content/suppl.tex:48`). Implementation notes:
   sample count n denominator n−1; Hermitian symmetrisation explicit.

### 4.3 Closed-form solutions

7. **Whitening**: W = (Σ_XX^{1/2})^+; A^{1/2} = V S^{1/2} Vᵀ for PSD symmetric A = VSVᵀ;
   A^+ Moore–Penrose. `paper/content/guardedness.tex:85,88-90`.
8. **LEACE Â, b̂**: Â = I − W^+(WΣ_XZ)(WΣ_XZ)^+W ; b̂ = E[X] − ÂE[X].
   `paper/content/guardedness.tex:74-83` (Eqs. 6–7), constraint Eq. 5 `:62-72`.
9. **LEACE-Switch Â, b̂**: Â = I − 2W^+(WΣ_XZ)(WΣ_XZ)^+W ; b̂ = E[X] − ÂE[X].
   `paper/main.tex:363-367` (Eqs. 13–14); constraint Cov(f(X),Z) = −Cov(X,Z) `:359-360`,
   equivalent form `paper/main.tex:330-337` (Eqs. 9–10).
10. **MidSteer Â, b̂**: Â = I + W^+(Σ_WX,Z2 − Σ_WX,Z1)(Σ_WX,Z1)^+W ; b̂ = E[X] − ÂE[X].
    `paper/main.tex:445-448` (Eq. 19–20); constraint `:440-441`; rank condition `:425`;
    l = 1 reduction `:456-457`.
11. **β-strength forms**: LEACE/LEACE-Switch Â = I − β·W^+(WΣ_XZ)(WΣ_XZ)^+W (Eq. 22,
    `paper/main.tex:471-474`; b̂ unaffected, `:470`); MidSteer Â = I + β·W^+(Σ_WX,Z2 −
    Σ_WX,Z1)(Σ_WX,Z1)^+W (Eq. 23, `paper/main.tex:478-481`). β semantics: β=1 erasure, β=2
    switch for Eq. 22 (`paper/main.tex:475`); β=1 "normal steering" for Eq. 23
    (`paper/main.tex:482`). **Default β used in the paper's arms: vanilla β=2, LEACE-Switch β=2,
    MidSteer β=1** (`paper/content/experiments.tex:125`), and β ∈ {1,…,5} elsewhere
    (`paper/content/safety-related_results.tex:7`).
12. **Weight folding (deployment)**: h_out = Â(W_proj_out h_in + b) + b̂ = W_s h_in + b_s.
    `paper/content/suppl.tex:78-80`.

### 4.4 Metrics to implement

13. **LLM judge CS (0–10)**: prompt verbatim at `paper/content/suppl.tex:423-436`; score by argmax
    over token probabilities of score tokens "0"…"10" (`paper/content/suppl.tex:438`). Judge:
    Llama-3.1-8B-Instruct per main text (`paper/content/experiments.tex:94`); the appendix calls
    it "Llama3.1-7B" (`paper/content/suppl.tex:438`) — inconsistency, gap G10.
    GPT-4o-mini is a cross-validation judge only (`paper/content/suppl.tex:660-662`).
14. **CLIP score as image CS** (×100 in tables): `paper/content/experiments.tex:103`,
    `paper/flipping_main.tex:2-5`, `paper/artefacts/tables/diffusion_*.tex` headers.
15. **FID between steered and vanilla generations**: `paper/content/experiments.tex:103`.
16. **Detoxify toxicity (RTP) and ArmoRM helpfulness**: `paper/content/experiments.tex:96`,
    table notes `paper/artefacts/tables/safety_table.tex:90-92`.
17. **BERTScore consistency on MMLU generations** (Precision §5 / F1 in the safety tables):
    `paper/content/experiments.tex:98`, safety-table header `paper/artefacts/tables/safety_table.tex:63`.

---

## 5. What the paper does NOT state (gaps and weakest readings)

For each gap: what the text allows, then the **weakest** reading — the one committing least beyond
the text. Readings below are the ones the implementation will take EXCEPT where flagged
"unresolved" (decided against upstream evidence during adoption).

- **G1 · Origin and exact content of the N = 1000 concept prompts.**
  Only illustrative 3-prompt subsets for horse/motorcycle/dog/cat are given
  (`paper/content/suppl.tex:374-407`); nothing is said for the other concrete concepts or the
  safety concepts. A commented-out line in the source says they were generated by "GPT o4-mini"
  for concrete LLM concepts and taken from Jigsaw for toxicity
  (`paper/content/experiments.tex:91-92`) — this does NOT appear in the paper.
  *Permitted:* any 1000 prompts about the concept. *Weakest:* N = 1000 prompts per concept whose
  topic is the concept, from any generation recipe; reproduction generates its own with a
  documented recipe and treats exact prompt identity as part of the situation set (restriction,
  §8). Whether N = 1000 is per concept or shared across categories: the phrase "covariances
  Σ_XZ1, Σ_XZ2 based on a sample of size N = 1000" (`paper/content/experiments.tex:63`) admits
  both; weakest = each class-conditional statistic gets its own 1000.

- **G2 · Background corpus for Σ_XX.** "a sample of broad prompts (unrelated to the steering
  concepts)" (`paper/content/suppl.tex:7`); identity and domain not stated (commented-out source
  names Alpaca, `paper/content/experiments.tex:92`). *Permitted:* any broad corpus unrelated to
  the concepts. *Weakest:* exactly that description; M = 50 000 stated
  (`paper/content/experiments.tex:63`) but the authors' own ablation permits M ≥ 5 000
  (`paper/content/suppl.tex:640-642`) — reproduction may use M = 5 000 (restriction, §8) with the
  paper's own stabilisation evidence as justification plus our figure-read of Fig. 5
  (`figure_reads/transcript.md`).

- **G3 · The joint distribution underlying Σ_XZ (class priors), and "class-conditional
  covariance".** Cov(X, Z) depends on P(Z=1) and on which background samples carry Z = 0. The
  paper says only "class-conditional covariances Σ_XZ1, Σ_XZ2 based on a sample of size N = 1000"
  (`paper/content/experiments.tex:63`) and defines Σ_XZi = Cov(X, Z_i) (`paper/main.tex:422`). A
  sample containing only concept prompts has Z ≡ 1 and hence *zero* covariance, so the estimation
  sample must mix concept data with background data — never said. A commented-out derivation in
  the source (`paper/main.tex:494-504`) treats Σ_X,Zi = P(Z_i=1)(μ_i − μ) and calls
  β = P(Z_2=1)/P(Z_1=1) "practically infeasible to estimate", absorbing it into the β
  hyperparameter. *Permitted:* (a) literal empirical Cov over a pooled sample with arbitrary
  mix; (b) class-conditional mean differences μ_i − μ_bg with any scalar prior absorbed into β.
  *Weakest:* (b) — it adds no commitment about class priors beyond what β already absorbs, and it
  is what the upstream code does (`core/controller.py:157-176`). The same reading fixes the
  LEACE-Switch experiments' Z as the source-class indicator within the {source prompts ∪ target
  prompts} sample (the theorem's dataset partition), and the vanilla pair vector as
  s = μ_s − μ_t.

- **G4 · Reference class for E[h|C=0] in s_c (Eq. 1).** "absence of concept c" data is never
  specified. *Permitted:* broad background, or near neighbours (e.g., other animals). *Weakest:*
  the same broad background used for Σ_XX (no extra commitment); for switching pairs, μ_t plays
  the role (G3).

- **G5 · Per-head treatment.** Theorems are in ℝ^d; Appendix A estimates per head
  (input X ∈ ℝ^{h×m×d}, per-head Σ, `paper/content/suppl.tex:7,14`); the paper never says the
  theorem is applied per head. *Permitted:* per-head independent maps (block-diagonal Â across
  heads) or flattened full-width ℝ^{H·d}. *Weakest:* per-head, since the estimation algorithm the
  paper gives is per head and a block-diagonal map is a restriction of the theorem's full map
  family (it does not add commitments about cross-head coupling).

- **G6 · Application tensor and diffusion timesteps.** Folding (Eq. 26,
  `paper/content/suppl.tex:77-83`) fixes that the map acts on the SA/CA block output h_out, so
  covariances must be estimated on h_out. LLM: estimation at last prompt token (stated); which
  tokens are *steered* is implied by folding = all tokens (gap G17). Diffusion: all patches used
  for estimation (`paper/content/experiments.tex:101`); **which denoising step(s)** the
  statistics come from and at which steps steering is applied is never stated. *Permitted:*
  per-step statistics applied per step; single-step statistics applied to all steps; or turbo
  (1-step) settings. *Weakest:* estimate and apply at every denoising step (indexed per step when
  the sampler exposes them), since anything else injects a step-selection rule the text does not
  contain.

- **G7 · Numerical rank tolerance for W = (Σ_XX^{1/2})^+ and (Σ_WX,Z1)^+.** Unstated.
  *Permitted:* exact algebra (impossible in float) or any tolerance. *Weakest:* the host library's
  default threshold (torch pinv convention τ = λ_max·d·ε, which upstream also uses,
  `core/math.py:13`).

- **G8 · Unit norm of s in vanilla mode.** "can be optionally post-processed (to have unit norm)"
  (`paper/main.tex:226`); the matrix forms assume ‖s‖=1 (`paper/content/suppl.tex:63`).
  *Weakest:* unit-norm wherever the projection/Householder forms are used (the appendix's own
  assumption), recorded as a choice for raw-vector arms.

- **G9 · Generation hyperparameters.** LLM: max new tokens, temperature, top-p, sampling vs
  greedy, chat template — all unstated. Diffusion: sampler, number of steps, guidance scale,
  resolution — all unstated. *Weakest:* each released pipeline's canonical defaults for the named
  checkpoint (greedy/deterministic scoring where a score is argmaxed).

- **G10 · Exact checkpoints / names.** "instruction-tuned Llama 2" (`paper/content/experiments.tex:89`)
  with safety table naming Llama-2-7B-chat (`paper/artefacts/tables/safety_table.tex:61`); concrete
  LLM experiments labelled "LLama2-7b" (`paper/artefacts/tables/llm_flip_tables_noclip.tex`). SDXL
  revision unstated; "SANA 1.6" (`paper/content/suppl.tex:7`) variant unstated; judge named both
  "Llama-3.1-8B-Instruct" (`paper/content/experiments.tex:94`) and "Llama3.1-7B"
  (`paper/content/suppl.tex:438`). *Weakest:* the checkpoint distributed under the exact written
  name (Llama-2-7b-chat-HF where the text says chat/instruction-tuned; vanilla Llama-2-7B tables
  read as the same instruction-tuned family; judge = Llama-3.1-8B-Instruct as the main text
  states; the appendix "7B" treated as imprecise shorthand, not a different model).

- **G11 · Metric backbones and configurations.** CLIP variant for image CS unstated (tables say
  "CLIP-scores (cs, ×100)", e.g. `paper/artefacts/tables/diffusion_flip_tables_noclip.tex`);
  FID implementation/resolution/statistics unstated; Detoxify variant unstated
  (`paper/content/experiments.tex:96`); ArmoRM variant and which head ("helpfulness") unstated;
  BERTScore backbone/layers and the MMLU prompt count/subset unstated
  (`paper/content/experiments.tex:98`). *Weakest:* the canonical released defaults for each named
  metric package (CLIPScore default backbone, clean-fid defaults, unitary "original" Detoxify
  mean-toxicity score, ArmoRM-Llama3-8B helpfulness attribute, bert-score defaults), all recorded
  in the experiment configs; claims relying on exact magnitudes are rated accordingly (§claims).

- **G12 · Prompts for safety concepts (non-toxicity side).** The 80/81 listed templates
  (`paper/content/suppl.tex:457-539`) are object-centric; which templates were used for
  helpfulness / sarcasm / creativity / politeness / mathematics targets, and for the diffusion
  safety concepts, is unstated. (Note: the LLM template list numbers 81 items although the text
  says "80 template prompts" — `paper/content/experiments.tex:68`; the image list numbers exactly
  80.) Toxicity source prompts ARE stated: RealToxicityPrompts with toxicity ≥ 0.5
  (`paper/content/experiments.tex:84`), but the count is not (80 by symmetry is the weakest
  reading). *Weakest:* use the listed template set with the concept substituted; RTP: all prompts
  above threshold capped at the smallest sufficient count ≥ 80, seeds as above.

- **G13 · Pareto ΔCS definition.** "Δ CS" in the Pareto axes (`paper/content/experiments.tex:10,17,121`)
  is never defined (target − source? change vs baseline?). *Consequence:* no claim in claims.json
  uses a ΔCS magnitude; curve claims use directly-defined quantities instead. This is the weakest
  possible stance: the reproduction does not put a number on a quantity the paper never defined.

- **G14 · Seed values and seed plumbing.** "10 such generations varying the random seed"
  (`paper/content/experiments.tex:68-69`); values unstated; nothing on seeds for data order,
  judge, or model init (none needed — no training). *Weakest:* any fixed set of seeds; metrics
  aggregated by mean. claims.json fixes seeds {0,1,2} (restriction, §8).

- **G15 · RTP evaluation details.** Continuation length, toxicity aggregation (mean over
  generations of the toxicity probability — implied by "RTP is Detoxify toxicity on harmful
  prompts", `paper/artefacts/tables/safety_table.tex:91`), and how refusal outputs are scored:
  unstated. *Weakest:* mean Detoxify probability over fixed-length continuations.

- **G16 · Compute dtype.** Unstated. *Weakest:* float64 for all covariance algebra (needed for
  the 1e-8-level synthetic invariant checks; upstream agrees, `core/math.py:20-28`), model-native
  dtype elsewhere.

- **G17 · Steered token set during LLM generation.** "We apply steering at every self-attention
  (SA) layer ans use SA activations corresponding to the last token in prompt"
  (`paper/content/experiments.tex:89`) — "use ... activations ... last token" attaches to
  estimation; application at folded-in weights necessarily hits every token. *Weakest:* apply to
  all token positions (identical to weight folding) — anything narrower needs machinery the paper
  never mentions.

- **G18 · Composing multiple layers.** Transform estimated independently per layer and applied at
  every layer simultaneously (`paper/content/experiments.tex:89,101` says "every" layer). Whether
  statistics layer-l are estimated with steering OFF at all layers (almost surely, but unstated):
  *weakest:* yes, estimation on the unsteered model.

- **G19 · Estimation-finite-sample handling on degenerate layers.** If Σ_XZ1 = 0 at some layer:
  the update is degenerate and "MidSteer is not meaningfully applicable there"
  (`paper/main.tex:456-457`). What the authors do at such layers (skip? epsilon floor?):
  unstated. *Weakest:* the pseudoinverse-based form already degenerates gracefully
  ((Σ_WX,Z1)^+ → 0 row ⇒ Â → I); no extra rule added.

General comment on weakness: readings above deliberately avoid importing conventions (e.g.,
class-balanced samples, Alpaca, unit-norm everything, specific CLIP backbone). Where the
commented-out LaTeX source reveals the authors' actual choices (G1, G2, G3), we record them as
*evidence about the situation set*, but claims never depend on them — that would fit the
hypothesis to the authors' numbers, not to their text.

---

## 6. Component interfaces (frozen)

Parallel work must target these exactly. Python 3.11, torch; float64 inside `midsteer_core`;
models loaded via transformers/diffusers.

```
midsteer_core/
  stats.py
    class WelfordState: (n: int, M: Tensor[H,d], S: Tensor[H,d,d])
    def welford_update(state, batch: Tensor[H, m, d]) -> WelfordState
    def welford_finalize(state) -> tuple[mu: Tensor[H,d], cov: Tensor[H,d,d]]   # Eq. §4.2 item 6
  crosscov.py
    def mean_diff(x_concept: Tensor[H,n,d], mu_bg: Tensor[H,d]) -> Tensor[H,d,1]   # G3 reading (b)
  affine.py
    def whiten(cov: Tensor[H,d,d]) -> tuple[W: Tensor[H,d,d], Wp: Tensor[H,d,d]]   # G7 tolerance
    def leace_update(W, Wp, sxz: Tensor[H,d,1]) -> Q: Tensor[H,d,d]     # Wp(u)(u)^+ W, Â = I − βQ
    def midsteer_update(W, Wp, sxz1: Tensor[H,d,1], sxz2: Tensor[H,d,1]) -> Q2     # Wp(u1−u2)(u1)^+W, Â = I − βQ2 ... note sign: Eq.23 uses (u2−u1); Q2 = −(above)
    def compose_A(update, beta) -> Â = I ± beta·update; def bias(mu, Â) -> b̂
    # exact operator forms per §4.3; sign conventions pinned to Eqs. 22/23 as written there
  steering_llm.py
    class AffineSteerer: registers forward hooks on every self_attn.o_proj output;
      forward(h: Tensor[B,T,H,d]) -> Tensor[B,T,H,d]  # Â h + b̂ per head, all tokens (G17)
    class VanillaSteerer: h -> (I − β s sᵀ) h, s unit-norm
  steering_diffusion.py
    same two classes for every cross-attention out-proj; input Tensor[B,P,H,d]
experiments/
  configs/{e1_synth.yaml, e2_llm_concrete.yaml, e3_llm_safety.yaml, e4_sdxl_h2m.yaml, e5_sdxl_safety.yaml}
  run_e1_synth.py -> results/e1_metrics.json
  ... one entrypoint per experiment, each writing results/<exp>_metrics.json with schema:
  {experiment, arm, beta, concept_pair, seed, metrics: {metric_name: float}, n_generations, config_hash}
eval/
  judge_cs.py     # Llama-3.1-8B-Instruct, prompt verbatim paper/content/suppl.tex:423-436, argmax over "0".."10"
  clip_cs.py      # CLIP score ×100 (backbone pinned in config, G11)
  fid.py          # steered vs vanilla image sets
  detoxify_score.py, armorm_score.py, bertscore_mmlu.py
```

Artifact pickles (estimation stage → disk), mirroring upstream layout so upstream scripts stay
usable: `artifacts/{model}/mu_neutral.pkl`, `sigma_neutral.pkl`,
`artifacts/{model}/steering_vector_{concept}.pkl`, each
`dict[step:int][place:str] -> list[Tensor[H,d]]` (sigma: `Tensor[H,d,d]`).

**Evaluation-quantity convention for claims.json:** every `measured.<arm>.<metric>` is evaluated
per seed in {0,1,2}; ordering claims pass iff the stated direction holds at *every* seed; value
claims pass iff the per-seed mean lies within `claimed ± tolerance`; `invariant`/`existence`
claims pass iff the predicate holds at every seed.

---

## 7. Arms the paper compares, with configuration

Four arms per the paper's own comparison (`paper/content/experiments.tex:49`, Table 1
`paper/flipping_main.tex:22-52`, Table 2 `paper/artefacts/tables/safety_table.tex:65-83`):

| Arm | Transform | β (default) | Config |
|---|---|---|---|
| `base` | identity | — | unsteered model |
| `vanilla` | Householder / projection via s = μ_s − μ_t (Eqs. 21, 24–25) | 2 for switching (`paper/content/experiments.tex:125`); grid {1..5} elsewhere | unit-norm s (G8) |
| `leace_switch` | Eq. 22 with Σ_XZ from source-class indicator (G3) | 2 default (`paper/content/experiments.tex:125`); grid {1..5} | per §6 |
| `midsteer` | Eq. 23 | 1 default (`paper/content/experiments.tex:125`); grid {1..5} | per §6 |

Erasure = `leace_switch` at β=1 / `midsteer` with constant Z_2 (`paper/main.tex:461`); covered by
the synthetic claim C3 rather than a separate model arm (restriction note below and §10).

**Experiments (claims coverage):**
- E1 synth (CPU): closed-form checks on synthetic Gaussian data (arms: the closed forms
  themselves). Carries C1–C3.
- E2 llm_concrete: Llama-2-7B-chat (G10) templates G.1, pairs horse→motorcycle and dog→cat,
  β grid per claim. Carries LLM-concrete claims (curve claim C20 is restricted to these two pairs;
  chihuahua→muffin excluded — the paper publishes no Llama-judge per-β table for it).
- E3 llm_safety: toxicity→helpfulness, Llama-2-7B-chat, RTP ≥ 0.5 prompts, β ∈ {3,5}. Carries
  safety-LLM claims.
- E4 sdxl_h2m: SDXL base, templates G.2, horse→motorcycle, β {2,2,1} for the three arms. Carries
  Table-1 claims.
- E5 sdxl_safety: violence→peace, β ∈ {3,5}. Carries Table-2b claims.

---

## 8. Restriction analysis (per arm): how the setup restricts the paper's

Only *situations* are narrowed; the definition of a correct measurement (metric identities,
metric definitions, constraint definitions) is never changed (Bennett child, def. 6 of
arXiv:2301.12987: situation set ⊆, correctness criterion =).

| Choice | Paper | Reproduction | Restriction? | Claims affected |
|---|---|---|---|---|
| Seeds per prompt | 10 (`paper/content/experiments.tex:69`) | 3 ({0,1,2}) | yes — subset of generation draws | all model-level claims; direction claims robust, magnitude claims rated low |
| Template prompts | 80(–81) (`paper/content/suppl.tex:457-631`) | full listed set | none | — |
| M (Σ_XX prompts) | 50 000 | 5 000 (may raise if budget allows) | yes — subset; justified by the paper's own ablation (`paper/content/suppl.tex:640-642`) + our Fig-5 read | E2–E5; bound: paper states metrics stabilise by 5 000 |
| N (Σ_XZ prompts) | 1 000 per concept | 1 000 per concept | none (G1 weakest reading) | — |
| β grid | {1,2,3,4,5} | {2,2,1} defaults for Table-1 claims; {3,5} for safety/curve claims | yes — subset of the grid | no claim references untested β |
| Models | Llama-2-7B-chat, Qwen2.5-7B/14B, SDXL, SANA | Llama-2-7B-chat, SDXL | yes — subset of architectures | Qwen/SANA-specific claims are not made (§10) |
| Concept pairs | 3 concrete pairs + 2 safety pairs | LLM concrete: horse→motorcycle, dog→cat (chihuahua→muffin excluded — no Llama-judge per-β table published); SDXL concrete: horse→motorcycle; both safety pairs kept | yes — subset | curve claim C20 covers 2 of 3 LLM pairs; magnitude claims scoped to the tested cells |
| Eval items per condition | 80×10 = 800 | 80(–81)×3 = 240–243 | yes — subset | bundled into sensitivities |
| Erasure experiments (App. L) | LLM+diffusion erasure tables | covered synthetically only | situation subset | erasure-specific model claims not made (§10) |

**What each arm still supports:** `base/vanilla/leace_switch/midsteer` at the tested (β, model,
pair) cells support all ordering and direction claims in claims.json at those cells (definitions
unchanged). **What they do not support:** exact CLIP/FID/CS magnitudes off the tested cell,
frontier-shape claims beyond the tested β set, and any Qwen/SANA behaviour — those claims are
either not made or rated low/medium (see claims.json `compute_invariance`).

---

## 9. claims.json

Written to `claims.json` (authoritative copy; embedded verbatim below for review). ≥3 seeds:
{0,1,2}. Every claim carries the paper sentence in `quote`, an on-disk `citation`, a `kind`, a
`compute_invariance` rating, and the settling arithmetic; high claims carry `sensitivity`.

See ./claims.json — kept byte-identical with the block below.

```json
{
  "schema_version": 1,
  "paper": "MidSteer: Optimal Affine Framework for Steering Generative Models (arXiv:2605.05220v3)",
  "arms": {
    "base":         {"transform": "identity", "beta": null},
    "vanilla":      {"transform": "householder_or_projection_eq21_24_25", "s": "mean_source_minus_mean_target_unit_norm", "beta_default_switching": 2, "beta_grid": [1, 2, 3, 4, 5]},
    "leace_switch": {"transform": "eq22_W+_(W Sxz)(W Sxz)^+ W", "beta_default": 2, "beta_grid": [1, 2, 3, 4, 5]},
    "midsteer":     {"transform": "eq23_W+_(Swz2 - Swz1)(Swz1)^+ W", "beta_default": 1, "beta_grid": [1, 2, 3, 4, 5]}
  },
  "seeds": [0, 1, 2],
  "experiments": {
    "e1_synth":        {"compute": "cpu", "purpose": "closed-form correctness", "claims": ["C1", "C2", "C3"]},
    "e2_llm_concrete": {"model": "Llama-2-7b-chat", "pairs": ["horse->motorcycle", "dog->cat"], "prompts": "paper/content/suppl.tex:457-539 templates", "metric_keys": ["src_cs_on_src", "tgt_cs_on_src", "src_cs_on_tgt", "tgt_cs_on_tgt", "unrel_cs", "bertp_mmlu"]},
    "e3_llm_safety":   {"model": "Llama-2-7b-chat", "switch": "toxicity->helpfulness", "metric_keys": ["rtp", "help", "unrel_cs", "mmlu_bert_f1"], "betas": [3, 5]},
    "e4_sdxl_h2m":     {"model": "SDXL-base-1.0", "switch": "horse->motorcycle", "metric_keys": ["horse_cs_on_horse", "moto_cs_on_horse", "horse_cs_on_moto", "moto_cs_on_moto", "cow_cs", "cow_fid", "pig_cs", "pig_fid", "dog_cs", "dog_fid", "legislator_cs", "legislator_fid"], "betas": {"vanilla": 2, "leace_switch": 2, "midsteer": 1}},
    "e5_sdxl_safety":  {"model": "SDXL-base-1.0", "switch": "violence->peace", "metric_keys": ["viol_cs", "peace_cs", "unrel_cs", "fid"], "betas": [3, 5]}
  },
  "evaluation_rule": "Every measured.<arm>.<metric> is evaluated per seed in seeds. Ordering claims pass iff the stated direction holds at every seed. Value claims pass iff the per-seed mean lies within claimed +/- tolerance. invariant/existence claims pass iff the predicate holds at every seed. Curve claims pass iff the comparison holds at every x at every seed.",
  "claims": [
    {
      "id": "C1", "kind": "invariant", "compute_invariance": "high",
      "quote": "has the following solution, almost surely:\n\\begin{align}\n    \\widehat A\n    &=\n    I - W^+(W\\Sigma_{XZ})(W\\Sigma_{XZ})^+W,",
      "citation": "paper/content/guardedness.tex:73",
      "name": "LEACE closed form enforces zero covariance and is minimal-disturbance; vanilla erasure is its standardized special case",
      "predicate": "For each seed: draw random PSD Sigma_XX (d=32), column vector Sigma_XZ in Im(Sigma_XX), sample 200000 pairs (X,Z) jointly Gaussian with those covariances. (i) ||Cov(A_hat X + b_hat, Z)||_F < 1e-6 with A_hat,b_hat from Eqs.6-7 (paper/content/guardedness.tex:74-83); (ii) E||A_hat X + b_hat - X||^2 <= min over comparator affine maps {I - c W+ (W Sxz)(W Sxz)+ W, c in linspace(0,2,41)} + 1e-6; (iii) under standardized data (E[X]=0, Sigma_XX=I): max_x ||f_delete(x,s) - (A_hat x + b_hat)||_inf < 1e-8 over 10000 samples with s from Eq.1 (Corollary 4.1, paper/main.tex:297-315).",
      "sensitivity": {"fixed_by_paper": true, "citation": "paper/content/guardedness.tex:56-85"}
    },
    {
      "id": "C2", "kind": "invariant", "compute_invariance": "high",
      "quote": "\\widehat{A} &= I - 2W^+(W \\Sigma_{XZ}) (W \\Sigma_{XZ})^+ W",
      "citation": "paper/main.tex:365",
      "name": "LEACE-Switch closed form flips the sign of Cov and is minimal-disturbance; Householder switching is its standardized special case",
      "predicate": "Setup as C1. (i) ||Cov(A_hat X + b_hat, Z) + Cov(X, Z)||_F < 1e-6 with A_hat from Eq.13 (paper/main.tex:363-367); (ii) E||A_hat X + b_hat - X||^2 <= min over comparator affine maps {I - c W+ (W Sxz)(W Sxz)+ W, c in linspace(0,4,81)} + 1e-6; (iii) under standardized data: max_x ||f_switch(x,s) - (A_hat x + b_hat)||_inf < 1e-8 (Corollary 4.3, paper/main.tex:377-394).",
      "sensitivity": {"fixed_by_paper": true, "citation": "paper/main.tex:343-367"}
    },
    {
      "id": "C3", "kind": "invariant", "compute_invariance": "high",
      "quote": "\\widehat{A} &= I + W^+ (\\Sigma_{WX, Z_2} - \\Sigma_{WX, Z_1})\\Sigma_{WX, Z_1}^+ W",
      "citation": "paper/main.tex:447",
      "name": "MidSteer closed form matches Cov(f(X),Z1) to Cov(X,Z2), is minimal-disturbance, and reduces to LEACE for constant Z2",
      "predicate": "Setup as C1 with l=1, Sigma_XZ1, Sigma_XZ2 random columns in Im(Sigma_XX), Sigma_XZ1 nonzero. (i) ||Cov(A_hat X + b_hat, Z1) - Cov(X, Z2)||_F < 1e-6 with A_hat from Eq.19 (paper/main.tex:445-448); (ii) E||A_hat X + b_hat - X||^2 <= min over comparator affine maps {I + c W+ (Swz2 - Swz1)(Swz1)+ W, c in linspace(0,2,41), W-completed to satisfy the constraint} + 1e-6; (iii) with Z2 constant (Sigma_XZ2 = 0): ||A_hat_midsteer - A_hat_leace||_F < 1e-8 (paper/main.tex:461).",
      "sensitivity": {"fixed_by_paper": true, "citation": "paper/main.tex:418-448"}
    },
    {
      "id": "C4", "kind": "ordering", "compute_invariance": "high", "experiment": "e4_sdxl_h2m",
      "quote": "In contrast, MidSteer keeps ``motorcycle'' intact.",
      "citation": "paper/content/experiments.tex:127",
      "name": "MidSteer keeps target concept intact on target prompts (vs vanilla)",
      "quantity": "measured.midsteer.moto_cs_on_moto - measured.vanilla.moto_cs_on_moto",
      "direction": ">0",
      "config": {"vanilla_beta": 2, "midsteer_beta": 1},
      "sensitivity": {"parameter": "n_eval_generations_per_concept_cell", "plausible": [80, 800], "survives": [240, 800], "note": "CLIP backbone unstated (gap G11); the paper's gap here is ~17.8 CLIP-x100 points, 6-10x the standard error of our per-seed mean at 240 generations, so the direction is robust to the backbone choice a careful reader could make (CLIPScore default ViT-B/32 through ViT-L/14)."}
    },
    {
      "id": "C5", "kind": "ordering", "compute_invariance": "high", "experiment": "e4_sdxl_h2m",
      "quote": "In contrast, MidSteer keeps ``motorcycle'' intact.",
      "citation": "paper/content/experiments.tex:127",
      "name": "MidSteer keeps target concept intact on target prompts (vs LEACE-Switch)",
      "quantity": "measured.midsteer.moto_cs_on_moto - measured.leace_switch.moto_cs_on_moto",
      "direction": ">0",
      "config": {"leace_switch_beta": 2, "midsteer_beta": 1},
      "sensitivity": {"parameter": "n_eval_generations_per_concept_cell", "plausible": [80, 800], "survives": [240, 800], "note": "Same backbone robustness argument as C4; paper gap ~17.4 CLIP-x100 points."}
    },
    {
      "id": "C6", "kind": "value", "compute_invariance": "low", "experiment": "e4_sdxl_h2m",
      "quote": "MidSteer (ours) & 1.0 \n& 51.2 & 68.7 \n& 51.9 & 70.7 & 12.7",
      "citation": "paper/flipping_main.tex:46-48",
      "name": "MidSteer target CS on motorcycle prompts equals the base value 70.7",
      "quantity": "measured.midsteer.moto_cs_on_moto",
      "claimed": 70.7,
      "tolerance": 3.0,
      "config": {"midsteer_beta": 1},
      "sensitivity": {"parameter": "clip_backbone_and_seed_count", "plausible": [80, 800], "survives": [240, 800], "note": "Exact magnitude leans on the unstated CLIP backbone (G11); rated low - tolerance covers backbone and seed-count reading error."}
    },
    {
      "id": "C7", "kind": "ordering", "compute_invariance": "high", "experiment": "e4_sdxl_h2m",
      "quote": "vanilla steering (CASteer) and LEACE fail when presented with prompt for the target concept (\"motorcycle\"), unable to distinguish between forward and reverse steering.",
      "citation": "paper/content/experiments.tex:108",
      "name": "Vanilla/LEACE-Switch re-induce the source concept on target prompts; MidSteer does not",
      "quantity": "measured.vanilla.horse_cs_on_moto - measured.midsteer.horse_cs_on_moto",
      "direction": ">0",
      "config": {"vanilla_beta": 2, "midsteer_beta": 1},
      "sensitivity": {"parameter": "n_eval_generations_per_concept_cell", "plausible": [80, 800], "survives": [240, 800], "note": "Paper gap 68.3 vs 51.9 (base 51.8): ~16 CLIP-x100 points; robust."}
    },
    {
      "id": "C8", "kind": "ordering", "compute_invariance": "high", "experiment": "e3_llm_safety",
      "quote": "For Llama-2-7B-chat, MidSteer achieves the strongest reduction in toxicity, decreasing the RTP toxicity score from $.371$ to $.281$ at $\\beta=5$",
      "citation": "paper/content/safety-related_results.tex:10",
      "name": "MidSteer reduces RTP toxicity more than LEACE-Switch at beta=5",
      "quantity": "measured.midsteer.rtp - measured.leace_switch.rtp",
      "direction": "<0",
      "config": {"beta": 5},
      "sensitivity": {"parameter": "n_rtp_prompts_with_toxicity_ge_0.5", "plausible": [80, 800], "survives": [240, 800], "note": "Detoxify variant unstated (G11); the two published Detoxify weight sets preserve this ordering for a .065 gap on a ~.35 base."}
    },
    {
      "id": "C9", "kind": "ordering", "compute_invariance": "high", "experiment": "e3_llm_safety",
      "quote": "decreasing the RTP toxicity score from $.371$ to $.281$ at $\\beta=5$",
      "citation": "paper/content/safety-related_results.tex:10",
      "name": "MidSteer reduces RTP toxicity below base at beta=5",
      "quantity": "measured.midsteer.rtp - measured.base.rtp",
      "direction": "<0",
      "config": {"beta": 5},
      "sensitivity": {"parameter": "n_rtp_prompts_with_toxicity_ge_0.5", "plausible": [80, 800], "survives": [240, 800]}
    },
    {
      "id": "C10", "kind": "value", "compute_invariance": "low", "experiment": "e3_llm_safety",
      "quote": "MidSteer-5  & \\textbf{.281}",
      "citation": "paper/artefacts/tables/safety_table.tex:71",
      "name": "MidSteer-5 RTP toxicity value 0.281",
      "quantity": "measured.midsteer.rtp",
      "claimed": 0.281,
      "tolerance": 0.05,
      "config": {"beta": 5},
      "sensitivity": {"parameter": "detoxify_variant_and_prompt_count", "plausible": [80, 800], "survives": [240, 800], "note": "Exact magnitude leans on unstated Detoxify variant, RTP prompt count and continuation length (G11/G15); rated low, tolerance is ~18% of base scale."}
    },
    {
      "id": "C11", "kind": "ordering", "compute_invariance": "high", "experiment": "e3_llm_safety",
      "quote": "while preserving helpfulness at the baseline level",
      "citation": "paper/content/safety-related_results.tex:10",
      "name": "Vanilla collapses helpfulness at beta=5 while MidSteer preserves it",
      "quantity": "measured.vanilla.help - measured.midsteer.help",
      "direction": "<0",
      "config": {"beta": 5},
      "sensitivity": {"parameter": "n_helpfulness_prompts", "plausible": [80, 800], "survives": [240, 800], "note": "ArmoRM checkpoint unstated (G11); paper gap .029 vs .117 = most of the .117 base scale, robust to any reasonable ArmoRM release."}
    },
    {
      "id": "C12", "kind": "ordering", "compute_invariance": "high", "experiment": "e3_llm_safety",
      "quote": "substantially degrades unrelated-concept preservation at higher strength, with the unrelated concept score dropping from $8.46$ to $3.11$",
      "citation": "paper/content/safety-related_results.tex:11",
      "name": "MidSteer preserves unrelated concepts where vanilla-5 collapses",
      "quantity": "measured.midsteer.unrel_cs - measured.vanilla.unrel_cs",
      "direction": ">0",
      "config": {"beta": 5},
      "sensitivity": {"parameter": "n_unrelated_generations_per_concept", "plausible": [80, 800], "survives": [240, 800], "note": "LLM-judge CS (Llama-3.1-8B-Instruct) per main text; paper gap 8.48 vs 3.11 on a 0-10 scale."}
    },
    {
      "id": "C13", "kind": "ordering", "compute_invariance": "medium", "experiment": "e3_llm_safety",
      "quote": "LEACE-Switch preserves general capabilities slightly better, as reflected by the highest MMLU BERT-F1 among interventions",
      "citation": "paper/content/safety-related_results.tex:12",
      "name": "LEACE-Switch has the highest MMLU BERT-F1 among intervention arms",
      "quantity": "measured.midsteer.mmlu_bert_f1 - measured.leace_switch.mmlu_bert_f1",
      "direction": "<0",
      "config": {"beta": 5},
      "sensitivity": {"parameter": "n_mmlu_prompts", "plausible": [200, 2000], "survives": [500, 2000], "note": "BERTScore backbone and MMLU subset unstated (G11); paper gap is .01, hence medium."}
    },
    {
      "id": "C14", "kind": "ordering", "compute_invariance": "high", "experiment": "e5_sdxl_safety",
      "quote": "Vanilla steering also reduces violence, but either underperforms MidSteer on source suppression or causes larger distortions",
      "citation": "paper/content/safety-related_results.tex:15",
      "name": "MidSteer suppresses violence below vanilla at beta=5",
      "quantity": "measured.midsteer.viol_cs - measured.vanilla.viol_cs",
      "direction": "<0",
      "config": {"beta": 5},
      "sensitivity": {"parameter": "n_eval_generations_per_concept_cell", "plausible": [80, 800], "survives": [240, 800], "note": "Paper gap 6.0 vs 20.8 CLIP-x100 points."}
    },
    {
      "id": "C15", "kind": "ordering", "compute_invariance": "high", "experiment": "e5_sdxl_safety",
      "quote": "MidSteer most effectively suppresses the source concept, reducing the violence score from $99.6$ to $6.0$",
      "citation": "paper/content/safety-related_results.tex:15",
      "name": "Vanilla also reduces violence relative to base (while underperforming MidSteer)",
      "quantity": "measured.vanilla.viol_cs - measured.base.viol_cs",
      "direction": "<0",
      "config": {"beta": 5},
      "sensitivity": {"parameter": "n_eval_generations_per_concept_cell", "plausible": [80, 800], "survives": [240, 800]}
    },
    {
      "id": "C16", "kind": "ordering", "compute_invariance": "high", "experiment": "e5_sdxl_safety",
      "quote": "LEACE-Switch yields the lowest FID among interventions but is substantially less effective at inducing the target concept.",
      "citation": "paper/content/safety-related_results.tex:15",
      "name": "MidSteer induces peace far better than LEACE-Switch at beta=5",
      "quantity": "measured.midsteer.peace_cs - measured.leace_switch.peace_cs",
      "direction": ">0",
      "config": {"beta": 5},
      "sensitivity": {"parameter": "n_eval_generations_per_concept_cell", "plausible": [80, 800], "survives": [240, 800], "note": "Paper gap 94.0 vs 54.5 CLIP-x100 points."}
    },
    {
      "id": "C17", "kind": "value", "compute_invariance": "low", "experiment": "e5_sdxl_safety",
      "quote": "MidSteer-5  & \\textbf{6.0} & \\textbf{94.0}",
      "citation": "paper/artefacts/tables/safety_table.tex:83",
      "name": "MidSteer-5 violence score 6.0",
      "quantity": "measured.midsteer.viol_cs",
      "claimed": 6.0,
      "tolerance": 5.0,
      "config": {"beta": 5},
      "sensitivity": {"parameter": "clip_backbone_and_seed_count", "plausible": [80, 800], "survives": [240, 800], "note": "Exact magnitude leans on unstated CLIP backbone (G11); rated low; tolerance ~5x the per-seed standard error."}
    },
    {
      "id": "C18", "kind": "ordering", "compute_invariance": "medium", "experiment": "e5_sdxl_safety",
      "quote": "LEACE-Switch yields the lowest FID among interventions",
      "citation": "paper/content/safety-related_results.tex:15",
      "name": "LEACE-Switch has lowest unrelated-concept FID among intervention arms at beta=3",
      "quantity": "measured.leace_switch.fid - min(measured.vanilla.fid, measured.midsteer.fid)",
      "direction": "<0",
      "config": {"beta": 3},
      "sensitivity": {"parameter": "n_unrelated_generations_per_concept", "plausible": [80, 800], "survives": [240, 800], "note": "FID on ~240 images per cell is noisy and its estimator is unstated (G11); paper gap at beta=3 is 88.9 vs 93.6/101.5."}
    },
    {
      "id": "C19", "kind": "ordering", "compute_invariance": "high", "experiment": "e2_llm_concrete",
      "quote": "In contrast, MidSteer keeps ``motorcycle'' intact.",
      "citation": "paper/content/experiments.tex:127",
      "name": "LLM: MidSteer keeps motorcycle intact on motorcycle prompts (Llama-2-7B-chat, judge CS)",
      "quantity": "measured.midsteer.tgt_cs_on_tgt - max(measured.vanilla.tgt_cs_on_tgt, measured.leace_switch.tgt_cs_on_tgt)",
      "direction": ">0",
      "config": {"vanilla_beta": 2, "leace_switch_beta": 2, "midsteer_beta": 1, "pair": "horse->motorcycle"},
      "sensitivity": {"parameter": "n_eval_generations_per_concept_cell", "plausible": [80, 800], "survives": [240, 800], "note": "Judge = Llama-3.1-8B-Instruct (G10). Paper numbers at default betas: 8.5 vs 7.9/5.4 (paper/artefacts/tables/llm_flip_tables_noclip.tex, horses_to_motorcycles); GPT-4o-mini judge agrees on ordering at all beta (paper/content/suppl.tex:664)."}
    },
    {
      "id": "C20", "kind": "curve", "compute_invariance": "high", "experiment": "e2_llm_concrete",
      "quote": "In each case, we see clear superiority of MidSteer over other steering approaches.",
      "citation": "paper/content/switching_suppl.tex:23",
      "name": "LLM source-concept suppression: MidSteer below both baselines across beta (Pareto dominance, source axis)",
      "quantity": "for pair in [horse->motorcycle, dog->cat]: [min(measured.vanilla.src_cs_on_src(beta), measured.leace_switch.src_cs_on_src(beta)) - measured.midsteer.src_cs_on_src(beta) for beta in x] (mean over the 2 pairs)",
      "x": [3, 4, 5],
      "comparison": "above",
      "reference": 0.0,
      "sensitivity": {"parameter": "n_eval_generations_per_concept_cell", "plausible": [80, 800], "survives": [240, 800], "note": "Pair set restricted to the two pairs with published Llama-3.1-8B-judge per-beta tables (paper/artefacts/tables/llm_flip_tables_noclip.tex); chihuahua->muffin excluded (only GPT-4o-mini tables exist). Backed by figure read: figure_reads/transcript.md Q1/Q2 (dominant method = MiDSteer)."}
    },
    {
      "id": "C21", "kind": "curve", "compute_invariance": "high", "experiment": "e4_sdxl_h2m",
      "quote": "We see that in each case MidSteer achieves much better balance between level of concept switch between $c_1$ and $c_2$ and preservation of other concepts across different values of $\\beta$.",
      "citation": "paper/content/experiments.tex:121",
      "name": "SDXL source-concept suppression: MidSteer below both baselines across beta (Pareto dominance, source axis)",
      "quantity": "[min(measured.vanilla.horse_cs_on_horse(beta), measured.leace_switch.horse_cs_on_horse(beta)) - measured.midsteer.horse_cs_on_horse(beta) for beta in x]",
      "x": [1, 2, 3, 4, 5],
      "comparison": "above",
      "reference": 0.0,
      "sensitivity": {"parameter": "n_eval_generations_per_concept_cell", "plausible": [80, 800], "survives": [240, 800], "note": "Paper margins >= 1.2 CLIP-x100 points at every beta (paper/artefacts/tables/diffusion_flip_tables_noclip.tex, Table 16). Backed by figure reads: figure_reads/transcript.md Q3/Q4 (dominant method = MiDSteer)."}
    },
    {
      "id": "C22", "kind": "curve", "compute_invariance": "medium", "experiment": "e2_llm_concrete",
      "quote": "Fig.~\\ref{fig:num_covs} shows that performance largely stabilizes around 5,000 prompts in the tested Llama-2-7B setting.",
      "citation": "paper/content/suppl.tex:642",
      "name": "Sigma_XX estimation with M>=5000 prompts matches M=50000 within tolerance",
      "quantity": "[measured.midsteer.bertp_mmlu(M) for M in x] (MidSteer, beta grid {2,3,4} as in ablation)",
      "x": [5000, 10000, 20000],
      "comparison": "matches",
      "claimed": "measured.midsteer.bertp_mmlu(M=50000)",
      "tolerance": 0.01,
      "note": "Our reproduction's M=5000 restriction (SPEC.md section 8) leans on this claim. Figure read: figure_reads/transcript.md Q5 (5000/10000/20000 cluster together = yes).",
      "sensitivity": {"parameter": "midsteer_beta_grid_for_ablation", "plausible": [2, 4], "survives": [2, 4]}
    }
  ],
  "not_tested": [
    "Appendix L erasure tables (paper/content/erasure_suppl.tex:1-20): erasure is the Z2-constant special case (paper/main.tex:461), tested synthetically in C3; running the full erasure model tables doubles generation cost for a strictly weaker claim.",
    "Qwen2.5-7B/14B and SANA results (paper/content/suppl_safety.tex; paper/artefacts/tables/llm_flip_tables_noclip.tex Qwen tables): architecture subset; two extra model families out of budget.",
    "GPT-4o-mini judge cross-validation (paper/content/suppl.tex:657-664): needs OPENAI_API_KEY; main-text judge (Llama-3.1-8B-Instruct) is what the paper's headline claims are computed with.",
    "Exact BERT-Precision/F1 magnitudes (paper/content/experiments.tex:98) and MMLU subset composition: BERTScore backbone and MMLU prompt count unstated (G11); only the ordering claim C13 touches this family.",
    "Qualitative figures (paper/img/teaser.png, paper/artefacts/pic_llms_main.tex): not quantitative claims.",
    "Full per-beta tables K.2/K.3 (paper/artefacts/tables/*_noclip.tex): locked into per-cell ordering claims at the tested betas only."
  ]
}
```

Gating note: gating is on the claims with compute_invariance = high (C1–C5, C7–C9, C11–C12,
C14–C16, C19–C21). Three of them — the synthetic closed-form checks C1–C3 — need only CPU
minutes and no model download, so the suite always has something decisive to say. C13, C18, C22
are medium (gap small or estimator noisy at our budget); C6, C10, C17 are low (exact magnitudes
that lean on unstated backbones).

## 10. Claims deliberately not tested

- Appendix L erasure tables (`paper/content/erasure_suppl.tex:1-20`): erasure is the Z_2-constant
  special case (`paper/main.tex:461`); tested synthetically (C3). Running the full erasure model
  tables doubles generation cost for a claim strictly weaker than the switching claims.
- Qwen2.5-7B/14B and SANA results (`paper/content/suppl_safety.tex`, `paper/artefacts/tables/*`):
  architecture subset; ordering claims for them would need 2 extra model families.
- GPT-4o-mini judge cross-validation (`paper/content/suppl.tex:657-664`): requires an OpenAI API
  key; the claim it backs ("ordering robust to judge choice") is already supported two ways in
  the paper and we test the main-text judge only. Recorded as untested rather than weakened.
- Exact BERT-Precision/F1 magnitudes on MMLU and "Unrel." CS averages across all models: rated
  too jumpy for value claims (BERTScore backbone unstated, G11); only the LLM-safety ordering
  claim C14 (LEACE best MMLU, medium) touches this family.
- Qualitative figures (Fig. 3 `paper/img/teaser.png`, Fig. 4 `paper/artefacts/pic_llms_main.tex`):
  not quantitative claims; used only as motivation.
- Full per-β tables (K.2, K.3): locked into per-cell ordering claims at tested β only.

## 11. Figure-reading log

`read-figure` exchanges backing the curve claims are committed at `figure_reads/transcript.md`.
Questions asked in constrained form ("Answer with exactly ..."); replies were one/two words as
required (no deliberation flags triggered). Readings:
- Fig. 2a (LLM, ΔCS vs unrelated CS): dominant method = MiDSteer.
- Fig. 2b (LLM, ΔCS vs 1−BERTPrecision MMLU): dominant = MiDSteer.
- Fig. 2c (SDXL, ΔCS vs unrelated CS): dominant = MiDSteer.
- Fig. 2d (SDXL, ΔCS vs unrelated FID): dominant = MiDSteer.
- Fig. 5 (Σ_XX prompt count): 5000/10000/20000 cluster together = "yes".
Each figure's underlying PDF was rendered to PNG at 150 dpi from the arXiv source artefacts.
