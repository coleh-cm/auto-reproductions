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
  (to be verified, not assumed, during adoption; verified against HEAD 0f3b31e on 2026-08-17):
  1. `core/controller.py` applies `intermediate_clipping=True` (projection scores clipped to ≥ 0,
     lines 216–217 and 260–261; constructor default `True` at line 99) in all steer paths.
     **Not in the paper.** Default ON in the class. The CLI opt-in help text reads
     "Apply intermediate clipping like CASteer for leace and midsteer"
     (`scripts/diffusion/run_with_steering.py:160`, `scripts/llm/run_with_steering.py:191`) —
     upstream's own description attributes the convention to CASteer, not to this paper. This is a
     candidate cause of a paper-vs-code mismatch; the paper's affine map is linear in (X−μ).
  2. `core/controller.py:157-176` uses class-conditional *mean differences* relative to a neutral
     mean (`mu_neutral`) instead of the literal `Cov(X, Z_i)` (which includes the class-prior
     factor `P(Z_i=1)`; see gap G3). Verified: `source_vector -= m_neutral`,
     `target_vector -= m_neutral`, then whitening by `sigma_minus_half`; the `leace` arm builds
     its update direction from the whitened **source − target** difference
     (`steering_vector = source_vector - target_vector`, `pinv(steering_vector)`), which is
     exactly the G3 pair reading with a balanced prior; the `midsteer` arm uses
     `pinv(source_vector)` as Eq. 23 prescribes.
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
  is what the upstream code does (`core/controller.py:157-176`, verified §1 item 2). The same
  reading fixes the LEACE-Switch experiments' Z as the source-class indicator within the
  {source prompts ∪ target prompts} sample (the theorem's dataset partition), and the vanilla
  pair vector as s = μ_s − μ_t.
  *Precision on what (b) commits to.* With class priors c_i = P(Z_i=1), the literal
  Σ_XZi = c_i(μ_i − μ), so the Eq. 23 update direction is (c_2 Wμ̂_2 − c_1 Wμ̂_1)(Wμ̂_1)^+ =
  ((c_2/c_1) Wμ̂_2 − Wμ̂_1)(Wμ̂_1)^+ : unequal priors change the update DIRECTION, so a prior
  ratio cannot in general be absorbed into the printed Eq. 23 β (which scales the whole
  correction (Σ_WX,Z2 − Σ_WX,Z1)). Reading (b) corresponds exactly to the literal formula
  specialised to balanced priors c_1 = c_2, which is also what "class-conditional covariances
  based on a sample of size N = 1000" per concept (`paper/content/experiments.tex:63`) most
  directly admits. Note the paper's commented-out alternative (β = c_2/c_1 multiplying only the
  target term, `paper/main.tex:494-504`) is a DIFFERENT parametrisation from printed Eq. 23; we
  implement the printed equation with reading (b), and record the unbalanced-mix case as an
  untested reading (§10). At β = 1 (the MidSteer default used in claims) reading (b) and the
  balanced literal reading coincide exactly.

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
| β grid | {1,2,3,4,5} safety; {1.0,1.5,...,5.0} in the Llama concrete tables (`paper/artefacts/tables/llm_flip_tables_noclip.tex`) | {2,2,1} defaults for Table-1 claims; {3,5} safety; {3,4,5}/{1..5} integer curve grids | yes — subset of the grid | no claim references untested β |
| Models | Llama-2-7B-chat, Qwen2.5-7B/14B, SDXL, SANA | Llama-2-7B-chat, SDXL | yes — subset of architectures | Qwen/SANA-specific claims are not made (§10) |
| Concept pairs | 3 concrete pairs + 2 safety pairs | LLM concrete: horse→motorcycle, dog→cat (chihuahua→muffin excluded — no Llama-judge per-β table published); SDXL concrete: horse→motorcycle; both safety pairs kept | yes — subset | curve claim C20 covers 2 of 3 LLM pairs; magnitude claims scoped to the tested cells |
| Eval items per condition | 80×10 = 800 | 80(–81)×3 = 240–243 | yes — subset | bundled into sensitivities |
| Erasure experiments (App. L) | LLM+diffusion erasure tables | covered synthetically only | situation subset | erasure-specific model claims not made (§10) |

**What each arm still supports:** `base/vanilla/leace_switch/midsteer` at the tested (β, model,
pair) cells support all ordering and direction claims in claims.json at those cells (definitions
unchanged). **What they do not support:** exact CLIP/FID/CS magnitudes off the tested cell,
frontier-shape claims beyond the tested β set, and any Qwen/SANA behaviour — those claims are
either not made or rated low/medium (see claims.json `compute_invariance`).

The same per-arm restriction statements exist in machine-readable form as the top-level
`restrictions` map in claims.json, keyed `{"<arm>": {"kind": "narrows_situations", "detail": ...}}`
for all four arms — every arm narrows only the situation set (seeds 3 of 10, β subset, model
subset, pair subset, M = 5 000 of 50 000 for the covariance arms under the Fig. 5 / C22
justification); no arm changes a metric, constraint or success criterion, so no entry is
`changes_correctness`.

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
    "base": {
      "transform": "identity",
      "beta": null
    },
    "vanilla": {
      "transform": "householder_or_projection_eq21_24_25",
      "s": "mean_source_minus_mean_target_unit_norm",
      "beta_default_switching": 2,
      "beta_grid": [
        1,
        2,
        3,
        4,
        5
      ]
    },
    "leace_switch": {
      "transform": "eq22_W+_(W Sxz)(W Sxz)^+ W",
      "beta_default": 2,
      "beta_grid": [
        1,
        2,
        3,
        4,
        5
      ]
    },
    "midsteer": {
      "transform": "eq23_W+_(Swz2 - Swz1)(Swz1)^+ W",
      "beta_default": 1,
      "beta_grid": [
        1,
        2,
        3,
        4,
        5
      ]
    }
  },
  "restrictions": {
    "base": {
      "kind": "narrows_situations",
      "detail": "Identity arm (no intervention). Situations narrowed only: 3 generation seeds {0,1,2} of the paper's 10 per prompt (subset of draws, paper/content/experiments.tex:69); model subset {Llama-2-7B-chat, SDXL} of the paper's 4 architectures (Llama-2-7B-chat, Qwen2.5-7B/14B, SDXL, SANA); concept-pair subset (LLM concrete: horse->motorcycle, dog->cat of 3; SDXL concrete: horse->motorcycle of 3; both safety pairs kept); full listed prompt template set (80 LLM G.1 / 80 image G.2, paper/content/suppl.tex:457-631); RTP prompts = RealToxicityPrompts toxicity>=0.5 as stated (paper/content/experiments.tex:84), count restricted like other eval cells. Metric definitions unchanged: judge CS 0-10 (Llama-3.1-8B-Instruct), CLIP score x100, Detoxify RTP, ArmoRM helpfulness, FID vs vanilla, BERT-P/F1 on MMLU generations."
    },
    "vanilla": {
      "kind": "narrows_situations",
      "detail": "Householder/projection steering per Eqs. 21/24-25 (paper/main.tex:466-469, paper/content/suppl.tex:64-73) with unit-norm s = mu_source - mu_target (weakest readings G3/G8, SPEC.md section 5). Same situation-narrowing as base, plus: beta subset of the paper's grid {1,2,3,4,5} per claim (default 2 for switching per paper/content/experiments.tex:125; {3,5} for safety and curve claims); steering-vector means estimated on the same prompt sets (N=1000 concept, background for mu_t role); upstream code's intermediate_clipping disabled because it appears nowhere in the paper (SPEC.md section 1, item 1). Metric and success definitions unchanged."
    },
    "leace_switch": {
      "kind": "narrows_situations",
      "detail": "Eq. 22 arm (paper/main.tex:471-474), Sigma_XZ from the source-class indicator under the mean-difference cross-covariance reading (weakest reading G3 with class priors absorbed into beta). Same situation-narrowing as base, plus: M = 5000 of the paper's 50000 prompts for Sigma_XX (subset; justified by the paper's own ablation Fig. 5 paper/content/suppl.tex:640-642 and claim C22), N = 1000 per concept unchanged (paper/content/experiments.tex:63), beta subset as vanilla (default 2), estimation on the unsteered model at every SA/CA layer as stated (paper/content/experiments.tex:89,101), pinv rank tolerance = host-library default (G7), upstream intermediate_clipping disabled (not in the paper). Transform, covariance constraint (Cov -> -Cov) and metrics exactly as in the paper; only the situation set is narrowed (Bennett child, SPEC.md section 8)."
    },
    "midsteer": {
      "kind": "narrows_situations",
      "detail": "Eq. 23 arm (paper/main.tex:478-481) with per-head block-diagonal maps (weakest reading G5). Same situation-narrowing as leace_switch (M = 5000 of 50000 with Fig.5/C22 justification, N = 1000 unchanged, seeds 3 of 10, model and pair subsets, beta subset of the paper's grid with default 1 per paper/content/experiments.tex:125, pinv tolerance = host default G7, upstream clipping disabled). No metric, constraint (Cov(f(X),Z1) = Cov(X,Z2)) or success-criterion change: what counts as a correct measurement is identical to the paper; only which situations are evaluated is narrowed. A run at beta or model or pair cells outside the tested subset cannot speak to claims about those cells and does not."
    }
  },
  "seeds": [
    0,
    1,
    2
  ],
  "experiments": {
    "e1_synth": {
      "compute": "cpu",
      "purpose": "closed-form correctness",
      "claims": [
        "C1",
        "C2",
        "C3"
      ]
    },
    "e2_llm_concrete": {
      "model": "Llama-2-7b-chat",
      "pairs": [
        "horse->motorcycle",
        "dog->cat"
      ],
      "prompts": "paper/content/suppl.tex:457-539 templates",
      "metric_keys": [
        "src_cs_on_src",
        "tgt_cs_on_src",
        "src_cs_on_tgt",
        "tgt_cs_on_tgt",
        "unrel_cs",
        "bertp_mmlu"
      ]
    },
    "e3_llm_safety": {
      "model": "Llama-2-7b-chat",
      "switch": "toxicity->helpfulness",
      "metric_keys": [
        "rtp",
        "help",
        "unrel_cs",
        "mmlu_bert_f1"
      ],
      "betas": [
        3,
        5
      ]
    },
    "e4_sdxl_h2m": {
      "model": "SDXL-base-1.0",
      "switch": "horse->motorcycle",
      "metric_keys": [
        "horse_cs_on_horse",
        "moto_cs_on_horse",
        "horse_cs_on_moto",
        "moto_cs_on_moto",
        "cow_cs",
        "cow_fid",
        "pig_cs",
        "pig_fid",
        "dog_cs",
        "dog_fid",
        "legislator_cs",
        "legislator_fid"
      ],
      "betas": {
        "vanilla": 2,
        "leace_switch": 2,
        "midsteer": 1
      }
    },
    "e5_sdxl_safety": {
      "model": "SDXL-base-1.0",
      "switch": "violence->peace",
      "metric_keys": [
        "viol_cs",
        "peace_cs",
        "unrel_cs",
        "fid"
      ],
      "betas": [
        3,
        5
      ]
    }
  },
  "evaluation_rule": "Every measured.<arm>.<metric> is evaluated per seed in seeds. Ordering claims pass iff the stated direction holds at every seed. Value claims pass iff the per-seed mean lies within claimed +/- tolerance. invariant/existence claims pass iff the predicate holds at every seed. Curve claims: `quantity` yields a sequence sampled at `x`; comparison above/below compares it elementwise to the `against` sequence; comparison matches compares it elementwise to `claimed` (read off the figure) within tolerance; comparison increasing/decreasing requires adjacent differences of the stated sign. All curve comparisons must hold at every x at every seed.",
  "claims": [
    {
      "id": "C1",
      "kind": "invariant",
      "compute_invariance": "high",
      "quote": "has the following solution, almost surely:\n\\begin{align}\n    \\widehat A\n    &=\n    I - W^+(W\\Sigma_{XZ})(W\\Sigma_{XZ})^+W,",
      "citation": "paper/content/guardedness.tex:73-77",
      "name": "LEACE closed form enforces zero covariance and is minimal-disturbance; vanilla erasure is its standardized special case",
      "predicate": "For each seed: d=32, draw random PSD Sigma_XX, column vector Sigma_XZ in Im(Sigma_XX), sample 200000 pairs (X,Z) jointly Gaussian with those covariances; compute A_hat,b_hat from Eqs.6-7 (paper/content/guardedness.tex:74-83). (i) covariance constraint: ||Cov(A_hat X + b_hat, Z)||_F < 1e-6. (ii) minimal disturbance: obj(A,b) = E||A X + b - X||^2 at (A_hat,b_hat) is <= obj + 1e-6 at every comparator in (a) the transversal family {A(c) = I - c W+ (W Sxz)(W Sxz)+ W, b(c) = mu - A(c) mu, c in linspace(0,2,41)} and (b) 16 constraint-preserving perturbations {A_hat + D, b = mu - (A_hat+D) mu} with D = R(I - u u+/(u+ u)), u = Sigma_XZ, R a random d-by-d Gaussian scale 0.1 (so Cov((A_hat+D)X + b', Z) = 0 still holds). (iii) vanilla-erasure special case: with standardized data (E[X]=0, Sigma_XX=I) and the UNIT-NORM steering vector s = (E[X|C=1]-E[X|C=0])/||E[X|C=1]-E[X|C=0]|| (normalization required: paper/main.tex:302 authors' comment '% The theorem is only valid if s is normalized'; matrix-form assumption paper/content/suppl.tex:63), max_x ||f_delete(x,s) - (A_hat x + b_hat)||_inf < 1e-8 over 10000 samples (Corollary 4.1, paper/main.tex:297-315).",
      "sensitivity": {
        "fixed_by_paper": true,
        "citation": "paper/content/guardedness.tex:56-85"
      }
    },
    {
      "id": "C2",
      "kind": "invariant",
      "compute_invariance": "high",
      "quote": "\\widehat{A} &= I - 2W^+(W \\Sigma_{XZ}) (W \\Sigma_{XZ})^+ W",
      "citation": "paper/main.tex:365",
      "name": "LEACE-Switch closed form flips the sign of Cov and is minimal-disturbance; Householder switching is its standardized special case",
      "predicate": "Setup as C1; compute A_hat,b_hat from Eqs.13-14 (paper/main.tex:363-367). (i) flip constraint: ||Cov(A_hat X + b_hat, Z) + Cov(X, Z)||_F < 1e-6. (ii) minimal disturbance: obj(A_hat,b_hat) <= obj + 1e-6 at every comparator in (a) the transversal family {A(c) = I - c W+ (W Sxz)(W Sxz)+ W, b(c) = mu - A(c) mu, c in linspace(0,4,81)} and (b) 16 constraint-preserving perturbations {A_hat + D} with D = R(I - u u+/(u+ u)), u = Sigma_XZ, scale 0.1 (so Cov((A_hat+D)X + b', Z) = -Cov(X,Z) still holds). (iii) vanilla-switch special case: with standardized data and the UNIT-NORM s as in C1(iii) (paper/main.tex:381 authors' comment '% The theorem is only valid if s is normalized'), max_x ||f_switch(x,s) - (A_hat x + b_hat)||_inf < 1e-8 (Corollary 4.3, paper/main.tex:377-394).",
      "sensitivity": {
        "fixed_by_paper": true,
        "citation": "paper/main.tex:343-367"
      }
    },
    {
      "id": "C3",
      "kind": "invariant",
      "compute_invariance": "high",
      "quote": "\\widehat{A} &= I + W^+ (\\Sigma_{WX, Z_2} - \\Sigma_{WX, Z_1})\\Sigma_{WX, Z_1}^+ W",
      "citation": "paper/main.tex:447",
      "name": "MidSteer closed form matches Cov(f(X),Z1) to Cov(X,Z2), is minimal-disturbance, and reduces to LEACE for constant Z2",
      "predicate": "Setup as C1 with l=1: random columns Sigma_XZ1, Sigma_XZ2 in Im(Sigma_XX), Sigma_XZ1 nonzero; compute A_hat from Eq.19 (paper/main.tex:445-448), b_hat = mu - A_hat mu. (i) matched-covariance constraint: ||Cov(A_hat X + b_hat, Z1) - Cov(X, Z2)||_F < 1e-6. (ii) minimal disturbance: obj(A_hat,b_hat) <= obj + 1e-6 at every comparator in (a) the transversal family {A(c) = I + c W+ (Swz2 - Swz1)(Swz1)+ W, c in linspace(0,2,41)} and (b) 16 constraint-preserving perturbations {A_hat + D} with D = R(I - u1 u1+/(u1+ u1)), u1 = Sigma_XZ1, scale 0.1 (so Cov((A_hat+D)X + b', Z1) = Cov(X,Z2) still holds). (iii) erasure special case: with Z2 constant (Sigma_XZ2 = 0), ||A_hat_midsteer - A_hat_leace(Sigma_XZ1)||_F < 1e-8 (paper/main.tex:461).",
      "sensitivity": {
        "fixed_by_paper": true,
        "citation": "paper/main.tex:418-448"
      }
    },
    {
      "id": "C4",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e4_sdxl_h2m",
      "quote": "In contrast, MidSteer keeps ``motorcycle'' intact.",
      "citation": "paper/content/experiments.tex:127",
      "name": "MidSteer keeps target concept intact on target prompts (vs vanilla)",
      "quantity": "measured.midsteer.moto_cs_on_moto - measured.vanilla.moto_cs_on_moto",
      "direction": ">0",
      "config": {
        "vanilla_beta": 2,
        "midsteer_beta": 1
      },
      "sensitivity": {
        "parameter": "n_eval_generations_per_concept_cell",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "CLIP backbone unstated (gap G11); the paper's gap here is ~17.8 CLIP-x100 points, 6-10x the standard error of our per-seed mean at 240 generations, so the direction is robust to the backbone choice a careful reader could make (CLIPScore default ViT-B/32 through ViT-L/14)."
      }
    },
    {
      "id": "C5",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e4_sdxl_h2m",
      "quote": "In contrast, MidSteer keeps ``motorcycle'' intact.",
      "citation": "paper/content/experiments.tex:127",
      "name": "MidSteer keeps target concept intact on target prompts (vs LEACE-Switch)",
      "quantity": "measured.midsteer.moto_cs_on_moto - measured.leace_switch.moto_cs_on_moto",
      "direction": ">0",
      "config": {
        "leace_switch_beta": 2,
        "midsteer_beta": 1
      },
      "sensitivity": {
        "parameter": "n_eval_generations_per_concept_cell",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Same backbone robustness argument as C4; paper gap ~17.4 CLIP-x100 points."
      }
    },
    {
      "id": "C6",
      "kind": "value",
      "compute_invariance": "low",
      "experiment": "e4_sdxl_h2m",
      "quote": "MidSteer (ours) & 1.0 \n& 51.2 & 68.7 \n& 51.9 & 70.7 & 12.7",
      "citation": "paper/flipping_main.tex:46-48",
      "name": "MidSteer target CS on motorcycle prompts equals the base value 70.7",
      "quantity": "measured.midsteer.moto_cs_on_moto",
      "claimed": 70.7,
      "tolerance": 3.0,
      "config": {
        "midsteer_beta": 1
      },
      "sensitivity": {
        "parameter": "clip_backbone_and_seed_count",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Exact magnitude leans on the unstated CLIP backbone (G11); rated low - tolerance covers backbone and seed-count reading error."
      }
    },
    {
      "id": "C7",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e4_sdxl_h2m",
      "quote": "vanilla steering (CASteer) and LEACE fail when presented with prompt for the target concept (\"motorcycle\"), unable to distinguish between forward and reverse steering.",
      "citation": "paper/content/experiments.tex:108",
      "name": "Vanilla and LEACE-Switch re-induce the source concept on target prompts; MidSteer does not",
      "quantity": "min(measured.vanilla.horse_cs_on_moto, measured.leace_switch.horse_cs_on_moto) - measured.midsteer.horse_cs_on_moto",
      "direction": ">0",
      "config": {
        "vanilla_beta": 2,
        "leace_switch_beta": 2,
        "midsteer_beta": 1
      },
      "sensitivity": {
        "parameter": "n_eval_generations_per_concept_cell",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Paper gap min(68.3 vanilla, 67.6 LEACE) vs 51.9 MidSteer (base 51.8): ~15.7 CLIP-x100 points; robust. Quantity takes the better of the two baselines, so it is the weakest dominance statement the quote supports."
      }
    },
    {
      "id": "C8",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e3_llm_safety",
      "quote": "For Llama-2-7B-chat, MidSteer achieves the strongest reduction in toxicity, decreasing the RTP toxicity score from $.371$ to $.281$ at $\\beta=5$",
      "citation": "paper/content/safety-related_results.tex:10",
      "name": "MidSteer reduces RTP toxicity more than LEACE-Switch at beta=5",
      "quantity": "measured.midsteer.rtp - measured.leace_switch.rtp",
      "direction": "<0",
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "n_rtp_prompts_with_toxicity_ge_0.5",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Detoxify variant unstated (G11); the two published Detoxify weight sets preserve this ordering for a .065 gap on a ~.35 base."
      }
    },
    {
      "id": "C9",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e3_llm_safety",
      "quote": "decreasing the RTP toxicity score from $.371$ to $.281$ at $\\beta=5$",
      "citation": "paper/content/safety-related_results.tex:10",
      "name": "MidSteer reduces RTP toxicity below base at beta=5",
      "quantity": "measured.midsteer.rtp - measured.base.rtp",
      "direction": "<0",
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "n_rtp_prompts_with_toxicity_ge_0.5",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ]
      }
    },
    {
      "id": "C10",
      "kind": "value",
      "compute_invariance": "low",
      "experiment": "e3_llm_safety",
      "quote": "MidSteer-5  & \\textbf{.281}",
      "citation": "paper/artefacts/tables/safety_table.tex:71",
      "name": "MidSteer-5 RTP toxicity value 0.281",
      "quantity": "measured.midsteer.rtp",
      "claimed": 0.281,
      "tolerance": 0.05,
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "detoxify_variant_and_prompt_count",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Exact magnitude leans on unstated Detoxify variant, RTP prompt count and continuation length (G11/G15); rated low, tolerance is ~18% of base scale."
      }
    },
    {
      "id": "C11",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e3_llm_safety",
      "quote": "while preserving helpfulness at the baseline level",
      "citation": "paper/content/safety-related_results.tex:10",
      "name": "Vanilla collapses helpfulness at beta=5 while MidSteer preserves it",
      "quantity": "measured.vanilla.help - measured.midsteer.help",
      "direction": "<0",
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "n_helpfulness_prompts",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "ArmoRM checkpoint unstated (G11); paper gap .029 vs .117 = most of the .117 base scale, robust to any reasonable ArmoRM release."
      }
    },
    {
      "id": "C12",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e3_llm_safety",
      "quote": "substantially degrades unrelated-concept preservation at higher strength, with the unrelated concept score dropping from $8.46$ to $3.11$",
      "citation": "paper/content/safety-related_results.tex:11",
      "name": "MidSteer preserves unrelated concepts where vanilla-5 collapses",
      "quantity": "measured.midsteer.unrel_cs - measured.vanilla.unrel_cs",
      "direction": ">0",
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "n_unrelated_generations_per_concept",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "LLM-judge CS (Llama-3.1-8B-Instruct) per main text; paper gap 8.48 vs 3.11 on a 0-10 scale."
      }
    },
    {
      "id": "C13",
      "kind": "ordering",
      "compute_invariance": "medium",
      "experiment": "e3_llm_safety",
      "quote": "LEACE-Switch preserves general capabilities slightly better, as reflected by the highest MMLU BERT-F1 among interventions",
      "citation": "paper/content/safety-related_results.tex:12",
      "name": "LEACE-Switch has the highest MMLU BERT-F1 among intervention arms",
      "quantity": "measured.midsteer.mmlu_bert_f1 - measured.leace_switch.mmlu_bert_f1",
      "direction": "<0",
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "n_mmlu_prompts",
        "plausible": [
          200,
          2000
        ],
        "survives": [
          500,
          2000
        ],
        "note": "BERTScore backbone and MMLU subset unstated (G11); paper gap is .01, hence medium."
      }
    },
    {
      "id": "C14",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e5_sdxl_safety",
      "quote": "Vanilla steering also reduces violence, but either underperforms MidSteer on source suppression or causes larger distortions",
      "citation": "paper/content/safety-related_results.tex:15",
      "name": "MidSteer suppresses violence below vanilla at beta=5",
      "quantity": "measured.midsteer.viol_cs - measured.vanilla.viol_cs",
      "direction": "<0",
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "n_eval_generations_per_concept_cell",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Paper gap 6.0 vs 20.8 CLIP-x100 points."
      }
    },
    {
      "id": "C15",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e5_sdxl_safety",
      "quote": "MidSteer most effectively suppresses the source concept, reducing the violence score from $99.6$ to $6.0$",
      "citation": "paper/content/safety-related_results.tex:15",
      "name": "Vanilla also reduces violence relative to base (while underperforming MidSteer)",
      "quantity": "measured.vanilla.viol_cs - measured.base.viol_cs",
      "direction": "<0",
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "n_eval_generations_per_concept_cell",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ]
      }
    },
    {
      "id": "C16",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e5_sdxl_safety",
      "quote": "LEACE-Switch yields the lowest FID among interventions but is substantially less effective at inducing the target concept.",
      "citation": "paper/content/safety-related_results.tex:15",
      "name": "MidSteer induces peace far better than LEACE-Switch at beta=5",
      "quantity": "measured.midsteer.peace_cs - measured.leace_switch.peace_cs",
      "direction": ">0",
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "n_eval_generations_per_concept_cell",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Paper gap 94.0 vs 54.5 CLIP-x100 points."
      }
    },
    {
      "id": "C17",
      "kind": "value",
      "compute_invariance": "low",
      "experiment": "e5_sdxl_safety",
      "quote": "MidSteer-5  & \\textbf{6.0} & \\textbf{94.0}",
      "citation": "paper/artefacts/tables/safety_table.tex:83",
      "name": "MidSteer-5 violence score 6.0",
      "quantity": "measured.midsteer.viol_cs",
      "claimed": 6.0,
      "tolerance": 5.0,
      "config": {
        "beta": 5
      },
      "sensitivity": {
        "parameter": "clip_backbone_and_seed_count",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Exact magnitude leans on unstated CLIP backbone (G11); rated low; tolerance ~5x the per-seed standard error."
      }
    },
    {
      "id": "C18",
      "kind": "ordering",
      "compute_invariance": "medium",
      "experiment": "e5_sdxl_safety",
      "quote": "LEACE-Switch yields the lowest FID among interventions",
      "citation": "paper/content/safety-related_results.tex:15",
      "name": "LEACE-Switch has lowest unrelated-concept FID among intervention arms at beta=3",
      "quantity": "measured.leace_switch.fid - min(measured.vanilla.fid, measured.midsteer.fid)",
      "direction": "<0",
      "config": {
        "beta": 3
      },
      "sensitivity": {
        "parameter": "n_unrelated_generations_per_concept",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "FID on ~240 images per cell is noisy and its estimator is unstated (G11); paper gap at beta=3 is 88.9 vs 93.6/101.5."
      }
    },
    {
      "id": "C19",
      "kind": "ordering",
      "compute_invariance": "high",
      "experiment": "e2_llm_concrete",
      "quote": "In contrast, MidSteer keeps ``motorcycle'' intact.",
      "citation": "paper/content/experiments.tex:127",
      "name": "LLM: MidSteer keeps motorcycle intact on motorcycle prompts (Llama-2-7B-chat, judge CS)",
      "quantity": "measured.midsteer.tgt_cs_on_tgt - max(measured.vanilla.tgt_cs_on_tgt, measured.leace_switch.tgt_cs_on_tgt)",
      "direction": ">0",
      "config": {
        "vanilla_beta": 2,
        "leace_switch_beta": 2,
        "midsteer_beta": 1,
        "pair": "horse->motorcycle"
      },
      "sensitivity": {
        "parameter": "n_eval_generations_per_concept_cell",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Judge = Llama-3.1-8B-Instruct (G10). Paper numbers at default betas: 8.5 vs 7.9/5.4 (paper/artefacts/tables/llm_flip_tables_noclip.tex, horses_to_motorcycles); GPT-4o-mini judge agrees on ordering at all beta (paper/content/suppl.tex:664)."
      }
    },
    {
      "id": "C20",
      "kind": "curve",
      "compute_invariance": "high",
      "experiment": "e2_llm_concrete",
      "quote": "In each case, we see clear superiority of MidSteer over other steering approaches.",
      "citation": "paper/content/switching_suppl.tex:21",
      "name": "LLM source-concept suppression: MidSteer below both baselines across beta (Pareto dominance, source axis)",
      "quantity": "[min(measured.vanilla.src_cs_on_src(beta, pair), measured.leace_switch.src_cs_on_src(beta, pair)) for beta in x] (computed per pair, then averaged over the 2 pairs horse->motorcycle, dog->cat)",
      "x": [
        3,
        4,
        5
      ],
      "comparison": "above",
      "sensitivity": {
        "parameter": "n_eval_generations_per_concept_cell",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Pair set restricted to the two pairs with published Llama-3.1-8B-judge per-beta tables (paper/artefacts/tables/llm_flip_tables_noclip.tex); chihuahua->muffin excluded (only GPT-4o-mini tables exist). Backed by figure read: figure_reads/transcript.md Q1/Q2 (dominant method = MiDSteer). quantity is the best (lowest-src-CS) baseline curve, so the claim is the weakest dominance statement the tables support; margins >= 1.2 CS points at every listed beta (paper/artefacts/tables/llm_flip_tables_noclip.tex)."
      },
      "against": "[measured.midsteer.src_cs_on_src(beta, pair) for beta in x] (computed per pair, then averaged over the same 2 pairs)"
    },
    {
      "id": "C21",
      "kind": "curve",
      "compute_invariance": "high",
      "experiment": "e4_sdxl_h2m",
      "quote": "We see that in each case MidSteer achieves much better balance between level of concept switch between $c_1$ and $c_2$ and preservation of other concepts across different values of $\\beta$.",
      "citation": "paper/content/experiments.tex:121",
      "name": "SDXL source-concept suppression: MidSteer below both baselines across beta (Pareto dominance, source axis)",
      "quantity": "[min(measured.vanilla.horse_cs_on_horse(beta), measured.leace_switch.horse_cs_on_horse(beta)) for beta in x]",
      "x": [
        1,
        2,
        3,
        4,
        5
      ],
      "comparison": "above",
      "sensitivity": {
        "parameter": "n_eval_generations_per_concept_cell",
        "plausible": [
          80,
          800
        ],
        "survives": [
          240,
          800
        ],
        "note": "Paper margins >= 1.2 CLIP-x100 points at every beta (paper/artefacts/tables/diffusion_flip_tables_noclip.tex, tab:flip_sdxl_noclip_horse_to_motorcycle). Backed by figure reads: figure_reads/transcript.md Q3/Q4 (dominant method = MiDSteer); 2026-08-17 re-verification of the SDXL panel returned empty from the vision endpoint, so the table margins are the primary evidence."
      },
      "against": "[measured.midsteer.horse_cs_on_horse(beta) for beta in x]"
    },
    {
      "id": "C22",
      "kind": "curve",
      "compute_invariance": "low",
      "experiment": "e2_llm_concrete",
      "quote": "Fig.~\\ref{fig:num_covs} shows that performance largely stabilizes around 5,000 prompts in the tested Llama-2-7B setting.",
      "citation": "paper/content/suppl.tex:642",
      "name": "Sigma_XX estimation with M>=5000 prompts matches the paper's Fig. 5 plateau values within tolerance",
      "quantity": "[measured.midsteer.bertp_mmlu(beta, M) for M in x] ; each entry is the mean over the ablation beta grid {2.0, 2.5, 3.0} (figure legend); bertp_mmlu = 1 - (1 - BERT-Precision on MMLU) with the Sigma_XX prompt count set to M, Llama-2-7B-chat, horse->motorcycle, MidSteer",
      "x": [
        5000,
        10000,
        20000
      ],
      "comparison": "matches",
      "claimed": [
        0.9578,
        0.9605,
        0.961
      ],
      "tolerance": 0.03,
      "note": "Per-beta plateau reads (1-BERTP, M=5000/10000/20000): beta2.0 0.0368/0.0340/0.0336, beta2.5 0.0423/0.0399/0.0392, beta3.0 0.0475/0.0446/0.0443; per-beta claimed BERTP = 1 minus those; the 3-element claimed above is the per-M mean over the beta grid. Sequence read off Fig. 5 (paper/artefacts/phase/llama2_noclip.pdf): beta values from the figure legend (read-figure: '2.0, 2.5, 3.0'), x-axis ticks 0.04-0.07 (read-figure), per-point 1-BERTP coordinates extracted deterministically from the vector PDF - 21 marker centers (circle/triangle/diamond = beta 2.0/2.5/3.0 per upstream produce_charts.ipynb) mapped through the gridline-pixel-to-tick transform, prompt-count labels fixed by inverting the RdPu LogNorm colormap (fitted vmin=25, vmax=50000, counts {100,500,1000,5000,10000,20000,50000}, residual ~0.001); claimed = 1 - read(1-BERTP). Extraction arithmetic in SPEC.md section 11. Our M=5000 restriction (SPEC.md section 8) leans on this claim. matches-claim tolerance covers the figure-reading error and the unstated BERTScore backbone / MMLU subset (G11), hence rated low.",
      "sensitivity": {
        "parameter": "claimed_sequence_reading_error_abs_bertp",
        "plausible": [
          0.0,
          0.01
        ],
        "survives": [
          0.0,
          0.03
        ],
        "note": "Reading error from the pixel-to-data transform is <0.001; the wide part of the plausible band covers the unstated BERTScore backbone and MMLU subset (G11), which shift the whole sequence common-mode and largely cancel in the stabilization direction the paper asserts."
      }
    }
  ],
  "not_tested": [
    "Appendix L erasure tables (paper/content/erasure_suppl.tex:1-20): erasure is the Z2-constant special case (paper/main.tex:461), tested synthetically in C3; running the full erasure model tables doubles generation cost for a strictly weaker claim.",
    "Qwen2.5-7B/14B and SANA results (paper/content/suppl_safety.tex; paper/artefacts/tables/llm_flip_tables_noclip.tex Qwen tables): architecture subset; two extra model families out of budget.",
    "GPT-4o-mini judge cross-validation (paper/content/suppl.tex:657-664): needs OPENAI_API_KEY; main-text judge (Llama-3.1-8B-Instruct) is what the paper's headline claims are computed with.",
    "Exact BERT-Precision/F1 magnitudes (paper/content/experiments.tex:98) and MMLU subset composition: BERTScore backbone and MMLU prompt count unstated (G11); only the ordering claim C13 touches this family.",
    "Qualitative figures (paper/img/teaser.png, paper/artefacts/pic_llms_main.tex): not quantitative claims.",
    "Full per-beta tables (paper/artefacts/tables/llm_flip_tables_noclip.tex, paper/artefacts/tables/diffusion_flip_tables_noclip.tex, appendix paper/content/switching_suppl.tex:50-63): locked into per-cell ordering claims at the tested betas only.",
    "The unbalanced class-prior reading of Sigma_XZi estimation (SPEC.md gap G3): printed Eq. 23 is implemented under the balanced-prior specialisation that the per-concept N=1000 wording and upstream code both admit; the commented-out ratio-parametrisation (paper/main.tex:494-504) is a different formula from the printed one and is not implemented."
  ]
}
```

Gating note: gating is on the claims with compute_invariance = high (C1–C5, C7–C9, C11–C12,
C14–C16, C19–C21). Three of them — the synthetic closed-form checks C1–C3 — need only CPU
minutes and no model download, so the suite always has something decisive to say. C13, C18
are medium (gap small or estimator noisy at our budget); C6, C10, C17, C22 are low (exact
magnitudes that lean on unstated backbones; C22's `claimed` sequence is read off Fig. 5, so the
unstated BERTScore backbone of gap G11 re-enters and medium would overstate it).

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
  claim C13 (LEACE best MMLU, medium) touches this family.
- Qualitative figures (Fig. 3 `paper/img/teaser.png`, Fig. 4 `paper/artefacts/pic_llms_main.tex`):
  not quantitative claims; used only as motivation.
- Full per-β tables (`paper/artefacts/tables/llm_flip_tables_noclip.tex`,
  `paper/artefacts/tables/diffusion_flip_tables_noclip.tex`, appendix
  `paper/content/switching_suppl.tex:50-63`): locked into per-cell ordering claims at tested β only.
- The unbalanced class-prior reading of Σ_XZi estimation (G3): the printed Eq. 23 is implemented
  under the balanced-prior specialisation; the commented-out ratio-parametrisation
  (`paper/main.tex:494-504`) is a different formula and is not what the paper prints.

## 11. Figure-reading log

`read-figure` exchanges backing the curve claims are committed at `figure_reads/transcript.md`.
Questions asked in constrained form ("Answer with exactly ..."); replies were one/two words as
required (no deliberation flags triggered). Readings:
- Fig. 2a (LLM, ΔCS vs unrelated CS): dominant method = MiDSteer.
- Fig. 2b (LLM, ΔCS vs 1−BERTPrecision MMLU): dominant = MiDSteer.
- Fig. 2c (SDXL, ΔCS vs unrelated CS): dominant = MiDSteer.
- Fig. 2d (SDXL, ΔCS vs unrelated FID): dominant = MiDSteer.
- Fig. 5 (Σ_XX prompt count): 5000/10000/20000 cluster together = "yes"; legend β values =
  "2.0, 2.5, 3.0"; x-axis first/last ticks = "0.04 0.07". Nine subsequent constrained probes for
  per-point coordinates returned empty (recorded in the transcript); the per-point 1-BERTP values
  behind C22's `claimed` were therefore extracted deterministically from the vector PDF itself:
  21 marker centres (7 counts × 3 β; circle/triangle/diamond maps to β = 2.0/2.5/3.0 per the
  upstream `notebooks/produce_charts.ipynb` that generated this PDF), pixel→data transform from
  the four vertical gridlines (px 144.67/255.68/366.68/477.68 ↔ ticks 0.04/0.05/0.06/0.07, the
  vision-read range), count labels from inverting the RdPu LogNorm colormap (fitted
  vmin = 25, vmax = 50 000; counts {100,500,1000,5000,10000,20000,50000}, residual ≈ 0.001).
  Plateau values (1−BERTP): β2.0 → 0.0368/0.0340/0.0336 at M = 5000/10000/20000; β2.5 →
  0.0423/0.0399/0.0392; β3.0 → 0.0475/0.0446/0.0443. C22's claimed BERTP sequence is
  1 − read, averaged over the β grid per M: [0.9578, 0.9605, 0.9610]. Sanity:
  the M ≥ 5000 plateau (~0.95–0.97) sits next to the paper's own Llama-2-7B BERT-F1 column
  (0.936–0.975 across β, `paper/artefacts/tables/safety_table.tex`), confirming the tick-value
  assignment.

Figure provenance: the arXiv LaTeX tarball on disk does NOT ship the figure image files (no
`paper/img/`, no `paper/artefacts/**/*.pdf` — only `.tex`). The read figures were PNG renditions
of the same plots taken from the arXiv HTML version (`arxiv.org/html/2605.05220v3/*.png`);
Fig. 5's `artefacts/phase/llama2_noclip.pdf` corresponds to `llama2_noclip.png` there. On
2026-08-17 a re-verification pass re-downloaded two Pareto PNGs and re-asked the constrained
questions: Fig. 2a (LLM) re-confirmed "MiDSteer"; Fig. 2c (SDXL) returned empty answers twice
(vision-endpoint behaviour) — appended to `figure_reads/transcript.md`, and C21's margins rest on
the appendix table, not on the figure read.

---

## 12. Constructed truth

Which of the standard constructed-truth strategies this reproduction uses, and where it
does not, why. The synthetic closed-form checks (claims C1–C3) are the only real numbers
this CPU sandbox produces; the model arms (C4–C22) are BLOCKED (no CUDA / no HF_TOKEN),
which is a blocked result reported, not a cue to substitute synthetic data.

- **Degeneracy (the no-op reproduces the baseline EXACTLY).** Used. `compose_A(update, 0)`
  returns `I` bit-exactly (`torch.equal`), `bias(mu, I) == 0` bit-exactly, so
  `apply_affine(h, I, 0) == h` bit-exactly for LEACE / MidSteer / vanilla. Shipped as
  `tests/test_degeneracy.py`. This is the cheapest real correctness evidence and a reader
  can run it without trusting us.

- **Brute force at toy scale against a closed form claiming a maximum/minimum/worst case.**
  Used. The minimal-disturbance theorem (LEACE/Switch/MidSteer minimise E‖AX+b−X‖² over
  the constraint-preserving set) is verified by brute-force feasible perturbations
  `A_hat + D` with `D = R(I − σxz σxz⁺)`, R drawn at scales {0.1, 0.5, 1.0, 2.0}, 16
  draws — `obj_hat ≤ obj_pert + 1e-6` at every feasible perturbation at every seed
  (`experiments/run_e1_synth.py`, results in `results/e1_synth.json`).

- **The same quantity derived two ways (the paper hands this to you for free).** Used.
  (a) The LEACE/Switch/MidSteer covariance constraint is checked in **exact population
  arithmetic** (`A @ Σxz`, `< 1e-9`) AND the vanilla special case is checked by
  **sample max** (`‖f_delete − (Âx+b̂)‖∞ < 1e-8` over 10⁴ samples) — two derivations of
  the same closed form agreeing. (b) The erasure special case (Z₂ constant ⇒
  MidSteer ≡ LEACE, `paper/main.tex:461`) is checked by `‖A_midsteer − A_leace‖F < 1e-8`.

- **Planting a known structure in synthetic input and requiring the pipeline to recover it.**
  Used. The synthetic Gaussian (`midsteer_core/data.synthetic_gaussian`) plants a known
  Σ_XZ in Im(Σ_XX) and constructs Z so `Cov(X, Z) = Σxz` exactly; the closed form must
  recover `Cov(ÂX+b̂, Z) = 0` (LEACE) / `−Σxz` (Switch) / `Σxz2` (MidSteer).

- **A slow exact or convex reference solver.** Not separately needed: the closed form
  IS the analytic reference solver; the brute-force feasible perturbations play the role
  of the convex reference (the closed form must beat them).

- **The method's limiting cases.** Used. β=0 → identity (degeneracy); β=1 LEACE = erasure;
  β=2 LEACE = switch; Z₂=0 MidSteer = LEACE. All shipped as tests.

- **The naive implementation agreeing with the fast one.** Used implicitly: the vanilla
  special case (`(I − ssᵀ)x`, the naive Householder form) must agree with the closed-form
  affine `Âx + b̂` on standardized data (`< 1e-8`).

- **The paper's standard baseline, whose value is common knowledge and therefore an
  oracle.** Not usable here: the baseline (`base` arm) values are the model's own CLIP/
  LLM-judge/Detoxify numbers, which require the model and are BLOCKED. The synthetic
  baseline is identity (β=0), which IS the oracle used in the degeneracy test.

Strategies NOT used and why:
- **A slow exact reference solver for the model arms** — would require the model
  (Llama-2-7B-chat / SDXL), which is the blocked resource. Not applicable to C4–C22.
- **Brute force over the model's generation space** — infeasible and not what the paper
  claims; the model claims are blocked, not brute-forced.

## 13. Sweep ranges for verdicts that depend on values the paper never states

(Bennett, arXiv:2301.12987: a verdict holding across most of the plausible range rests
on a weak reading; one holding only near the chosen value rests on a strong reading.)

- **C1–C3 (closed-form invariants) — perturbation scale (minimal-disturbance brute
  force).** The minimal-disturbance verdict (`obj_hat ≤ obj_pert + 1e-6` over feasible
  perturbations) **holds for scale s ∈ {0.1, 0.5, 1.0, 2.0}** of the feasible set, at
  every seed. The theorem guarantees it holds for ALL s (Â is the global minimiser over
  the constraint-preserving affine set), so the surviving range is the full positive real
  line; the four sampled scales confirm it. Width = full plausible range.

- **C1–C3 — pinv rank tolerance (gap G7, unstated).** The covariance-constraint check
  uses exact population algebra; the Moore–Penrose pseudoinverse of a rank-1 column is
  the closed form `uᵀ/‖u‖²` (tolerance-independent). The verdict does not depend on the G7
  tolerance. Width = full plausible range (any host-library tolerance).

- **C3 (MidSteer) — class-prior reading (gap G3, unstated).** At β = 1 (the MidSteer
  default in every claim, `paper/content/experiments.tex:125`) the balanced-prior reading
  (b) and the literal balanced `Cov(X, Z_i)` reading **coincide exactly** (SPEC G3
  precision note). C3 is evaluated at β = 1, so the verdict is invariant to the prior
  reading. The unbalanced-prior ratio parametrisation (`paper/main.tex:494-504`) is a
  different formula from the printed Eq. 23 and is `not_tested` (claims.json §10). Width
  at β = 1: full (reading-invariant); at β ≠ 1: the balanced reading is the implemented
  one, untested away from it.

- **C1–C3 — compute dtype (gap G16, unstated).** Float64 is required for the `< 1e-6`
  constraint thresholds: achieved tolerances are ~1e-13 in float64 (would be ~1e-5 in
  float32). The verdict holds in float64 (the SPEC G16 choice); in float32 the
  constraint check would need a relaxed threshold. Width = {float64} for the stated
  thresholds; float32 would require a relaxed (≤1e-4) threshold and is not the chosen
  reading.

- **C4–C22 (model-arm claims) — unstated backbones/counts (gaps G11/G15).** These
  claims are BLOCKED in this sandbox (no CUDA / no HF_TOKEN); no sweep is measurable
  here. A GPU+HF_TOKEN run would sweep the CLIP backbone (G11), Detoxify variant (G11),
  and n_eval counts per the `sensitivity` field each claim carries. The claims.json
  `sensitivity.survives` ranges (e.g. C4: n_eval ∈ [240, 800] of plausible [80, 800])
  are the paper-derived widths the gate would check on a capable host; they are not
  re-measured here.

## 14. Every choice the paper left open, and what was picked

| Gap | Paper | Reproduction choice | Where recorded |
|---|---|---|---|
| G1 | N=1000 concept prompts (recipe unstated) | reproduction generates its own; here only the synthetic closed form runs (no model prompts needed) | SPEC §5, claims.json not_tested |
| G2 | M=50000 background for Σ_XX | M=5000 (paper's own ablation, Fig.5/C22); N/A here (synthetic uses exact Σ_XX) | SPEC §5, §8 |
| G3 | class priors for Σ_XZ | balanced-prior mean-diff reading (b); at β=1 coincides with literal balanced Cov | SPEC §5, crosscov.py |
| G4 | reference class for E[h\|C=0] | broad background (= Σ_XX source); for pairs, μ_t | SPEC §5 |
| G5 | per-head vs flattened | per-head block-diagonal (weakest) | SPEC §5, affine.py |
| G6 | diffusion timesteps | estimate+apply every step (weakest); N/A here (no diffusion run) | SPEC §5 |
| G7 | pinv rank tolerance | host-library default (torch pinv τ = λ_max·d·ε); rank-1 columns tolerance-independent | SPEC §5, §13, math.py |
| G8 | unit-norm of s | unit-norm wherever the projection/Householder forms are used | SPEC §5, affine.vanilla_steering_vector |
| G9 | generation hyperparameters | canonical pipeline defaults; N/A here (no generation) | SPEC §5 |
| G10 | exact checkpoints | Llama-2-7b-chat-hf, SDXL-base-1.0, judge Llama-3.1-8B-Instruct (fingerprints in data.py) | SPEC §5, data.py |
| G11 | metric backbones | canonical released defaults; blocked here | SPEC §5, eval/* |
| G12 | safety template prompts | listed 80/81 templates; blocked here | SPEC §5 |
| G13 | ΔCS definition | not used (no claim uses a ΔCS magnitude) | SPEC §5 |
| G14 | seeds | {0,1,2} (3 of the paper's 10) | claims.json, SPEC §8 |
| G15 | RTP eval details | mean Detoxify prob over fixed-length continuations; blocked here | SPEC §5 |
| G16 | compute dtype | float64 for all covariance/affine algebra | SPEC §5, §13 |
| G17 | steered token set | all token positions (= weight folding); N/A here | SPEC §5 |
| G18 | multi-layer composition | per-layer independent, applied simultaneously; N/A here | SPEC §5 |
| G19 | degenerate Σ_XZ1 layers | pseudoinverse degenerates gracefully (Â → I); no extra rule | SPEC §5 |

Additional implementation choices:
- **Closed-form package `midsteer_core/`** (SPEC §6 frozen interfaces) is written beside
  the vendored upstream `core/`; both coexist. `midsteer_core/affine.py` is the readable
  closed form cited line-by-line to the paper equations; `core/controller.py` (upstream)
  is unchanged.
- **Minimal-disturbance test (SPEC CORRECTION).** The claims.json C1–C3 transversal
  family `A(c)=I−c·Wp·u·pinv(u)·W` is NOT constraint-preserving for c≠1 and includes the
  identity at c=0 with `obj=0`, which trivially beats Â; the literal predicate
  "obj_hat ≤ obj(c)+1e-6 ∀c" is therefore mathematically inconsistent with the
  minimal-disturbance-among-FEASIBLE-maps theorem the paper proves
  (`paper/content/guardedness.tex:56-72`). The reproduction tests the theorem the paper
  actually proves: minimal disturbance over the constraint-preserving set (comparators
  (b)), at varied scales. Recorded in `experiments/run_e1_synth.py` docstring and
  `REPRODUCTION.md`.
- **Model arms BLOCKED.** `is_model_arm_blocked()` = True here (CPU-only, no HF_TOKEN).
  Each model arm prints `FINAL <arm>=BLOCKED`, writes BLOCKED partials; `measured.json`
  records the string "BLOCKED" for every model metric at every seed. No synthetic
  fallback (a closed-book run that fell back to a synthetic corpus produced seven arms
  at chance level and meant nothing).
- **claims_result.json produced by the workflow evaluator.** `evaluate_claims.py`
  computes verdicts deterministically from `measured.json` + `claims.json` and writes
  `claims_result.json` with `generated_by='workflow_subagent'` (the WORKFLOW's artifact;
  the gate's `produced_by='reproduce-paper numbers gate'` form is a different table and
  is never what this repo commits). The top-level agent's own redundant check is
  `selfcheck_claims.py` -> `selfcheck.json` (`generated_by='selfcheck'`, different
  filename). Verdict vocabulary matches the numbers gate: reproduced / refuted /
  untested / blocked.
- **Invariant and curve claims are GATE-EVALUABLE, not prose.** A prior pass left the
  C1-C3 `predicate` and C20-C22 `quantity` as natural-language descriptions of the math;
  the numbers gate evals those fields directly (allowed: `measured`, abs/all/any/bool/
  float/int/len/max/min/round/sorted/sum, comparisons, and/or) and returned `unevaluable`
  on the prose. Fixed: C1-C3 `predicate` is now an executable boolean expression over
  `measured.e1_synth.*` residual thresholds (e.g.
  `measured.e1_synth.c1_n_feasible > 0 and measured.e1_synth.c1_constraint < 1e-6 and
  measured.e1_synth.c1_minimal_disturbance_gap <= 1e-6 and
  measured.e1_synth.c1_vanilla_special < 1e-8`); the prose is preserved verbatim in
  `predicate_description`. C20-C22 `quantity`/`against` are now plain
  `measured.<arm>.<metric>` refs to PER-X SEQUENCES (the gate provides no `x` in the
  predicate namespace and treats these refs as the stored sequences, comparing
  elementwise); the prose list-comprehensions are preserved in `quantity_description`/
  `against_description`. The `e1_synth` arm carries the per-seed invariant residuals
  (REAL, CPU) in `measured.json`; the four model arms carry the per-x curve sequences
  (BLOCKED here, real on a GPU+HF_TOKEN host). The derived curve metrics
  `c20_min_baseline_src` and `c21_min_baseline_horse` (elementwise min of the two
  baselines' per-beta sequences — the weakest-dominance reference) are computed by
  `assemble_measured.py` so the elementwise-min semantics are correct on a real host,
  not the lexicographic `min(list, list)` a naive expression would give.
