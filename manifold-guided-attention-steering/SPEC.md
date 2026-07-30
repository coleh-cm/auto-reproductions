# SPEC: Manifold-Guided Attention Steering (MAGS)

- **Paper:** *Manifold-Guided Attention Steering* — Li, Guruprasad, Sengupta, Satish, D'Antoni, Yu (UCSD), 2026
- **arXiv:** 2605.21770 (v1, 2026-05-20) · **paper_ref:** c8a82a40-c60b-43ce-863f-37fb55cb3e8e · **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Authoritative source for all maths/tables/numbers:** `paper/latex_src/neurips_2026.tex` (arXiv e-print LaTeX). PDF text extraction `paper/paper_pdf_extracted.txt` is prose-only and is NOT trusted for equations. All citations below are `paper/latex_src/neurips_2026.tex:<line-range>` and are grep-able.

## 0. Upstream code

- Paper contains **no** code link (grep for `github|code available|href` over the LaTeX source: no matches).
- arXiv abstract page for 2605.21770 lists no code link.
- GitHub searches (2026-07-29, via api.github.com): `"manifold guided attention steering"` → 0 repos; `"MAGS attention steering language model"` → 0; `"attention steering correctness manifold"` → 0; `mags steering` within org `Rose-STL-Lab` (68 repos) → 0; user `i6li` → 0.
- **Verdict: no upstream code exists. We implement from scratch.**

## 1. The method as an algorithm

There is **no training loss and no gradient update** — MAGS is a training-free inference-time intervention. The "fit" is descriptive statistics (means + SVD) over collected activations. The "update rule" is the in-place activation correction of Eq. (9).

### Phase A — Offline manifold construction (per model × benchmark)

**Inputs.**

- A corpus of paired traces: problems `P = {p_1..p_N}`, `S ≤ 8` sampled traces `T_i = {τ_i,1..τ_i,S}` per problem, binary label `y_τ ∈ {0,1}` (1 = correct final answer). Only problems with `|T_i^+| ≥ 1` **and** `|T_i^-| ≥ 1` are kept (tex:L398–399).
- A monitored layer set `Lmon` (Llama diagnostic used `{8,16,24,31}`; tex:L296) and, per layer, all `H` heads.
- Fresh forward pass capture of per-head outputs `A_τ^{(l,h)} ∈ R^{L_τ×d_h}` for every trace `τ` and every `(l,h)` in the monitored set (tex:L179–185).

**Steps.** For every monitored head `(l,h)`:

1. Per-problem per-class means over all token steps: `μ_{c,i}^{(l,h)}, μ_{e,i}^{(l,h)} ∈ R^{d_h}` — Eq. (2), tex:L192–199.
2. Contrastive difference `δ_i^{(l,h)} = μ_{e,i}^{(l,h)} − μ_{c,i}^{(l,h)} ∈ R^{d_h}` — Eq. (3), tex:L201–207.
3. Stack `D^{(l,h)} ∈ R^{N×d_h}` whose i-th row is `δ_i` (the paper prints the transpose `D^⊤ ∈ R^{d_h×N}`) — Eq. (4), tex:L220–228.
4. Compact SVD `D^{(l,h)} = U Σ V^⊤` with `U ∈ R^{N×r}`, `Σ ∈ R^{r×r}`, `V^⊤ ∈ R^{r×d_h}`, `r = min(N, d_h)`; take `B^{(l,h)} = (V_{:,1:k})^⊤ ∈ R^{k×d_h}` (rows orthonormal) — Eq. (5), tex:L229–243. **This is the only "learning": an SVD. No loss, no iterations.**
5. Global correct centroid `μ_c^{(l,h)} ∈ R^{d_h}` over all problems × correct traces × token steps — Eq. (6), tex:L245–254.
6. Threshold calibration: compute `d_t^{(l,h)}` (Eq. 7) over **every token step of every correct train trace**; set `τ^{(l,h)}` to the `q`-th percentile of that pooled set — Eq. (8), tex:L279–285. (See §4: `q` is never given a value.)
7. Head selection: on a held-out problem split, per head compute the trajectory-level score as the **mean** of `{d_t}` over the trajectory, and the AUROC between that score and the trace label `y_τ`; rank all monitored heads by AUROC and keep top-`K` (tex:L304–305). *Note the aggregation tension — see §4 item 6.*

**Artifacts:** `{B^{(l,h)}, μ_c^{(l,h)}, τ^{(l,h)}, auroc^{(l,h)}}` for the selected top-`K` heads (Algorithm 1 `Require`, tex:L352).

### Phase B — Inference with steering (Algorithm 1, tex:L348–374)

Per decode step `t` (after the normal forward pass computes attention for the step):

for each monitored head `(l,h)` in layer order:
  1. `v ← a_t^{(l,h)} − μ_c^{(l,h)}`                       (halo: R^{d_h})
  2. `d ← ‖ B^{(l,h)} v ‖²`                               (scalar, cost O(k·d_h))
  3. if `d > τ^{(l,h)}`: `a_t^{(l,h)} ← a_t^{(l,h)} − α · B^{(l,h)⊤} (B^{(l,h)} v)` **in place, before the o_proj `W_O^{(l)}` consumes the head output** (Eq. 9, tex:L307–317; α ∈ (0,1], tex:L317)

At `α=1` this is exactly `μ_c + P_⊥(a_t − μ_c)` with `P_⊥ = I − B^⊤B` — Eq. (10), tex:L319–330 — i.e. discard only the error-subspace component of the centred activation. The correction is applied to the concat-of-heads input of `W_O^{(l)}` via a pre-hook on that projection (tex:L356–357, L367–368).

Per-step overhead `O(K·k·d_h)` (tex:L376).

### MAGS^u (multi-objective union, molecular task)

For each objective `c_j` (validity; binding affinity), fit an independent manifold per §A and select its own top-`K` head set `H_j`. At inference, steer the **union** of head sets; each head is corrected only through its own objective's manifold (tex:L378–379). (Collision handling across objectives: §4 item 12.)

## 2. Symbols, with shapes

| Symbol | Shape | Meaning |
|---|---|---|
| `N` | scalar | # retained problems (papers' eval N: MATH-500 500, GSM8K 1319, HumanEval 164, MBPP 427 — tex:L710–713, L756–759; fit-N is the retained train-problem count) |
| `S` | scalar | traces sampled per problem, `S ≤ 8` (tex:L399) |
| `L_τ` | scalar | # tokens of trace τ considered (see §4 item 7 on prompt vs generated tokens) |
| `L`, `H` | scalars | model layers, heads/layer |
| `d_h` | scalar | head output dim (Llama-3.1-8B: 128 = 4096/32; Gemma-4-E4B-it: 256; GPT-OSS-20B: 64) |
| `a_t^{(l,h,τ)}` | `[d_h]` | head `h`'s attention output (A·V product) at token t, layer l, trace τ — **before** `W_O` (tex:L356–357) |
| `A_τ^{(l,h)}` | `[L_τ, d_h]` | per-trace stack of the above — Eq. (1), tex:L179–185 |
| `μ_{c,i}^{(l,h)}`, `μ_{e,i}^{(l,h)}` | `[d_h]` | per-class means weighted by token count — Eq. (2) |
| `δ_i^{(l,h)}` | `[d_h]` | contrastive difference — Eq. (3) |
| `D^{(l,h)}` | `[N, d_h]` (paper prints `D^⊤: [d_h, N]`) | difference matrix — Eq. (4) |
| `U, Σ, V^⊤` | `[N,r], [r,r], [r,d_h]`, `r=min(N,d_h)` | compact SVD of `D` (tex:L231) |
| `B^{(l,h)}` | `[k, d_h]` | error-subspace basis (top-k rows of V^⊤, orthonormal) — Eq. (5) |
| `μ_c^{(l,h)}` | `[d_h]` | global correct centroid — Eq. (6) |
| `d_t^{(l,h)}` | scalar | `‖B(a_t − μ_c)‖²` — Eq. (7) |
| `τ^{(l,h)}` | scalar | q-th percentile of correct-trace scores — Eq. (8). Symbol collision: same letter as trace τ elsewhere; context disambiguates |
| `k` | scalar | subspace rank (UNSTATED, §4 item 1) |
| `K` | scalar | # monitored/steered heads (§4 item 2) |
| `q` | scalar | threshold percentile (UNSTATED, §4 item 1) |
| `α` | scalar ∈ (0,1] | steering strength — Eq. (9) |
| `P_⊥^{(l,h)}` | `[d_h, d_h]` | `I − B^⊤B`, projector on complement — tex:L319 |
| `W_O^{(l)}` | `[d_model, H·d_h]` | attention output projection (concat over heads) |
| `x_{1:prompt}` | `[T_prompt]` int | prompt token ids (Algorithm 1 Require) |

## 3. Equations to implement (with citations)

All in `paper/latex_src/neurips_2026.tex`:

| # | Equation | Citation | Notes |
|---|---|---|---|
| (1) | `A_τ^{(l,h)} ∈ R^{L_τ×d_h}` | L179–185 | activation capture layout |
| (2) | per-class means `μ_{c,i}, μ_{e,i}` | L192–199 (`\label{eq:mu}`) | token-count weighted average over traces of one problem |
| (3) | `δ_i = μ_{e,i} − μ_{c,i}` | L201–207 (`\label{eq:diff_vec}`) | per problem |
| (4) | `D^⊤ = [δ_1 … δ_P] ∈ R^{d_h×N}` | L220–228 (`\label{eq:diff_matrix_transpose}`) | printed as transpose; we store `D: [N, d_h]` |
| (5) | `B = (V_{:,1:k})^⊤ ∈ R^{k×d_h}` from `D = UΣV^⊤` | L229–243 (`\label{eq:basis}`) | compact SVD; `torch.linalg.svd(D, full_matrices=False)` gives `Vh == V^⊤`; `B = Vh[:k]` |
| (6) | global `μ_c` | L245–254 (`\label{eq:global_mean}`) | token-count weighting across problems and traces |
| (7) | `d_t = ‖B(a_t − μ_c)‖²` | L266–275 (`\label{eq:proximity}`) | trigger score |
| (8) | fire iff `d_t > τ^{(l,h)}`; `τ` = q-th pctile over correct-trace token steps | L279–285 (`\label{eq:trigger}`) | compare against **pooled per-token** scores, not per-trace |
| (9) | `ã = a − α B^⊤ B (a − μ_c)` | L308–317 (`\label{eq:correction}`) | in place, before `W_O` |
| (10) | `ã = μ_c + P_⊥(a − μ_c)` | L319–330 (`\label{eq:correction_proj}`) | α=1 special case; implement (9), property-test against (10) |
| (11–13) | Proposition 1 + proof | L335–346, L585–600 | `<ã,v> = <a,v>` ∀v: Bv=0; use as a **unit invariant test**: correction must not alter any complement direction |
| Alg. 1 | full decode loop | L348–374 | monitor in layer order; hook = pre-hook on `W_O`; complete forward, then sample x_{t+1} |

Validation experiment (§3.4, tex:L294–298): fit on 70% of problems (problem-level split so traces of one problem never cross splits), evaluate per-head trajectory AUROC on the 30% held-out, trajectory score = **max** over `{d_t}` of the trace, threshold chosen on the train split by maximizing balanced accuracy. Figure 3 (tex:L287–292) marks heads with AUROC > 0.65. This experiment doubles as our **degeneracy-check harness**: if learned B's carry detectable signal (many heads with AUROC substantially > 0.5), the manifold fit is sane.

## 4. What the paper does NOT state (gaps → defaults we adopt)

These are exactly the places where we must **not** pretend to know the paper's choice. Defaults marked `→` are reproduction decisions, not paper facts.

1. **`k` (subspace rank) — never given.** Only hint: the visualisation projects onto the "top-4 principal components of the contrastive error subspace" (tex:L564). `→ k=4` for main runs; sweep `{2,4,8,16}` in sensitivity.
2. **`q` (threshold percentile, Eq. 8) — never given.** `→ q=95` (percentile over pooled correct-trace token scores); sweep `{90,95,99}`.
3. **`K` (monitored/steered heads) for the main tables — never given.** Ablation (tex:L605, L613–628) covers only top-1/top-3 on Llama/MATH-500; both reach 0.530 at α=1.0. `→ K=3`.
4. **`α` for main-table MAGS rows other than MATH-500/Llama — never given.** Ablation's best is α=1.0. `→ α=1.0` everywhere; MATH-500 ablation curve reproduced for the quarterly check.
5. **Monitored layer set for steering** — the diagnostic uses layers `{8,16,24,31}` for Llama-32l (tex:L296); whether steering restricts to those layers or searches all `L×H` heads at all layers is unstated. For Gemma-4-E4B-it (42 layers) / GPT-OSS-20B (24 layers) no monitored set is given anywhere. `→ Llama: {8,16,24,31}; Gemma: {10,21,31,41} (evenly spaced); GPT-OSS: {6,12,18,23}`; top-K selection ranked within all heads of the monitored layers.
6. **Aggregation inconsistency**: §3.4 uses **max** over steps for the trajectory AUROC (tex:L298); §4 head selection uses **mean** proximity over the trajectory (tex:L305). We implement both; max for the AUROC diagnostic, mean for production head selection (as §4 literally says).
7. **Which token positions count as the trace** (`t=1..L_τ`): prompt tokens included or generated-only is unstated. The shared prompt would cancel in `δ` per problem but would shift `μ_c` (Eq. 6). `→ generated tokens only` for both means and μ_c.
  - **Round-20 correction (verified finding-1, recorded honestly):** "generated tokens only" is enforced at capture time by recording **decode steps only (seq==1)** in `_CaptureHook` (`mags/capture.py`). `model.generate(use_cache=True)` issues 1 prefill (seq>1) + (N-1) decodes (seq==1) for N generated tokens; the prefill forward's last position is the activation at the LAST PROMPT-token position (it attends only to prompt tokens) and is a prompt-context activation that SPEC §4.7 explicitly excludes. The prior code captured the prefill-last as row 0 of every trace's activation stack, so it entered the per-class means (Eq.2), the global centroid `μ_c` (Eq.6) and the threshold pool (Eq.8). It also created a fit/inference inconsistency: the inference controller (`mags/steering.py:45`) skips prefill (`if seq != 1: return None`), so the threshold was calibrated on a score distribution containing the prefill-last's score while inference never emits that score. Decode-only capture removes both: `T = N-1` rows (one per decode step), aligned to `gen_ids[:-1]` (the last generated token has no decode forward). `capture_trace` returns the aligned gen-id prefix so the stored `token_ids` length matches `A`'s `T`. This makes fit and inference score the identical set of positions and keeps `μ_c` free of the prompt-position term.
8. **Held-out split for head selection**: fraction and relation to the 70/30 diagnostic split unstated. `→ 70/15/15 problem-level split: manifold-train / head-select(held-out) / report-only AUROC test`.
9. **Evaluation decoding config** — greedy vs sampling, temperature, top-p, max_new_tokens, #samples per problem: **all unstated** (bootstrap N equals the full test set size, and HumanEval is pass@1). `→ greedy (do_sample=False), max_new_tokens=1024 (math), 512 (code), one completion per problem`.
10. **Trace-collection sampling config** (temperature etc. of the ≤8 samples) — unstated. `→ temperature=1.0, top_p=0.95, n=8, stop at EOS`; keep exactly the paper's keep-if-both-ripen rule.
11. **One pair vs all traces**: text says "one correct solution and one incorrect" (tex:L399) but Eq. (2) sums over *all* traces in `T_i^±`. `→ implement Eq. (2) as written (all retained traces)`; a one-pair variant is a review-time variant, not the default.
12. **MAGS^u collision handling**: "naturally yields disjoint head sets" (tex:L379) — behaviour when two objectives select the same head is unstated. `→ each physical head is steered once; if selected by two objectives, the objective with the higher held-out AUROC owns it`.
13. **Prompt templates / answer extraction** — unstated. `→ model's HuggingFace chat template; MATH/GSM8K scored by `math_verify`-style boxed/number extraction; GSM8K strip `####`; code scored by executing canonical tests (HumanEval: official harness; MBPP: sanitized split test asserts; N=427 matches MBPP-sanitized test)`. 
  - **Round-20 correction (verified finding-2, recorded honestly):** the chat template applies to the natural-language-instruction benchmarks **MATH-500, GSM8K, MBPP** only. **HumanEval is EXCLUDED**: its prompt is a raw function signature and the canonical HumanEval protocol is completion-style (no chat priming), so the instruct model is run as a raw-continuation LM there. Implemented as `CHAT_TEMPLATE_BENCHMARKS = {"MATH-500","GSM8K","MBPP"}` in `mags/generation.py`; the flag is passed to BOTH eval (`generate`/`cd_generate`) and the manifold fit (`capture_trace`), because the contrastive error subspace must be fit on traces in the SAME prompt format as eval (an APPS-as-completion fit for the HumanEval manifold matches HumanEval's completion-style eval; an APPS-as-chat fit for the MBPP manifold matches MBPP's chat eval). Base models without a chat template (distilgpt2 smoke) fall back to raw tokenization (`_tokenize_prompt`). The prior code tokenized every prompt raw with no chat template anywhere, violating this decision and silently deflating the three natural-language benchmarks on instruct models.
14. **Perplexity protocol** — PPL of what, under which model: unstated. `→ perplexity of the generated completion tokens under the *unsteered* base model, averaged over problems` (secondary metric, not gated).
15. **ITI adaptation mechanics** beyond "correct/incorrect traces as contrast sets" (tex:L394) — probe type, intervened activations: unstated. `→ per-head logistic-probe on the trace-mean head output, top-K heads by held-out probe accuracy (K ∈ {24,48,96}); intervention `a += α·σ_h·v_h` every decode step (original ITI convention, Li et al. 2023); α ∈ {0.5,1,5}`.
16. **Angular Steering adaptation mechanics** — "fixed 2D rotation in the mean-difference span across all layers" (tex:L394): extraction points, raw-vs-normalised activations, offset-vs-target rotation: unstated. `→ Vu & Nguyen (arXiv:2510.26243) FIXED-OFFSET rotation (their Eq. 1, "Rotation by an Offset Angle": every activation rotated by the SAME constant angle θ; θ=0 is the identity), with plane Span(d_feat, d_PC0) where d_feat = mean_incorrect − mean_correct (the contrastive direction, tex:L394) and d_PC0 = PC1 of the per-(l,h) candidate difference-in-means directions (Vu & Nguyen §4.5, NOT PC1 of pooled raw activations); correct-vs-incorrect traces as the two contrast sets; rotate all layers; sweep the offset angle 0°..330° step 30° (tex:L654–665); default 30° (best per ablation)`. **Why fixed-offset and NOT the target-angle form (Eq. 2):** the ablation Table 4(c) signature is diagnostic — 0° ≈ unsteered (0.488 vs 0.478), 360°-periodic, worst at 90–120° (0.206). A target-angle form at θ=0 would force EVERY activation onto d_feat (the *incorrect* direction, d_feat = μ_e − μ_c) and catastrophically crash accuracy, not yield ≈unsteered; only the constant-offset form is identity at θ=0 and matches the signature. (The prior round's "target-angle" decision here was wrong on this point; corrected round-18, see REPRODUCTION.md.) AS reference code exists at github.com/lone17/angular-steering (cited by that paper, usable as reference for plane construction).
  - **IMPLEMENTATION DECISION (deviation, recorded honestly):** the reproduction's hook infrastructure is the per-head attention output **before `W_O`** (the only interception point the MAGS `HookRegistry` exposes; SPEC §5.3). Rather than build a separate, unvalidatable-in-this-environment residual-stream (`d_model`) hook path for a single baseline, we fit `(d_feat, d_PC0)` in **per-head attention-output space (`d_h`)** from the same contrastive activation set MAGS captures (restricted to the fit/train split, SPEC §4.8 — no held-out leakage), and apply the fixed-offset rotation to `x_heads` (the `W_O` pre-hook input) at **every layer** via `hook_layers='all'`. This is a **head-output variant of AS**, NOT the residual-stream form Vu & Nguyen specify: a head-output rotation projected through `W_O` is a linear image of a residual-stream rotation only inside the column space of `W_O` and cannot move the orthogonal complement, so the plane dimension (`d_h`=128 vs `d_model`=4096 for Llama), magnitude and effect differ. Consequently the AS arm's accuracy/PPL numbers are **not directly comparable** to the paper's AS rows (Table 1/2, e.g. MATH-500/Llama 0.506); the arm is retained as a same-hook-infrastructure contrast and labelled a head-output AS variant in `mags/baselines.py:194-198` and REPRODUCTION.md. The paper (tex:L394) states only the contrast-SET adaptation (correct/incorrect traces for prompt pairs) and is silent on the intervention point, so this is a reproduction choice, recorded here. The faithful residual-stream form (pre-hook on each decoder layer's `hidden_states` input, `d_model`-dim plane) is the reference design if a residual-stream hook path is later added.
17. **Contrastive Decoding specifics** — amateur model per family and CD hyperparameters: unstated. `→ α_plausibility=0.1, β=0.5 (CD paper defaults); amateur: Llama-3.2-1B-Instruct for Llama-3.1-8B-Instruct, gemma-3-1b-it for Gemma-4-E4B-it; excluded for GPT-OSS (paper's own exclusion, tex:L527)`.
18. **Molecular task specifics** — target protein, molecule prompts, SMILES corpus for contrastive pairs, high/low binding-affinity cutoff, AutoDock-GPU parameters, GPT-OSS precision: **all unstated**. `→ will be pinned at implementation time from AutoDock-GPU defaults; treat Table 3 as a stretch target.`
19. **Seeds** — only the bootstrap seed (=42, tex:L700) is given. Trace-sampling and any eval sampling seeds: unstated. `→ seed=42 everywhere needed`, recorded in run manifest.
20. **`ℓ_bip` (attention-shift visualisation layer, tex:L678) and KDE bandwidth (Fig 4) — unstated (visualisation only, not gated).**
21. **Precision**: weights fp16 on RTX 4090 for the two 8B-class models (tex:L692); GPT-OSS ran on H200 (tex:L693) — precision unstated (Hub default is mxfp4 for experts).
22. **Model naming**: "Gemma-4-E4b-it" (tex:L387) is `google/gemma-4-E4B-it` on HF (exists as of 2026-07; 42 layers, 8 heads, head_dim 256, bf16). Llama-3.1-8B-Instruct config (32 layers, 32 Q-heads / 8 KV-heads, head_dim 128) is the standard public architecture. GPT-OSS-20B: 24 layers, 64 heads, head_dim 64.
  - **Round-19 correction (verified against the real `google/gemma-4-E4B-it` config on `meta` device, no weights, `transformers` 5.14.1):** the repo loads as `Gemma4ForConditionalGeneration` (multimodal: text + vision + audio). The text stack is `model.model.language_model.layers[l].self_attn.o_proj` (a `Gemma4TextModel` of 42 layers). `num_attention_heads=8` is CONSTANT across all layers, but `head_dim` is NOT: `sliding_attention` layers use 8×256 (o_proj.in=2048), `full_attention` layers (indices 5,11,17,23,29,35,41) use 8×512 (o_proj.in=4096). The paper/SPEC §4.22 statement "8 heads, head_dim 256" is true only for the sliding layers. This broke two things in the prior implementation (both caught by `tests/test_gemma4_adapter.py`, fixed round-19): (a) `_gemma4_o_proj` checked `hasattr(model,"language_model")` on the TOP object (False for the multimodal wrapper) then `model.layers` (absent) → `AttributeError`, so MAGS could not resolve a single hook target on the real Gemma model; (b) the hook's single-head_dim assert `flat==H*dh` (4096==8*256) crashed on full-attention layer 41, which IS in the SPEC §4.5 default monitored set [10,21,31,41]. Fixes: `_gemma4_o_proj` resolves `model.model.language_model.layers[l]...` (with text-only/bare fallbacks); the hook derives `dh = flat // n_heads` per layer; `_infer_layout` reads `config.text_config`; Angular Steering fits one plane per distinct head_dim (Gemma-4 → two planes, dh 256 and 512) so pooling 256- and 512-dim contrastive activations no longer raises.
 23. **MathInstruct subset for the MATH-500 contrastive corpus.** The paper names "Math-Instruct" (tex:L299, tex:L398-399) citing `yue2023mammoth`, i.e. `TIGER-Lab/MathInstruct` (262k rows), a MIX of MATH-train, GSM8K-derived, aqua_rat, mathqa, numglue, TheoremQA, college_math, and camel items (both CoT and PoT). It does not state which subset to use. `→ keep only MATH-sourced rows (source field contains "MATH": `MATH_train.json` CoT+PoT ≈ 21.9k, `math50k_camel.json` ≈ 49.5k, `college_math.json` ≈ 1.8k)` and discard GSM8K-grade-school / aqua_rat / mathqa / numglue / TheoremQA rows. Rationale: MATH-500 is competition-style math; the contrastive manifold should capture MATH-style error drift, and mixing in grade-school word problems (GSM8K) and code-style PoT items would dilute the error-direction signal with off-distribution problem content. This is the same on-distribution principle behind Remark 1 (tex:L258-261). Implemented in `mags/data/loaders.py::load_mathinstruct` (source-field filter, lines 111-114). A full-mix variant is a review-time knob, not the default.
  - **REALIZED keep-set (recorded honestly, round-19):** gold must be extractable so a generated trace can be graded correct/incorrect against it; the MATH grader (`mags.grading.grade_math`) extracts a trailing `\boxed{...}` from the generated trace and compares to the gold. So the keep-set is the subset of the MATH-sourced rows whose reference `output` yields a clean gold:
    - **MATH_train CoT (11,237 kept):** trailing `\boxed{...}` — gold/trace graders match. This is the bulk.
    - **college_math (1,840 kept, 100%):** multiple-choice; every row ends with "The answer is `<letter>`." → gold = the letter; a generated trace is graded by boxed-letter match.
    - **math50k_camel (90 kept of 49,484):** only the ~90 rows that themselves carry a `\boxed{...}` are kept; the rest are free-form prose with no reliable single-answer marker (e.g. 'D ≈ 1.46497.', or non-single-answer items). **Excluded** — parsing a gold from prose would inject noisy contrastive labels and corrupt the error-direction manifold (the very thing Remark 1 warns against).
    - **MATH_train PoT (1 kept of 10,632):** Python programs whose gold is the executed `print` stdout. **Excluded** — grading a generated PoT trace needs a program-execution grader distinct from the MATH boxed grader and out of scope for the MATH-500 (boxed-answer) manifold; executing arbitrary model-generated code is also a sandboxing concern.
    - **Realized total ≈ 13,168 problems** (not the ~73k the source-field filter alone would admit). This is a narrower-but-clean contrastive corpus; 13k problems is ample for fitting a low-rank (k=4) error subspace. Implemented in `mags/data/loaders.py::_extract_mathinstruct_gold` (boxed-then-MC-letter). The earlier docstring claim "each row's output ends with a boxed answer" was false and is removed.

 25. **`transformers` library version (the pin that lets the paper's models load).** The paper never states a library version. The reproduction's `MODEL_REGISTRY` (§2) names three model families: `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-4-E4B-it`, `openai/gpt-oss-20b`. Their AutoModel classes are `LlamaForCausalLM` (in `transformers` since 4.x), `Gemma4ForConditionalGeneration`, and `GptOssForCausalLM`. **Round-23 finding:** the lock previously pinned `transformers==4.57.1`, whose AutoModel registry has **neither** `Gemma4ForConditionalGeneration` **nor** `GptOssForCausalLM` (verified: `4.57.1` exposes only `Gemma`/`Gemma2`/`Gemma3`/`Gemma3n`). So under the old pin the reproduction could not load 2 of the 3 paper model families on *any* host — a latent bug masked because (a) the no-GPU sandbox BLOCKs every arm before model load, so the missing-class error never surfaced, and (b) `tests/test_gemma4_adapter.py` `pytest.skip`s when the class is absent, so the 4 real-architecture adapter tests silently became 4 skipped tests for 18 rounds (the round-22 REPRODUCTION.md even miscounted them as "52 passed"; the true count under 4.57.1 was 48 passed + 4 skipped). `→ pin transformers==5.14.1` (the first release whose registry has both `Gemma4ForConditionalGeneration` and `GptOssForCausalLM`), and the transitive `huggingface-hub` bump it requires (0.36.2 → 1.25.1). Verified round-23: the 4 Gemma-4 adapter tests now RUN and PASS against the real `google/gemma-4-E4B-it` config on `torch`'s `meta` device (no 16 GB download, no GPU) — genuinely 52 passed, 0 skipped — and `smoke.sh` and all equation-invariant/degeneracy tests stay green; the `run_arm.sh` offline/cache fast-fail still emits `FINAL <arm>=BLOCKED` under `huggingface-hub` 1.x. The pin change is reproducibility infrastructure (the paper's models are unloadable without it), not a method knob.
 24. **Per-arm invocation wrapper (`run_arm.sh`).** The numbers gate iterates every key of `arms.json` and runs that key's command *individually* (not `run_all_arms.sh`). A bare `python -m mags.run ...` command has no outer fallback, so any environment failure to print `FINAL <arm>=<value>` (no `python` on PATH, `mags` not importable from the gate's CWD, import error, hang) leaves the arm with no FINAL line. `→ every `arms.json` command is `sh run_arm.sh <arm-id> <mags.run args...>`; the wrapper `cd`s to repo root, resolves `python`, runs the real `python -m mags.run`, re-emits the run's `FINAL <arm-id>...` lines, and falls back to one honest `FINAL <arm-id>=BLOCKED` (a literal string, never a fabricated number) only when the real run produced none. This is reproducibility infrastructure, not a paper knob: the method hyperparameters (`--arm`, `--iti-K`, `--iti-alpha`, `--angle-deg`) are unchanged, and on a GPU host the real numeric value passes through unchanged.
  - **Round-26 correction (the recurring `values: []` gate failure, root-caused):** the gate reads `manifold-guided-attention-steering/arms.json` by path and runs each command string from the **repository root** (the parent holding the per-paper subfolders), NOT from the reproduction folder. A bare `sh run_arm.sh …` (relative) is not found from the repo root → `sh` exits 2 with zero stdout → the gate reports every arm "missing a FINAL line" with `values: []`. `→ every `arms.json` command is now prefixed with a CWD-resolver: `cd manifold-guided-attention-steering 2>/dev/null || true; sh run_arm.sh <arm-id> …`. From the repo root the `cd` enters the reproduction folder; from the reproduction folder the `cd` fails harmlessly (`|| true` keeps it non-fatal under `set -e`) and `sh run_arm.sh` runs in-place. Verified by a gate simulation (`cwd=repo root`): `found=45 missing=0` (was `found=0 missing=45`). The folder name is the fixed paper slug the gate itself uses to locate `arms.json`, so it is not fragile.

## 5. Component interfaces (frozen — parallel agents build against these)

All tensors little-endian numpy arrays on disk; fp32 for fitted parameters, fp16 only for raw activations.

1. **`TraceStore` (trace collection output)** — JSONL, one record per retained problem:
   `{problem_id, benchmark, prompt_text, gold, traces: [{text, token_ids: [T_gen], correct: bool, act_path}]}`.
   `act_path` → `.npz`: `A: [T_gen, L_mon, H, d_h] fp16` aligned to `token_ids`; `layers: [L_mon] int32`.
   Positions are **generated tokens only** (§4 item 7).
2. **`ManifoldBank` (fit output, one file per model×benchmark)** — `.npz` + JSON manifest:
   per selected/monitored head: `B__l{l}_h{h}: [k, d_h] fp32`, `muc__l{l}_h{h}: [d_h] fp32`, `thresh__l{l}_h{h}: fp32 scalar`, `auroc__l{l}_h{h}: fp32 scalar`.
   Manifest: `{model_id, benchmark, k, q, K, alpha, layers_monitored, split_seeds, n_problems_fit, n_problems_select, selected_heads: [[l,h], ...], git_sha}`.
3. **`HookRegistry` (steering runtime)** — model-family adapter exposing, per layer `l`, a `nn.Module` whose input is `[bsz, seq, H·d_h]` with head-contiguous layout (view `[bsz, seq, H, d_h]`):
   - Llama: `model.layers[l].self_attn.o_proj`
   - Gemma-4 (text stack): `model.language_model.layers[l].self_attn.o_proj` *(exact attr verified at implementation time against transformers version)*
   - GPT-OSS: `model.layers[l].self_attn.o_proj` *(attention output is 64 heads × 64 dims, contiguous)*
   Corrections apply to **decode steps only** (`seq==1` chunks / last position of prefill: §4 item 9 follow-up — prefill positions are not steered, they are only scored).
4. **`SteeringController`** — derives from `ManifoldBank`: `score_hook(pre_input) -> modified_pre_input`; per monitored head per decode position implements Eq. (7)–(9) in fp32 with the in-place semantics of Algorithm 1. Batch size 1 for eval; scores recorded to a per-run `steering_log.jsonl` (`{problem, t, head, d, fired}`) for diagnostics.
5. **`BaselineSteerer` interface** — same hook injection point; three implementations: `ITISteerer` (K, α as §4.15), `AngularSteerer` (fixed-offset angle as §4.16, plane from candidate directions at ALL layers, applied at every layer), `ContrastiveDecoder` (expert/amateur `generate` wrapper computing CD logits, no hooks), `UnsteeredRunner` (no hooks).
6. **`EvalHarness`** — `run(benchmark, model, arm_charge) -> {n, acc, ppl, per_problem: [{id, correct: 0/1, ppl}], path}`; math: boxed-extraction exact match; code: sandboxed test execution, timeout 10 s; GSM8K: numeric match. Bootstrap CIs: percentile method, B=10_000, seed=42 (tex:L700) over `per_problem.correct`.
7. **`arms.json`** — root of repo; schema `{benchmark, model, arm, metric, claimed, claimed_ci95, config, notes}`. This file is the numbers-gate contract (§6).
8. **Run manifest** — every eval run writes `runs/<ts>__<arm>__<benchmark>__<model>/manifest.json` with seeds, model revision SHAs, generation config, hooks config, git SHA.

## 6. Arms (the comparisons the paper makes → arms.json)

**Reasoning/code**: 2 models × 4 benchmarks × 5 methods = **40 arms**. Methods: `unsteered`, `iti`, `angular-steering`, `contrastive-decoding`, `mags`. Claimed values from Table 1 (tex:L420–441) and Table 2 (tex:L466–487); CIs from tex:L717–743 / L763–789.

**Molecular**: 1 model (`openai/gpt-oss-20b`) × 5 methods — `unsteered`, `angular-steering`, `iti`, `mags`, `mags-u` (CD excluded by the paper, tex:L527) = **5 arms**, two metrics each (validity %, docking kcal/mol), Table 3 (tex:L539–543).

Arm configs:
- `mags`: `k=4 (§4.1 default), q=95 (§4.2 default), K=3, α=1.0`, per-§5.5 hooks on monitored layers.
- `mags-u`: two manifolds (validity, affinity), union head set, same k/q/K/α defaults.
- `iti`: `K=96, α=0.5` (best of tex:L640–642, matches main-table MATH-500 0.498).
- `angular-steering`: `angle=30°` (best of tex:L654–665, matches main-table 0.506).
- `contrastive-decoding`: `α_p=0.1, β=0.5`, amateur per §4.17.

The full arm table (45 rows) with claimed accuracy/PPL/validity/affinity per benchmark×model is written to `arms.json` alongside this SPEC. The ablation grid (tex:L613–668: MAGS 6 configs, ITI 9, AS 12 on MATH-500/Llama) is a **secondary** arm set used for the sensitivity check, not for the headline comparison.

Claimed headline (what the numbers gate checks first): MATH-500, Llama-3.1-8B-Instruct — unsteered 0.478 vs MAGS 0.530 (Δ +5.2 pts); Gemma — 0.614 vs 0.648 (Δ +3.4 pts).

## 7. Hazards / things that will otherwise silently be wrong

- SVD convention: `torch.linalg.svd` returns `Vh` = `V^⊤`; Eq. (5) takes its top rows directly — no transpose of the principal components. `B` rows must be orthonormal (assert ‖B B^⊤ − I‖ < 1e-4).
- `D` is `[N, d_h]` (rows = problems). Taking top singular vectors of `D^⊤` instead would produce vectors in problem-space — wrong axis, trains fine, wrong method.
- `μ_c` token-count weighting (denominator `Σ_i Σ_{τ∈T_i^+} L_τ`) must pool token counts, not trace counts.
- Threshold pool: Eq. (8) percentiles act over **per-token** scores pooled across correct traces — not per-trace maxima or means.
- The correction subtracts the projection of the **centred** activation: `B^⊤B(a − μ_c)`, and then **re-adds nothing centre-wise** — it does NOT subtract `B^⊤B·a`. Off-by-centreing is a silent method change.
- Hook must fire on the input to `W_O` (pre-o_proj), i.e. attention outputs pre-projection, per tex:L356–357 — not on the residual stream and not post-o_proj.
- Decode-only steering (§4.9): prefill pass-through must not be rewritten, or prompts get re-centred too.
- Gemma-4-E4B-it: `H=8` heads × `d_h=256` = 2048 ≠ `d_model=2560` — do not derive d_h as hidden/H; read `config.head_dim`. **Round-19 addendum:** `d_h` is NOT constant across layers on the real Gemma-4 (256 sliding, 512 full-attention); the hook derives `d_h = o_proj.in_features // n_heads` per layer (SPEC §4.22), and Angular Steering fits one plane per distinct `d_h`.
