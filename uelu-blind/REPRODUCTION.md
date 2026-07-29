# Reproduction: A Structural Interpretation of GELU and Threshold-Transmission Activations via the First-Order Loss Function

- **Paper**: Roberto Rossi (2026), *A Structural Interpretation of GELU and Threshold-Transmission Activations via the First-Order Loss Function*
- **paper_ref**: 2607.03664
- **project_id**: blind-eval
- **Date setup**: 2026-07-29

## Status

**Current stage: workspace setup complete — implementation not yet started.**

- [x] Reproduction folder created: `/root/auto-reproductions/uelu-blind`
- [x] Folder path recorded in `$HOME/.repro_dir` (no trailing newline)
- [x] Paper source saved to `paper/main.tex` (full LaTeX source, supplied verbatim with the workflow objective)
- [ ] SPEC.md (method, equations, hyperparameters, "not stated" list)
- [ ] Implementation of activations (GELU, ReLU, SiLU, DGELU, UELU-β, TUELU) and the five experiments
- [ ] Smallest end-to-end run produces a parsed number
- [ ] Adversarial review loop clean
- [ ] Readiness gates walked
- [ ] Final numbers and publish

## Source notes

- The objective listed `arxiv_id: unknown` and supplied the full LaTeX source directly, so there was
  **no arXiv e-print fetch to perform** — nothing failed; the paper on disk *is* the LaTeX source and
  equations are exact (no PDF-extraction risk applies).
- Preamble macros to resolve when quoting the paper:
  - `\best{x}` → bold math (marks the best entry in results tables).
  - `normalcdf(x)` → a pgfmath approximation of the standard-normal CDF used only for plotting
    Figure 1; it is not part of any method equation.
- The authors' code-availability paragraph was redacted from the supplied source; no upstream code
  repository is referenced. At implementation time, before writing new code, search the web/GitHub for
  an author repository anyway.

## What the reproduction must cover (from the paper)

Five controlled experiments comparing activations {ReLU, SiLU, GELU, DGELU, UELU(β=√(π/2)≈1.25),
UELU/hard swish (β=3), TUELU(learned shared β via softplus, init √(π/2))}, five seeds each:

1. **MLP-Mixer / CIFAR-100**: 4×4 patches, embed 192, 6 blocks, token width 96, channel width 384;
   5k/45k validation/training split, random crop + horizontal flip, AdamW lr 3e-4, wd 0.05,
   label smoothing 0.1, batch 128, cosine decay, 20 epochs. Paper: UELU/TUELU ≈ 51.28–51.29 test acc.
2. **Compact ViT / CIFAR-100**: 4×4 patches, cls token, learned pos-emb, 192-dim, 6 pre-norm blocks,
   6 heads, FFN 384; same optimiser schedule as Mixer. Paper: TUELU 49.41 test acc, β≈1.14.
3. **Tiny char-GPT / Tiny Shakespeare**: 4 blocks, 4 heads, dim 128, ctx 128, dropout 0.1;
   AdamW lr 3e-4→3e-5 cosine, wd 0.1, clip 1.0, batch 64, 100 warmup iters, 10,000 iters.
   Paper: TUELU val perplexity 4.575, β≈0.70.
4. **TinyStories token-GPT**: 8k byte-level BPE, ≤50k train / 5k val stories; 6 blocks, 6 heads,
   dim 384, ctx 256, dropout 0.1, batch 32; optimiser matches #3. Paper: TUELU ppl 6.863, β≈1.02.
5. **WikiText-2 token-GPT**: raw train/val splits, 8k BPE; 4 blocks, 4 heads, dim 256, ctx 256,
   dropout 0.1, batch 32; optimiser matches #3. Paper: TUELU ppl 82.816, β≈0.887.

Region occupancy (closed / transition / open percentages) must be logged for all uniform-threshold arms.

## Log

- **2026-07-29** — Workspace initialised. Repo `auto-reproductions` was already cloned at `$HOME`
  (remote `https://github.com/coleh-cm/auto-reproductions`, token-credentialed); pulled to latest.
  `uelu-blind/` created, `paper/main.tex` written verbatim, this file started. Committed.
