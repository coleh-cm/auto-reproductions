"""Activation capture for trace collection (SPEC §5.1, Phase A input).

Runs the base model over a set of prompts (sampled or greedy) and records, per
generated token, the per-head pre-``W_O`` input for every monitored layer. Returns
per-trace activations aligned to the generated tokens (prompt tokens excluded per
SPEC §4.7). Also records the generated text and token ids and the binary correctness
label (graded externally).
"""
from __future__ import annotations
import numpy as np
import torch


class _CaptureHook:
    """Records the last-position o_proj input at every forward call.

    For prefill (seq>1) we keep only the last position (it produces the first
    generated token's logits); for decode (seq==1) we keep the single position.
    Concatenated in call order this yields exactly L_tau activations, one per
    generated token (SPEC §4.7: generated tokens only).

    ``n_heads`` is constant across layers for every supported family (Gemma-4: 8
    everywhere). ``head_dim`` can vary per layer (Gemma-4: 256 sliding, 512 full),
    so each captured array keeps its own ``[H, dh]`` and the per-layer shape is
    learned from the first capture (the ``head_dim`` arg is only the empty-buffer
    fallback, never the reshape authority — the hook derives dh per layer).
    """
    def __init__(self, monitored_layers, n_heads, head_dim):
        self.monitored = set(monitored_layers)
        self.buffer = {l: [] for l in monitored_layers}
        self.H = n_heads
        self.dh = head_dim
        self._layer_shape = {}   # layer -> (H, dh) learned from first capture

    def __call__(self, layer, x_heads):
        bsz, seq, H, dh = x_heads.shape
        if layer not in self.monitored:
            return None
        # SPEC §4.7 (generated tokens only): capture DECODE steps only (seq==1).
        # model.generate(use_cache=True) issues one prefill (seq>1) per trace whose
        # last position is the activation at the LAST PROMPT-token position — that is a
        # prompt-context activation (attends only to prompt tokens) which SPEC §4.7
        # explicitly excludes from the means and the global centroid mu_c (Eq.6),
        # warning it would shift mu_c. Skipping prefill here also keeps the FIT
        # consistent with the inference controller (mags/steering.py:45), which only
        # scores/steers decode steps (seq==1) — so the threshold pool (Eq.8) is
        # calibrated on the same decode-step score distribution the controller emits.
        # The activation captured at decode step k is the head output at the position
        # of generated token k-1; pooling all captured rows yields one activation per
        # decode step (per generated token after the first), aligned to generated tokens.
        if seq != 1:
            return None
        self._layer_shape[layer] = (H, dh)
        last = x_heads[:, -1:, :, :].detach().to(torch.float32).cpu().numpy()
        self.buffer[layer].append(last[0, 0])   # [H, dh]
        return None

    def stacked(self):
        # per layer [T, H, dh]; dh is per-layer (Gemma-4 full-attention dh=512).
        out = {}
        for l in self.monitored:
            if self.buffer[l]:
                out[l] = np.stack(self.buffer[l], axis=0)
            else:
                H, dh = self._layer_shape.get(l, (self.H, self.dh))
                out[l] = np.zeros((0, H, dh), dtype=np.float32)
        return out


@torch.no_grad()
def capture_trace(model, tok, model_id, hook_registry, prompt_text, max_new_tokens,
                  do_sample=True, temperature=1.0, top_p=0.95, seed=42,
                  use_chat_template=False):
    """Generate one trace from ``prompt_text`` and capture per-head activations.

    ``hook_registry`` must be attached with this capture callback. Returns
    (text, token_ids, acts) where acts[layer] = [T_gen, H, dh] fp32.

    ``use_chat_template`` (SPEC §4.13): the manifold must be fit on traces in the
    SAME prompt format as eval, so the contrastive error subspace captures errors of
    the eval distribution. Pass the benchmark's chat-template flag here too.
    """
    torch.manual_seed(seed)
    from .generation import _truncate_prompt, _tokenize_prompt
    ids = _tokenize_prompt(tok, prompt_text, use_chat_template).to(model.device)
    ids = _truncate_prompt(model, ids, max_new_tokens)
    n_prompt = ids.shape[1]
    capture = _CaptureHook(hook_registry.layout.monitored_layers,
                           hook_registry.layout.n_heads, hook_registry.layout.head_dim)
    hook_registry.attach(capture)
    try:
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )
    finally:
        hook_registry.detach()
    gen_ids = out[0, n_prompt:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    acts = capture.stacked()
    # Decode-only capture (SPEC §4.7): acts has T = (#decode forwards) rows, one per
    # decode step. Each decode step k's activation sits at the position of generated
    # token k-1, so the captured rows align to gen_ids[:-1] (the last generated token
    # has no decode forward). Return the aligned gen-id prefix so the stored token_ids
    # length matches A's T (build_head_activations pools rows and ignores token_ids, but
    # a length mismatch would be a latent store inconsistency). ``text`` is the FULL
    # completion (all generated tokens) for grading.
    any_T = next((a.shape[0] for a in acts.values() if a.shape[0] > 0), 0)
    aligned = gen_ids[:any_T]
    return text, aligned.cpu().numpy(), acts
