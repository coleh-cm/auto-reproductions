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
    """
    def __init__(self, monitored_layers, n_heads, head_dim):
        self.monitored = set(monitored_layers)
        self.buffer = {l: [] for l in monitored_layers}
        self.H = n_heads
        self.dh = head_dim

    def __call__(self, layer, x_heads):
        bsz, seq, H, dh = x_heads.shape
        if layer not in self.monitored:
            return None
        # last position only
        last = x_heads[:, -1:, :, :].detach().to(torch.float32).cpu().numpy()
        self.buffer[layer].append(last[0, 0])   # [H, dh]
        return None

    def stacked(self):
        # per layer [T, H, dh]
        return {l: np.stack(self.buffer[l], axis=0) if self.buffer[l]
                else np.zeros((0, self.H, self.dh), dtype=np.float32)
                for l in self.monitored}


@torch.no_grad()
def capture_trace(model, tok, model_id, hook_registry, prompt_text, max_new_tokens,
                  do_sample=True, temperature=1.0, top_p=0.95, seed=42):
    """Generate one trace from ``prompt_text`` and capture per-head activations.

    ``hook_registry`` must be attached with this capture callback. Returns
    (text, token_ids, acts) where acts[layer] = [T_gen, H, dh] fp32.
    """
    torch.manual_seed(seed)
    ids = tok(prompt_text, return_tensors="pt").input_ids.to(model.device)
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
    return text, gen_ids.cpu().numpy(), acts
