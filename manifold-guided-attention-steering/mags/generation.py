"""Generation loop with steering + perplexity (SPEC §4.9, §4.14).

Greedy / sampled generation with an installed W_O pre-hook controller. Decode-only
steering: the controller passes through prefill (seq>1) and modifies only decode
steps (seq==1), per SPEC §4.9.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tok, prompt_text, controller, max_new_tokens=1024,
             do_sample=False, temperature=1.0, top_p=0.95, seed=42):
    """Generate text with ``controller`` attached as the W_O pre-hook callback.

    Uses model.generate (use_cache=True) so decode forwards are seq==1 and the
    controller's prefill pass-through + decode-modify semantics hold. Returns the
    decoded completion (prompt excluded) and the generated token ids.
    """
    torch.manual_seed(seed)
    ids = tok(prompt_text, return_tensors="pt").input_ids.to(model.device)
    n_prompt = ids.shape[1]
    # Install the controller through the model's HookRegistry when available.
    if hasattr(model, "_mags_registry") and controller is not None:
        model._mags_registry.attach(controller)
    try:
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )
    finally:
        if hasattr(model, "_mags_registry") and controller is not None:
            model._mags_registry.detach()
        if hasattr(controller, "flush_log"):
            controller.flush_log()
    gen_ids = out[0, n_prompt:]
    return tok.decode(gen_ids, skip_special_tokens=True), gen_ids.cpu().numpy()


@torch.no_grad()
def cd_generate(expert, amateur, tok, prompt_text, max_new_tokens=1024,
                alpha_plausibility=0.1, beta=0.5, seed=42):
    """Greedy Contrastive Decoding (SPEC §4.17). Expert and amateur share the
    tokenizer; we keep a parallel KV cache for the amateur and adapt expert logits."""
    from .baselines import ContrastiveDecoder
    torch.manual_seed(seed)
    cd = ContrastiveDecoder(expert, amateur, tok, alpha_plausibility, beta)
    ids = tok(prompt_text, return_tensors="pt").input_ids.to(expert.device)
    n_prompt = ids.shape[1]
    # prefill both
    out_e = expert(ids, use_cache=True)
    out_a = amateur(ids, use_cache=True)
    past_e = out_e.past_key_values
    past_a = out_a.past_key_values
    logits_e = out_e.logits[:, -1, :]
    logits_a = out_a.logits[:, -1, :]
    generated = []
    for _ in range(max_new_tokens):
        score = cd.adapted_logits(logits_e, logits_a)   # [1, V]
        nxt = score.argmax(dim=-1, keepdim=True)        # [1,1]
        generated.append(int(nxt.item()))
        if nxt.item() == tok.eos_token_id:
            break
        out_e = expert(nxt, past_key_values=past_e, use_cache=True)
        out_a = amateur(nxt, past_key_values=past_a, use_cache=True)
        past_e = out_e.past_key_values
        past_a = out_a.past_key_values
        logits_e = out_e.logits[:, -1, :]
        logits_a = out_a.logits[:, -1, :]
    gen_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0)
    return tok.decode(gen_ids[0], skip_special_tokens=True), gen_ids[0].cpu().numpy()


@torch.no_grad()
def perplexity_of(model, tok, completion_text):
    """Perplexity of ``completion_text`` under ``model`` (token-level, mean over tokens).

    SPEC §4.14: PPL of the generated completion under the (unsteered) base model,
    averaged over problems. Secondary metric; not gated.
    """
    ids = tok(completion_text, return_tensors="pt").input_ids.to(model.device)
    if ids.shape[1] < 2:
        return float("nan")
    out = model(ids, use_cache=False)
    logits = out.logits[:, :-1, :]
    targets = ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(-tok_logp.mean()).item())
