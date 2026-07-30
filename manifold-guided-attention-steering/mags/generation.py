"""Generation loop with steering + perplexity (SPEC §4.9, §4.14).

Greedy / sampled generation with an installed W_O pre-hook controller. Decode-only
steering: the controller passes through prefill (seq>1) and modifies only decode
steps (seq==1), per SPEC §4.9.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

# SPEC §4.13: the model's HuggingFace chat template is the prompt-format decision for
# the natural-language-instruction benchmarks. HumanEval is the exception: its prompt
# is a raw function signature and the canonical HumanEval protocol is completion-style
# (no chat priming), so the instruct model is run as a raw-continuation LM there. This
# set drives ``use_chat_template`` for BOTH eval and the manifold fit (capture_trace):
# the contrastive error subspace must be fit on traces in the SAME prompt format as
# eval, or it captures errors of a different distribution (SPEC §4.13 + finding-2 fix).
CHAT_TEMPLATE_BENCHMARKS = {"MATH-500", "GSM8K", "MBPP"}


def _tokenize_prompt(tok, prompt_text, use_chat_template):
    """Tokenize ``prompt_text``, applying the model's chat template when requested.

    SPEC §4.13 mandates the HF chat template for instruct models on the
    natural-language benchmarks. A base model (e.g. distilgpt2 smoke) has no chat
    template; ``apply_chat_template`` is absent or raises, so we fall back to raw
    tokenization rather than crashing — the chat template is a property of the
    tokenizer, not the benchmark, and base models are only ever used for the
    code-path smoke (never for paper numbers).
    """
    if not use_chat_template:
        return tok(prompt_text, return_tensors="pt").input_ids
    try:
        messages = [{"role": "user", "content": prompt_text}]
        ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=False,
        )
        if ids is None or (hasattr(ids, "shape") and ids.shape[-1] == 0):
            return tok(prompt_text, return_tensors="pt").input_ids
        return ids
    except (TypeError, ValueError, AttributeError, RuntimeError):
        # no chat template configured (base model) -> raw tokenization
        return tok(prompt_text, return_tensors="pt").input_ids


def _max_positions(model):
    """Best-effort model context length. Falls back to a large value when the
    attribute is absent (modern models set it on config)."""
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "max_seq_len",
                 "model_max_length"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 1 << 30


def _truncate_prompt(model, ids, max_new_tokens):
    """Left-truncate prompt token ids so prompt + max_new_tokens fits the model's
    context window. Long training prompts (e.g. APPS questions) can exceed a
    model's max_position_embeddings on small models and would otherwise raise an
    IndexError in the position-embedding lookup; the paper's 8B/20B models have
    >=8k context so this is inert for the real runs (it only prevents a crash on
    very long prompts). Left-truncation preserves the most recent prompt context,
    matching the paper's instruction-tuned chat-template usage."""
    cap = _max_positions(model)
    room = max(1, cap - int(max_new_tokens))
    if ids.shape[1] > room:
        ids = ids[:, -room:]
    return ids


@torch.no_grad()
def generate(model, tok, prompt_text, controller, max_new_tokens=1024,
             do_sample=False, temperature=1.0, top_p=0.95, seed=42,
             use_chat_template=False):
    """Generate text with ``controller`` attached as the W_O pre-hook callback.

    Uses model.generate (use_cache=True) so decode forwards are seq==1 and the
    controller's prefill pass-through + decode-modify semantics hold. Returns
    (completion_text, gen_ids, prompt_ids) where gen_ids are the generated token
    ids (prompt excluded) and prompt_ids are the tokenized prompt. prompt_ids is
    returned so perplexity_of can compute CONDITIONAL PPL of the completion given
    the prompt under the unsteered base model (SPEC §4.14); the bare completion
    ids alone would give an unconditional PPL the paper never reports.

    ``use_chat_template`` (SPEC §4.13): wrap the prompt in the model's HF chat
    template for the natural-language-instruction benchmarks before generation;
    HumanEval (completion-style) passes False. Base models without a chat
    template fall back to raw tokenization (see _tokenize_prompt).
    """
    torch.manual_seed(seed)
    ids = _tokenize_prompt(tok, prompt_text, use_chat_template).to(model.device)
    ids = _truncate_prompt(model, ids, max_new_tokens)
    n_prompt = ids.shape[1]
    prompt_ids = ids[0].cpu().numpy()
    # Install the controller through the model's HookRegistry when available.
    if hasattr(model, "_mags_registry") and controller is not None:
        layers = getattr(controller, "hook_layers", None)
        model._mags_registry.attach(controller, layers=layers)
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
    return tok.decode(gen_ids, skip_special_tokens=True), gen_ids.cpu().numpy(), prompt_ids


@torch.no_grad()
def cd_generate(expert, amateur, tok, prompt_text, max_new_tokens=1024,
                alpha_plausibility=0.1, beta=1.0, seed=42, use_chat_template=False):
    """Greedy Contrastive Decoding (SPEC §4.17). Expert and amateur share the
    tokenizer; we keep a parallel KV cache for the amateur and adapt expert logits.

    ``use_chat_template`` (SPEC §4.13): same convention as generate(); CD is run on
    the same benchmarks so it must use the same prompt format as the other arms or
    its logits/distribution comparison is against a different prompt distribution.
    """
    from .baselines import ContrastiveDecoder
    torch.manual_seed(seed)
    cd = ContrastiveDecoder(expert, amateur, tok, alpha_plausibility, beta)
    ids = _tokenize_prompt(tok, prompt_text, use_chat_template).to(expert.device)
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
    return (tok.decode(gen_ids[0], skip_special_tokens=True),
            gen_ids[0].cpu().numpy(), ids[0].cpu().numpy())


@torch.no_grad()
def perplexity_of(model, tok, completion_text=None, token_ids=None, prompt_ids=None):
    """Perplexity of a completion under ``model`` (token-level, mean over tokens).

    SPEC §4.14: PPL of the generated completion under the (unsteered) base model,
    averaged over problems. Secondary metric; not gated.

    CONDITIONAL PPL (the protocol the paper's ~1.1-1.2 values imply, tex:L420-441):
    the NLL of the completion tokens GIVEN the prompt under the unsteered model.
    Pass ``prompt_ids`` (the prompt's token ids) together with the generated
    ``token_ids``; the model is run on prompt+completion and the loss is averaged
    over the completion positions only. Without ``prompt_ids`` this falls back to
    the UNCONDITIONAL PPL of the bare completion (kept for backward compatibility
    with smoke/diagnostics, but NOT what the paper reports).

    Prefer passing the actual generated ``token_ids`` (avoids re-tokenization, which
    can shift boundaries / drop EOS and yield a PPL not equal to the PPL of the
    generated tokens). Falls back to re-tokenizing ``completion_text`` only if ids
    are unavailable.
    """
    import torch as _t
    if token_ids is None:
        ids = tok(completion_text, return_tensors="pt").input_ids
    else:
        ids = _t.as_tensor(token_ids, dtype=_t.long).unsqueeze(0)
    ids = ids.to(model.device)
    if prompt_ids is not None:
        p_ids = _t.as_tensor(prompt_ids, dtype=_t.long).unsqueeze(0).to(model.device)
        n_prompt = p_ids.shape[1]
        full = _t.cat([p_ids, ids], dim=1)
    else:
        n_prompt = 0
        full = ids
    full = full.to(model.device)
    n_gen = ids.shape[1]
    if n_gen < 1:
        return float("nan")
    out = model(full, use_cache=False)
    logits = out.logits[:, :-1, :]            # predicts token at next position
    targets = full[:, 1:]
    # Completion token i (0-indexed) is at full position n_prompt + i; it is
    # predicted by the logit at position n_prompt + i - 1, i.e. logits index
    # (n_prompt + i - 1) and target index (n_prompt + i - 1).
    start = n_prompt - 1
    end = n_prompt + n_gen - 1
    if start < 0:
        start = 0
    logits_gen = logits[:, start:end, :]
    targets_gen = targets[:, start:end]
    logp = F.log_softmax(logits_gen, dim=-1)
    tok_logp = logp.gather(-1, targets_gen.unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(-tok_logp.mean()).item())
