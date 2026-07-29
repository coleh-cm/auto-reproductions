"""Round-20 regression tests: two verified faithfulness findings fixed.

1. capture.py must capture DECODE steps only (seq==1), never the prefill-last
   (prompt-position) activation. SPEC §4.7 (generated tokens only): the prefill-last
   is a prompt-context activation that would shift the global centroid mu_c (Eq.6)
   and the threshold pool (Eq.8), and the inference controller never scores it
   (steering.py skips seq>1), so including it is a fit/inference inconsistency.

2. generation.py must apply the model's HF chat template for the natural-language
   benchmarks (MATH-500/GSM8K/MBPP) and NOT for HumanEval (completion-style). SPEC §4.13.
   Base models without a chat template (distilgpt2) fall back to raw tokenization.
"""
import os
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MAGS_SMOKE_MODEL") == "none", reason="smoke model disabled",
)
SMOKE_MODEL = os.environ.get("MAGS_SMOKE_MODEL", "distilgpt2")


def _load():
    from mags.models import load_model
    return load_model(SMOKE_MODEL)


def test_capture_is_decode_only():
    """Finding-1 fix: _CaptureHook records only seq==1 forwards, not the prefill.

    With max_new_tokens=N and use_cache=True, model.generate issues 1 prefill
    (seq>1) + (N-1) decodes (seq==1). The hook must capture exactly N-1 rows
    (the decodes), NOT N (which would include the prefill-last prompt-position
    activation). The prefill-last attends only to prompt tokens and would shift
    mu_c (Eq.6) and the threshold pool (Eq.8) — a SPEC §4.7 violation.
    """
    from mags.model_adapter import HookRegistry  # noqa: F401
    from mags.capture import _CaptureHook
    model, tok = _load()
    model._mags_tokenizer = tok
    reg = model._mags_registry
    monitored = reg.layout.monitored_layers
    prompt = "The answer to 2+2 is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    N = 5
    cap = _CaptureHook(monitored, reg.layout.n_heads, reg.layout.head_dim)
    reg.attach(cap)
    try:
        with torch.no_grad():
            model.generate(ids, max_new_tokens=N, do_sample=False,
                           pad_token_id=tok.eos_token_id, use_cache=True)
    finally:
        reg.detach()
    acts = cap.stacked()
    for l in monitored:
        A = acts[l]
        # N max_new_tokens -> 1 prefill + (N-1) decodes -> capture N-1 rows.
        assert A.shape[0] == N - 1, (
            f"layer {l}: captured {A.shape[0]} rows, expected {N - 1} (decode-only); "
            "prefill-last prompt activation leaked into the manifold (SPEC §4.7).")


def test_capture_trace_aligned_gen_ids_length():
    """Finding-1 fix: capture_trace returns gen_ids whose length matches the
    captured T (decode-only), so the stored token_ids/A stay aligned."""
    from mags.capture import capture_trace
    model, tok = _load()
    model._mags_tokenizer = tok
    reg = model._mags_registry
    prompt = "The capital of France is"
    text, ids, acts = capture_trace(
        model, tok, SMOKE_MODEL, reg, prompt, max_new_tokens=6,
        do_sample=False, seed=42, use_chat_template=False,
    )
    # text is the FULL completion (all 6 generated tokens); ids is aligned to acts.
    any_T = next(a.shape[0] for a in acts.values() if a.shape[0] > 0)
    assert len(ids) == any_T, (
        f"gen_ids len {len(ids)} != captured T {any_T}; store would be misaligned.")
    # text still decodes to the full completion (grading uses text, not ids).
    assert len(text) > 0


def test_chat_template_set_and_humaneval_excluded():
    """Finding-2: CHAT_TEMPLATE_BENCHMARKS = {MATH-500, GSM8K, MBPP}; HumanEval excluded."""
    from mags.generation import CHAT_TEMPLATE_BENCHMARKS
    assert CHAT_TEMPLATE_BENCHMARKS == {"MATH-500", "GSM8K", "MBPP"}, (
        "HumanEval must be EXCLUDED (completion-style); MATH/GSM8K/MBPP included.")
    assert "HumanEval" not in CHAT_TEMPLATE_BENCHMARKS


def test_tokenize_prompt_chat_fallback_on_base_model():
    """Finding-2: a base model with no chat template (distilgpt2) must fall back to
    raw tokenization rather than crash; the result must equal raw tokenization."""
    from mags.generation import _tokenize_prompt
    _, tok = _load()
    prompt = "Solve: what is 2+2?"
    raw = tok(prompt, return_tensors="pt").input_ids
    # distilgpt2 has no chat template -> fallback path.
    out = _tokenize_prompt(tok, prompt, use_chat_template=True)
    assert out.shape == raw.shape, (
        "base-model chat-template fallback should equal raw tokenization")
    assert torch.equal(out, raw), (
        "base-model chat-template fallback diverged from raw tokenization")


def test_tokenize_prompt_no_chat_equals_raw():
    """Finding-2: use_chat_template=False is exactly raw tokenization (HumanEval path)."""
    from mags.generation import _tokenize_prompt
    _, tok = _load()
    prompt = "def f(x):\n    return"
    raw = tok(prompt, return_tensors="pt").input_ids
    out = _tokenize_prompt(tok, prompt, use_chat_template=False)
    assert torch.equal(out, raw)
