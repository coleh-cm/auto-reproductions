"""Model loading helper. Loads a HF causal LM + tokenizer for any family in the
registry, attaches a HookRegistry, and pins the dtype (SPEC §4.21)."""
from __future__ import annotations
import torch
from .model_adapter import HookRegistry
from . import config


def load_model(model_id: str, device: str | None = None, dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = config.get_model_spec(model_id)
    prec = dtype if dtype is not None else spec.get("precision")
    torch_dtype = {
        "float16": torch.float16, "bfloat16": torch.bfloat16,
        "float32": torch.float32, None: None,
    }[prec]
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device is not None:
        kwargs["device_map"] = device
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Could not load model {model_id!r}: {e!r}. This is the expected BLOCKER "
            f"in a no-GPU / no-gated-token sandbox; the paper's models are Llama-3.1-8B-"
            f"Instruct (gated), Gemma-4-E4B-it, GPT-OSS-20B (need >=40GB VRAM)."
        )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # attach the hook registry
    model._mags_registry = HookRegistry(model, model_id)
    return model, tok
