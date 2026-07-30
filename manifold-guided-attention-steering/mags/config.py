"""Configuration and model-family registry for the MAGS reproduction.

Defaults for parameters the paper leaves UNSTATED are recorded here (see SPEC.md §4).
Every default is a reproduction decision, NOT a paper fact; the paper's own choices are
spelled out in arms.json / SPEC.md.
"""
from __future__ import annotations

# --- Hyperparameter defaults (paper-stated where given, reproduction defaults otherwise) ---
# SPEC §4.1: k never given; only hint = "top-4 principal components" (tex:L564). -> k=4.
DEFAULT_K_RANK = 4
# SPEC §4.2: q (Eq.8 percentile) never given. -> q=95.
DEFAULT_Q_PERCENTILE = 95
# SPEC §4.3: K (monitored heads) never given for main tables; ablation top-1/top-3 (tex:L605).
# Both reach 0.530 at alpha=1.0. -> K=3.
DEFAULT_K_HEADS = 3
# SPEC §4.4: alpha for main-table MAGS rows (except MATH-500/Llama) never given; ablation best=1.0.
DEFAULT_ALPHA = 1.0
# SPEC §4.10: trace sampling config unstated. -> temp=1.0, top_p=0.95, n=8.
DEFAULT_N_SAMPLES = 8
DEFAULT_SAMPLING_TEMP = 1.0
DEFAULT_SAMPLING_TOP_P = 0.95
# SPEC §4.9: eval decoding config unstated; HumanEval is pass@1. -> greedy, one completion.
DEFAULT_DO_SAMPLE = False
DEFAULT_MAX_NEW_TOKENS = {"math": 1024, "code": 512}
# SPEC §4.19: only bootstrap seed (=42) given. -> seed=42 everywhere.
DEFAULT_SEED = 42
# SPEC §6 (bootstrap): percentile method, B=10000, seed=42 (tex:L699-700).
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42

# SPEC §4.15: ITI adaptation.
ITI_K_GRID = [24, 48, 96]
ITI_ALPHA_GRID = [0.5, 1.0, 5.0]
ITI_DEFAULT_K = 96
ITI_DEFAULT_ALPHA = 0.5
# SPEC §4.16: Angular Steering adaptation.
AS_ANGLE_GRID = list(range(0, 360, 30))
AS_DEFAULT_ANGLE_DEG = 30
# SPEC §4.17: Contrastive Decoding.
CD_DEFAULT_ALPHA_PLAUS = 0.1
# CD amateur log-prob coefficient. Li et al. 2023 (li2023contrastive, the paper
# the MAGS paper cites at tex:L396) defines CD-score = log p_exp - log p_ama
# (Eq.3, coefficient 1 on BOTH log-probs); there is NO beta parameter. The
# MAGS paper does not restate the objective, so the CD paper is authoritative.
# The amateur temperature tau (a SEPARATE op, softmax(logits_ama/tau); =1.0
# for OPT/Llama-class) is not a coefficient on log p. We use 1.0 (the plain
# log-ratio); tau=1.0 is a no-op, so no amateur-temperature machinery is needed.
CD_DEFAULT_BETA = 1.0
CD_AMATEUR = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-4-E4B-it": "google/gemma-3-1b-it",
}

# --- Model-family registry (SPEC §2, §4.22, §5.3) ---
# head_dim is read from config at runtime where possible; these are the documented values
# used for sanity checks and for the smoke fallback model.
MODEL_REGISTRY = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "family": "llama",
        "layers": 32,
        "heads": 32,
        "head_dim": 128,
        "monitored_layers": [8, 16, 24, 31],   # SPEC §4.5 / tex:L296
        "precision": "float16",
        "o_proj_path": lambda m, l: m.layers[l].self_attn.o_proj,
    },
    "google/gemma-4-E4B-it": {
        "family": "gemma4",
        "layers": 42,
        "heads": 8,
        "head_dim": 256,                          # SPEC §7: d_h != hidden/H
        "monitored_layers": [10, 21, 31, 41],     # SPEC §4.5 default
        "precision": "bfloat16",
        # text stack: SPEC §5.3; resolved against transformers at load time.
        "o_proj_path": lambda m, l: _gemma4_o_proj(m, l),
    },
    "openai/gpt-oss-20b": {
        "family": "gptoss",
        "layers": 24,
        "heads": 64,
        "head_dim": 64,
        "monitored_layers": [6, 12, 18, 23],     # SPEC §4.5 default
        "precision": None,                        # native (mxfp4 experts)
        "o_proj_path": lambda m, l: m.layers[l].self_attn.o_proj,
    },
    # Smoke / test fallback: a small open model that runs on CPU. NOT a paper model.
    "distilgpt2": {
        "family": "gpt2",
        "layers": 6,
        "heads": 12,
        "head_dim": 64,
        "monitored_layers": [0, 2, 5],
        "precision": "float32",
        "o_proj_path": lambda m, l: m.transformer.h[l].attn.c_proj,
    },
    "sshleifer/tiny-gpt2": {
        "family": "gpt2",
        "layers": 2,
        "heads": 2,
        "head_dim": 1,
        "monitored_layers": [0, 1],
        "precision": "float32",
        "o_proj_path": lambda m, l: m.transformer.h[l].attn.c_proj,
    },
}


def _gemma4_o_proj(model, layer):
    """Resolve the attention output projection for the Gemma-4 text stack.

    Verified round-19 against the REAL ``google/gemma-4-E4B-it`` repo (config-only,
    no weights, ``transformers`` 5.14.1 ``AutoModelForCausalLM`` →
    ``Gemma4ForConditionalGeneration``). The text stack lives at
    ``model.model.language_model.layers[l].self_attn.o_proj`` (the multimodal wrapper
    ``Gemma4ForConditionalGeneration`` has a ``Gemma4Model`` at ``.model`` whose
    ``.language_model`` is the ``Gemma4TextModel``). A text-only ``Gemma4ForCausalLM``
    load exposes ``model.model.layers[l]``; a bare ``Gemma4TextModel`` exposes
    ``model.layers[l]``. The prior version checked ``hasattr(model, "language_model")``
    on the TOP object — which is False for the multimodal wrapper — and then
    ``model.layers[l]`` — which does not exist either — so it raised
    ``AttributeError: 'Gemma4ForConditionalGeneration' object has no attribute 'layers'``
    and MAGS could not resolve a single hook target on the real Gemma model. The paths
    below are tried most-specific-first and validated by tests/test_gemma4_adapter.py
    (builds the real config on ``meta`` device, no 16 GB download).
    """
    # multimodal Gemma4ForConditionalGeneration: model.model.language_model.layers[l]
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer].self_attn.o_proj
    # text-only Gemma4ForCausalLM: model.model.layers[l]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer].self_attn.o_proj
    # bare Gemma4TextModel: model.layers[l]
    if hasattr(model, "layers"):
        return model.layers[layer].self_attn.o_proj
    # legacy: a wrapper that already exposes language_model at top level
    if hasattr(model, "language_model"):
        return model.language_model.layers[layer].self_attn.o_proj
    raise AttributeError(
        f"could not resolve Gemma-4 o_proj on {type(model).__name__}: expected "
        f"model.model.language_model.layers[l].self_attn.o_proj "
        f"(multimodal) or model.model.layers[l].self_attn.o_proj (text-only)"
    )


def get_model_spec(model_id: str) -> dict:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"unknown model_id {model_id!r}; add it to mags.config.MODEL_REGISTRY")
    return MODEL_REGISTRY[model_id]


def monitored_layers_for(model_id: str) -> list:
    return list(MODEL_REGISTRY[model_id]["monitored_layers"])
