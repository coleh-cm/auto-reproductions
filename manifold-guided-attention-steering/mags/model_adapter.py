"""Model-family adapter (HookRegistry, SPEC §5.3).

Exposes, per model family, the attention output projection ``W_O`` for each layer and a
view that reshapes its ``[bsz, seq, H*d_h]`` input to head-contiguous ``[bsz, seq, H, d_h]``.

The per-head attention output ``a_t^{(l,h)}`` is, by construction in every supported
transformers architecture, exactly the input to ``W_O`` reshaped to ``[..., H, d_h]``:
the layer computes ``context = (softmax(QK^T)V)`` shaped ``[bsz, H, seq, d_h]``, transposes
to ``[bsz, seq, H, d_h]`` and reshapes (flatten last two dims, C-order) to
``[bsz, seq, H*d_h]`` before ``W_O``. The reshape is head-major, so ``view[..., H, d_h]``
recovers heads losslessly. (Llama/Llama-arch: ``attn_output.view(bsz,q_len,H,dh).transpose(1,2)``
then ``.reshape(bsz,q_len,H*dh)``; Gemma-4 / GPT-OSS follow the same convention; GPT-2
``c_proj`` input is ``[bsz, seq, H*d_h]`` after the same reshape.)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
import torch.nn as nn

from . import config


@dataclass
class LayerHeadLayout:
    """Static geometry of one model's attention stack."""
    n_layers: int
    n_heads: int
    head_dim: int
    monitored_layers: list


class HookRegistry:
    """Registers/unregisters forward pre-hooks on ``W_O`` of monitored layers.

    The hook receives the pre-projection input ``x`` of shape ``[bsz, seq, H*d_h]`` and
    may return a modified tensor of the same shape. Hook callbacks are called with
    ``(layer_index, x)`` and must return either ``None`` (pass-through) or a modified
    ``[bsz, seq, H*d_h]`` tensor. The registry handles the head-reshape bookkeeping
    and passes head-contiguous ``[bsz, seq, H, d_h]`` views to callbacks so a controller
    can index per head without re-deriving the layout.
    """

    def __init__(self, model: nn.Module, model_id: str):
        self.model = model
        self.model_id = model_id
        spec = config.get_model_spec(model_id)
        self.spec = spec
        self.layout = _infer_layout(model, model_id, spec)
        self._handles = []
        # set during a forward pass so a controller can read current step context
        self.callback: Callable | None = None

    # -- geometry --
    def _infer_layout(self):  # pragma: no cover (helper relocated below)
        ...

    def get_o_proj(self, layer: int) -> nn.Module:
        spec = self.spec
        return spec["o_proj_path"](self.model, layer)

    # -- hook lifecycle --
    def attach(self, callback: Callable, layers=None):
        """Attach ``callback`` as a pre-hook on ``W_O`` of the requested layers.

        ``callback(layer: int, x_heads: Tensor[bsz,seq,H,d_h]) -> Optional[Tensor[bsz,seq,H,d_h]]``
        receives a head-contiguous *view*; if it returns a tensor, that tensor (already
        head-contiguous) is reshaped back to ``[bsz, seq, H*d_h]`` and fed to ``W_O``.

        ``layers`` selects which layers get hooks:
          - None (default): the monitored layer subset (MAGS / ITI / capture — these
            are targeted methods that only touch their selected heads; monitoring
            only top-K heads keeps MAGS overhead O(K·k·d_h), tex:L376).
          - "all": every layer in the stack (Angular Steering — the paper applies "a
            fixed 2D rotation ... across all layers", tex:L394; SPEC §4.16/§5.5).
          - an explicit list of layer indices.
        """
        self.detach()
        self.callback = callback
        if layers is None:
            target_layers = self.layout.monitored_layers
        elif layers == "all":
            target_layers = list(range(self.layout.n_layers))
        else:
            target_layers = [l for l in layers if l < self.layout.n_layers]
        for layer in target_layers:
            o_proj = self.get_o_proj(layer)
            handle = o_proj.register_forward_pre_hook(
                self._make_hook(layer), prepend=True, with_kwargs=True
            )
            self._handles.append(handle)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.callback = None

    def _make_hook(self, layer: int):
        registry = self

        def hook(module, args, kwargs):
            # o_proj is called as o_proj(x) (positional) in all supported families.
            x = args[0] if args else kwargs.get("hidden_states") or kwargs.get("input")
            if x is None:
                return None
            x = x  # [bsz, seq, H*d_h]
            bsz, seq, flat = x.shape
            H = registry.layout.n_heads
            # SPEC §7 (round-19): head_dim is NOT constant across all layers on every
            # model. The REAL google/gemma-4-E4B-it has TWO attention geometries:
            # sliding_attention layers use 8 heads x 256 (o_proj.in=2048), full_attention
            # layers use 8 heads x 512 (o_proj.in=4096). num_heads (8) is constant;
            # head_dim varies. Deriving dh = flat // H per layer (from the actual W_O
            # input) is correct for every supported family (Llama 32x128, GPT-OSS
            # 64x64, distilgpt2 12x64, Gemma4 8x{256,512}) and removes the prior
            # single-head_dim assumption that crashed the assert on Gemma4 full-attention
            # layers (4096 != 8*256). Verified by tests/test_gemma4_adapter.py.
            assert flat % H == 0, (
                f"layer {layer}: W_O input flat={flat} not divisible by n_heads={H}"
            )
            dh = flat // H
            x_heads = x.view(bsz, seq, H, dh)
            if registry.callback is None:
                return None
            modified = registry.callback(layer, x_heads)
            if modified is None:
                return None
            # flatten back to head-major [bsz, seq, H*d_h]
            new_flat = modified.reshape(bsz, seq, H * dh).to(x.dtype)
            # preserve device/contiguity
            if args:
                return (new_flat,) + args[1:], kwargs
            return (new_flat,), kwargs

        return hook


def _infer_layout(model: nn.Module, model_id: str, spec: dict) -> LayerHeadLayout:
    """Prefer config.json values read from the loaded model; fall back to registry.

    For multi-modal families (Gemma-4 loads as ``Gemma4ForConditionalGeneration``),
    the top-level ``model.config`` has null geometry; the real numbers live in
    ``config.text_config`` (round-19: verified against the real
    ``google/gemma-4-E4B-it`` repo — text_config gives 42 layers / 8 heads /
    head_dim 256 for the sliding layers). ``head_dim`` here is the SLIDING-layer
    default; the hook derives the actual per-layer head_dim from the W_O input
    (Gemma-4 full-attention layers use head_dim 512), so this value is only a
    fallback / empty-buffer shape, never the reshape authority.
    """
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None)
    n_layers = (_first(text_cfg, ["num_hidden_layers", "n_layer"])
               or _first(cfg, ["num_hidden_layers", "n_layer"]) or spec["layers"])
    n_heads = (_first(text_cfg, ["num_attention_heads", "num_heads", "n_head"])
               or _first(cfg, ["num_attention_heads", "num_heads", "n_head"])
               or spec["heads"])
    # SPEC §7: read head_dim from config, do NOT derive as hidden/H (Gemma-4: 8*256 != 2560).
    head_dim = (
        getattr(text_cfg, "head_dim", None)
        or getattr(cfg, "head_dim", None)
        or getattr(cfg, "d_head", None)
        or (spec["head_dim"] if spec.get("head_dim") else None)
    )
    if head_dim is None:
        hidden = (_first(text_cfg, ["hidden_size", "d_model"])
                  or _first(cfg, ["hidden_size", "n_embd", "d_model"])
                  or (n_heads * spec["head_dim"]))
        head_dim = hidden // n_heads
    monitored = list(spec["monitored_layers"])
    # clamp monitored layers to the model's actual depth (smoke models etc.)
    monitored = [l for l in monitored if l < n_layers]
    return LayerHeadLayout(int(n_layers), int(n_heads), int(head_dim), monitored)


def _first(cfg, keys):
    if cfg is None:
        return None
    for k in keys:
        v = getattr(cfg, k, None)
        if v is not None:
            return v
    return None
