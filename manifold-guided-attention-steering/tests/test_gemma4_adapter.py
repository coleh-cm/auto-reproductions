"""Real-model adapter validation for the Gemma-4 arm (round-19).

These tests build the REAL ``google/gemma-4-E4B-it`` architecture on ``torch``'s
``meta`` device (config-only — NO 16 GB weight download, NO GPU) and assert the
MAGS hook infrastructure actually resolves and reshapes on the real model. They
catch exactly the two bugs 18 rounds of distilgpt2-only smoke could not, because
distilgpt2 (``model.transformer.h[l].attn.c_proj``, uniform 12x64) shares neither
Gemma-4's module path nor its heterogeneous head_dim:

  1. ``_gemma4_o_proj`` must resolve ``W_O`` on the multimodal
     ``Gemma4ForConditionalGeneration`` (text stack at
     ``model.model.language_model.layers[l].self_attn.o_proj``). The prior
     version checked ``hasattr(model, "language_model")`` on the TOP object
     (False) then ``model.layers`` (absent) and raised
     ``AttributeError: 'Gemma4ForConditionalGeneration' object has no attribute
     'layers'`` — MAGS could not resolve a single hook target on the real model.

  2. The hook must reshape the W_O input with a PER-LAYER head_dim. The real
     Gemma-4 has TWO attention geometries: ``sliding_attention`` layers are
     8 heads x 256 (o_proj.in=2048), ``full_attention`` layers are 8 heads x 512
     (o_proj.in=4096). num_heads (8) is constant; head_dim varies. The prior
     single-head_dim assert ``flat == H*dh`` (4096 == 8*256) crashed on the
     full-attention layers — and layer 41 (full_attention) is in the SPEC's
     default Gemma monitored set [10, 21, 31, 41].

These tests need network only to fetch ``config.json`` (a few KB); they never
download weights. They are skipped (not failed) when the config cannot be
fetched (offline gate / no network), so a no-network gate host is not penalised
for an environment limitation — but on any host with the config cached or
network, the real-model contract is enforced.
"""
from __future__ import annotations
import pytest
import numpy as np

_HAS = None  # cached (model_on_meta, cfg)


def _load_meta():
    """Build the real Gemma-4 model on meta device (no weights). Cached per session."""
    global _HAS
    if _HAS is not None:
        return _HAS
    import torch
    try:
        from transformers import Gemma4ForConditionalGeneration, Gemma4Config
    except Exception as e:  # pragma: no cover
        pytest.skip(f"transformers lacks Gemma4 classes: {e!r}")
    try:
        cfg = Gemma4Config.from_pretrained("google/gemma-4-E4B-it")
        with torch.device("meta"):
            model = Gemma4ForConditionalGeneration(cfg)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot fetch/build real Gemma-4 config (offline?): {e!r}")
    _HAS = (model, cfg)
    return _HAS


def test_gemma4_o_proj_resolves_on_real_multimodal_model():
    """Bug 1: the real repo loads as Gemma4ForConditionalGeneration; the text
    stack is at model.model.language_model.layers[l].self_attn.o_proj."""
    from mags.config import _gemma4_o_proj, get_model_spec
    model, cfg = _load_meta()
    spec = get_model_spec("google/gemma-4-E4B-it")
    # must resolve on BOTH a sliding and a full-attention layer without raising
    op_slide = _gemma4_o_proj(model, 0)     # sliding_attention
    op_full = _gemma4_o_proj(model, 41)    # full_attention (in default monitored set)
    assert op_slide is not None and op_full is not None
    # the resolved modules must be the REAL attention output projections
    lm = model.model.language_model
    assert op_slide is lm.layers[0].self_attn.o_proj
    assert op_full is lm.layers[41].self_attn.o_proj


def test_gemma4_layout_reads_text_config():
    """The multimodal top config has null geometry; the layout must read
    text_config -> 42 layers, 8 heads (head_dim is per-layer, see next test)."""
    from mags.model_adapter import _infer_layout
    from mags.config import get_model_spec
    model, cfg = _load_meta()
    spec = get_model_spec("google/gemma-4-E4B-it")
    lay = _infer_layout(model, "google/gemma-4-E4B-it", spec)
    assert lay.n_layers == 42
    assert lay.n_heads == 8           # constant across sliding/full layers
    assert lay.monitored_layers == [10, 21, 31, 41]   # SPEC §4.5 default, all < 42


def test_gemma4_per_layer_head_dim_heterogeneous():
    """Bug 2: head_dim is NOT constant. Sliding layers are 8x256 (o_proj.in 2048);
    full-attention layers are 8x512 (o_proj.in 4096). The default monitored set
    [10,21,31,41] mixes both (41 is full_attention)."""
    from mags.config import _gemma4_o_proj
    model, cfg = _load_meta()
    tc = cfg.text_config
    full_idx = [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]
    slide_idx = [i for i, t in enumerate(tc.layer_types) if t == "sliding_attention"]
    assert 41 in full_idx and 10 in slide_idx and 21 in slide_idx and 31 in slide_idx
    # sliding: 8 heads x 256
    assert _gemma4_o_proj(model, 10).in_features == 8 * 256
    # full: 8 heads x 512
    assert _gemma4_o_proj(model, 41).in_features == 8 * 512


def test_gemma4_hook_reshapes_per_layer_head_dim():
    """The hook derives dh = flat // H per layer, so it does NOT crash on the
    full-attention layer 41 (flat 4096, H 8 -> dh 512). The old assert
    `flat == H*dh` (with the layout's single head_dim 256) would crash here."""
    import torch
    from mags.model_adapter import HookRegistry
    from mags.config import _gemma4_o_proj
    model, cfg = _load_meta()
    reg = HookRegistry(model, "google/gemma-4-E4B-it")
    H = reg.layout.n_heads
    seen = {}

    def cb(layer, x_heads):
        bsz, seq, Hc, dh = x_heads.shape
        seen[layer] = (Hc, dh)
        assert Hc == H
        return None

    reg.attach(cb, layers=[10, 41])
    # drive the REAL hook closure (_make_hook) directly with a dummy W_O input at
    # each layer's real flat dim. (Calling op(x) is not viable on a meta-device
    # model — Linear forward needs real weights — so we invoke the hook closure
    # the registry registers, which is the code under test.)
    for layer in (10, 41):
        op = _gemma4_o_proj(model, layer)
        flat = op.in_features
        x = torch.zeros(1, 1, flat)
        hook_fn = reg._make_hook(layer)
        hook_fn(op, (x,), {})
    reg.detach()
    # sliding layer 10 -> dh 256; full-attention layer 41 -> dh 512
    assert seen[10] == (8, 256)
    assert seen[41] == (8, 512)


def test_gemma4_as_bank_fits_one_plane_per_head_dim():
    """AS must fit one plane per distinct head_dim so it does not crash pooling
    256- and 512-dim activations (the monitored set [10,21,31,41] has both)."""
    from mags.baselines import fit_as_bank
    from mags.manifold import HeadProblemActivations
    rng = np.random.default_rng(7)
    # two layers, two head_dims (256 and 512) like Gemma-4's mix
    all_acts = {}
    for l, dh in ((10, 256), (41, 512)):
        ha = HeadProblemActivations()
        for pid in ("p0", "p1", "p2"):
            ha.correct[pid] = [rng.normal(size=(5, dh)).astype(np.float32)
                               for _ in range(2)]
            ha.incorrect[pid] = [rng.normal(size=(5, dh)).astype(np.float32)
                                 + rng.normal(size=dh) * 3
                                 for _ in range(2)]
        all_acts[(l, 0)] = ha
    asb = fit_as_bank(all_acts, ["p0", "p1", "p2"], model_id="google/gemma-4-E4B-it",
                      benchmark="b", angle_deg=30, layers_monitored=[10, 41])
    dhs = sorted(p.dh for p in asb.planes)
    assert dhs == [256, 512], f"AS must emit one plane per head_dim, got {dhs}"
    for p in asb.planes:
        assert p.d_feat.shape == (p.dh,)
        assert p.d_pc0.shape == (p.dh,)
        # orthonormal basis
        assert abs(float(p.d_feat @ p.d_pc0)) < 1e-5
        assert abs(float(np.linalg.norm(p.d_feat) - 1.0)) < 1e-5
        assert abs(float(np.linalg.norm(p.d_pc0) - 1.0)) < 1e-5


def test_gemma4_as_controller_selects_plane_by_head_dim():
    """The controller picks the plane matching the layer's head_dim and rotates;
    a layer with an unseen head_dim pass-throughs (identity at theta=0 semantics)."""
    import torch
    from mags.baselines import fit_as_bank, AngularSteeringController
    from mags.manifold import HeadProblemActivations
    rng = np.random.default_rng(8)
    all_acts = {}
    for l, dh in ((10, 256), (41, 512)):
        ha = HeadProblemActivations()
        for pid in ("p0", "p1"):
            ha.correct[pid] = [rng.normal(size=(5, dh)).astype(np.float32) for _ in range(2)]
            ha.incorrect[pid] = [rng.normal(size=(5, dh)).astype(np.float32) + 1.0
                                 for _ in range(2)]
        all_acts[(l, 0)] = ha
    asb = fit_as_bank(all_acts, ["p0", "p1"], model_id="google/gemma-4-E4B-it",
                      benchmark="b", angle_deg=30, layers_monitored=[10, 41])
    ctrl = AngularSteeringController(asb, angle_deg=30.0)
    # a 256-dim head output (layer 10 geometry) -> rotated by the 256 plane
    x256 = torch.from_numpy(rng.normal(size=(1, 1, 8, 256)).astype(np.float32))
    out256 = ctrl(10, x256)
    assert out256 is not None and out256.shape == (1, 1, 8, 256)
    # a 512-dim head output (layer 41 geometry) -> rotated by the 512 plane
    x512 = torch.from_numpy(rng.normal(size=(1, 1, 8, 512)).astype(np.float32))
    out512 = ctrl(41, x512)
    assert out512 is not None and out512.shape == (1, 1, 8, 512)
    # an unseen head_dim (e.g. 128) -> pass-through (None), not a crash
    x128 = torch.from_numpy(rng.normal(size=(1, 1, 8, 128)).astype(np.float32))
    assert ctrl(5, x128) is None
