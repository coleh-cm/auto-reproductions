"""Degeneracy test (research-code skill): the method at its no-op setting must
reproduce the baseline EXACTLY.

For MAGS the no-op settings are:
  (a) alpha = 0  -> Eq.(9) becomes a_t - 0 = a_t  (correction is identically zero)
  (b) threshold = +inf  -> Eq.(8) never fires  (no head is ever modified)

Under both, MAGS generation must be TOKEN-IDENTICAL to the unsteered baseline on the
same prompts. This is the cheapest real correctness evidence: a single mismatch means
the hook is corrupting the forward pass even when it claims not to. It runs on the
small smoke model (distilgpt2) so it is CPU-fast and needs no gated model.

This test also doubles as the "decode-only steering" check (SPEC §4.9): prefill
positions are never modified, so the first generated token (whose logits come from the
prefill forward) must also match the unsteered run.
"""
import os
import numpy as np
import pytest
import torch

# Skip the whole module if no smoke model is available in the sandbox.
pytestmark = pytest.mark.skipif(
    os.environ.get("MAGS_SMOKE_MODEL") == "none",
    reason="smoke model disabled",
)

SMOKE_MODEL = os.environ.get("MAGS_SMOKE_MODEL", "distilgpt2")


def _load():
    from mags.models import load_model
    return load_model(SMOKE_MODEL)


def _make_bank(model, model_id, k=2, q=95):
    """Fit a tiny real manifold from activations captured on a few prompts.

    We capture real per-head activations (real forward path) and label traces
    'correct'/'incorrect' by an arbitrary deterministic hash so the contrastive set
    is non-empty. The numbers are meaningless (smoke), but the manifold and the
    steering hook are the real code path.
    """
    from mags.model_adapter import HookRegistry
    from mags.capture import _CaptureHook
    from mags.manifold import (
        HeadProblemActivations, difference_matrix, fit_basis, global_correct_centroid,
        fit_manifold_bank, HeadManifold,
    )
    reg = model._mags_registry
    tok = None
    prompts = [
        "The answer to 2+2 is",
        "def add(a, b):\n    return",
        "Once upon a time",
        "The capital of France is",
    ]
    ha_per_head = {}
    pids = [f"p{i}" for i in range(len(prompts))]
    # capture activations for each prompt (a single greedy "trace" per prompt)
    import torch
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = model._mags_tokenizer(p, return_tensors="pt").input_ids.to(model.device) \
                if hasattr(model, "_mags_tokenizer") else None
            cap = _CaptureHook(reg.layout.monitored_layers, reg.layout.n_heads,
                               reg.layout.head_dim)
            reg.attach(cap)
            try:
                out = model.generate(ids, max_new_tokens=12, do_sample=False,
                                      pad_token_id=model._mags_tokenizer.eos_token_id,
                                      use_cache=True)
            finally:
                reg.detach()
            acts = cap.stacked()  # {layer: [T,H,dh]}
            # label by parity: even prompts 'correct', odd 'incorrect'
            for l in reg.layout.monitored_layers:
                A = acts[l]   # [T,H,dh]
                for h in range(reg.layout.n_heads):
                    key = (l, h)
                    if key not in ha_per_head:
                        ha_per_head[key] = HeadProblemActivations()
                    a_head = A[:, h, :]
                    if i % 2 == 0:
                        ha_per_head[key].correct.setdefault(pids[i], []).append(a_head)
                    else:
                        ha_per_head[key].incorrect.setdefault(pids[i], []).append(a_head)
    # need both classes for each problem -> restructure: each pid has only one class.
    # Instead, give every problem both a 'correct' and 'incorrect' trace by duplicating
    # the activation with a small perturbation.
    ha_per_head = {}
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = model._mags_tokenizer(p, return_tensors="pt").input_ids.to(model.device)
            cap = _CaptureHook(reg.layout.monitored_layers, reg.layout.n_heads,
                               reg.layout.head_dim)
            reg.attach(cap)
            try:
                out = model.generate(ids, max_new_tokens=12, do_sample=False,
                                      pad_token_id=model._mags_tokenizer.eos_token_id,
                                      use_cache=True)
            finally:
                reg.detach()
            acts = cap.stacked()
            for l in reg.layout.monitored_layers:
                A = acts[l]
                for h in range(reg.layout.n_heads):
                    key = (l, h)
                    if key not in ha_per_head:
                        ha_per_head[key] = HeadProblemActivations()
                    base = A[:, h, :]
                    ha_per_head[key].correct.setdefault(pids[i], []).append(base)
                    ha_per_head[key].incorrect.setdefault(pids[i], []).append(
                        base + 0.5 * rng.normal(size=base.shape))
    bank = fit_manifold_bank(
        ha_per_head, pids[:3], pids[3:], model_id=model_id, benchmark="smoke",
        k=k, q=q, K=3, alpha=1.0, layers_monitored=reg.layout.monitored_layers,
        split_seed=42,
    )
    return bank


def test_degeneracy_alpha_zero_matches_unsteered():
    """MAGS with alpha=0 must be token-identical to the unsteered baseline."""
    model, tok = _load()
    model._mags_tokenizer = tok
    bank = _make_bank(model, SMOKE_MODEL)
    from mags.steering import MAGSController, NoOpController
    from mags.generation import generate
    prompts = [
        "The answer to 2+2 is",
        "def add(a, b):\n    return",
        "Once upon a time there was a",
    ]
    for p in prompts:
        txt_base, ids_base, _ = generate(model, tok, p, NoOpController(), max_new_tokens=16)
        ctrl = MAGSController(bank, alpha=0.0)   # no-op: correction term multiplied by 0
        txt_mag, ids_mag, _ = generate(model, tok, p, ctrl, max_new_tokens=16)
        assert ids_base.tolist() == ids_mag.tolist(), \
            f"alpha=0 mismatch on prompt {p!r}: MAGS diverged from unsteered"


def test_degeneracy_untriggerable_matches_unsteered():
    """MAGS with threshold=+inf (never fires) must be token-identical to unsteered."""
    model, tok = _load()
    model._mags_tokenizer = tok
    bank = _make_bank(model, SMOKE_MODEL)
    from mags.steering import MAGSController, NoOpController
    from mags.generation import generate
    # push every selected head's threshold to +inf so Eq.(8) never fires
    for h in bank.selected_heads:
        bank.heads[tuple(h)].threshold = float("inf")
    prompts = [
        "The answer to 2+2 is",
        "def add(a, b):\n    return",
        "Once upon a time there was a",
    ]
    for p in prompts:
        txt_base, ids_base, _ = generate(model, tok, p, NoOpController(), max_new_tokens=16)
        ctrl = MAGSController(bank, alpha=1.0)   # full strength, but never triggers
        txt_mag, ids_mag, _ = generate(model, tok, p, ctrl, max_new_tokens=16)
        assert ids_base.tolist() == ids_mag.tolist(), \
            f"untriggerable mismatch on prompt {p!r}: MAGS diverged from unsteered"


def test_prefill_not_steered():
    """Decode-only steering (SPEC §4.9): the first generated token (from prefill
    logits) is unchanged by a controller that only modifies decode steps, even with
    a forced trigger on every head."""
    model, tok = _load()
    model._mags_tokenizer = tok
    bank = _make_bank(model, SMOKE_MODEL)
    from mags.steering import MAGSController, NoOpController
    from mags.generation import generate
    # force-trigger every selected head by setting threshold to -inf
    for h in bank.selected_heads:
        bank.heads[tuple(h)].threshold = float("-inf")
    p = "The answer to 2+2 is"
    _, ids_base, _ = generate(model, tok, p, NoOpController(), max_new_tokens=4)
    ctrl = MAGSController(bank, alpha=1.0)
    _, ids_mag, _ = generate(model, tok, p, ctrl, max_new_tokens=4)
    # first generated token comes from prefill (seq>1) forward -> controller passed through
    assert ids_base.tolist()[0] == ids_mag.tolist()[0], \
        "prefill was steered; controller must pass-through seq>1 (SPEC §4.9)"


def test_hook_return_value_is_consumed():
    """The degeneracy tests above pass even if the W_O pre-hook silently ignored the
    controller's return value (both no-op settings return None regardless), so they
    do NOT by themselves prove the active path is wired up. This test does: a
    controller that returns a clearly-modified tensor (zeros) MUST change the
    generated tokens vs the unsteered baseline. A silent hook-ignored bug (e.g. the
    registry not feeding the returned tensor back into W_O) would leave tokens
    identical and fail here. This is the invariant the paper's Algorithm 1 line 9
    ("Replace a_t with a_tilde in the input to W_O") implies: the correction must
    actually reach W_O."""
    model, tok = _load()
    model._mags_tokenizer = tok
    import torch
    from mags.generation import generate
    from mags.steering import NoOpController

    class GarbageController:
        """Returns a zeroed head tensor on every decode step. If the hook return
        value is consumed, this corrupts generation and diverges from baseline."""
        hook_layers = None

        def __call__(self, layer, x_heads):
            if x_heads.shape[1] != 1:
                return None  # prefill pass-through
            return torch.zeros_like(x_heads)

    p = "Once upon a time there was a"
    _, ids_base, _ = generate(model, tok, p, NoOpController(), max_new_tokens=16)
    _, ids_garb, _ = generate(model, tok, p, GarbageController(), max_new_tokens=16)
    assert ids_base.tolist() != ids_garb.tolist(), (
        "A controller that zeros every head's W_O input did NOT change generation; "
        "the W_O pre-hook return value is being ignored (Algorithm 1 line 9 broken)."
    )


def test_active_correction_changes_tokens():
    """A force-triggered MAGS correction with a non-trivial magnitude must change
    the generated tokens vs the unsteered baseline. This proves the ACTIVE path
    (Eq.9 correction actually applied), not just the no-op path, is exercised end
    to end. Without this, a manifold that computes the right B/mu_c but whose
    correction never reaches W_O would pass the degeneracy tests and still be a
    no-op in practice. We make the correction non-trivial by pushing mu_c away from
    the activations so the centered vector v = a - mu_c is large (Eq.9 scales
    alpha * B^T B v)."""
    model, tok = _load()
    model._mags_tokenizer = tok
    import numpy as np
    bank = _make_bank(model, SMOKE_MODEL)
    from mags.steering import MAGSController, NoOpController
    from mags.generation import generate
    # force-trigger every selected head and push mu_c far so the correction is large
    for h in bank.selected_heads:
        bank.heads[tuple(h)].threshold = float("-inf")
        bank.heads[tuple(h)].mu_c = (bank.heads[tuple(h)].mu_c + 50.0).astype(np.float32)
    p = "Once upon a time there was a"
    _, ids_base, _ = generate(model, tok, p, NoOpController(), max_new_tokens=16)
    _, ids_mag, _ = generate(model, tok, p, MAGSController(bank, alpha=1.0),
                             max_new_tokens=16)
    assert ids_base.tolist() != ids_mag.tolist(), (
        "A force-triggered MAGS correction with a large (a-mu_c) did not change "
        "generation; the active correction (Eq.9) is not reaching W_O."
    )
