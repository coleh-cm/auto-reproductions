"""Degeneracy test for CASteer (research-code skill: the method at its no-op
setting must reproduce the baseline EXACTLY).

CASteer's no-op setting is beta (strength) = 0: the erasure update
    c <- c - beta * <s, c> * s      (Eq. 6)
    c <- c - beta * s               (Eq. 4, constant mode)
collapses to c <- c for every patch, i.e. the steered output is bit-identical
to the baseline (vanilla) output. If this fails, the implementation is wrong
and any downstream gap is plumbing, not the method.
"""
import torch

from core.controller import CrossAttentionOutputSteering


def _unit_store(d):
    s = torch.randn(d)
    s = s / s.norm()
    return s, {0: {"down": [s.view(1, 1, d)]}}


def test_degeneracy_strength_zero_dotproduct_bitexact():
    """beta=0, dotproduct mode: steered output == input, bit-exact (float32)."""
    torch.manual_seed(0)
    d = 320
    _, store = _unit_store(d)
    ctrl = CrossAttentionOutputSteering(
        source_concepts=[store], target_concepts=[None], strength=0.0,
        device=torch.device("cpu"), intermediate_clipping=False,
        use_first_diffusion_step=True, num_layers=1, steering_mode='dotproduct',
    )
    c = torch.randn(2, 4, 1, d)
    out = ctrl.forward(c.clone(), diffusion_step=0, place_in_unet="down", block_index=0)
    assert torch.equal(out, c), "beta=0 dotproduct must be identity"


def test_degeneracy_strength_zero_constant_bitexact():
    """beta=0, constant mode: steered output == input, bit-exact (float32)."""
    torch.manual_seed(1)
    d = 320
    _, store = _unit_store(d)
    ctrl = CrossAttentionOutputSteering(
        source_concepts=[store], target_concepts=[None], strength=0.0,
        device=torch.device("cpu"), intermediate_clipping=True,
        use_first_diffusion_step=True, num_layers=1, steering_mode='constant',
    )
    c = torch.randn(2, 4, 1, d)
    out = ctrl.forward(c.clone(), diffusion_step=0, place_in_unet="down", block_index=0)
    assert torch.equal(out, c), "beta=0 constant must be identity"


def test_degeneracy_inactive_control_is_identity():
    """When the control is inactive, VectorControl.__call__ returns the input unchanged."""
    torch.manual_seed(2)
    d = 64
    _, store = _unit_store(d)
    ctrl = CrossAttentionOutputSteering(
        source_concepts=[store], target_concepts=[None], strength=2.0,
        device=torch.device("cpu"), intermediate_clipping=True,
        use_first_diffusion_step=True, num_layers=1, steering_mode='dotproduct',
    )
    ctrl.active = False
    c = torch.randn(2, 4, 1, d)
    out = ctrl(c, place_in_unet="down")
    assert torch.equal(out, c)


def test_degeneracy_rejects_nonzero_offset():
    """Negative: the degeneracy check (torch.equal vs the input) REJECTS a known-wrong
    output -- a steered output carrying a non-zero offset at beta=0. This is the same
    defect as mutations.json: degeneracy_offset (return vector + steering_delta + 0.001)
    and proves the bit-exact check can catch a broken no-op rather than passing silently."""
    torch.manual_seed(3)
    d = 64
    c = torch.randn(2, 4, 1, d)
    # beta=0 must be identity; a defect that adds a non-zero offset is NOT bit-identical.
    broken = c + 0.001
    assert not torch.equal(broken, c), (
        "degeneracy check must reject a non-zero offset at beta=0")
