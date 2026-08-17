"""Method-core invariants for CASteer (SPEC E1-E12, claims.json claim `house`).

Pure-CPU, no model download. These assert what the paper's *equations* imply:
- Householder norm preservation under beta=2, unit s (claim `house`,
  experiments.tex:21-22).
- Unit-norm construction f_norm = v/||v|| (SPEC U1, method_2.tex:81 / 125).
- Eq.6 no-clip == (I - beta s s^T) c on the conditional CFG half.
- Eq.7 clip steers only positive-projection patches.
- Eq.4 constant-alpha mode (SPEC U13).
- Eq.9 multi-concept average is the elementwise mean, not re-normalized (U9).
- CFG: only the conditional half (batch index B//2) is steered.
"""
import math

import torch

from core.controller import (
    CrossAttentionOutputSteering,
    average_concept_vectors,
    EPS,
)


def _unit_store(s, d):
    """Build a single-step/down/single-block store from a unit vector s in R^d."""
    return {0: {"down": [s.view(1, 1, d)]}}


def test_householder_norm_preservation():
    """SPEC claim `house` (experiments.tex:21-22): (I - 2 s s^T) is a Householder
    reflection, so it preserves ||c||_2. Check 100 random unit s and random c
    in dims {320,640,1280}, float64, max error < 1e-5."""
    torch.manual_seed(0)
    for d in (320, 640, 1280):
        for _ in range(100):
            s = torch.randn(d, dtype=torch.float64)
            s = s / s.norm()
            c = torch.randn(d, dtype=torch.float64)
            c_new = c - 2.0 * (s @ c) * s  # (I - 2 s s^T) c
            assert abs(c_new.norm().item() - c.norm().item()) < 1e-5


def test_construction_is_unit_norm():
    """SPEC U1: f_norm(v) = v/||v|| (the only reading consistent with the
    projection-length and Householder statements). Construction yields a unit vector."""
    torch.manual_seed(1)
    raw = torch.randn(640)
    sv = raw / raw.norm().clamp(min=1e-8)
    assert abs(sv.norm().item() - 1.0) < 1e-6


def test_eq6_noclip_matches_matrix_form():
    """Eq.6 (method_2.tex:126-130), no clipping: the elementwise no-clip path AND
    the precomputed matrix form (Eq.5, P = I - beta * s * s^T) BOTH equal
    (I - beta * s * s^T) c. With beta=2 and a unit s this is the Householder
    reflection, so ||c|| is preserved.

    This exercises BOTH code paths that consume `strength`:
      - steer_with_clipping (intermediate_clipping=False) reads self.strength
        directly, so it catches a wrong strength on the elementwise path;
      - steer_matrix_form reads the precomputed P = I - beta * s * s^T (controller.py
        `res = strength * (v @ pinv(v))`), so it catches the half_strength_projection
        mutation, which scales `res` by 0.5 -> P = I - 1 * s * s^T: the matrix form
        then yields (I - s s^T) c, which neither matches (I - 2 s s^T) c nor
        preserves ||c||. A test that only checks the elementwise path is blind to
        that mutation, because steer_with_clipping never reads P.
    """
    torch.manual_seed(2)
    d = 320
    s = torch.randn(d)
    s = s / s.norm()
    store = _unit_store(s, d)
    ctrl = CrossAttentionOutputSteering(
        source_concepts=[store], target_concepts=[None], strength=2.0,
        device=torch.device("cpu"), intermediate_clipping=False,
        use_first_diffusion_step=True, num_layers=1, steering_mode='dotproduct',
    )
    c = torch.randn(2, 5, 1, d)

    # (1) Elementwise no-clip path (steer_with_clipping via forward, CFG cond half).
    out = ctrl.forward(c.clone(), diffusion_step=0, place_in_unet="down", block_index=0)
    cond_in = c[1, :, 0, :]
    cond_out = out[1, :, 0, :]
    expected_cond = cond_in - 2.0 * (cond_in @ s).unsqueeze(-1) * s
    assert torch.allclose(cond_out, expected_cond, atol=1e-5)
    assert torch.allclose(cond_out.norm(dim=-1), cond_in.norm(dim=-1), atol=1e-4)

    # (2) Matrix form (steer_matrix_form) consumes the precomputed P; it must also
    # equal (I - 2 s s^T) c and preserve ||c|| on EVERY row. The
    # half_strength_projection mutation changes P to (I - 1 s s^T), which fails
    # both checks here (output diverges by ~0.2, norm by ~0.03 -- far above the
    # 1e-4 tolerance).
    b, P = ctrl.casteer_vectors[0][0]["down"][0]
    assert P.shape[-1] == d and P.shape[-2] == d, "P must be a d x d projection"
    m_in = c.clone()
    m_out = ctrl.steer_matrix_form(m_in, b, P)
    # (I - 2 s s^T) c broadcast over [B, seq, 1, d]
    m_expected = m_in - 2.0 * (m_in @ s).unsqueeze(-1) * s
    assert torch.allclose(m_out, m_expected, atol=1e-4), (
        f"matrix form must equal (I - 2 s s^T) c; max diff "
        f"{(m_out - m_expected).abs().max().item()}")
    assert torch.allclose(m_out.norm(dim=-1), m_in.norm(dim=-1), atol=1e-4), (
        f"matrix form must preserve ||c|| (Householder); max norm diff "
        f"{(m_out.norm(dim=-1) - m_in.norm(dim=-1)).abs().max().item()}")


def test_eq7_clip_only_positive_projections():
    """Eq.7 (method_2.tex:141-147): clipping steers only patches whose projection
    onto s is positive; patches with non-positive projection are unchanged."""
    torch.manual_seed(3)
    d = 64
    s = torch.zeros(d); s[0] = 1.0  # unit axis vector -> projection = c[0]
    store = _unit_store(s, d)
    ctrl = CrossAttentionOutputSteering(
        source_concepts=[store], target_concepts=[None], strength=2.0,
        device=torch.device("cpu"), intermediate_clipping=True,
        use_first_diffusion_step=True, num_layers=1, steering_mode='dotproduct',
    )
    # craft a conditional-half row with one positive- and one negative-projection patch
    c = torch.zeros(2, 2, 1, d)
    c[1, 0, 0, 0] = 3.0   # projection +3 -> steered: 3 - 2*3 = -3 along s (c[0])
    c[1, 1, 0, 0] = -2.0  # projection -2 -> clipped to 0 -> unchanged
    out = ctrl.forward(c.clone(), diffusion_step=0, place_in_unet="down", block_index=0)
    assert abs(out[1, 0, 0, 0].item() - (-3.0)) < 1e-5   # steered
    assert abs(out[1, 1, 0, 0].item() - (-2.0)) < 1e-5   # untouched (clip no-op)


def test_constant_mode_eq4():
    """Eq.4 (method_2.tex:90-93, supplementary.tex:544), SPEC U13: constant alpha
    subtracts alpha*s from EVERY patch; clip flag is a no-op for alpha>0."""
    torch.manual_seed(4)
    d = 48
    s = torch.randn(d); s = s / s.norm()
    store = _unit_store(s, d)
    for clip in (False, True):
        ctrl = CrossAttentionOutputSteering(
            source_concepts=[store], target_concepts=[None], strength=2.0,
            device=torch.device("cpu"), intermediate_clipping=clip,
            use_first_diffusion_step=True, num_layers=1, steering_mode='constant',
        )
        c = torch.randn(2, 3, 1, d)
        out = ctrl.forward(c.clone(), diffusion_step=0, place_in_unet="down", block_index=0)
        cond_in = c[1, :, 0, :]
        cond_out = out[1, :, 0, :]
        expected = cond_in - 2.0 * s  # same shift for every patch
        assert torch.allclose(cond_out, expected, atol=1e-5), f"constant mode clip={clip}"


def test_multi_concept_average_eq9():
    """Eq.9 (supplementary.tex:1068-1070), SPEC U9: average of individually-normalized
    per-concept stores is the elementwise mean, NOT re-normalized (||mean|| <= 1)."""
    torch.manual_seed(5)
    d = 32
    stores = []
    for _ in range(3):
        v = torch.randn(d); v = v / v.norm()
        stores.append({0: {"down": [v.view(1, 1, d)]}})
    avg = average_concept_vectors(stores)
    mean_vec = avg[0]["down"][0].view(-1)
    expected_mean = torch.stack([st[0]["down"][0].view(-1) for st in stores]).mean(dim=0)
    assert torch.allclose(mean_vec, expected_mean, atol=1e-6)
    assert mean_vec.norm().item() <= 1.0 + 1e-6  # not re-normalized


def test_cfg_only_conditional_half_steered():
    """SPEC U3 / controller: only batch index B//2: (conditional CFG branch) is modified;
    the unconditional half (index 0) is untouched."""
    torch.manual_seed(6)
    d = 32
    s = torch.randn(d); s = s / s.norm()
    store = _unit_store(s, d)
    ctrl = CrossAttentionOutputSteering(
        source_concepts=[store], target_concepts=[None], strength=2.0,
        device=torch.device("cpu"), intermediate_clipping=False,
        use_first_diffusion_step=True, num_layers=1, steering_mode='dotproduct',
    )
    c = torch.randn(2, 4, 1, d)
    out = ctrl.forward(c.clone(), diffusion_step=0, place_in_unet="down", block_index=0)
    assert torch.equal(out[0], c[0])  # uncond half exactly preserved (float32, dtype-preserving)


def test_controller_preserves_shape():
    """VectorControl.__call__ asserts output shape == input shape."""
    torch.manual_seed(7)
    d = 16
    s = torch.randn(d); s = s / s.norm()
    store = _unit_store(s, d)
    ctrl = CrossAttentionOutputSteering(
        source_concepts=[store], target_concepts=[None], strength=2.0,
        device=torch.device("cpu"), intermediate_clipping=False,
        use_first_diffusion_step=True, num_layers=1, steering_mode='dotproduct',
    )
    c = torch.randn(2, 7, 1, d)
    out = ctrl(c, place_in_unet="down")
    assert out.shape == c.shape


def test_householder_rejects_half_strength():
    """Negative: the Householder norm-preservation invariant (claim `house`,
    experiments.tex:21-22) REJECTS a known-wrong projection. The half_strength_projection
    defect (mutations.json) applies (I - 1*s s^T) instead of (I - 2*s s^T); this does NOT
    preserve ||c||, so the | ||out|| - ||c|| | < 1e-5 invariant fails -- proving the
    check catches a broken reflection rather than passing silently."""
    torch.manual_seed(8)
    d = 320
    s = torch.randn(d, dtype=torch.float64)
    s = s / s.norm()
    c = torch.randn(d, dtype=torch.float64)
    # (I - 1*s s^T) c -- the half-strength (broken) form, NOT a Householder reflection.
    out = c - 1.0 * (s @ c) * s
    assert abs(out.norm().item() - c.norm().item()) > 1e-5, (
        "half-strength projection must fail the norm-preservation invariant")


def test_multi_concept_no_renormalize_at_steer():
    """SPEC U9 / review A3: the Eq.9 multi-concept average is used AS-IS at steer
    time -- NOT re-normalized to a unit vector. The paper says "simply averaging"
    (supplementary.tex:1068-1071); re-normalizing the sub-unit mean would multiply
    effective suppression by ~1/||mean|| (~2.6x for 7 near-orthogonal concepts).

    This test builds a 3-concept averaged store (||mean|| < 1) and checks that
    steer_with_clipping (no clip, Eq.6) applies (I - beta * b b^T) c with the
    SUB-UNIT b -- i.e. the effective suppression is WEAKER than the unit-vector
    case. A regression that re-normalizes b at steer time would make the output
    equal the unit-vector Householder reflection and FAIL the weaker-suppression
    check here.
    """
    torch.manual_seed(9)
    d = 64
    # 3 near-orthogonal unit concept vectors
    stores = []
    for _ in range(3):
        v = torch.randn(d); v = v / v.norm()
        stores.append(v)
    mean_vec = torch.stack(stores).mean(dim=0)  # sub-unit
    assert mean_vec.norm().item() < 0.9, "mean of 3 random unit vectors should be sub-unit"
    avg_store = {0: {"down": [mean_vec.view(1, 1, d)]}}

    ctrl = CrossAttentionOutputSteering(
        source_concepts=[avg_store], target_concepts=[None], strength=2.0,
        device=torch.device("cpu"), intermediate_clipping=False,
        use_first_diffusion_step=True, num_layers=1, steering_mode='dotproduct',
    )
    c = torch.randn(2, 5, 1, d)
    out = ctrl.forward(c.clone(), diffusion_step=0, place_in_unet="down", block_index=0)
    cond_in = c[1, :, 0, :]
    cond_out = out[1, :, 0, :]
    # Paper-literal: c_new = (I - beta * b b^T) c with the sub-unit b (NOT b/||b||).
    expected = cond_in - 2.0 * (cond_in @ mean_vec).unsqueeze(-1) * mean_vec
    assert torch.allclose(cond_out, expected, atol=1e-5), (
        "steer must use the sub-unit mean b directly (no re-normalization); "
        f"max diff {(cond_out - expected).abs().max().item()}")
    # The unit-normalized form would apply (I - 2 * b_hat b_hat^T) c, which
    # suppresses MORE; assert the sub-unit output differs from that stronger form.
    b_hat = mean_vec / mean_vec.norm()
    stronger = cond_in - 2.0 * (cond_in @ b_hat).unsqueeze(-1) * b_hat
    assert not torch.allclose(cond_out, stronger, atol=1e-3), (
        "sub-unit steer must NOT equal the re-normalized (stronger) steer -- "
        "re-normalization at steer time is the A3 defect")


def test_multi_concept_no_renormalize_constant_mode():
    """Same A3/U9 check for the constant-alpha Eq.4 path: the constant shift is
    alpha * b (sub-unit), NOT alpha * b_hat (unit). A regression that re-normalizes
    b would shift by a LARGER amount and fail this test."""
    torch.manual_seed(10)
    d = 48
    stores = []
    for _ in range(3):
        v = torch.randn(d); v = v / v.norm()
        stores.append(v)
    mean_vec = torch.stack(stores).mean(dim=0)
    assert mean_vec.norm().item() < 0.9
    avg_store = {0: {"down": [mean_vec.view(1, 1, d)]}}
    ctrl = CrossAttentionOutputSteering(
        source_concepts=[avg_store], target_concepts=[None], strength=2.0,
        device=torch.device("cpu"), intermediate_clipping=True,
        use_first_diffusion_step=True, num_layers=1, steering_mode='constant',
    )
    c = torch.randn(2, 3, 1, d)
    out = ctrl.forward(c.clone(), diffusion_step=0, place_in_unet="down", block_index=0)
    cond_in = c[1, :, 0, :]
    cond_out = out[1, :, 0, :]
    expected = cond_in - 2.0 * mean_vec  # sub-unit shift
    assert torch.allclose(cond_out, expected, atol=1e-5), (
        "constant mode must use the sub-unit mean b (no re-normalization)")
