"""Baseline-bank tests: ITI probe fit + static intervention, AS target-angle plane,
APPS loader + grader. Lock in the reviewer-driven fixes (SPEC §4.15-4.18)."""
import numpy as np
import pytest

from mags.manifold import HeadProblemActivations
from mags.baselines import (
    fit_iti_bank, ITIController, fit_as_bank, AngularSteeringController, ASBank,
)


def _make_all_acts(rng, n_heads_per_layer=4, layers=(0, 1), n_problems=12, d_h=8,
                   traces=2, T=8):
    """Real contrastive signal: incorrect traces shift along a per-head direction."""
    out = {}
    pids = [f"p{i}" for i in range(n_problems)]
    for l in layers:
        for h in range(n_heads_per_layer):
            w = rng.normal(size=d_h); w /= np.linalg.norm(w)
            ha = HeadProblemActivations()
            for i, pid in enumerate(pids):
                c = [rng.normal(scale=0.3, size=(T, d_h)) for _ in range(traces)]
                e = [rng.normal(scale=0.3, size=(T, d_h)) + 1.5 * w for _ in range(traces)]
                ha.correct[pid] = c
                ha.incorrect[pid] = e
            out[(l, h)] = ha
    return out, pids


def test_iti_bank_fits_probes_and_reaches_large_K():
    rng = np.random.default_rng(0)
    all_acts, pids = _make_all_acts(rng, n_heads_per_layer=4, layers=(0, 1))
    bank = fit_iti_bank(all_acts, pids[:8], pids[8:], model_id="m", benchmark="b",
                       K=8, alpha=0.5, layers_monitored=[0, 1])
    # ALL monitored heads are stored (8 heads), so K in {24,48,96} is reachable upstream
    assert len(bank.heads) == 8, "ITI bank must persist ALL monitored heads (SPEC §4.15)"
    assert len(bank.selected_heads) == 8
    # selected heads are the top-K by held-out probe ACCURACY (not MAGS AUROC)
    sel_keys = {tuple(h) for h in bank.selected_heads}
    sel_acc = [bank.heads[k].accuracy for k in sel_keys]
    nonsel_acc = [v.accuracy for k, v in bank.heads.items() if k not in sel_keys]
    if nonsel_acc:
        assert min(sel_acc) >= max(nonsel_acc) - 1e-9
    # every selected head has a unit direction + a fixed sigma
    for hd in bank.heads.values():
        assert np.isclose(np.linalg.norm(hd.direction), 1.0, atol=1e-5)
        assert 0.0 <= hd.sigma <= 1.0


def test_iti_intervention_is_static_and_oriented():
    """SPEC §4.15: a += alpha * sigma_h * v_h every step (static; no dependence on the
    current activation). v_h oriented toward the CORRECT class."""
    rng = np.random.default_rng(1)
    all_acts, pids = _make_all_acts(rng, n_heads_per_layer=2, layers=(0,), d_h=6)
    bank = fit_iti_bank(all_acts, pids[:8], pids[8:], model_id="m", benchmark="b",
                       K=2, alpha=0.5, layers_monitored=[0])
    # force-select both heads for the test
    bank.selected_heads = [list(k) for k in bank.heads.keys()]
    ctrl = ITIController(bank, alpha=0.5)
    import torch
    # two different activations must receive the SAME shift (static intervention)
    x1 = torch.zeros(1, 1, 2, 6)
    x2 = torch.ones(1, 1, 2, 6) * 5.0
    out1 = ctrl(0, x1.clone()).numpy()
    out2 = ctrl(0, x2.clone()).numpy()
    shift1 = out1 - x1.numpy()
    shift2 = out2 - x2.numpy()
    assert np.allclose(shift1, shift2, atol=1e-6), \
        "ITI intervention must be STATIC (independent of the current activation, SPEC §4.15)"
    # the shift for each head is alpha * sigma_h * v_h
    for h_idx, (l, h) in enumerate(bank.selected_heads):
        hd = bank.heads[(l, h)]
        expected = 0.5 * hd.sigma * hd.direction
        assert np.allclose(shift1[0, 0, h_idx], expected, atol=1e-5)


def test_as_bank_uses_difference_in_means_d_feat():
    """SPEC §4.16: d_feat = difference-in-means (correct vs incorrect), not the centroid."""
    rng = np.random.default_rng(2)
    all_acts, pids = _make_all_acts(rng, n_heads_per_layer=2, layers=(0,), d_h=6)
    asb = fit_as_bank(all_acts, model_id="m", benchmark="b", angle_deg=30,
                      layers_monitored=[0])
    p = asb.plane                       # single global plane (layer=-1)
    assert p.layer == -1
    # recompute the true difference-in-means direction (pooled over the layer's heads)
    cs = [t for (l, h), ha in all_acts.items() if l == 0
          for pid in ha.correct for t in ha.correct[pid]]
    iss = [t for (l, h), ha in all_acts.items() if l == 0
          for pid in ha.incorrect for t in ha.incorrect[pid]]
    c_mean = np.concatenate(cs, axis=0).mean(axis=0)
    i_mean = np.concatenate(iss, axis=0).mean(axis=0)
    true_dfeat = (i_mean - c_mean)
    # the stored d_feat must align with the true contrastive direction (after orthonorm)
    cos = float(p.d_feat @ true_dfeat / (np.linalg.norm(true_dfeat) + 1e-12))
    assert cos > 0.95, f"d_feat must be the difference-in-means direction (cos={cos})"


def test_as_rotation_is_target_angle_not_fixed_offset():
    """SPEC §4.16: rotate so the activation's plane-angle becomes the target, i.e.
    the result angle == target regardless of the input angle (target-angle form)."""
    rng = np.random.default_rng(3)
    all_acts, pids = _make_all_acts(rng, n_heads_per_layer=1, layers=(0,), d_h=6)
    asb = fit_as_bank(all_acts, model_id="m", benchmark="b", angle_deg=45,
                      layers_monitored=[0])
    ctrl = AngularSteeringController(asb, angle_deg=45.0)
    import torch
    p = asb.plane
    # place activation purely along d_feat (angle 0) and along d_pc0 (angle 90)
    for start_deg in (0, 30, 90, 200):
        a = np.cos(np.deg2rad(start_deg)) * p.d_feat + np.sin(np.deg2rad(start_deg)) * p.d_pc0
        x = torch.from_numpy(a[None, None, None, :]).float()  # [1,1,1,d_h] (1 head)
        out = ctrl(0, x).numpy()
        a_out = out[0, 0, 0]
        ang_out = np.degrees(np.arctan2(a_out @ p.d_pc0, a_out @ p.d_feat))
        assert abs(((ang_out - 45 + 180) % 360) - 180) < 1e-2, \
            f"target-angle form: output angle {ang_out} != 45 (start {start_deg})"


def test_as_applies_the_same_global_plane_at_every_layer():
    """Paper tex:L394 / SPEC §4.16: AS is "a fixed 2D rotation ... across all
    layers". The controller must (a) request hooks on ALL layers (hook_layers='all')
    and (b) apply the SAME global plane at every layer it is called with — not
    pass through non-monitored layers. The prior per-layer-monitored-only impl
    rotated only 4 of 32 layers, which the review flagged as a major paper-fidelity
    deviation."""
    rng = np.random.default_rng(5)
    all_acts, pids = _make_all_acts(rng, n_heads_per_layer=1, layers=(0, 1), d_h=6)
    asb = fit_as_bank(all_acts, model_id="m", benchmark="b", angle_deg=30,
                      layers_monitored=[0, 1])
    ctrl = AngularSteeringController(asb, angle_deg=30.0)
    assert ctrl.hook_layers == "all", "AS must request hooks on ALL layers"
    import torch
    p = asb.plane
    # the SAME plane is applied at every layer index, including ones NOT monitored
    for layer in (0, 1, 5, 17, 31):
        a = np.cos(np.deg2rad(10)) * p.d_feat + np.sin(np.deg2rad(10)) * p.d_pc0
        x = torch.from_numpy(a[None, None, None, :]).float()
        out = ctrl(layer, x)
        assert out is not None, f"AS must rotate at layer {layer} (not pass-through)"
        a_out = out.numpy()[0, 0, 0]
        ang_out = np.degrees(np.arctan2(a_out @ p.d_pc0, a_out @ p.d_feat))
        assert abs(((ang_out - 30 + 180) % 360) - 180) < 1e-2, \
            f"layer {layer}: output angle {ang_out} != 30 (global plane not applied)"


def test_cd_plausibility_mask_is_relative_to_expert_max():
    """Li et al. 2023 adaptive plausibility (cited by the paper, tex:L396): the plausible
    set is {x : p_expert(x) >= alpha_p * max_x' p_expert(x')} — a RELATIVE threshold
    (alpha_p of the expert's own maximum), NOT an absolute cutoff p_e >= alpha_p. An
    absolute cutoff collapses the plausible set to ~1-2 tokens over a large vocab and
    makes the amateur penalty inert (CD degrades to the greedy expert). This test pins
    the relative form by checking a token whose expert prob is well below alpha_p but
    above alpha_p*max stays IN the plausible set, and one below alpha_p*max is OUT."""
    import torch
    import torch.nn.functional as F
    from mags.baselines import ContrastiveDecoder
    # a fake vocab of 5 tokens; expert strongly favors token 0
    expert_logits = torch.tensor([[0.0, -1.0, -2.0, -8.0, -10.0]])
    amateur_logits = torch.tensor([[-5.0, -5.0, -5.0, -5.0, -5.0]])
    cd = ContrastiveDecoder(None, None, None, alpha_plausibility=0.1, beta=0.5)
    score = cd.adapted_logits(expert_logits, amateur_logits)
    p_e = F.softmax(expert_logits, dim=-1)
    p_max = float(p_e.max())
    # token 1: p_e ~0.21 > 0.1*0.71 ~0.071  => stays plausible (relative form keeps it);
    #   under the OLD absolute form (p_e < 0.1) it would ALSO stay, so this alone is
    #   not the discriminator — see the token-2 assertion below.
    # token 2: p_e ~0.087. Absolute cutoff 0.1 would MASK it; relative threshold 0.071
    #   keeps it. So under the relative form token 2 must be UNMASKED (finite score),
    #   while under the absolute form it would be -inf. This is the discriminator.
    assert torch.isfinite(score[0, 2]), \
        "relative plausibility: token 2 (p_e~0.087 > 0.1*p_max~0.071) must stay plausible; " \
        "the absolute form (p_e < 0.1) would wrongly mask it"
    # token 3: p_e ~1.5e-4 << 0.1*p_max~0.071 => masked under BOTH forms (sanity)
    assert torch.isinf(score[0, 3]) and score[0, 3] < 0, \
        "token far below alpha_p*max must be masked"
    # the relative threshold value itself is what we use, not alpha_p alone
    assert float(p_e[0, 2]) < 0.1, "token 2 must be below the absolute 0.1 line (so the " \
        "absolute form would mask it); the relative form is what keeps it"


def test_as_persists_and_reloads(tmp_path):
    rng = np.random.default_rng(4)
    all_acts, pids = _make_all_acts(rng, layers=(0, 1))
    asb = fit_as_bank(all_acts, model_id="m", benchmark="b", angle_deg=30,
                      layers_monitored=[0, 1])
    npz = str(tmp_path / "as.npz")
    asb.save(npz)
    asb2 = ASBank.load(npz)
    assert asb2.layers_monitored == asb.layers_monitored
    assert asb2.plane.layer == -1
    assert np.allclose(asb2.plane.d_feat, asb.plane.d_feat)
    assert np.allclose(asb2.plane.d_pc0, asb.plane.d_pc0)
