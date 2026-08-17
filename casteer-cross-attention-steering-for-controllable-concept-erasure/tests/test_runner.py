"""Runner configuration tests (review F3: per-task steering-vector mapping).

CASteer erases a concept that depends on the TASK being run, not on the arm.
The arm only fixes the model, beta, clip flag, mode, and per-step-vs-first-step;
the steering VECTOR is selected by the task (snoopy->Snoopy, style->Van Gogh,
i2p->nudity, i2p_overall->7-class average, coco->nudity). The previous
implementation hardcoded vector='nudity' for every steered SD-1.4 arm, which
would have steered snoopy/style runs with the nudity vector and measured the
wrong quantity.
"""
import os

import pytest
import torch

from core.runner import (
    ARM_CONFIG, TASK_VECTOR, I2P_OVERALL_CONCEPTS,
    _resolve_task_vector, build_controller,
)


def test_task_vector_covers_every_task_in_arm_config():
    """Every task listed in any ARM_CONFIG[arm]['tasks'] must have a
    TASK_VECTOR entry -- otherwise build_controller would raise KeyError at run
    time."""
    all_tasks = set()
    for arm, cfg in ARM_CONFIG.items():
        all_tasks.update(cfg.get("tasks", []))
    # 'i2p_overall' is the 7-class-average subtask; arms list 'i2p' and the
    # driver splits nudity vs overall by metric. Both must be present.
    for task in all_tasks:
        assert task in TASK_VECTOR, f"task {task!r} missing from TASK_VECTOR"
    assert "i2p_overall" in TASK_VECTOR, "i2p_overall (7-class avg) must be mappable"


def test_i2p_overall_is_the_seven_concept_average():
    """experiments.tex:39 + supplementary.tex:1071: the I2P 'all inappropriate'
    erasure averages 7 per-concept vectors: hate, harassment, violence,
    self-harm, sexual, shocking, illegal activity. The 7th concept is
    'illegal activity' per the appendix's explicit enumeration
    (supplementary.tex:1071) -- the main text paraphrases it as 'illegal
    content' (experiments.tex:39); the prior 'illegal' matched neither paper
    variant (review F-12e, fixed)."""
    assert I2P_OVERALL_CONCEPTS == [
        "hate", "harassment", "violence", "self-harm",
        "shocking", "sexual", "illegal activity",
    ]
    assert len(I2P_OVERALL_CONCEPTS) == 7
    assert isinstance(TASK_VECTOR["i2p_overall"], list)


def test_snoopy_style_tasks_use_their_own_vectors_not_nudity():
    """review F3: snoopy -> 'Snoopy', style -> 'Van Gogh', NOT 'nudity'."""
    assert TASK_VECTOR["snoopy"] == "Snoopy"
    assert TASK_VECTOR["style"] == "Van Gogh"
    assert TASK_VECTOR["i2p"] == "nudity"
    assert TASK_VECTOR["coco"] == "nudity"  # supplementary.tex:544


def test_resolve_task_vector_single_concept_loads_named_file(tmp_path):
    """_resolve_task_vector loads {concept}.pt for single-concept tasks."""
    vdir = str(tmp_path)
    s = torch.randn(16); s = s / s.norm()
    store = {0: {"down": [s.view(1, 1, 16)]}}
    torch.save(store, os.path.join(vdir, "Snoopy.pt"))
    out = _resolve_task_vector("snoopy", torch.device("cpu"), vdir)
    assert out[0]["down"][0].shape == (1, 1, 16)


def test_resolve_task_vector_missing_file_raises(tmp_path):
    """No OK-on-empty: a missing vector file raises FileNotFoundError, never
    silently substitutes."""
    with pytest.raises(FileNotFoundError):
        _resolve_task_vector("snoopy", torch.device("cpu"), str(tmp_path))


def test_resolve_task_vector_i2p_overall_averages_seven_files(tmp_path):
    """review F3 + Eq.9: the i2p_overall task composes the average of the 7
    per-concept stores on the fly (no production caller existed before this
    fix). The averaged store must equal average_concept_vectors of the 7."""
    from core.controller import average_concept_vectors
    vdir = str(tmp_path)
    stores = []
    for concept in I2P_OVERALL_CONCEPTS:
        v = torch.randn(16); v = v / v.norm()
        st = {0: {"down": [v.view(1, 1, 16)]}}
        torch.save(st, os.path.join(vdir, f"{concept}.pt"))
        stores.append(st)
    out = _resolve_task_vector("i2p_overall", torch.device("cpu"), vdir)
    expected = average_concept_vectors(stores)
    assert torch.allclose(out[0]["down"][0], expected[0]["down"][0], atol=1e-6)


def test_resolve_task_vector_i2p_overall_missing_one_concept_raises(tmp_path):
    """If any of the 7 per-concept files is missing, the i2p_overall resolution
    raises (no silent partial average)."""
    vdir = str(tmp_path)
    for concept in I2P_OVERALL_CONCEPTS[:-1]:  # omit the last one
        v = torch.randn(16); v = v / v.norm()
        torch.save({0: {"down": [v.view(1, 1, 16)]}}, os.path.join(vdir, f"{concept}.pt"))
    with pytest.raises(FileNotFoundError):
        _resolve_task_vector("i2p_overall", torch.device("cpu"), vdir)


def test_build_controller_vanilla_arm_returns_none():
    """build_controller for a vanilla arm (beta=None) returns None regardless of task."""
    assert build_controller("sd14", "i2p", torch.device("cpu")) is None
    assert build_controller("sdxl", "i2p", torch.device("cpu")) is None


def test_build_controller_steered_arm_uses_per_task_vector(tmp_path, monkeypatch):
    """build_controller(casteer_clip, 'snoopy', ...) must load Snoopy.pt, NOT
    nudity.pt, from the PER-MODEL vector subdir (review F3 + F7: vectors are
    loaded from <vector_dir>/<vector_model>; casteer_clip's vector_model is
    'sd14', so the file is read from <vdir>/sd14/Snoopy.pt). Verifies the F3 fix
    end-to-end: the controller is built with the per-task store."""
    vdir = str(tmp_path)
    # Per-model vector store: sd14 vectors live under <vdir>/sd14/.
    sd14_dir = os.path.join(vdir, "sd14")
    os.makedirs(sd14_dir, exist_ok=True)
    snoopy_v = torch.ones(16); snoopy_v = snoopy_v / snoopy_v.norm()
    nudity_v = torch.zeros(16); nudity_v[0] = 1.0
    torch.save({0: {"down": [snoopy_v.view(1, 1, 16)]}}, os.path.join(sd14_dir, "Snoopy.pt"))
    torch.save({0: {"down": [nudity_v.view(1, 1, 16)]}}, os.path.join(sd14_dir, "nudity.pt"))

    ctrl = build_controller("casteer_clip", "snoopy", torch.device("cpu"), vector_dir=vdir)
    # The loaded steering vector b should be Snoopy (all-equal), NOT nudity (axis).
    b, P = ctrl.casteer_vectors[0][0]["down"][0]
    b_flat = b.view(-1)
    # Snoopy is uniform; nudity is an axis vector. Check it's the Snoopy one.
    assert abs(b_flat[0].item() - b_flat[1].item()) < 1e-6, "must be the Snoopy (uniform) vector"
    assert abs(b_flat[0].item() - (1.0 / 16**0.5)) < 1e-5


def test_build_controller_per_model_vector_dir_isolates_sdxl_from_sd14(tmp_path):
    """review F7: sdxl_casteer_clip's vector_model is 'sdxl-turbo', so its
    vectors load from <vdir>/sdxl-turbo/, NOT the shared <vdir>/ or <vdir>/sd14/.
    A shared store would let sdxl_casteer_clip silently load SD-1.4's nudity.pt
    (dim-mismatch crash at best, silent wrong-model steering at worst)."""
    vdir = str(tmp_path)
    # SD-1.4 nudity vector (the wrong-model trap) at the OLD shared location
    # and at <vdir>/sd14 -- neither should be loaded by sdxl_casteer_clip.
    sd14_dir = os.path.join(vdir, "sd14"); os.makedirs(sd14_dir, exist_ok=True)
    turbo_dir = os.path.join(vdir, "sdxl-turbo"); os.makedirs(turbo_dir, exist_ok=True)
    sd14_v = torch.zeros(16); sd14_v[0] = 1.0  # axis vector (the trap)
    torch.save({0: {"down": [sd14_v.view(1, 1, 16)]}}, os.path.join(sd14_dir, "nudity.pt"))
    torch.save({0: {"down": [sd14_v.view(1, 1, 16)]}, 1: {"down": [sd14_v.view(1, 1, 16)]}},
               os.path.join(vdir, "nudity.pt"))
    turbo_v = torch.ones(16); turbo_v = turbo_v / turbo_v.norm()  # uniform (correct)
    torch.save({0: {"down": [turbo_v.view(1, 1, 16)]}}, os.path.join(turbo_dir, "nudity.pt"))

    ctrl = build_controller("sdxl_casteer_clip", "i2p", torch.device("cpu"), vector_dir=vdir)
    b, _ = ctrl.casteer_vectors[0][0]["down"][0]
    b_flat = b.view(-1)
    # Must be the SDXL-Turbo uniform vector, NOT the SD-1.4 axis vector.
    assert abs(b_flat[0].item() - b_flat[1].item()) < 1e-6, "must be the sdxl-turbo (uniform) vector"
    assert abs(b_flat[0].item() - (1.0 / 16**0.5)) < 1e-5


def test_arm_config_vector_model_is_set_for_steered_arms():
    """Every steered arm (beta is not None) must declare vector_model so
    build_controller can route to the per-model store. A regression that drops
    vector_model would make build_controller raise (caught here)."""
    for arm, cfg in ARM_CONFIG.items():
        if cfg["beta"] is not None:
            assert cfg.get("vector_model") is not None, (
                f"steered arm {arm!r} must declare vector_model (per-model vector "
                "store, review F7)")
        else:
            assert cfg.get("vector_model") is None, (
                f"vanilla arm {arm!r} must have vector_model=None")


def test_arm_config_has_no_single_vector_field():
    """The ARM_CONFIG no longer carries a single 'vector' field (per-arm);
    vectors are per-task via TASK_VECTOR. A regression that re-introduces
    vector='nudity' on every arm would defeat the F3 fix."""
    for arm, cfg in ARM_CONFIG.items():
        assert "vector" not in cfg, (
            f"arm {arm!r} must not carry a per-arm 'vector' (per-task via TASK_VECTOR); "
            "hardcoding vector='nudity' is the F3 defect")
