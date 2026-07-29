"""Degeneracy test: the FGSM adversarial-training method at its no-op setting
(eps=0) must reproduce the baseline (plain training) EXACTLY.

This is the cheapest real correctness evidence: if it fails, the wiring
(input-gradient, stop-grad, the alpha-mixing, RNG plumbing) is wrong, and we
know in minutes without a full run. See research-code skill: "the method at its
no-op setting must reproduce the baseline exactly".

Why it must hold bit-for-bit here (no dropout): at eps=0, eta = 0, so
x_tilde == x; the adversarial loss J~ = alpha*J(x) + (1-alpha)*J(x) = J(x), and
because x_tilde is the same tensor values as x (detached copy), the parameter
gradient is alpha*g + (1-alpha)*g = g (with alpha=0.5, 0.5g+0.5g == g in
IEEE-754). The adversarial branch's autograd.grad for the input gradient
consumes no RNG, so weight init + batch shuffling stay in lock-step with the
baseline arm. Therefore the trained checkpoint is bit-identical.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgsm_repro.data import load_mnist
from fgsm_repro.models import MaxoutMLP
from fgsm_repro.train import TrainConfig, train

ROOT = Path(__file__).resolve().parents[1]
STEPS = 60  # enough to move the weights off init; small for test speed


def _train_copy(adv: bool, eps: float, seed: int = 0) -> dict:
    """Train a small maxout net and return its best-state-dict (the checkpoint)."""
    torch.manual_seed(seed)  # MaxoutMLP weight init draws from the GLOBAL rng
    data = load_mnist(ROOT / "data", seed=seed)
    m = MaxoutMLP(units=64, pieces=5, seed=seed)  # small for speed; no dropout
    cfg = TrainConfig(
        batch_size=100, lr=0.1, momentum=0.5, max_epochs=1, seed=seed,
        adv_train=adv, eps=eps, alpha=0.5, early_stop="clean", patience=100,
        max_steps=STEPS, init_seed=seed,
    )
    res = train(m, cfg, data)
    return res.best_state_dict


def test_degeneracy_eps0_equals_baseline_checkpoint():
    """Strongest check: the best-state-dict (monitor-best checkpoint) is
    bit-identical at eps=0 vs the baseline arm."""
    base = _train_copy(adv=False, eps=0.0)
    adv0 = _train_copy(adv=True, eps=0.0)
    assert set(base.keys()) == set(adv0.keys())
    for k in base:
        assert torch.equal(base[k], adv0[k]), (
            f"checkpoint differs at eps=0 for tensor {k}: "
            f"max abs diff {(base[k]-adv0[k]).abs().max().item()}"
        )


def test_degeneracy_eps0_equals_baseline_accuracy_subprocess():
    """End-to-end via the CLI: `--lambda 0` and `--baseline` print the same line."""
    env_run = lambda extra: subprocess.run(
        [sys.executable, str(ROOT / "run_experiment.py"), "--steps", str(STEPS),
         "--seed", "0", *extra],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    r_base = env_run(["--baseline"])
    r_adv0 = env_run(["--lambda", "0"])
    assert r_base.returncode == 0, f"baseline failed: {r_base.stderr[-800:]}"
    assert r_adv0.returncode == 0, f"adv eps0 failed: {r_adv0.stderr[-800:]}"
    line_b = [l for l in r_base.stdout.splitlines() if l.startswith("FINAL accuracy=")][0]
    line_a = [l for l in r_adv0.stdout.splitlines() if l.startswith("FINAL accuracy=")][0]
    assert line_b == line_a, f"degeneracy broken via CLI: {line_b!r} vs {line_a!r}"


def test_eps0_perturbation_is_zero():
    """At eps=0 the method perturbs nothing: x_adv == x."""
    from fgsm_repro.attacks import fgsm
    torch.manual_seed(1)
    m = MaxoutMLP(units=16, pieces=5, seed=1)
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    x_adv = fgsm(m, x, y, 0.0)
    assert torch.equal(x_adv, x)
