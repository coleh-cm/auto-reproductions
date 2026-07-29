"""Degeneracy test — the cheapest real correctness evidence.

The FGSM adversarial-training method (Algorithm B, E7) has a no-op setting:
``eps = 0``.  There ``x + 0*sign(grad) == x`` exactly, so

    J~(theta, x, y) = alpha J + (1-alpha) J(x) = J(theta, x, y)

and the method MUST reproduce the baseline (clean training) EXACTLY.  This
file asserts that at two levels:

  1. Cost level: ``adversarial_train_cost(..., eps=0) == cross_entropy_cost``
     bit-for-bit (E7 reduces to J).
  2. Training level: ``train(adv_train=True, eps=0)`` and ``train(adv_train=False)``
     produce bit-identical ``best_state_dict`` under the same seed (dropout OFF
     so no mask divergence).
  3. CLI level: ``run_experiment.py --lambda 0`` (the method at its no-op) and
     ``run_experiment.py --baseline`` (clean reference) print the SAME
     ``FINAL accuracy=...`` line.

If any of these fails the implementation is wrong — and we know in seconds,
without a full paper-scale run.  (research-code skill: "the most valuable
test in research code, and the one most often skipped".)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT / "src"))

from fgsm_repro.models import MaxoutMLP, SoftmaxRegression
from fgsm_repro.objectives import adversarial_train_cost, cross_entropy_cost
from fgsm_repro.train import TrainConfig, train


def _tiny_data(n_train=200, n_valid=80, n_test=80, seed=0):
    from fgsm_repro.data import MNISTData
    g = torch.Generator().manual_seed(seed)
    def make(n):
        x = torch.rand(n, 784, generator=g, dtype=torch.float32)
        y = torch.randint(0, 10, (n,), generator=g, dtype=torch.int64)
        x[torch.arange(n), y] += 0.5
        return x, y
    return MNISTData(x_train=make(n_train)[0], y_train=make(n_train)[1],
                     x_valid=make(n_valid)[0], y_valid=make(n_valid)[1],
                     x_test=make(n_test)[0], y_test=make(n_test)[1])


def _state_equal(a, b):
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        ta, tb = a[k], b[k]
        if ta.dtype != tb.dtype or ta.shape != tb.shape or not torch.equal(ta, tb):
            return False
    return True


def test_cost_degeneracy_eps0_equals_clean():
    """E7 at eps=0 must equal J exactly (bit-for-bit)."""
    torch.manual_seed(0)
    m = SoftmaxRegression(784, 10)
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    Jc = cross_entropy_cost(m, x, y)
    Jadv = adversarial_train_cost(m, x, y, 0.0, 0.5)
    assert torch.equal(Jc, Jadv), f"E7(eps=0)={Jadv} != J={Jc}"


def test_train_degeneracy_eps0_equals_baseline_bitidentical():
    """train(adv_train=True, eps=0) == train(adv_train=False) bit-identical.

    Dropout OFF (include 1.0/1.0) so no mask divergence between the method's
    extra forward and the clean forward.  This is the degeneracy the
    run_experiment.py harness relies on (SPEC.md section 6 item 22).
    """
    data = _tiny_data()
    common = dict(batch_size=32, lr=0.1, momentum=0.5, max_epochs=10, seed=0,
                  max_steps=5, early_stop="clean", patience=100)
    torch.manual_seed(0)
    m_clean = MaxoutMLP(units=8, pieces=2, dropout_input_include=1.0,
                        dropout_hidden_include=1.0, seed=0)
    res_clean = train(m_clean, TrainConfig(adv_train=False, eps=0.25, **common), data)
    torch.manual_seed(0)
    m_adv = MaxoutMLP(units=8, pieces=2, dropout_input_include=1.0,
                      dropout_hidden_include=1.0, seed=0)
    res_adv = train(m_adv, TrainConfig(adv_train=True, eps=0.0, **common), data)
    assert res_clean.steps_run == res_adv.steps_run == 5
    assert _state_equal(res_clean.best_state_dict, res_adv.best_state_dict), (
        "Degeneracy FAILED: method at eps=0 != baseline under same seed")


def test_train_degeneracy_drops_with_dropout_on():
    """Sanity (not a failure gate): with dropout ON the method's extra forward
    draws a different mask, so eps=0 no longer reproduces the baseline
    bit-for-bit.  This documents WHY run_experiment.py disables dropout by
    default.  We assert the divergence is non-zero (the degeneracy mechanism
    is dropout-mask divergence, not a no-op)."""
    data = _tiny_data()
    common = dict(batch_size=32, lr=0.1, momentum=0.5, max_epochs=10, seed=0,
                  max_steps=5, early_stop="clean", patience=100)
    torch.manual_seed(0)
    m_clean = MaxoutMLP(units=8, pieces=2, dropout_input_include=0.8,
                        dropout_hidden_include=0.5, seed=0)
    res_clean = train(m_clean, TrainConfig(adv_train=False, eps=0.25, **common), data)
    torch.manual_seed(0)
    m_adv = MaxoutMLP(units=8, pieces=2, dropout_input_include=0.8,
                      dropout_hidden_include=0.5, seed=0)
    res_adv = train(m_adv, TrainConfig(adv_train=True, eps=0.0, **common), data)
    assert not _state_equal(res_clean.best_state_dict, res_adv.best_state_dict), (
        "Expected dropout-on eps=0 to diverge from baseline (mask divergence)")


def test_run_experiment_cli_degeneracy():
    """CLI: `run_experiment.py --lambda 0` (method at no-op) must print the
    SAME `FINAL accuracy=...` line as `--baseline` (clean reference)."""
    runner = REPRO_ROOT / "run_experiment.py"
    env = {"PYTHONPATH": str(REPRO_ROOT / "src")}
    common = ["--steps", "5", "--units", "16", "--seed", "0"]
    r1 = subprocess.run([sys.executable, str(runner), "--lambda", "0", *common],
                        capture_output=True, text=True, cwd=str(REPRO_ROOT), env=env)
    r2 = subprocess.run([sys.executable, str(runner), "--baseline", *common],
                        capture_output=True, text=True, cwd=str(REPRO_ROOT), env=env)
    assert r1.returncode == 0 and r2.returncode == 0, (r1.stderr, r2.stderr)
    line1 = r1.stdout.strip().splitlines()[-1]
    line2 = r2.stdout.strip().splitlines()[-1]
    assert line1.startswith("FINAL accuracy="), line1
    assert line2.startswith("FINAL accuracy="), line2
    assert line1 == line2, f"CLI degeneracy: method eps=0='{line1}' != baseline='{line2}'"


def test_run_experiment_cli_determinism():
    """Same args twice -> identical output (seed control)."""
    runner = REPRO_ROOT / "run_experiment.py"
    env = {"PYTHONPATH": str(REPRO_ROOT / "src")}
    args = ["--lambda", "0.25", "--steps", "5", "--units", "16", "--seed", "0"]
    r1 = subprocess.run([sys.executable, str(runner), *args],
                        capture_output=True, text=True, cwd=str(REPRO_ROOT), env=env)
    r2 = subprocess.run([sys.executable, str(runner), *args],
                        capture_output=True, text=True, cwd=str(REPRO_ROOT), env=env)
    assert r1.stdout.strip() == r2.stdout.strip(), "non-deterministic CLI"
