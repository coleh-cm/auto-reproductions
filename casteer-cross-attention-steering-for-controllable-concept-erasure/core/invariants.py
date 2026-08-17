"""CPU-runnable invariant measurements for the CASteer reproduction.

The numbers gate evaluates each claim's `quantity`/`against`/`predicate` as a
Python expression over `measured.<arm>.<metric>`. The `house` invariant
(experiments.tex:21-22) is pure maths -- it does NOT need a GPU or any diffusion
generation -- so this module computes it deterministically from a seed and
returns real numeric values that `run_all_arms` writes into measured.json under
the `casteer_noclip` arm (the method arm whose beta=2 erasure matrix the claim
is about). This lets the gate settle `house` as `reproduced` on a CPU-only host
instead of `unevaluable` (no expression) or `blocked` (no GPU).

Claim `house` (experiments.tex:21-22): with beta=2 and a unit steering vector s,
the erasure matrix (I - 2 s s^T) is a Householder reflection and preserves
||c||_2. SPEC U1: construction yields a unit vector (v/||v||). The test mirrors
`tests/test_method_core.py::test_householder_norm_preservation` and
`selfcheck_claims.check_house` exactly: 100 random unit s x random c in dims
{320,640,1280} (float64), max | ||(I-2 s s^T)c|| - ||c|| | < 1e-5, plus a
unit-norm construction check.
"""
from __future__ import annotations

import torch

# The arm whose measured dict carries the house metrics. The Householder
# property is a property of the CASteer beta=2 erasure matrix (used by both
# casteer arms); we attach it to the primary method arm `casteer_noclip`.
HOUSE_ARM = "casteer_noclip"

# Metrics this module produces (declared in claims.json `metrics`).
HOUSE_METRICS = ("house_max_norm_err", "house_unit_norm_ok", "house_pass")


def compute_house_invariant(seed: int) -> dict:
    """Deterministic Householder norm-preservation measurement for `seed`.

    Returns {house_max_norm_err: float, house_unit_norm_ok: int(0|1),
    house_pass: int(0|1)}. house_pass==1  <=>  the claim holds for this seed.
    """
    torch.manual_seed(int(seed))
    max_err = 0.0
    for d in (320, 640, 1280):
        for _ in range(100):
            s = torch.randn(d, dtype=torch.float64)
            s = s / s.norm()
            c = torch.randn(d, dtype=torch.float64)
            c_new = c - 2.0 * (s @ c) * s  # (I - 2 s s^T) c
            max_err = max(max_err, abs(c_new.norm().item() - c.norm().item()))
    raw = torch.randn(640, dtype=torch.float64)
    sv = raw / raw.norm().clamp(min=1e-8)  # SPEC U1: f_norm = v/||v||
    unit_ok = 1 if abs(sv.norm().item() - 1.0) < 1e-6 else 0
    passed = 1 if (max_err < 1e-5 and unit_ok == 1) else 0
    return {
        "house_max_norm_err": max_err,
        "house_unit_norm_ok": unit_ok,
        "house_pass": passed,
    }


def cpu_invariant_metrics(arm: str, seed: int) -> dict:
    """CPU-computable invariant metrics for `arm` at `seed`.

    Returns {} for arms that carry no CPU invariant. The diffusion metrics for
    these arms remain BLOCKED on a CPU host; only the pure-math invariants are
    filled in here, so the gate can settle them.
    """
    if arm == HOUSE_ARM:
        return compute_house_invariant(seed)
    return {}
