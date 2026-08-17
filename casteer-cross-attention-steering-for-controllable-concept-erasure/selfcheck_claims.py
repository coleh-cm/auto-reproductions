"""Self-check evaluator for the CASteer reproduction.

This is OUR OWN evaluator (it writes selfcheck.json). It is deliberately NOT
claims_result.json, which is produced by the workflow's numbers gate, not by
us. We use it to check our claims as we go -- finding a claim unevaluable early
is cheaper than the gate finding it.

For each claim in claims.json:
- `house` (invariant): execute the predicate (pure math) -> PASS/FAIL.
- every other claim (ordering/value/curve): read measured.json; if every
  measured value the claim references is BLOCKED, the verdict is BLOCKED with
  the recorded reason; otherwise compute the claim's quantity and compare.

Usage:  python selfcheck_claims.py
Writes: selfcheck.json
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

REPO = os.path.dirname(os.path.abspath(__file__))


def check_house() -> dict:
    """SPEC claim `house` (experiments.tex:21-22): with beta=2 and unit s,
    (I - 2 s s^T) preserves ||c||_2; construction is unit-norm (U1)."""
    torch.manual_seed(0)
    max_err = 0.0
    for d in (320, 640, 1280):
        for _ in range(100):
            s = torch.randn(d, dtype=torch.float64)
            s = s / s.norm()
            c = torch.randn(d, dtype=torch.float64)
            c_new = c - 2.0 * (s @ c) * s
            max_err = max(max_err, abs(c_new.norm().item() - c.norm().item()))
    raw = torch.randn(640, dtype=torch.float64)
    sv = raw / raw.norm().clamp(min=1e-8)
    unit_ok = abs(sv.norm().item() - 1.0) < 1e-6
    passed = max_err < 1e-5 and unit_ok
    return {"verdict": "PASS" if passed else "FAIL",
            "measured": {"max_norm_err": max_err, "unit_norm_ok": unit_ok},
            "predicate": "max | ||(I-2 s s^T)c|| - ||c|| | < 1e-5 (float64, 100 trials x 3 dims) and ||v/||v|||| == 1"}


def referenced_values(claim: dict, measured: dict) -> list:
    """All measured.<arm>.<metric> values a claim references (from quantity/against)."""
    refs = []
    texts = []
    for k in ("quantity", "against"):
        v = claim.get(k)
        if isinstance(v, str):
            texts.append(v)
        elif isinstance(v, list):
            texts.extend(str(x) for x in v)
    for t in texts:
        for arm, metric in re.findall(r"measured\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)", t):
            for seed in measured.get(arm, {}):
                val = measured[arm][seed].get(metric)
                refs.append((arm, seed, metric, val))
    return refs


def main():
    with open(os.path.join(REPO, "claims.json")) as f:
        claims = json.load(f)
    with open(os.path.join(REPO, "measured.json")) as f:
        measured = json.load(f)
    with open(os.path.join(REPO, "measured_blocked_reasons.json")) as f:
        reasons = json.load(f)

    out = {}
    for c in claims["claims"]:
        cid = c["id"]
        if c["kind"] == "invariant":
            if cid == "house":
                out[cid] = check_house()
            else:
                out[cid] = {"verdict": "BLOCKED", "reason": "invariant predicate not implemented in selfcheck"}
            continue
        refs = referenced_values(c, measured)
        vals = [r[3] for r in refs]
        if not vals:
            out[cid] = {"verdict": "BLOCKED", "reason": "no measured references parsed"}
            continue
        if all(v == "BLOCKED" for v in vals):
            # gather a reason
            r0 = refs[0]
            reason = reasons.get(r0[0], {}).get(r0[1], {}).get(r0[2], "BLOCKED (no GPU; paper full config infeasible on CPU)")
            out[cid] = {"verdict": "BLOCKED", "reason": reason,
                        "kind": c["kind"], "references": [{"arm": a, "seed": s, "metric": m, "value": v} for a, s, m, v in refs]}
        else:
            out[cid] = {"verdict": "UNEVALUATED", "reason": "some measured values present; full evaluation is the numbers gate's job",
                        "references": [{"arm": a, "seed": s, "metric": m, "value": v} for a, s, m, v in refs]}

    # summary
    counts = {}
    for v in out.values():
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    out["_summary"] = {"counts": counts, "n_claims": len(claims["claims"])}

    with open(os.path.join(REPO, "selfcheck.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out["_summary"], indent=2))


if __name__ == "__main__":
    main()
