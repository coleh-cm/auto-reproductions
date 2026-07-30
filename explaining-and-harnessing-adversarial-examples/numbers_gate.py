#!/usr/bin/env python3
"""numbers_gate.py — evaluate claims.json against measured.json, write claims_result.json.

The numbers gate. For each claim in ``claims.json`` it resolves the
``measured.<arm>.<metric>`` tokens from ``measured.json`` (per seed) and renders a
verdict:

  - ``pass``    the assertion holds at every seed the reproduction ran.
  - ``fail``    the assertion does not hold at some seed the reproduction ran.
  - ``blocked`` a referenced metric is missing or carries the ``BLOCKED`` sentinel
                (the environment could not produce a number); the claim is NOT
                adjudicated.

Semantics (mirrors claims.json['evaluation']):

  - ``ordering``       the ``quantity`` (an expression over measured.<arm>.<metric>)
                       must satisfy ``direction`` at EVERY seed.
  - ``value``          ``abs(quantity - claimed) <= tolerance`` at every seed.
  - ``existence`` /    the boolean ``predicate`` must hold at every seed.
    ``invariant``

  - A claim whose expression aggregates across seeds — ``min(measured.x.y)``,
    ``max(measured.x.y)`` or ``mean(measured.x.y)`` wrapping a SINGLE measured
    token — is evaluated ONCE over the per-seed list (``min``/``max`` reduce the
    list; ``mean`` is ``sum/len``). Every other expression is evaluated per seed.
    A multi-argument ``min(a, b)`` / ``max(a, b)`` is element-wise (per seed),
    NOT a cross-seed aggregation.

  - ``seeds`` per-claim (default claims.json['seeds']) selects the seeds; only the
    seeds actually present in measured.json are adjudicated (a sub-scale run that
    could not finish all five M5 seeds adjudicates the three that ran, and the
    claim's note already records the scale gap).

  - Metric names may contain dots (e.g. ``l1_0.0025_train_error``,
    ``l1_2.5e-05_test_error``), so the canonical ``measured.<arm>.<metric>``
    tokens are taken from claims.json's ``arms`` block and matched as literal
    strings (longest first), never by regex.

The gate's load-bearing verdict is over the ``compute_invariance == "high"``
claims (the assertions that survive this reproduction's CPU sub-scale); the
``low`` claims are informational and several are *expected* to fail at sub-scale
(they need the paper's full 1600-unit / 5-seed budget). All three verdict
categories are reported so a reader can see which is which.

Output: ``claims_result.json`` — ``{summary, claims, not_tested, generated_from}``.
Prints one ``FINAL <verdict>=<count>`` line per verdict category and one
``FINAL gate=PASS|FAIL`` line (gate passes iff every HIGH claim is adjudicated
``pass`` with none blocked).
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parent
CLAIMS_PATH = REPRO_ROOT / "claims.json"
MEASURED_PATH = REPRO_ROOT / "measured.json"
OUT_PATH = REPRO_ROOT / "claims_result.json"

BLOCKED = "BLOCKED"


def build_canonical_tokens(claims_doc: dict) -> dict:
    """token -> (arm, metric) for every measured.<arm>.<metric> in claims.json.

    Metric names may contain dots (l1_0.0025_train_error), so we cannot regex-parse
    them; we read arm/metric straight from claims.json's arms block. Returned dict
    is ordered longest-token-first so substitution never partly overlaps.
    """
    pairs = {}
    for arm, spec in claims_doc.get("arms", {}).items():
        for metric in spec.get("metrics", {}):
            pairs[f"measured.{arm}.{metric}"] = (arm, metric)
    return dict(sorted(pairs.items(), key=lambda kv: len(kv[0]), reverse=True))


def _safe_eval(expr: str) -> object:
    """Eval an arithmetic/boolean expression with min/max/mean/abs only.

    measured.<arm>.<metric> tokens MUST already be substituted out of ``expr``
    (dotted names are not valid Python identifiers). Only the contract's allowed
    names (min, max, mean, abs, arithmetic, comparisons, and, or) are in scope.
    """
    allowed_names = {"min": min, "max": max, "mean": statistics.mean,
                     "abs": abs, "True": True, "False": False}
    code = compile(expr, "<claim-expr>", "eval")
    for name in code.co_names:
        if name not in allowed_names:
            raise ValueError(f"disallowed name in expression: {name!r}")
    return eval(code, {"__builtins__": {}}, allowed_names)


def _tokens_in(expr: str, canonical: dict) -> list:
    """Canonical tokens present as substrings of expr (longest first)."""
    return [t for t in canonical if t in expr]


def _substitute(expr: str, token_values: dict, canonical: dict) -> str:
    """Replace each canonical token with repr of its mapped value (longest first)."""
    out = expr
    for token in canonical:  # already sorted longest-first
        if token in token_values and token in out:
            out = out.replace(token, repr(token_values[token]))
    leftover = [t for t in canonical if t in out and t not in token_values]
    if leftover:
        raise ValueError(f"unresolved measured token in expression: {leftover}")
    return out


def is_aggregate(expr: str, canonical: dict) -> bool:
    """True iff min/max/mean wraps a SINGLE canonical token (cross-seed aggregation)."""
    for fn in ("min", "max", "mean"):
        for token in canonical:
            if f"{fn}({token})" in expr:
                return True
    return False


def _measured_scalar(measured: dict, arm: str, metric: str, seed) -> object:
    """measured[arm][str(seed)][metric]; BLOCKED sentinel if missing."""
    arm_block = measured.get(arm)
    if not isinstance(arm_block, dict):
        return BLOCKED
    seed_block = arm_block.get(str(seed))
    if not isinstance(seed_block, dict):
        return BLOCKED
    return seed_block.get(metric, BLOCKED)


def _available_seeds(measured: dict, arm: str) -> set:
    arm_block = measured.get(arm)
    if not isinstance(arm_block, dict):
        return set()
    return {int(s) for s in arm_block.keys() if str(s).isdigit()}


def evaluate_claim(claim: dict, measured: dict, top_seeds: list,
                   canonical: dict) -> dict:
    kind = claim["kind"]
    seeds_requested = claim.get("seeds", top_seeds)
    if kind in ("existence", "invariant"):
        expr, is_bool = claim["predicate"], True
    else:
        expr, is_bool = claim["quantity"], False

    tokens = _tokens_in(expr, canonical)
    if not tokens:
        return {"verdict": "blocked", "reason": "no measured token found in expression",
                "seeds_evaluated": []}

    common = set(seeds_requested)
    for t in tokens:
        common &= _available_seeds(measured, canonical[t][0])
    seeds = sorted(common)
    if not seeds:
        return {"verdict": "blocked",
                "reason": "no requested seed has data for all referenced metrics",
                "seeds_evaluated": []}

    for s in seeds:
        for t in tokens:
            arm, metric = canonical[t]
            if _measured_scalar(measured, arm, metric, s) == BLOCKED:
                return {"verdict": "blocked",
                        "reason": f"BLOCKED cell at seed {s}: {t}",
                        "seeds_evaluated": []}

    try:
        if is_aggregate(expr, canonical):
            token_values = {}
            for t in tokens:
                arm, metric = canonical[t]
                token_values[t] = [_measured_scalar(measured, arm, metric, s) for s in seeds]
            sub = _substitute(expr, token_values, canonical)
            result = _safe_eval(sub)
            if is_bool:
                ok = bool(result)
                return {"verdict": "pass" if ok else "fail",
                        "seeds_evaluated": seeds,
                        "aggregate_value": bool(result), "seeds": seeds}
            val = float(result)
            ok = abs(val - claim["claimed"]) <= claim["tolerance"]
            return {"verdict": "pass" if ok else "fail", "seeds_evaluated": seeds,
                    "aggregate_value": val, "claimed": claim["claimed"],
                    "tolerance": claim["tolerance"],
                    "abs_diff": abs(val - claim["claimed"]), "seeds": seeds}

        # per-seed evaluation
        per_seed = []
        for s in seeds:
            token_values = {}
            for t in tokens:
                arm, metric = canonical[t]
                token_values[t] = _measured_scalar(measured, arm, metric, s)
            sub = _substitute(expr, token_values, canonical)
            val = _safe_eval(sub)
            if is_bool:
                ok = bool(val)
            elif kind == "value":
                ok = abs(float(val) - claim["claimed"]) <= claim["tolerance"]
            else:  # ordering
                v = float(val)
                d = claim["direction"]
                ok = (v < 0 if d == "<0" else v > 0 if d == ">0"
                      else v <= 0 if d == "<=0" else v >= 0 if d == ">=0"
                      else v == 0 if d == "==0" else (_ for _ in ()).throw(
                          ValueError(f"unknown direction {d!r}")))
            per_seed.append({"seed": s, "value": val, "ok": ok})
        detail = {"per_seed": per_seed}
        if kind == "value":
            detail["claimed"], detail["tolerance"] = claim["claimed"], claim["tolerance"]
        elif kind == "ordering":
            detail["direction"] = claim["direction"]
        verdict = "pass" if all(p["ok"] for p in per_seed) else "fail"
        return {"verdict": verdict, "seeds_evaluated": seeds, **detail}
    except Exception as e:
        return {"verdict": "blocked", "reason": f"eval error: {e}",
                "seeds_evaluated": seeds}


def main() -> int:
    claims_doc = json.loads(CLAIMS_PATH.read_text())
    measured = json.loads(MEASURED_PATH.read_text())
    top_seeds = claims_doc["seeds"]
    canonical = build_canonical_tokens(claims_doc)

    results = []
    for claim in claims_doc["claims"]:
        r = evaluate_claim(claim, measured, top_seeds, canonical)
        results.append({
            "id": claim["id"], "kind": claim["kind"],
            "compute_invariance": claim["compute_invariance"],
            "citation": claim.get("citation"), "verdict": r["verdict"],
            "detail": {k: v for k, v in r.items() if k != "verdict"},
        })

    def counts(level=None):
        sel = results if level is None else [c for c in results
                                             if c["compute_invariance"] == level]
        return {v: sum(1 for c in sel if c["verdict"] == v)
                for v in ("pass", "fail", "blocked")}

    high = counts("high")
    summary = {
        "total": len(results),
        "pass": counts()["pass"], "fail": counts()["fail"], "blocked": counts()["blocked"],
        "high": high, "medium": counts("medium"), "low": counts("low"),
        "high_total": sum(high.values()),
        "gate_pass": high["pass"] == sum(high.values()) and high["blocked"] == 0,
        "note": "The gate's load-bearing verdict is high (claims that survive this "
                "reproduction's CPU sub-scale). low claims are informational; several "
                "are expected to fail at sub-scale (annotated in claims.json).",
    }

    out = {
        "summary": summary,
        "claims": results,
        "not_tested": claims_doc.get("not_tested", []),
        "generated_from": {
            "claims": CLAIMS_PATH.name, "measured": MEASURED_PATH.name,
            "canonical_tokens": list(canonical.keys()),
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH.name}")
    for v in ("pass", "fail", "blocked"):
        print(f"FINAL {v}={summary[v]}")
    print(f"FINAL high_pass={high['pass']} high_fail={high['fail']} "
          f"high_blocked={high['blocked']}")
    print(f"FINAL gate={'PASS' if summary['gate_pass'] else 'FAIL'}")
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
