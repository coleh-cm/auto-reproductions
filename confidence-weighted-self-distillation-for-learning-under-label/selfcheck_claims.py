#!/usr/bin/env python3
"""Self-evaluate the claims in claims.json against measured.json.

This is a LOCAL self-check (writes selfcheck.json), NOT the workflow's numbers
gate: the gate owns claims_result.json and refuses any copy it did not
produce, so this script deliberately writes a different filename. It exists
so a claim that is unevaluable is found early (here, cheaply) rather than by
the gate, and so a reader can re-run it. A grader that cannot run raises;
a missing/BLOCKED metric blocks the claim rather than fabricating a verdict.

Verdicts per claim per seed: pass / fail / blocked.
  ordering            : quantity > 0 (direction ">") at every seed
  value               : abs(quantity - claimed) <= tolerance at every seed
  existence/invariant : the predicate (evaluated over measured tokens) is
                        true at every seed; a missing or "BLOCKED" metric
                        blocks the claim for that seed

Usage: python selfcheck_claims.py [--strict]
  --strict: exit 1 if any claim is not pass at every seed.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path


def _num(x):
    """Return (float, is_blocked). 'BLOCKED' or non-numeric -> blocked."""
    if isinstance(x, str) and x.upper() == "BLOCKED":
        return None, True
    if isinstance(x, bool) or x is None:
        return None, True
    try:
        return float(x), False
    except (TypeError, ValueError):
        return None, True


def _resolve(measured, arm, seed, expr):
    """Resolve an expression like 'measured.cwsd.accuracy - measured.baseline.accuracy'
    against measured.json. Returns (float_or_None, blocked_bool)."""
    g = {"__builtins__": {}}
    env = {}
    for a, seeds in measured.items():
        env[a] = {}
        for s, metrics in seeds.items():
            env[a][s] = metrics
    # Build a per-seed evaluator: replace measured.<arm>.<metric> with the value
    def at(seedstr):
        locals_ = {}
        for a, seeds in measured.items():
            locals_[a] = {}
            for s, metrics in seeds.items():
                locals_[a][s] = metrics
        # Rewrite measured.arm.metric -> arm['<seed>']['metric']
        import re
        e = expr
        for a in measured:
            for m in measured[a].get(seedstr, {}):
                pass
        # simpler: eval with measured.* as a namespace
        class NS: pass
        ns = NS()
        for a, seeds in measured.items():
            sub = NS()
            setattr(ns, a, sub)
            for s, metrics in seeds.items():
                setattr(sub, s, metrics)
        # measured.X.Y -> ns.X.<seed>.Y  -- but the expression uses measured.X.Y
        # which means per-seed; we substitute the seed's block.
        # Replace 'measured.<arm>.<metric>' -> the value for this seed
        pattern = re.compile(r"measured\.(\w+)\.(\w+)")
        def repl(mo):
            a, met = mo.group(1), mo.group(2)
            seed_block = measured.get(a, {}).get(seedstr, {})
            v = seed_block.get(met)
            val, blocked = _num(v)
            return "None" if blocked or val is None else repr(val)
        e2 = pattern.sub(repl, expr)
        try:
            return eval(e2, {"__builtins__": {}}, {})
        except Exception:
            return None
    # evaluate per seed
    seeds = list(measured.get(list(measured)[0], {}).keys()) if measured else []
    return at, seeds


def evaluate(claims_path, measured_path):
    claims = json.loads(Path(claims_path).read_text())
    measured = json.loads(Path(measured_path).read_text())
    seeds = [str(s) for s in claims.get("seeds", [])]
    results = []
    for c in claims.get("claims", []):
        kind = c.get("kind")
        cid = c.get("id")
        per_seed = {}
        for seed in seeds:
            if kind == "ordering":
                expr = c.get("quantity") or c.get("predicate")
                direction = c.get("direction", ">")
                # direction may be ">0", ">", "<0", "<", ">=", "<=", "==", "!="
                op = direction.rstrip("0").strip() or direction
                try:
                    val = _eval_expr(expr, measured, seed)
                except Exception:
                    val = None
                if val is None:
                    per_seed[seed] = "blocked"
                else:
                    cmp = {
                        ">": lambda v: v > 0,
                        "<": lambda v: v < 0,
                        ">=": lambda v: v >= 0,
                        "<=": lambda v: v <= 0,
                        "==": lambda v: v == 0,
                        "!=": lambda v: v != 0,
                    }.get(op, lambda v: v > 0)
                    per_seed[seed] = "pass" if cmp(val) else "fail"
            elif kind == "value":
                expr = c.get("quantity") or c.get("predicate")
                claimed = float(c.get("claimed"))
                tol = float(c.get("tolerance"))
                try:
                    val = _eval_expr(expr, measured, seed)
                except Exception:
                    val = None
                if val is None:
                    per_seed[seed] = "blocked"
                else:
                    per_seed[seed] = "pass" if abs(val - claimed) <= tol else "fail"
            elif kind in ("invariant", "existence"):
                pred = c.get("predicate")
                try:
                    val = _eval_expr(pred, measured, seed)
                except Exception:
                    val = None
                if val is None:
                    per_seed[seed] = "blocked"
                else:
                    per_seed[seed] = "pass" if bool(val) else "fail"
            else:
                per_seed[seed] = "blocked"
        all_pass = all(v == "pass" for v in per_seed.values()) and bool(per_seed)
        results.append({
            "id": cid, "kind": kind, "compute_invariance": c.get("compute_invariance"),
            "per_seed": per_seed, "verdict": "pass" if all_pass else (
                "blocked" if all(v == "blocked" for v in per_seed.values()) else "fail"),
        })
    return {"seeds": seeds, "claims": results}


def _eval_expr(expr, measured, seed):
    import re
    pattern = re.compile(r"measured\.(\w+)\.(\w+)")
    def repl(mo):
        a, met = mo.group(1), mo.group(2)
        block = measured.get(a, {}).get(seed, {})
        v = block.get(met)
        val, blocked = _num(v)
        if blocked or val is None:
            raise KeyError(f"blocked/missing metric measured.{a}.{met} at seed {seed}")
        return repr(val)
    e = pattern.sub(repl, expr)
    return eval(e, {"__builtins__": {}}, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="claims.json")
    ap.add_argument("--measured", default="measured.json")
    ap.add_argument("--out", default="selfcheck.json")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    res = evaluate(args.claims, args.measured)
    Path(args.out).write_text(json.dumps(res, indent=2))
    n_pass = sum(1 for c in res["claims"] if c["verdict"] == "pass")
    n_fail = sum(1 for c in res["claims"] if c["verdict"] == "fail")
    n_block = sum(1 for c in res["claims"] if c["verdict"] == "blocked")
    print(f"selfcheck: {n_pass} pass / {n_fail} fail / {n_block} blocked "
          f"of {len(res['claims'])} claims (seeds {res['seeds']})", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    if args.strict and (n_fail or n_block):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
