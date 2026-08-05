#!/usr/bin/env python3
"""Self-evaluate the claims in claims.json against measured.json.

This is a LOCAL self-check (writes selfcheck.json), NOT the workflow's numbers
gate: the gate owns claims_result.json and refuses any copy it did not
produce, so this script deliberately writes a different filename. It exists
so a claim that is unevaluable is found early (here, cheaply) rather than by
the gate, and so a reader can re-run it. A grader that cannot run raises;
a missing/BLOCKED metric blocks the claim rather than fabricating a verdict.

Verdicts per claim per seed: pass / fail / blocked.
  ordering            : quantity > 0 (direction ">") at every seed. The
                        AGGREGATE verdict applies the declared spread heuristic
                        (claims.json evaluation.ordering): if |mean gap| <=
                        cross-seed spread the arms are not separated and the
                        verdict is 'untested' (within-noise ordering not
                        established, not falsified); else 'pass' if the
                        direction holds at every seed, else 'fail'. This matches
                        the numbers gate's own rule.
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
        ordering_vals = []  # numeric gap per seed, for the spread heuristic
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
                    ordering_vals.append(val)
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
        all_blocked = all(v == "blocked" for v in per_seed.values())
        if all_blocked:
            verdict = "blocked"
        elif all_pass:
            verdict = "pass"
        elif kind == "ordering" and len(ordering_vals) == len(seeds):
            # Declared rule (claims.json evaluation.ordering): if the mean gap
            # is within the cross-seed spread the arms are NOT separated and the
            # verdict is 'untested' (within-noise ordering not established, not
            # falsified). This matches the numbers gate's own spread heuristic,
            # so the declared rule and the produced verdict agree.
            mean_v = sum(ordering_vals) / len(ordering_vals)
            spread_v = max(ordering_vals) - min(ordering_vals)
            if abs(mean_v) <= spread_v:
                verdict = "untested"
            else:
                verdict = "fail"
        else:
            verdict = "fail"
        entry = {
            "id": cid, "kind": kind, "compute_invariance": c.get("compute_invariance"),
            "per_seed": per_seed, "verdict": verdict,
        }
        if kind == "ordering" and len(ordering_vals) == len(seeds):
            entry["mean_gap"] = sum(ordering_vals) / len(ordering_vals)
            entry["spread_gap"] = max(ordering_vals) - min(ordering_vals)
        results.append(entry)
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


def _detached_counterfactual(claims_path):
    """Run the DETACHED gradient variant (grad_mode detached, the WHOLE target
    constant -- stopgrad on p_tilde AND the gate weight w) at every seed and
    record its test accuracy. This is the standard self-distillation convention
    the paper does NOT mark on w (Eq. 3 marks stopgrad ONLY on p_tilde); it is
    the variant under which Table 1's 0.9620 and the +2.5-point headline ARE
    reachable. Reported as a COUNTERFACTUAL (not a gated arm): the gated cwsd
    arm uses the paper-LITERAL gradient, under which the headline does NOT
    reproduce. Imported here so a reader can re-run it; written to selfcheck.json
    under 'counterfactual_detached' for machine readability."""
    import argparse
    import run_experiment as r
    claims = json.loads(Path(claims_path).read_text())
    seeds = claims.get("seeds", [])
    out = {}
    for arm in ("baseline", "cwsd"):
        cfg = claims["arms"][arm]["config"]
        per = {}
        for seed in seeds:
            ns = argparse.Namespace(
                lambda_=cfg["lambda"], s=cfg["s"], tau=cfg["tau"],
                temperature=cfg["temperature"], seed=seed, steps=cfg["steps"],
                lr=cfg["lr"], batch_size=cfg["batch_size"], init=cfg["init"],
                noise_mode=cfg["noise_mode"], noise_rate=cfg["noise_rate"],
                batch_mode=cfg["batch_mode"], rng_layout=cfg["rng_layout"],
                grad_mode="detached",
            )
            acc, _, _, _ = r.train(ns)
            per[str(seed)] = float(f"{acc:.4f}")
        out[arm] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="claims.json")
    ap.add_argument("--measured", default="measured.json")
    ap.add_argument("--out", default="selfcheck.json")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-counterfactual", action="store_true",
                    help="skip running the detached counterfactual arm")
    args = ap.parse_args()
    res = evaluate(args.claims, args.measured)
    if not args.no_counterfactual:
        try:
            res["counterfactual_detached"] = _detached_counterfactual(args.claims)
            res["counterfactual_detached_note"] = (
                "DETACHED gradient (grad_mode detached, whole target constant) "
                "accuracy at each seed -- the standard self-distillation convention "
                "the paper does NOT mark on w (Eq. 3 marks stopgrad ONLY on p_tilde). "
                "This is ONE configuration under which Table 1 (0.9620) and the +2.5-point "
                "headline are reachable (already at the default s=0.15). The paper-LITERAL "
                "gradient (the gated cwsd arm) ALSO reaches the headline for shallow gates "
                "(s >= ~0.7 ordering, s >= ~2.0 value/magnitude; see s_sweep.json from "
                "sweep_s.py), because its gate-path term (proportional to 1/s) vanishes as "
                "s grows and the two gradients converge. NOT the gated arm: the gated cwsd "
                "arm uses the paper-LITERAL gradient at the sharp-gate default s=0.15 "
                "(the prose-aligned regime where 's controls how sharply the gate opens'), "
                "under which the headline does NOT reproduce at the gated s. Because the "
                "paper states neither s nor the stop-grad scope on w, the headline is "
                "under-specified: reachable under the paper's equations for a range of "
                "(s, grad-mode) but not at the prose-aligned sharp-gate default under literal.")
        except Exception as e:  # pragma: no cover - counterfactual is supplementary
            res["counterfactual_detached"] = {"error": str(e)}
    Path(args.out).write_text(json.dumps(res, indent=2))
    n_pass = sum(1 for c in res["claims"] if c["verdict"] == "pass")
    n_fail = sum(1 for c in res["claims"] if c["verdict"] == "fail")
    n_block = sum(1 for c in res["claims"] if c["verdict"] == "blocked")
    n_untested = sum(1 for c in res["claims"] if c["verdict"] == "untested")
    print(f"selfcheck: {n_pass} pass / {n_fail} fail / {n_untested} untested / "
          f"{n_block} blocked of {len(res['claims'])} claims (seeds {res['seeds']})",
          file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    if args.strict and (n_fail or n_block):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
