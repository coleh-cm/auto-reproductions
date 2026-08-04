#!/usr/bin/env python3
"""numbers_gate.py — evaluate claims.json against measured.json, write claims_result.json.

The numbers gate. For each claim in claims.json it resolves the
``measured.<arm>.<metric>`` tokens from measured.json (per seed) and renders a
verdict:

  - ``pass``    the assertion holds at every seed the reproduction ran.
  - ``fail``    the assertion does not hold at some seed the reproduction ran.
  - ``blocked`` a referenced metric is missing or carries the ``BLOCKED``
                sentinel (the environment could not produce a number); the
                claim is NOT adjudicated.

Semantics mirror claims.json['evaluation_rules']:
  - ordering : quantity (an expression over measured.<arm>.<metric>) must
               satisfy `direction` at EVERY seed.
  - value    : mean of quantity over the claim's seeds must satisfy
               |mean - claimed| <= tolerance.  (Per the rules, value claims
               compare the MEAN over seeds, not per-seed; an aggregate
               `mean(...)`/`min(...)`/`max(...)` wrapping a single token is
               also honored as an explicit cross-seed aggregate.)
  - existence/invariant : the boolean `predicate` must hold at every seed.
  - curve    : quantity (and, for above/below/crosses, `against`) resolve to
               SEQUENCES stored in measured.json under the arm (one value per x
               in the order of the claim's `x` list); sample at `x`; diffs =
               quantity - against:
                 above     = all sampled diffs > 0
                 below     = all sampled diffs < 0
                 crosses   = first sampled diff > 0 and last < 0
                 increasing= last > first with >=80% of consecutive diffs >= 0
                 matches   = elementwise |sampled - claimed| <= tolerance
               The comparison must hold at EVERY seed.

Tokens are derived from the claim expressions themselves: every
``measured.<arm>.<metric>`` substring where <arm> is a key of
claims.json['arms']. This avoids depending on a separate metric registry and
handles metrics whose names contain dots (the known arm name is the split
point, longest-first).

Output: claims_result.json = {produced_by, summary, claims, not_tested,
generated_from}. Prints one FINAL line per verdict category and one
FINAL gate=PASS|FAIL line (gate passes iff every HIGH claim is adjudicated
``pass`` with none blocked). Exits non-zero on a blocked HIGH claim or a
failed HIGH claim — the gate's load-bearing verdict is over
compute_invariance == "high".

produced_by stamps the file with the gate's identity: claims_result.json must
be written BY this gate and never by hand.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
CLAIMS_PATH = REPO / "claims.json"
MEASURED_PATH = REPO / "measured.json"
OUT_PATH = REPO / "claims_result.json"

BLOCKED = "BLOCKED"


def build_canonical_tokens(claims_doc: dict) -> dict:
    """token -> (arm, metric) for every measured.<arm>.<metric> referenced by
    any claim, where <arm> is a key of claims.json['arms']. Sorted longest-first
    so substitution never partly overlaps (arm names are matched as literal
    strings, not regex)."""
    arm_names = list(claims_doc.get("arms", {}).keys())
    # longest arm name first so 'maxout_large_adv' wins over 'maxout_adv' etc.
    arm_names_sorted = sorted(arm_names, key=len, reverse=True)
    tokens = {}
    seen = set()

    def harvest(expr):
        if not isinstance(expr, str):
            return
        for arm in arm_names_sorted:
            prefix = f"measured.{arm}."
            i = 0
            while True:
                j = expr.find(prefix, i)
                if j < 0:
                    break
                end = j + len(prefix)
                # metric = up to the next char that ends a python identifier
                # tail: stop at ) , space + - * / < > = & | ] and end of string
                k = end
                while k < len(expr) and (expr[k].isalnum() or expr[k] in "_.[]'\""):
                    k += 1
                token = expr[j:k]
                if token and token not in seen:
                    seen.add(token)
                    metric = expr[end:k]
                    tokens[token] = (arm, metric)
                i = k if k > i else end

    for c in claims_doc.get("claims", []):
        harvest(c.get("quantity"))
        harvest(c.get("predicate"))
        harvest(c.get("against"))
        harvest(c.get("reference"))
        harvest(c.get("x"))
    return dict(sorted(tokens.items(), key=lambda kv: len(kv[0]), reverse=True))


def _safe_eval(expr: str) -> object:
    """Eval an arithmetic/boolean expression with min/max/mean/abs/count only.
    measured tokens MUST already be substituted out. Only the contract's names
    are in scope."""
    allowed = {"min": min, "max": max, "mean": statistics.mean,
               "abs": abs, "count": lambda it: len(list(it)),
               "True": True, "False": False}
    code = compile(expr, "<claim-expr>", "eval")
    for name in code.co_names:
        if name not in allowed:
            raise ValueError(f"disallowed name in expression: {name!r}")
    return eval(code, {"__builtins__": {}}, allowed)


def _tokens_in(expr: str, canonical: dict) -> list:
    return [t for t in canonical if t in expr]


def _substitute(expr: str, token_values: dict, canonical: dict) -> str:
    out = expr
    for token in canonical:  # longest-first
        if token in token_values and token in out:
            out = out.replace(token, repr(token_values[token]))
    leftover = [t for t in canonical if t in out and t not in token_values]
    if leftover:
        raise ValueError(f"unresolved measured token in expression: {leftover}")
    return out


def is_aggregate(expr: str, canonical: dict) -> bool:
    """True iff min/max/mean wraps a SINGLE canonical token (cross-seed agg)."""
    for fn in ("min", "max", "mean"):
        for token in canonical:
            if f"{fn}({token})" in expr:
                return True
    return False


def _measured_value(measured: dict, arm: str, metric: str, seed):
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


def _sample_at_x(seq, x_full, x_want):
    """Return seq sampled at the x values in x_want (by exact match, falling
    back to nearest). x_full is the claim's `x` list (the index labels)."""
    out = []
    for xv in x_want:
        # exact match first
        if xv in x_full:
            out.append(seq[x_full.index(xv)])
            continue
        # nearest
        best_i = min(range(len(x_full)), key=lambda i: abs(x_full[i] - xv))
        out.append(seq[best_i])
    return out


def _curve_verdict(comparison, q, r, claimed, tolerance):
    n = len(q)
    if comparison == "crosses":
        if r is None:
            raise ValueError("crosses needs an `against` curve sequence")
        d0, dn = q[0] - r[0], q[-1] - r[-1]
        return d0 > 0 and dn < 0
    if comparison == "above":
        if r is not None:
            return all(qi - ri > 0 for qi, ri in zip(q, r))
        c = float(claimed)
        return all(qi - c > 0 for qi in q)
    if comparison == "below":
        if r is not None:
            return all(qi - ri < 0 for qi, ri in zip(q, r))
        c = float(claimed)
        return all(qi - c < 0 for qi in q)
    if comparison == "increasing":
        if n < 2:
            return False
        if not (q[-1] > q[0]):
            return False
        diffs = [q[i + 1] - q[i] for i in range(n - 1)]
        nonneg = sum(1 for d in diffs if d >= 0)
        return nonneg >= 0.8 * len(diffs)
    if comparison == "decreasing":
        if n < 2:
            return False
        if not (q[-1] < q[0]):
            return False
        diffs = [q[i + 1] - q[i] for i in range(n - 1)]
        nonpos = sum(1 for d in diffs if d <= 0)
        return nonpos >= 0.8 * len(diffs)
    if comparison == "matches":
        if not isinstance(claimed, list) or len(claimed) != n:
            raise ValueError("matches needs `claimed` as a list of the same length")
        return all(abs(qi - ci) <= tolerance for qi, ci in zip(q, claimed))
    raise ValueError(f"unknown curve comparison {comparison!r}")


def evaluate_curve_claim(claim, measured, top_seeds, canonical):
    seeds_requested = claim.get("seeds", top_seeds)
    comparison = claim["comparison"]
    q_tok = claim["quantity"]
    r_tok = claim.get("against", claim.get("reference"))
    x_list = claim["x"]
    if q_tok not in canonical:
        return {"verdict": "blocked", "reason": "quantity is not a measured token",
                "seeds_evaluated": []}
    q_arm, q_metric = canonical[q_tok]
    r_arm = r_metric = None
    if r_tok is not None:
        if r_tok not in canonical:
            return {"verdict": "blocked", "reason": "against is not a measured token",
                    "seeds_evaluated": []}
        r_arm, r_metric = canonical[r_tok]
        if r_arm != q_arm:
            return {"verdict": "blocked",
                    "reason": "curve claims must draw quantity/against from ONE arm",
                    "seeds_evaluated": []}
    arm = q_arm
    seeds = sorted(set(seeds_requested) & _available_seeds(measured, arm))
    if not seeds:
        return {"verdict": "blocked", "reason": f"no seed data for arm {arm}",
                "seeds_evaluated": []}
    per_seed = []
    for s in seeds:
        q_raw = _measured_value(measured, arm, q_metric, s)
        r_raw = _measured_value(measured, arm, r_metric, s) if r_tok else None
        if q_raw == BLOCKED or (r_tok and r_raw == BLOCKED):
            return {"verdict": "blocked", "reason": f"BLOCKED cell at seed {s}",
                    "seeds_evaluated": []}
        if not isinstance(q_raw, list):
            return {"verdict": "blocked",
                    "reason": f"{arm}.{q_metric} at seed {s} is not a sequence",
                    "seeds_evaluated": []}
        if r_tok and not isinstance(r_raw, list):
            return {"verdict": "blocked",
                    "reason": f"{arm}.{r_metric} at seed {s} is not a sequence",
                    "seeds_evaluated": []}
        # sample at x
        q = _sample_at_x(q_raw, x_list, x_list)
        r = _sample_at_x(r_raw, x_list, x_list) if r_tok else None
        try:
            ok = _curve_verdict(comparison, q, r, claim.get("claimed"),
                                float(claim.get("tolerance", 0.0)))
        except Exception as e:
            return {"verdict": "blocked", "reason": f"seed {s}: {e}",
                    "seeds_evaluated": []}
        per_seed.append({"seed": s, "ok": bool(ok), "n_samples": len(q),
                         "q_first": q[0], "q_last": q[-1]})
    verdict = "pass" if all(p["ok"] for p in per_seed) else "fail"
    return {"verdict": verdict, "seeds_evaluated": [p["seed"] for p in per_seed],
            "comparison": comparison, "per_seed": per_seed}


def evaluate_claim(claim, measured, top_seeds, canonical, claims_doc=None):
    kind = claim["kind"]
    if kind == "curve":
        return evaluate_curve_claim(claim, measured, top_seeds, canonical)
    seeds_requested = claim.get("seeds", top_seeds)
    if kind in ("existence", "invariant"):
        expr, is_bool = claim["predicate"], True
    else:
        expr, is_bool = claim["quantity"], False

    tokens = _tokens_in(expr, canonical)
    if not tokens:
        return {"verdict": "blocked", "reason": "no measured token in expression",
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
            if _measured_value(measured, arm, metric, s) == BLOCKED:
                return {"verdict": "blocked",
                        "reason": f"BLOCKED cell at seed {s}: {t}",
                        "seeds_evaluated": []}

    try:
        if is_aggregate(expr, canonical):
            token_values = {}
            for t in tokens:
                arm, metric = canonical[t]
                token_values[t] = [_measured_value(measured, arm, metric, s) for s in seeds]
            sub = _substitute(expr, token_values, canonical)
            result = _safe_eval(sub)
            if is_bool:
                return {"verdict": "pass" if bool(result) else "fail",
                        "seeds_evaluated": seeds, "aggregate_value": bool(result)}
            val = float(result)
            ok = abs(val - claim["claimed"]) <= claim["tolerance"]
            return {"verdict": "pass" if ok else "fail", "seeds_evaluated": seeds,
                    "aggregate_value": val, "claimed": claim["claimed"],
                    "tolerance": claim["tolerance"],
                    "abs_diff": abs(val - claim["claimed"])}

        # value claim: compare MEAN over seeds (per evaluation_rules)
        if kind == "value" and not is_bool:
            per_seed_vals = []
            for s in seeds:
                tv = {t: _measured_value(measured, *canonical[t], s) for t in tokens}
                sub = _substitute(expr, tv, canonical)
                per_seed_vals.append(float(_safe_eval(sub)))
            mean_val = statistics.mean(per_seed_vals)
            ok = abs(mean_val - claim["claimed"]) <= claim["tolerance"]
            return {"verdict": "pass" if ok else "fail", "seeds_evaluated": seeds,
                    "mean": mean_val, "per_seed": per_seed_vals,
                    "claimed": claim["claimed"], "tolerance": claim["tolerance"],
                    "abs_diff": abs(mean_val - claim["claimed"])}

        # ordering / existence / invariant: per-seed, must hold at EVERY seed
        per_seed = []
        for s in seeds:
            tv = {t: _measured_value(measured, *canonical[t], s) for t in tokens}
            sub = _substitute(expr, tv, canonical)
            val = _safe_eval(sub)
            if is_bool:
                ok = bool(val)
            else:  # ordering
                v = float(val)
                d = claim["direction"]
                ok = (v < 0 if d == "<0" else v > 0 if d == ">0"
                      else v <= 0 if d == "<=0" else v >= 0 if d == ">=0"
                      else v == 0 if d == "==0"
                      else (_ for _ in ()).throw(ValueError(f"unknown direction {d!r}")))
            per_seed.append({"seed": s, "value": val, "ok": ok})
        detail = {"per_seed": per_seed}
        if kind == "ordering":
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
        r = evaluate_claim(claim, measured, top_seeds, canonical, claims_doc)
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
        "note": "The gate's load-bearing verdict is over compute_invariance == 'high'. "
                "low claims are informational; several are expected to fail at sub-scale "
                "(annotated in claims.json).",
    }

    out = {
        "produced_by": "numbers_gate.py",
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
