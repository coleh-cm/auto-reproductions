"""Claims evaluator — the ONLY workflow component that writes claims_result.json (the
verdict table). This is the WORKFLOW's artifact (generated_by='workflow_subagent'); the
top-level agent's own redundant check is selfcheck_claims.py -> selfcheck.json.

Reads claims.json + measured.json. Produces a verdict per claim per the claims.json
evaluation_rule, using the numbers-gate vocabulary reproduced / refuted / untested / blocked:

  ordering:   per-seed quantity; UNTESTED if |mean| <= cross-seed spread (max-min) (the
              arms are not separated, a within-noise ordering is not established);
              otherwise reproduced if `direction` holds at EVERY seed, refuted if it
              fails at any seed.
  value:      reproduced iff |per-seed-MEAN - claimed| <= tolerance; refuted otherwise.
  invariant:  `predicate` is a Python boolean expression over measured.<arm>.<metric>
              tokens (comparisons, and/or, arithmetic; abs/all/any/bool/float/int/len/
              max/min/round/sorted/sum and `measured`, nothing else) evaluated at EVERY
              seed; reproduced iff true at every seed, refuted if false at any seed.
  curve:      `quantity` (and `against`/`claimed`) are measured refs whose values are
              per-x SEQUENCES; elementwise comparison (above/below/matches/increasing/
              decreasing) at every x at every seed.

If ANY measured.<arm>.<metric> referenced by a claim is the string "BLOCKED" (or a
BLOCKED entry of a sequence, or absent), the verdict is 'blocked'. NO success path
reports reproduced on an empty or BLOCKED result.
"""
from __future__ import annotations
import json
import os
import re

REPO = os.path.dirname(os.path.abspath(__file__))

# Numbers-gate verdict vocabulary.
REPRODUCED = "reproduced"
REFUTED = "refuted"
UNTESTED = "untested"
BLOCKED = "blocked"

# Builtins the gate allows in predicate/quantity expressions.
_SAFE_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bool': bool, 'float': float, 'int': int,
    'len': len, 'max': max, 'min': min, 'round': round, 'sorted': sorted, 'sum': sum,
}


def _load(path):
    with open(path) as f:
        return json.load(f)


def _refs(quantity, against=None):
    """Return sorted list of (arm, metric) referenced in a claim's quantity/against/
    predicate strings. Sorted (not a set) so the blocked-detail arm name and any per-ref
    loop order are deterministic and claims_result.json is reproducible across runs."""
    text = (quantity or '') + '\n' + (against or '')
    return sorted(set(re.findall(r'measured\.(\w+)\.(\w+)', text)))


def _is_blocked(measured, arm, metric, seed):
    """True if the referenced measured value is BLOCKED (or absent). A sequence metric
    (curve, one value per x) is BLOCKED if ANY entry is BLOCKED."""
    try:
        v = measured[arm][str(seed)][metric]
    except (KeyError, TypeError):
        return True
    if v is None:
        return True
    if isinstance(v, list):
        return any(x == "BLOCKED" or x is None for x in v)
    return v == "BLOCKED"


def _seed_value(measured, arm, metric, seed):
    """Return the measured value for (arm, metric, seed); None if BLOCKED/absent."""
    try:
        v = measured[arm][str(seed)][metric]
    except (KeyError, TypeError):
        return None
    if v == "BLOCKED" or v is None:
        return None
    return v


def _blocked_result(detail):
    return {'verdict': BLOCKED, 'measured': 'BLOCKED', 'spread': 'BLOCKED', 'detail': detail}


def _eval_scalar(expr, measured, seed):
    """Evaluate a scalar quantity expression (ordering/value) for one seed.
    Returns a float, or None if a referenced metric is BLOCKED/absent."""
    refs = re.findall(r'measured\.(\w+)\.(\w+)', expr)
    local = {}
    for arm, metric in refs:
        v = _seed_value(measured, arm, metric, seed)
        if v is None:
            return None
        local[f"measured_{arm}_{metric}"] = float(v)
    safe = re.sub(r'measured\.(\w+)\.(\w+)', r'measured_\1_\2', expr)
    env = dict(_SAFE_BUILTINS)
    env.update(local)
    try:
        return float(eval(safe, {'__builtins__': {}}, env))
    except Exception:
        return None


def _eval_bool(predicate, measured, seed):
    """Evaluate an invariant predicate (boolean expression) for one seed.
    Returns True/False, or None if a referenced metric is BLOCKED/absent."""
    refs = re.findall(r'measured\.(\w+)\.(\w+)', predicate)
    local = {}
    for arm, metric in refs:
        v = _seed_value(measured, arm, metric, seed)
        if v is None:
            return None
        local[f"measured_{arm}_{metric}"] = float(v)
    safe = re.sub(r'measured\.(\w+)\.(\w+)', r'measured_\1_\2', predicate)
    env = dict(_SAFE_BUILTINS)
    env.update(local)
    try:
        return bool(eval(safe, {'__builtins__': {}}, env))
    except Exception:
        return None


def _eval_seq(expr, measured, seed):
    """Evaluate a curve quantity/against expression for one seed -> a list (one value
    per x), or None if a referenced metric is BLOCKED/absent. The expression is a plain
    measured.<arm>.<metric> ref whose stored value is the per-x sequence; we also support
    safe combinators (min/max/...) over sequences for derived curves."""
    refs = re.findall(r'measured\.(\w+)\.(\w+)', expr)
    local = {}
    for arm, metric in refs:
        v = _seed_value(measured, arm, metric, seed)
        if v is None:
            return None
        local[f"measured_{arm}_{metric}"] = v
    safe = re.sub(r'measured\.(\w+)\.(\w+)', r'measured_\1_\2', expr)
    env = dict(_SAFE_BUILTINS)
    env.update(local)
    try:
        out = eval(safe, {'__builtins__': {}}, env)
    except Exception:
        return None
    if out is None:
        return None
    if isinstance(out, list):
        if any(x == "BLOCKED" or x is None for x in out):
            return None
        return out
    # scalar wrapped as a 1-element sequence
    return [out]


def _eval_ordering(claim, measured, seeds):
    quantity = claim['quantity']
    direction = claim['direction']
    refs = _refs(quantity)
    for arm, metric in refs:
        for s in seeds:
            if _is_blocked(measured, arm, metric, s):
                return _blocked_result(
                    f'arm {arm}.{metric} blocked in this sandbox (no CUDA / no HF_TOKEN)')
    per_seed = []
    for s in seeds:
        val = _eval_scalar(quantity, measured, s)
        if val is None:
            return _blocked_result('could not evaluate quantity (metric not measured)')
        per_seed.append(val)
    mean = sum(per_seed) / len(per_seed)
    spread = max(per_seed) - min(per_seed)
    # within-noise: arms not separated -> untested (not falsified, not reproduced)
    if abs(mean) <= spread:
        return {'verdict': UNTESTED, 'measured': mean, 'spread': per_seed,
                'detail': f'|mean {mean:.4g}| <= spread {spread:.4g}: arms not separated'}
    if direction == '>0':
        ok = all(v > 0 for v in per_seed)
    elif direction == '<0':
        ok = all(v < 0 for v in per_seed)
    elif direction == '>=0':
        ok = all(v >= 0 for v in per_seed)
    elif direction == '<=0':
        ok = all(v <= 0 for v in per_seed)
    else:
        return _blocked_result(f'unknown direction {direction}')
    return {'verdict': REPRODUCED if ok else REFUTED, 'measured': mean, 'spread': per_seed,
            'detail': f'direction {direction} holds at every seed: {ok} (mean {mean:.4g}, spread {spread:.4g})'}


def _eval_value(claim, measured, seeds):
    quantity = claim['quantity']
    claimed = claim['claimed']
    tol = claim['tolerance']
    refs = _refs(quantity)
    for arm, metric in refs:
        for s in seeds:
            if _is_blocked(measured, arm, metric, s):
                return _blocked_result(f'arm {arm}.{metric} blocked in this sandbox')
    per_seed = []
    for s in seeds:
        v = _eval_scalar(quantity, measured, s)
        if v is None:
            return _blocked_result('could not evaluate quantity')
        per_seed.append(v)
    mean = sum(per_seed) / len(per_seed)
    gap = abs(mean - claimed)
    ok = gap <= tol
    return {'verdict': REPRODUCED if ok else REFUTED, 'measured': mean, 'claimed': claimed,
            'tolerance': tol, 'gap': gap, 'spread': per_seed,
            'detail': f'mean {mean:.4g} vs claimed {claimed} +/- {tol} (gap {gap:.4g})'}


def _eval_invariant(claim, measured, seeds):
    """C1/C2/C3: evaluate the executable `predicate` (a boolean expression over
    measured.<arm>.<metric> residuals) at every seed. reproduced iff true at every seed;
    refuted if false at any seed; blocked if any referenced metric is BLOCKED/absent."""
    predicate = claim['predicate']
    refs = _refs(predicate)
    for arm, metric in refs:
        for s in seeds:
            if _is_blocked(measured, arm, metric, s):
                return _blocked_result(
                    f'invariant metric {arm}.{metric} blocked/absent (no CUDA / no HF_TOKEN)')
    per_seed_bool = []
    for s in seeds:
        b = _eval_bool(predicate, measured, s)
        if b is None:
            return _blocked_result('could not evaluate predicate (metric not measured)')
        per_seed_bool.append(b)
    ok = all(per_seed_bool)
    return {'verdict': REPRODUCED if ok else REFUTED,
            'measured': 'true_at_every_seed' if ok else 'false_at_some_seed',
            'spread': per_seed_bool,
            'detail': f'predicate {"true" if ok else "false"} at every seed: {per_seed_bool}'}


def _eval_curve(claim, measured, seeds):
    quantity = claim['quantity']
    against = claim.get('against')
    comparison = claim['comparison']
    x = claim['x']
    refs = _refs(quantity, against)
    for arm, metric in refs:
        for s in seeds:
            if _is_blocked(measured, arm, metric, s):
                return _blocked_result(
                    f'curve metric {arm}.{metric} blocked in this sandbox (no CUDA / no HF_TOKEN)')
    per_seed_q = []
    per_seed_a = []
    for s in seeds:
        q = _eval_seq(quantity, measured, s)
        if q is None or len(q) != len(x):
            return _blocked_result('curve quantity not measured / wrong length')
        per_seed_q.append([float(v) for v in q])
        if comparison in ('above', 'below'):
            a = _eval_seq(against, measured, s)
            if a is None or len(a) != len(x):
                return _blocked_result('curve against not measured / wrong length')
            per_seed_a.append([float(v) for v in a])

    if comparison == 'above':
        ok = all(all(q[i] > a[i] for i in range(len(x)))
                 for q, a in zip(per_seed_q, per_seed_a))
        detail = 'quantity > against at every x every seed'
    elif comparison == 'below':
        ok = all(all(q[i] < a[i] for i in range(len(x)))
                 for q, a in zip(per_seed_q, per_seed_a))
        detail = 'quantity < against at every x every seed'
    elif comparison == 'matches':
        claimed = claim['claimed']
        tol = claim['tolerance']
        ok = all(all(abs(q[i] - float(claimed[i])) <= tol for i in range(len(x)))
                 for q in per_seed_q)
        detail = f'|quantity - claimed| <= {tol} at every x every seed'
    elif comparison == 'increasing':
        ok = all(all(q[i + 1] > q[i] for i in range(len(x) - 1)) for q in per_seed_q)
        detail = 'quantity strictly increasing at every x every seed'
    elif comparison == 'decreasing':
        ok = all(all(q[i + 1] < q[i] for i in range(len(x) - 1)) for q in per_seed_q)
        detail = 'quantity strictly decreasing at every x every seed'
    else:
        return _blocked_result(f'unknown curve comparison {comparison}')
    return {'verdict': REPRODUCED if ok else REFUTED, 'measured': per_seed_q,
            'spread': per_seed_q, 'detail': detail}


def evaluate(claims_path, measured_path, e1_path=None):
    claims = _load(claims_path)
    if not claims.get('claims'):
        raise RuntimeError("evaluate_claims: claims.json has no claims (no success on empty)")
    measured = _load(measured_path) if os.path.exists(measured_path) else {}
    if not measured:
        raise RuntimeError("evaluate_claims: measured.json missing or empty (no success on empty)")
    seeds = claims['seeds']
    verdicts = []
    for cl in claims['claims']:
        kind = cl['kind']
        if kind == 'ordering':
            r = _eval_ordering(cl, measured, seeds)
        elif kind == 'value':
            r = _eval_value(cl, measured, seeds)
        elif kind == 'invariant':
            r = _eval_invariant(cl, measured, seeds)
        elif kind == 'curve':
            r = _eval_curve(cl, measured, seeds)
        else:
            r = _blocked_result(f'unknown kind {kind}')
        v = {
            'id': cl['id'], 'kind': kind, 'quote': cl.get('quote', ''),
            'citation': cl.get('citation', ''), 'compute_invariance': cl.get('compute_invariance'),
            'verdict': r['verdict'], 'measured': r.get('measured'),
            'claimed': cl.get('claimed'), 'tolerance': cl.get('tolerance'),
            'spread': r.get('spread'), 'detail': r.get('detail'),
        }
        if 'gap' in r:
            v['gap'] = r['gap']
        verdicts.append(v)
    summary = {REPRODUCED: sum(1 for v in verdicts if v['verdict'] == REPRODUCED),
               REFUTED: sum(1 for v in verdicts if v['verdict'] == REFUTED),
               UNTESTED: sum(1 for v in verdicts if v['verdict'] == UNTESTED),
               BLOCKED: sum(1 for v in verdicts if v['verdict'] == BLOCKED)}
    return {
        'schema_version': 1,
        'paper_ref': '0e6756c9-d827-4749-a8d7-3140a8c98a25',
        'project_id': '06910d54-1d99-4864-8bff-ab3007a0c70e',
        'generated_by': 'workflow_subagent',
        'verdicts': verdicts,
        'summary': summary,
    }


def main():
    out = os.path.join(REPO, 'claims_result.json')
    result = evaluate(os.path.join(REPO, 'claims.json'),
                      os.path.join(REPO, 'measured.json'))
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"claims_result.json: reproduced={result['summary'][REPRODUCED]} "
          f"refuted={result['summary'][REFUTED]} untested={result['summary'][UNTESTED]} "
          f"blocked={result['summary'][BLOCKED]}")
    for v in result['verdicts']:
        print(f"  {v['id']:4s} {v['kind']:9s} {v['verdict']:10s} {v['detail']}")


if __name__ == '__main__':
    main()
