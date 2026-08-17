"""Claims evaluator — the ONLY component that writes claims_result.json (the verdict
table). This is the WORKFLOW's artifact (generated_by='workflow_subagent'); the
top-level agent's own redundant check is selfcheck_claims.py -> selfcheck.json.

Reads claims.json, measured.json, results/e1_synth.json. Produces a verdict per claim
per the claims.json evaluation_rule:
  ordering:   pass iff the stated direction holds at EVERY seed
  value:      pass iff the per-seed MEAN lies within claimed +/- tolerance
  invariant:  pass iff the predicate holds at EVERY seed (C1-C3 via results/e1_synth.json)
  curve:      sequence sampled at x; 'above'/'below' vs against elementwise; 'matches' vs
              claimed within tolerance; 'increasing'/'decreasing' adjacent diffs of stated
              sign — all at every x at every seed

If ANY measured.<arm>.<metric> referenced by a claim is the string "BLOCKED", the verdict
is 'blocked'. NO success path reports pass on an empty result.
"""
from __future__ import annotations
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def _load(path):
    with open(path) as f:
        return json.load(f)


def _refs(quantity, against=None):
    """Return set of (arm, metric) referenced in a claim's quantity/against strings."""
    text = (quantity or '') + '\n' + (against or '')
    return set(re.findall(r'measured\.(\w+)\.(\w+)', text))


def _is_blocked(measured, arm, metric, seed):
    """True if the referenced measured value is BLOCKED (or absent)."""
    try:
        v = measured[arm][str(seed)][metric]
    except (KeyError, TypeError):
        return True
    return v == "BLOCKED" or v is None


def _seed_values(measured, arm, metric, seeds):
    """Return list of per-seed values for (arm, metric); None if any BLOCKED/absent."""
    vals = []
    for s in seeds:
        try:
            v = measured[arm][str(s)][metric]
        except (KeyError, TypeError):
            return None
        if v == "BLOCKED" or v is None:
            return None
        vals.append(float(v))
    return vals


def _eval_ordering(claim, measured, seeds):
    quantity = claim['quantity']
    direction = claim['direction']
    refs = _refs(quantity)
    # any BLOCKED ref -> blocked
    for arm, metric in refs:
        for s in seeds:
            if _is_blocked(measured, arm, metric, s):
                return {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
                        'detail': f'arm {arm}.{metric} blocked in this sandbox (no CUDA / no HF_TOKEN)'}
    # evaluate the quantity expression per seed; require direction at every seed
    per_seed = []
    for s in seeds:
        val = _eval_expr(quantity, measured, s)
        if val is None:
            return {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
                    'detail': 'could not evaluate quantity (metric not measured)'}
        per_seed.append(val)
    if direction == '>0':
        ok = all(v > 0 for v in per_seed)
    elif direction == '<0':
        ok = all(v < 0 for v in per_seed)
    elif direction == '>=0':
        ok = all(v >= 0 for v in per_seed)
    elif direction == '<=0':
        ok = all(v <= 0 for v in per_seed)
    else:
        return {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
                'detail': f'unknown direction {direction}'}
    return {'verdict': 'pass' if ok else 'fail', 'measured': _summarise(per_seed),
            'spread': per_seed, 'detail': f'direction {direction} holds at every seed: {ok}'}


def _eval_value(claim, measured, seeds):
    quantity = claim['quantity']
    claimed = claim['claimed']
    tol = claim['tolerance']
    refs = _refs(quantity)
    for arm, metric in refs:
        for s in seeds:
            if _is_blocked(measured, arm, metric, s):
                return {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
                        'detail': f'arm {arm}.{metric} blocked in this sandbox'}
    per_seed = []
    for s in seeds:
        v = _eval_expr(quantity, measured, s)
        if v is None:
            return {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
                    'detail': 'could not evaluate quantity'}
        per_seed.append(v)
    mean = sum(per_seed) / len(per_seed)
    ok = abs(mean - claimed) <= tol
    return {'verdict': 'pass' if ok else 'fail', 'measured': mean, 'claimed': claimed,
            'tolerance': tol, 'spread': per_seed,
            'detail': f'mean {mean:.4g} vs claimed {claimed} +/- {tol}'}


def _eval_invariant(claim, e1):
    """C1/C2/C3: read results/e1_synth.json per-seed predicate pass bools."""
    cid = claim['id']
    if e1 is None:
        return {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
                'detail': 'results/e1_synth.json missing'}
    per_seed_pass = []
    details = []
    metrics_agg = {}
    for seed, seed_res in e1.items():
        if cid not in seed_res:
            return {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
                    'detail': f'{cid} missing from e1_synth.json seed {seed}'}
        cell = seed_res[cid]
        per_seed_pass.append(bool(cell['pass']))
        details.append(cell['detail'])
        for k, val in cell.get('metrics', {}).items():
            metrics_agg.setdefault(k, []).append(val)
    ok = all(per_seed_pass) and len(per_seed_pass) > 0
    return {'verdict': 'pass' if ok else 'fail', 'measured': 'PASS' if ok else 'FAIL',
            'spread': per_seed_pass, 'detail': '; '.join(details),
            'metrics': metrics_agg}


def _eval_curve(claim, measured, seeds):
    quantity = claim['quantity']
    x = claim['x']
    comparison = claim['comparison']
    refs = _refs(quantity, claim.get('against'))
    for arm, metric in refs:
        for s in seeds:
            if _is_blocked(measured, arm, metric, s):
                return {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
                        'detail': f'curve metric {arm}.{metric} blocked in this sandbox'}
    # Build the quantity sequence per seed at each x. Curve quantities reference
    # measured.<arm>.<metric>(beta, ...) — a parametric sweep. In this sandbox every
    # such metric is BLOCKED, so the blocked branch above already returned.
    # (On a GPU host a real sweep would populate measured[arm][seed][f'{metric}@beta={x}']
    # or equivalent; that path is not reachable here.)
    return {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
            'detail': 'curve sweep not available in this sandbox (model metrics BLOCKED)'}


def _eval_expr(expr, measured, seed):
    """Evaluate a claims quantity expression like
    'measured.midsteer.moto_cs_on_moto - measured.vanilla.moto_cs_on_moto' or
    'min(measured.a.x, measured.b.y) - measured.c.z' for one seed.
    Returns a float, or None if a referenced metric is BLOCKED/absent.
    """
    # find all measured.arm.metric references; if any is BLOCKED -> None
    refs = re.findall(r'measured\.(\w+)\.(\w+)', expr)
    local = {}
    for arm, metric in refs:
        try:
            v = measured[arm][str(seed)][metric]
        except (KeyError, TypeError):
            return None
        if v == "BLOCKED" or v is None:
            return None
        local[f"measured_{arm}_{metric}"] = float(v)
    # build a python expression with measured.arm.metric -> measured_arm_metric
    safe = re.sub(r'measured\.(\w+)\.(\w+)', r'measured_\1_\2', expr)
    # allow min, max, abs, len
    env = {'min': min, 'max': max, 'abs': abs, 'len': len}
    env.update(local)
    try:
        return float(eval(safe, {'__builtins__': {}}, env))
    except Exception:
        return None


def _summarise(vals):
    if not vals:
        return 'BLOCKED'
    if all(v == vals[0] for v in vals):
        return vals[0]
    return {'mean': sum(vals) / len(vals), 'min': min(vals), 'max': max(vals)}


def evaluate(claims_path, measured_path, e1_path):
    claims = _load(claims_path)
    if not claims.get('claims'):
        raise RuntimeError("evaluate_claims: claims.json has no claims (no success on empty)")
    measured = _load(measured_path) if os.path.exists(measured_path) else {}
    if not measured:
        raise RuntimeError("evaluate_claims: measured.json missing or empty (no success on empty)")
    e1 = _load(e1_path) if os.path.exists(e1_path) else None
    seeds = claims['seeds']
    verdicts = []
    for cl in claims['claims']:
        kind = cl['kind']
        if kind == 'ordering':
            r = _eval_ordering(cl, measured, seeds)
        elif kind == 'value':
            r = _eval_value(cl, measured, seeds)
        elif kind == 'invariant':
            r = _eval_invariant(cl, e1)
        elif kind == 'curve':
            r = _eval_curve(cl, measured, seeds)
        else:
            r = {'verdict': 'blocked', 'measured': 'BLOCKED', 'spread': 'BLOCKED',
                 'detail': f'unknown kind {kind}'}
        v = {
            'id': cl['id'], 'kind': kind, 'quote': cl.get('quote', ''),
            'citation': cl.get('citation', ''), 'verdict': r['verdict'],
            'measured': r.get('measured'), 'claimed': cl.get('claimed'),
            'tolerance': cl.get('tolerance'), 'spread': r.get('spread'),
            'detail': r.get('detail'),
        }
        if 'metrics' in r:
            v['metrics'] = r['metrics']
        verdicts.append(v)
    summary = {'pass': sum(1 for v in verdicts if v['verdict'] == 'pass'),
               'fail': sum(1 for v in verdicts if v['verdict'] == 'fail'),
               'blocked': sum(1 for v in verdicts if v['verdict'] == 'blocked')}
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
                      os.path.join(REPO, 'measured.json'),
                      os.path.join(REPO, 'results', 'e1_synth.json'))
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"claims_result.json: pass={result['summary']['pass']} "
          f"fail={result['summary']['fail']} blocked={result['summary']['blocked']}")
    for v in result['verdicts']:
        print(f"  {v['id']:4s} {v['kind']:9s} {v['verdict']:7s} {v['detail']}")


if __name__ == '__main__':
    main()
