"""Assemble measured.json from the per-experiment partial files + the metric union
parsed from claims.json. Structure: {arm: {seed_str: {metric: value_or_BLOCKED}}}.

The metric union is every metric referenced by any claim (measured.<arm>.<metric> in
any 'quantity'/'against') AND every experiments[*].metric_keys entry for every arm of
that experiment, so measured.json covers every arm and every seed in claims.json with
the string BLOCKED in place of any value this environment cannot produce.

Arms = claims.json arms; seeds = claims.json seeds. Every model metric is BLOCKED here
(no CUDA / no HF_TOKEN). e1_synth contributes no arm/metric entries (its invariants have
no arm/metric) — it is consumed separately by evaluate_claims.py via results/e1_synth.json.

Completeness gate: exit nonzero if measured.json is missing any arm/seed/metric.
"""
from __future__ import annotations
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def _metric_union(claims):
    """Collect the full set of metric names referenced across claims + experiments."""
    metrics = set()
    text_blobs = []
    for cl in claims.get('claims', []):
        text_blobs.append(cl.get('quantity', ''))
        if cl.get('against'):
            text_blobs.append(cl['against'])
    blob = '\n'.join(text_blobs)
    for arm, metric in re.findall(r'measured\.(\w+)\.(\w+)', blob):
        metrics.add(metric)
    # also include every experiment's metric_keys (covers metrics not in any claim quantity
    # but listed for the arm, e.g. cow_cs/pig_cs which appear in e4 metric_keys)
    for exp_name, exp in claims.get('experiments', {}).items():
        for m in exp.get('metric_keys', []):
            metrics.add(m)
    return sorted(metrics)


def _curve_metric_xlists(claims):
    """Map each curve-claim metric name to its x-list (one value per x in the claim's order).

    A curve claim's quantity references measured.<arm>.<metric>(beta, ...) parametric in the
    x-list; the measured sequence must be one value per x, same shape at every seed. If two
    curve claims share a metric, the x-lists must agree (asserted).
    """
    out = {}
    for cl in claims.get('claims', []):
        if cl.get('kind') != 'curve':
            continue
        x = cl['x']
        refs = set(re.findall(r'measured\.(\w+)\.(\w+)', cl.get('quantity', '') + (cl.get('against') or '')))
        for _, metric in refs:
            if metric in out and out[metric] != x:
                raise RuntimeError(f"curve metric {metric} has conflicting x-lists: {out[metric]} vs {x}")
            out[metric] = list(x)
    return out


def main():
    claims_path = os.path.join(REPO, 'claims.json')
    with open(claims_path) as f:
        claims = json.load(f)
    arms = list(claims['arms'].keys())
    seeds = [str(s) for s in claims['seeds']]
    metrics = _metric_union(claims)
    curve_x = _curve_metric_xlists(claims)

    def _default(metric):
        # curve metrics are sequences (one value per x, same shape at every seed);
        # all other metrics are scalars. This sandbox produces BLOCKED for every model metric.
        if metric in curve_x:
            return ["BLOCKED"] * len(curve_x[metric])
        return "BLOCKED"

    # load partials
    partials = {}
    for exp in ['e2_llm_concrete', 'e3_llm_safety', 'e4_sdxl_h2m', 'e5_sdxl_safety']:
        p = os.path.join(REPO, 'results', f'{exp}_partial.json')
        if os.path.exists(p):
            with open(p) as f:
                partials[exp] = json.load(f)

    # assemble: every arm x seed x metric. Default BLOCKED (scalar or sequence); fill from
    # partials where present (in this sandbox all are BLOCKED).
    measured = {}
    for arm in arms:
        measured[arm] = {}
        for seed in seeds:
            measured[arm][seed] = {m: _default(m) for m in metrics}
    # merge any real (non-BLOCKED) values from partials (in this sandbox all are BLOCKED,
    # so the shape-correct default — sequence for curve metrics, scalar otherwise — is kept)
    for exp, part in partials.items():
        for arm, seed_dict in part.items():
            if arm not in measured:
                continue
            for seed, md in seed_dict.items():
                if seed not in measured[arm]:
                    continue
                for m, v in md.items():
                    if m not in measured[arm][seed]:
                        continue
                    # only overwrite with a real value; keep the shape-correct BLOCKED default otherwise
                    if isinstance(v, list):
                        if any(x != "BLOCKED" and x is not None for x in v):
                            measured[arm][seed][m] = v
                    elif v != "BLOCKED" and v is not None:
                        measured[arm][seed][m] = v

    # completeness gate
    missing = []
    for arm in arms:
        for seed in seeds:
            for m in metrics:
                if m not in measured[arm][seed]:
                    missing.append(f"{arm}.{seed}.{m}")
    if missing:
        sys.stderr.write("measured.json completeness gate FAILED, missing: "
                         + ", ".join(missing[:20]) + f" ({len(missing)} total)\n")
        # write anyway so it can be inspected, then exit nonzero
        out = os.path.join(REPO, 'measured.json')
        with open(out, 'w') as f:
            json.dump(measured, f, indent=2)
        sys.exit(1)

    out = os.path.join(REPO, 'measured.json')
    with open(out, 'w') as f:
        json.dump(measured, f, indent=2)

    # print one FINAL <arm>=... line per arm
    for arm in arms:
        # primary metric: first non-BLOCKED across seeds if any, else BLOCKED
        val = "BLOCKED"
        for seed in seeds:
            for m in metrics:
                v = measured[arm][seed][m]
                if isinstance(v, list):
                    if any(x != "BLOCKED" for x in v):
                        val = v
                        break
                elif v != "BLOCKED":
                    val = v
                    break
            if val != "BLOCKED":
                break
        print(f"FINAL {arm}={val}")

    sys.stderr.write(f"measured.json: {len(arms)} arms x {len(seeds)} seeds x {len(metrics)} metrics "
                     f"(curve metrics as sequences per x); all model metrics BLOCKED in this sandbox.\n")


if __name__ == '__main__':
    main()
