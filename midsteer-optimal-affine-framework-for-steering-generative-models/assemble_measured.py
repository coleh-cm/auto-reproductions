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


def main():
    claims_path = os.path.join(REPO, 'claims.json')
    with open(claims_path) as f:
        claims = json.load(f)
    arms = list(claims['arms'].keys())
    seeds = [str(s) for s in claims['seeds']]
    metrics = _metric_union(claims)

    # load partials
    partials = {}
    for exp in ['e2_llm_concrete', 'e3_llm_safety', 'e4_sdxl_h2m', 'e5_sdxl_safety']:
        p = os.path.join(REPO, 'results', f'{exp}_partial.json')
        if os.path.exists(p):
            with open(p) as f:
                partials[exp] = json.load(f)

    # assemble: every arm x seed x metric. Default BLOCKED; fill from partials where present.
    measured = {}
    for arm in arms:
        measured[arm] = {}
        for seed in seeds:
            measured[arm][seed] = {m: "BLOCKED" for m in metrics}
    # merge any real (non-BLOCKED) values from partials (in this sandbox all are BLOCKED)
    for exp, part in partials.items():
        for arm, seed_dict in part.items():
            if arm not in measured:
                continue
            for seed, md in seed_dict.items():
                if seed not in measured[arm]:
                    continue
                for m, v in md.items():
                    if m in measured[arm][seed]:
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
                if v != "BLOCKED":
                    val = v
                    break
            if val != "BLOCKED":
                break
        print(f"FINAL {arm}={val}")

    sys.stderr.write(f"measured.json: {len(arms)} arms x {len(seeds)} seeds x {len(metrics)} metrics; "
                     f"all model metrics BLOCKED in this sandbox.\n")


if __name__ == '__main__':
    main()
