"""Assemble measured.json from the per-experiment partial files + e1_synth invariants
+ the metric union parsed from claims.json. Structure: {arm: {seed_str: {metric: value}}}.

The metric union is every metric referenced by any claim (measured.<arm>.<metric> in
any 'quantity'/'against'/'predicate') AND every experiments[*].metric_keys entry, so
measured.json covers every arm and every seed in claims.json with the string BLOCKED
in place of any value this environment cannot produce.

Arms = claims.json arms (the 4 model arms + e1_synth, the CPU closed-form invariant
arm). Seeds = claims.json seeds. Every MODEL metric is BLOCKED here (no CUDA / no
HF_TOKEN); no synthetic fallback. The e1_synth arm is REAL: its per-seed invariant
residuals (c1_constraint, c1_minimal_disturbance_gap, ... ) are read from
results/e1_synth.json so the invariant claims' executable predicates can threshold
measured evidence. The derived curve metrics c20_min_baseline_src and
c21_min_baseline_horse (elementwise min of the two baselines' per-beta sequences, the
weakest-dominance reference for C20/C21) are computed from the baseline sequences when
they are real and BLOCKED (sequence) when any source is BLOCKED.

Completeness gate: exit nonzero if measured.json is missing any arm/seed/metric.
"""
from __future__ import annotations
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def _metric_union(claims):
    """Collect the full set of metric names referenced across claims + experiments.

    Scans every claim's `quantity`, `against` AND `predicate` (invariant claims carry
    their threshold expression in `predicate`, so the e1_synth residual metrics must be
    in the union) plus every experiment's metric_keys.
    """
    metrics = set()
    text_blobs = []
    for cl in claims.get('claims', []):
        text_blobs.append(cl.get('quantity', ''))
        if cl.get('against'):
            text_blobs.append(cl['against'])
        if cl.get('predicate'):
            text_blobs.append(cl['predicate'])
    blob = '\n'.join(text_blobs)
    for arm, metric in re.findall(r'measured\.(\w+)\.(\w+)', blob):
        metrics.add(metric)
    for exp_name, exp in claims.get('experiments', {}).items():
        for m in exp.get('metric_keys', []):
            metrics.add(m)
    return sorted(metrics)


def _curve_metric_xlists(claims):
    """Map each curve-claim metric name to its x-list (one value per x in the claim's order).

    A curve claim's quantity/against reference measured.<arm>.<metric> whose stored value
    is a per-x SEQUENCE (one value per x, same shape at every seed). If two curve claims
    share a metric, the x-lists must agree (asserted).
    """
    out = {}
    for cl in claims.get('claims', []):
        if cl.get('kind') != 'curve':
            continue
        x = cl['x']
        refs = set(re.findall(r'measured\.(\w+)\.(\w+)',
                              cl.get('quantity', '') + '\n' + (cl.get('against') or '')))
        for _, metric in refs:
            if metric in out and out[metric] != x:
                raise RuntimeError(f"curve metric {metric} has conflicting x-lists: {out[metric]} vs {x}")
            out[metric] = list(x)
    return out


def _seq_blocked(seq):
    return any(x == "BLOCKED" or x is None for x in seq)


def _elementwise_min(seq_a, seq_b, xlen):
    """Elementwise min of two per-x sequences. BLOCKED (sequence) if either source is
    BLOCKED/absent; otherwise [min(seq_a[i], seq_b[i]) for i]."""
    if seq_a is None or seq_b is None or _seq_blocked(seq_a) or _seq_blocked(seq_b):
        return ["BLOCKED"] * xlen
    return [min(float(a), float(b)) for a, b in zip(seq_a, seq_b)]


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
        # all other metrics are scalars. Model metrics are BLOCKED here.
        if metric in curve_x:
            return ["BLOCKED"] * len(curve_x[metric])
        return "BLOCKED"

    # load model-experiment partials (all BLOCKED in this sandbox)
    partials = {}
    for exp in ['e2_llm_concrete', 'e3_llm_safety', 'e4_sdxl_h2m', 'e5_sdxl_safety']:
        p = os.path.join(REPO, 'results', f'{exp}_partial.json')
        if os.path.exists(p):
            with open(p) as f:
                partials[exp] = json.load(f)

    # load e1_synth invariants (REAL CPU evidence): {seed: {C1: {metrics: {...}}, C2:..., C3:...}}
    e1_path = os.path.join(REPO, 'results', 'e1_synth.json')
    e1 = {}
    if os.path.exists(e1_path):
        with open(e1_path) as f:
            e1 = json.load(f)

    # assemble: every arm x seed x metric. Default BLOCKED (scalar or sequence); fill from
    # partials where present (model arms BLOCKED here).
    measured = {}
    for arm in arms:
        measured[arm] = {}
        for seed in seeds:
            measured[arm][seed] = {m: _default(m) for m in metrics}
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
                    if isinstance(v, list):
                        if any(x != "BLOCKED" and x is not None for x in v):
                            measured[arm][seed][m] = v
                    elif v != "BLOCKED" and v is not None:
                        measured[arm][seed][m] = v

    # e1_synth arm: flatten the per-claim invariant residuals into measured[e1_synth][seed].
    # These are the metrics the C1/C2/C3 predicates threshold. Real numbers; never BLOCKED
    # in this CPU sandbox (the closed form runs). A missing seed/claim -> raise (no silent
    # success on an empty result).
    if 'e1_synth' in measured:
        if not e1:
            raise RuntimeError("assemble_measured: e1_synth arm declared but results/e1_synth.json "
                               "missing/empty (no success on empty)")
        for seed in seeds:
            if seed not in e1:
                raise RuntimeError(f"assemble_measured: e1_synth seed {seed} missing from e1_synth.json")
            for claim in ('C1', 'C2', 'C3'):
                if claim not in e1[seed]:
                    raise RuntimeError(f"assemble_measured: {claim} missing from e1_synth.json seed {seed}")
                cm = e1[seed][claim].get('metrics', {})
                for mname, mval in cm.items():
                    if mname not in measured['e1_synth'][seed]:
                        # an invariant residual metric the predicate references must be in
                        # the union; if not, the claim cannot be evaluated.
                        raise RuntimeError(f"assemble_measured: invariant metric {mname} not in metric union")
                    if mval is None:
                        raise RuntimeError(f"assemble_measured: {claim}.{mname} is None (no success on empty)")
                    measured['e1_synth'][seed][mname] = float(mval)

    # Derived curve metrics: elementwise min of the two baselines' per-beta sequences,
    # the weakest-dominance reference for C20 (LLM src_cs_on_src, x=[3,4,5]) and C21
    # (SDXL horse_cs_on_horse, x=[1..5]). Stored under the midsteer arm (the claim compares
    # FOR midsteer). BLOCKED sequence when any source sequence is BLOCKED.
    def _derive_min(arm_out, metric_out, arm_a, metric_a, arm_b, metric_b, xkey):
        xlen = len(curve_x[xkey])
        for seed in seeds:
            sa = measured.get(arm_a, {}).get(seed, {}).get(metric_a)
            sb = measured.get(arm_b, {}).get(seed, {}).get(metric_b)
            measured[arm_out][seed][metric_out] = _elementwise_min(sa, sb, xlen)

    if 'c20_min_baseline_src' in metrics:
        _derive_min('midsteer', 'c20_min_baseline_src',
                    'vanilla', 'src_cs_on_src', 'leace_switch', 'src_cs_on_src', 'c20_min_baseline_src')
    if 'c21_min_baseline_horse' in metrics:
        _derive_min('midsteer', 'c21_min_baseline_horse',
                    'vanilla', 'horse_cs_on_horse', 'leace_switch', 'horse_cs_on_horse', 'c21_min_baseline_horse')

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
        out = os.path.join(REPO, 'measured.json')
        with open(out, 'w') as f:
            json.dump(measured, f, indent=2)
        sys.exit(1)

    out = os.path.join(REPO, 'measured.json')
    with open(out, 'w') as f:
        json.dump(measured, f, indent=2)

    # one FINAL <arm>=<value> line per arm (the arm summary a reader sees in the log).
    for arm in arms:
        if arm == 'e1_synth':
            # REAL arm: worst (max) covariance-constraint residual across C1/C2/C3 and seeds
            # — a measured number proving the closed-form ran; not a verdict word.
            worst = 0.0
            for seed in seeds:
                for k in ('c1_constraint', 'c2_flip_constraint', 'c3_matched_cov_constraint'):
                    v = measured['e1_synth'][seed].get(k)
                    if isinstance(v, (int, float)):
                        worst = max(worst, float(v))
            print(f"FINAL {arm}={worst:.6e}")
            continue
        # model arms: first non-BLOCKED across seeds if any, else BLOCKED (all BLOCKED here)
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
                     f"(curve metrics as sequences per x); model metrics BLOCKED, e1_synth invariants REAL.\n")


if __name__ == '__main__':
    main()
