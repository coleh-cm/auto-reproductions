"""TraceStore: on-disk layout for collected contrastive traces (SPEC §5.1).

JSONL one record per retained problem:
  {problem_id, benchmark, prompt_text, gold, extra,
   traces: [{text, token_ids:[T_gen], correct:bool, act_path}]}

``act_path`` -> .npz with ``A:[T_gen, L_mon, H, d_h]`` fp16 aligned to token_ids, and
``layers:[L_mon]`` int32. Positions are generated tokens only (SPEC §4.7).

Builds the per-(l,h) ``HeadProblemActivations`` structure consumed by mags.manifold.
"""
from __future__ import annotations
import json
import os
import numpy as np
from dataclasses import dataclass
from .loaders import Problem
from ..manifold import HeadProblemActivations


def save_trace_store(path_jsonl, problems_with_traces, act_dir):
    """``problems_with_traces``: list of (Problem, list of (text, token_ids, acts, correct)).
    ``acts``: dict layer->[T,H,dh]. We stack layers into [T,L_mon,H,dh] fp16."""
    os.makedirs(act_dir, exist_ok=True)
    os.makedirs(os.path.dirname(path_jsonl) or ".", exist_ok=True)
    layers = None
    with open(path_jsonl, "w") as f:
        for prob, traces in problems_with_traces:
            rec = {
                "problem_id": prob.id, "benchmark": prob.benchmark,
                "prompt_text": prob.prompt_text, "gold": prob.gold,
                "extra": prob.extra or {}, "traces": [],
            }
            for j, (text, token_ids, acts, correct) in enumerate(traces):
                if layers is None:
                    layers = sorted(acts.keys())
                stacked = np.stack([acts[l] for l in layers], axis=1)  # [T,L_mon,H,dh]
                ap = os.path.join(act_dir, f"{prob.id.replace('/','_')}__{j}.npz")
                np.savez_compressed(ap, A=stacked.astype(np.float16),
                                    layers=np.asarray(layers, dtype=np.int32))
                rec["traces"].append({
                    "text": text, "token_ids": [int(x) for x in token_ids],
                    "correct": bool(correct), "act_path": ap,
                })
            if rec["traces"]:
                f.write(json.dumps(rec) + "\n")
    return path_jsonl


def load_trace_store(path_jsonl) -> list:
    """Returns list of dicts (one per problem) with loaded activations in fp32."""
    out = []
    with open(path_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for tr in rec["traces"]:
                z = np.load(tr["act_path"])
                tr["A"] = z["A"].astype(np.float32)   # [T,L_mon,H,dh]
                tr["layers"] = list(int(x) for x in z["layers"])
            out.append(rec)
    return out


def build_head_activations(records: list, monitored_layers: list) -> dict:
    """Reconstruct the (l,h) -> HeadProblemActivations structure from a loaded TraceStore.

    Splits each trace's activations by head; groups by problem and correctness label.
    """
    out = {}
    for rec in records:
        pid = rec["problem_id"]
        for tr in rec["traces"]:
            A = tr["A"]              # [T, L_mon, H, dh]
            layers = tr["layers"]
            for li, l in enumerate(layers):
                if l not in monitored_layers:
                    continue
                A_l = A[:, li, :, :]   # [T, H, dh]
                H, dh = A_l.shape[1], A_l.shape[2]
                for h in range(H):
                    key = (l, h)
                    if key not in out:
                        out[key] = HeadProblemActivations()
                    ha = out[key]
                    a_head = A_l[:, h, :]      # [T, dh]
                    if tr["correct"]:
                        ha.correct.setdefault(pid, []).append(a_head)
                    else:
                        ha.incorrect.setdefault(pid, []).append(a_head)
    return out


def retain_paired_problems(records: list) -> list:
    """Keep only problems with >=1 correct AND >=1 incorrect trace (tex:L399)."""
    out = []
    for rec in records:
        has_c = any(t["correct"] for t in rec["traces"])
        has_i = any(not t["correct"] for t in rec["traces"])
        if has_c and has_i:
            out.append(rec)
    return out
