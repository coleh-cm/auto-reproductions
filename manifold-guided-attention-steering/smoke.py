"""Smoke entrypoint: the same MAGS code path at a size that finishes in a couple of
minutes on CPU. It is NOT evidence about the paper -- it only proves the path
(fit -> steer -> grade) runs end to end on a real (tiny, open) model and a real
(tiny) eval set. Per the research-code rule, smoke output is never reported as a
result.

Pipeline:
  1. capture real per-head activations on a few prompts (real forward path)
  2. fit a contrastive error manifold (real SVD/centroid/threshold/head-selection)
  3. run unsteered + MAGS-steered generation on a few MATH-500 problems
  4. grade with the real math grader
  5. print exactly one FINAL line:  FINAL smoke=<value>
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch

SMOKE_MODEL = os.environ.get("MAGS_SMOKE_MODEL", "distilgpt2")


def main():
    from mags.models import load_model
    from mags.capture import _CaptureHook
    from mags.manifold import (
        HeadProblemActivations, fit_manifold_bank,
    )
    from mags.steering import MAGSController, NoOpController
    from mags.generation import generate
    from mags.grading import grade_math
    from mags.data.loaders import load_math500
    from mags.run import _model_cached, _hf_hub_dir

    # Fast cache precheck (mirrors mags.run round-8 fix): if the tiny smoke
    # model is not in the HF cache, fail to BLOCKED in <1s instead of hanging
    # in online `from_pretrained` on a blackholed network (the gate carries an
    # HF token, which would otherwise enable online mode). On a host that
    # pre-cached distilgpt2 (or has working network + no token, where smoke.sh
    # forces offline and a cached copy loads) the real smoke path runs.
    if not _model_cached(SMOKE_MODEL):
        print("FINAL smoke=BLOCKED")
        sys.stderr.write(
            f"BLOCKED[smoke]: smoke model {SMOKE_MODEL!r} not in HF cache "
            f"({_hf_hub_dir()}); pre-cache with `huggingface-cli download "
            f"{SMOKE_MODEL}` to run the smoke path.\n")
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model, tok = load_model(SMOKE_MODEL)
    model._mags_tokenizer = tok
    reg = model._mags_registry
    monitored = reg.layout.monitored_layers

    # 1-2. fit a tiny manifold from captured activations (label by perturbation).
    fit_prompts = [
        "The answer to 2+2 is",
        "def add(a, b):\n    return",
        "Once upon a time",
        "The capital of France is",
    ]
    pids = [f"p{i}" for i in range(len(fit_prompts))]
    ha_per_head = {}
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i, p in enumerate(fit_prompts):
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            cap = _CaptureHook(monitored, reg.layout.n_heads, reg.layout.head_dim)
            reg.attach(cap)
            try:
                model.generate(ids, max_new_tokens=12, do_sample=False,
                               pad_token_id=tok.eos_token_id, use_cache=True)
            finally:
                reg.detach()
            acts = cap.stacked()
            for l in monitored:
                A = acts[l]
                for h in range(reg.layout.n_heads):
                    key = (l, h)
                    if key not in ha_per_head:
                        ha_per_head[key] = HeadProblemActivations()
                    base = A[:, h, :]
                    ha_per_head[key].correct.setdefault(pids[i], []).append(base)
                    ha_per_head[key].incorrect.setdefault(pids[i], []).append(
                        base + 0.5 * rng.normal(size=base.shape))
    bank = fit_manifold_bank(
        ha_per_head, pids[:3], pids[3:], model_id=SMOKE_MODEL, benchmark="smoke",
        k=min(4, reg.layout.head_dim), q=95, K=3, alpha=1.0,
        layers_monitored=monitored, split_seed=42,
    )

    # 3-4. eval on a few real MATH-500 problems (real data, real grader).
    problems = load_math500()[:4]
    corrects = []
    for prob in problems:
        prompt = prob.prompt_text + "\nPlease box your final answer: $\\boxed{}$"
        _, ids_b, _ = generate(model, tok, prompt, NoOpController(), max_new_tokens=32)
        ctrl = MAGSController(bank, alpha=1.0)
        _, ids_m, _ = generate(model, tok, prompt, ctrl, max_new_tokens=32)
        # grade the steered completion (the path under test)
        txt = tok.decode(ids_m, skip_special_tokens=True)
        corrects.append(int(grade_math(txt, prob.gold)))

    acc = float(np.mean(corrects))
    print(f"FINAL smoke={acc:.4f}")


if __name__ == "__main__":
    main()
