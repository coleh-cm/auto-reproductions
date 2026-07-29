"""EvalHarness (SPEC §5.6): run one arm over a benchmark, compute accuracy + PPL,
bootstrap 95% percentile CIs (B=10000, seed=42, tex:L699-700)."""
from __future__ import annotations
import json
import os
import numpy as np


def bootstrap_ci(corrects: list, B: int = 10000, seed: int = 42, ci: float = 0.95):
    """Percentile bootstrap CI over per-problem 0/1 correctness (tex:L699-700)."""
    arr = np.asarray(corrects, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = arr[rng.integers(0, n, size=(B, n))].mean(axis=1)
    lo, hi = np.percentile(samples, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(arr.mean()), [float(lo), float(hi)]


def run_arm(model, tok, model_id, controller, benchmark, problems, max_new_tokens,
            ppl_model=None, completion_limit=None, grading=None):
    """Generate one completion per problem with ``controller`` attached, grade, and
    (optionally) compute PPL of each completion under ``ppl_model`` (SPEC §4.14).

    Returns dict {n, acc, ci95, ppl, per_problem:[{id, correct, ppl, completion}]}.
    ``grading`` defaults to mags.grading.grade.
    """
    from .generation import generate, perplexity_of, CHAT_TEMPLATE_BENCHMARKS
    if grading is None:
        from .grading import grade as grading
    use_chat = benchmark in CHAT_TEMPLATE_BENCHMARKS
    per_problem = []
    corrects = []
    ppls = []
    for i, prob in enumerate(problems):
        # Reset per-problem controller state (decode-step index, problem id) so
        # the steering_log records the per-problem decode index `t` and the
        # problem id (Algorithm 1, SPEC §5.4). Without this, a controller reused
        # across problems accumulates `_decode_step` monotonically and logs every
        # record under problem=None. No-op controllers ignore the call.
        begin = getattr(controller, "begin_problem", None)
        if begin is not None:
            begin(prob.id)
        completion, gen_ids, prompt_ids = generate(
            model, tok, prob.prompt_text, controller,
            max_new_tokens=max_new_tokens, do_sample=False,
            use_chat_template=use_chat,
        )
        if completion_limit:
            completion = completion[:completion_limit]
        ok = grading(benchmark, completion, prob)
        ppl = float("nan")
        if ppl_model is not None:
            try:
                # SPEC §4.14: CONDITIONAL PPL of the completion given the prompt,
                # under the unsteered base model (ppl_model). Using the exact
                # generated token ids (no re-tokenization) + the prompt ids gives
                # the NLL of the completion conditioned on the prompt — the
                # protocol the paper's ~1.1-1.2 values imply (tex:L420-441).
                ppl = perplexity_of(ppl_model, tok, token_ids=gen_ids,
                                    prompt_ids=prompt_ids)
            except Exception:
                ppl = float("nan")
        corrects.append(int(ok))
        ppls.append(ppl)
        per_problem.append({"id": prob.id, "correct": int(ok), "ppl": ppl,
                            "completion": completion})
    acc, ci = bootstrap_ci(corrects)
    valid_ppls = [p for p in ppls if not np.isnan(p)]
    mean_ppl = float(np.mean(valid_ppls)) if valid_ppls else float("nan")
    return {"n": len(problems), "acc": acc, "ci95": ci, "ppl": mean_ppl,
            "per_problem": per_problem}
