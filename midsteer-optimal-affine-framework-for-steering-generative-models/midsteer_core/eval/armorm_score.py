"""ArmoRM helpfulness, SPEC section 4.4 item 16 (paper/content/experiments.tex:96).
ArmoRM-Llama3-8B helpfulness attribute; variant/head unstated (gap G11)."""
from __future__ import annotations
from midsteer_core.data import BlockedException, is_model_arm_blocked


def _mean_helpfulness(scores) -> float:
    scores = list(scores)
    if not scores:
        raise ValueError("armorm _mean_helpfulness: empty input (no success on empty)")
    return float(sum(scores) / len(scores))


def armorm_helpfulness(prompts, responses):
    """Return per-response ArmoRM helpfulness. Blocked here."""
    if is_model_arm_blocked():
        raise BlockedException("armorm_helpfulness blocked: ArmoRM backbone needs CUDA + HF_TOKEN")
    raise BlockedException("armorm_helpfulness real path not wired")
