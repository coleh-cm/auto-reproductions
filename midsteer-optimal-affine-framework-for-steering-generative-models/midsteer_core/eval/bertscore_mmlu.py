"""BERTScore Precision/F1 on MMLU generations, SPEC section 4.4 item 17
(paper/content/experiments.tex:98). Backbone/layers and MMLU subset unstated (gap G11)."""
from __future__ import annotations
from midsteer_core.data import BlockedException, is_model_arm_blocked


def _bertscore_aggregate(per_pair_scores) -> float:
    """Aggregate per-pair BERTScore to a corpus mean (Precision or F1)."""
    per_pair_scores = list(per_pair_scores)
    if not per_pair_scores:
        raise ValueError("bertscore _bertscore_aggregate: empty input (no success on empty)")
    return float(sum(per_pair_scores) / len(per_pair_scores))


def bertscore_p(cands, refs):
    """BERTScore Precision. Blocked (backbone) here."""
    if is_model_arm_blocked():
        raise BlockedException("bertscore_p blocked: BERTScore backbone needs CUDA + HF_TOKEN")
    raise BlockedException("bertscore_p real path not wired")


def bertscore_f1(cands, refs):
    """BERTScore F1. Blocked (backbone) here."""
    if is_model_arm_blocked():
        raise BlockedException("bertscore_f1 blocked: BERTScore backbone needs CUDA + HF_TOKEN")
    raise BlockedException("bertscore_f1 real path not wired")
