"""CLIP score x100 (image concept score), SPEC section 4.4 item 14.

paper/content/experiments.tex:103, paper/flipping_main.tex:2-5: tables report
'CLIP-scores (cs, x100)'. Backbone unstated (gap G11); default openai/clip-vit-base-patch32.
"""
from __future__ import annotations
from midsteer_core.data import BlockedException, is_model_arm_blocked


def _scale100(similarity: float) -> float:
    """CLIPScore convention: cosine similarity x100 (paper tables are x100)."""
    return float(similarity) * 100.0


def clip_cs(images, concept, backbone='openai/clip-vit-base-patch32'):
    """Return CLIP score x100 for each image against the concept text. Blocked here."""
    if is_model_arm_blocked():
        raise BlockedException(f"clip_cs blocked: backbone {backbone} needs CUDA + HF_TOKEN")
    raise BlockedException(f"clip_cs real path not wired (backbone {backbone})")
