"""LLM judge concept score (0-10), SPEC section 4.4 item 13.

Prompt verbatim from paper/content/suppl.tex:423-436. Score = argmax over token
probabilities of the score tokens "0".."10" from a forward pass of the judge
(paper/content/suppl.tex:438). Main-text judge: Llama-3.1-8B-Instruct
(paper/content/experiments.tex:94; the appendix's "Llama3.1-7B" is gap G10 shorthand).
"""
from __future__ import annotations
import torch
from midsteer_core.data import BlockedException, is_model_arm_blocked

# Prompt VERBATIM from paper/content/suppl.tex:423-436 (the \begin{verbatim}...\end{verbatim}
# block). Mid-line newlines and trailing spaces are preserved exactly as the paper prints
# them; the block ends with a trailing blank line before \end{verbatim}.
JUDGE_PROMPT = (
    '\n'
    'Score the text on a scale of 0 to 10, where the score reflects the extent \n'
    'to which it demonstrates relevance to the concept of {concept}. A score of 0 \n'
    'indicates the text is completely irrelevant and shows no connection to {concept}, \n'
    'whereas a score of 10 signifies strong and explicit relevance to {concept}. For \n'
    'texts that suggest varying degrees of relevance, use the intermediate numbers \n'
    '(1 through 9) to represent the gradation of connection to {concept}. \n'
    'Output just a single number.\n'
    '\n'
    'Text to evaluate: ```\n'
    '{text}\n'
    '```\n'
    '\n'
)

DEFAULT_JUDGE = 'meta-llama/Llama-3.1-8B-Instruct'


def _argmax_score_from_logits(logits: torch.Tensor, token_ids: list[int]) -> float:
    """Given next-token logits and the token ids for '0'..'10', return the argmax score.

    logits: 1-D tensor of vocab logits for the next token. token_ids: the 11 ids for '0'..'10'.
    Returns the float value (0.0..10.0) of the highest-probability score token.
    """
    if len(token_ids) != 11:
        raise ValueError("token_ids must be the 11 ids for '0'..'10'")
    probs = torch.softmax(logits[token_ids], dim=-1)
    best = int(torch.argmax(probs).item())
    return float(best)


def judge_cs(texts, concept, model_id=DEFAULT_JUDGE):
    """Score each text on 0-10 for the concept. Raises BlockedException in this sandbox."""
    if is_model_arm_blocked():
        raise BlockedException(
            f"judge_cs blocked: judge backbone {model_id} needs CUDA + HF_TOKEN")
    # Real path (not reachable in this sandbox): load tokenizer+model, format JUDGE_PROMPT,
    # forward pass, gather '0'..'10' token logits, argmax. Fingerprint: DEFAULT_JUDGE is
    # the main-text Llama-3.1-8B-Instruct per paper/content/experiments.tex:94.
    raise BlockedException(f"judge_cs real path not wired (model {model_id})")


# Fixtures for the instrument test.
POSITIVE_TEXT = ("A thorough discussion of motorcycles: their engines, two wheels, "
                 "the open road, and motorcycle racing culture.")
POSITIVE_CONCEPT = "motorcycle"
NEGATIVE_TEXT = "A recipe for chocolate cake with flour, sugar, eggs, and cocoa."
NEGATIVE_CONCEPT = "motorcycle"
