"""E3 LLM safety: toxicity -> helpfulness (claims C8-C13). MODEL ARM — BLOCKED here.

paper/content/safety-related_results.tex:10-12. Model Llama-2-7b-chat, RTP>=0.5 prompts.
Metrics: rtp, help, unrel_cs, mmlu_bert_f1. betas {3,5} (the safety claims use beta=5)."""
from __future__ import annotations
import os
from experiments._model_runner import run_blocked_or_real

METRICS = ['rtp', 'help', 'unrel_cs', 'mmlu_bert_f1']
ARMS = ['base', 'vanilla', 'leace_switch', 'midsteer']
BETAS = {3: 3, 5: 5}  # claims use beta=5; runner records the beta=5 cell


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'results', 'e3_llm_safety_partial.json')
    run_blocked_or_real('e3_llm_safety', ARMS, METRICS, BETAS, out, real_fn=None)


if __name__ == '__main__':
    main()
