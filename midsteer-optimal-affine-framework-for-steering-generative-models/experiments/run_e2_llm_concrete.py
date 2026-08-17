"""E2 LLM concrete concept switching (claims C19, C20). MODEL ARM — BLOCKED in this sandbox.

paper/content/experiments.tex:89, paper/content/suppl.tex:457-539 templates.
Model: Llama-2-7b-chat (G10). Pairs: horse->motorcycle, dog->cat.
Metrics: src_cs_on_src, tgt_cs_on_src, src_cs_on_tgt, tgt_cs_on_tgt, unrel_cs, bertp_mmlu.
"""
from __future__ import annotations
import os
from experiments._model_runner import run_blocked_or_real

METRICS = ['src_cs_on_src', 'tgt_cs_on_src', 'src_cs_on_tgt', 'tgt_cs_on_tgt', 'unrel_cs', 'bertp_mmlu']
ARMS = ['base', 'vanilla', 'leace_switch', 'midsteer']
BETAS = {'vanilla': 2, 'leace_switch': 2, 'midsteer': 1, 'base': None}


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'results', 'e2_llm_concrete_partial.json')
    run_blocked_or_real('e2_llm_concrete', ARMS, METRICS, BETAS, out,
                        real_fn=None)  # real path needs Llama-2-7b-chat + HF_TOKEN


if __name__ == '__main__':
    main()
