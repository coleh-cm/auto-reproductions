"""E5 SDXL safety: violence -> peace (claims C14-C18). MODEL ARM — BLOCKED here.

paper/content/safety-related_results.tex:15, paper/artefacts/tables/safety_table.tex:83.
Model SDXL-base-1.0. Metrics: viol_cs, peace_cs, unrel_cs, fid. betas {3,5} (claims use beta=5)."""
from __future__ import annotations
import os
from experiments._model_runner import run_blocked_or_real

METRICS = ['viol_cs', 'peace_cs', 'unrel_cs', 'fid']
ARMS = ['base', 'vanilla', 'leace_switch', 'midsteer']
BETAS = {3: 3, 5: 5}


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'results', 'e5_sdxl_safety_partial.json')
    run_blocked_or_real('e5_sdxl_safety', ARMS, METRICS, BETAS, out, real_fn=None)


if __name__ == '__main__':
    main()
