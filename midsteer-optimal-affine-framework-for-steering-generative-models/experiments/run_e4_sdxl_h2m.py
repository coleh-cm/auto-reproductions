"""E4 SDXL horse->motorcycle (claims C4-C7, C21). MODEL ARM — BLOCKED here.

paper/content/experiments.tex:101, paper/flipping_main.tex:22-52 (Table 1).
Model SDXL-base-1.0. Metrics per claims.json e4_sdxl_h2m.metric_keys.
betas {vanilla:2, leace_switch:2, midsteer:1} per paper/content/experiments.tex:125."""
from __future__ import annotations
import os
from experiments._model_runner import run_blocked_or_real

METRICS = ['horse_cs_on_horse', 'moto_cs_on_horse', 'horse_cs_on_moto', 'moto_cs_on_moto',
           'cow_cs', 'cow_fid', 'pig_cs', 'pig_fid', 'dog_cs', 'dog_fid',
           'legislator_cs', 'legislator_fid']
ARMS = ['base', 'vanilla', 'leace_switch', 'midsteer']
BETAS = {'vanilla': 2, 'leace_switch': 2, 'midsteer': 1, 'base': None}


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'results', 'e4_sdxl_h2m_partial.json')
    run_blocked_or_real('e4_sdxl_h2m', ARMS, METRICS, BETAS, out, real_fn=None)


if __name__ == '__main__':
    main()
