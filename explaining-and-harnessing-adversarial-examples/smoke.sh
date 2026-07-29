#!/usr/bin/env bash
# smoke.sh — prove the code path runs end-to-end. Same code path as
# run_all_arms.sh but at a size that finishes in well under a couple of
# minutes, printing exactly one FINAL line.
#
# This is a PATH-PROVER, NOT evidence about the paper: 200 SGD steps leaves
# the maxout net undertrained, so the accuracy here is meaningless. It only
# demonstrates that the data pipeline, the model, the FGSM adversarial-
# training cost (Algorithm B), the training loop, and the eval metric all
# run and produce a well-formed `FINAL adversarial=<float>` line.
# (Per the reproduction contract: never report smoke output as a result.)
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# 200 steps of the method arm (Algorithm B, eps=0.25). Tiny but exercises the
# full adversarial-training path (input-gradient probe + mixed loss + SGD).
python run_experiment.py --lambda 0.25 --steps 200 --seed 0 --units 64 \
  --pieces 5 --batch-size 100 --lr 0.1 --alpha 0.5
