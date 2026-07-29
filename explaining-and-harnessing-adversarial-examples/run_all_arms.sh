#!/usr/bin/env bash
# run_all_arms.sh — run every arm of the paper's headline comparison at the
# paper's configuration, each printing exactly one gate line.
#
# Paper: Goodfellow, Shlens, Szegedy, "Explaining and Harnessing Adversarial
# Examples" (ICLR 2015, arXiv:1412.6572v3). The comparison the paper makes
# (tex:492-494: "from 0.94% without adversarial training to 0.84%") is clean
# maxout training (arm `baseline`) vs FGSM adversarial training (arm
# `adversarial`, Algorithm B, eps=0.25, alpha=0.5).
#
# Each arm command (mirroring arms.json) prints exactly one line to STDOUT:
#     FINAL <arm name>=<clean test accuracy>
# Progress goes to STDERR only, so stdout is exactly the two gate lines.
# The gate pairs each line to its arm by name; arms.json is the source of
# truth for the names and commands.
#
# This is the graded-harness (degeneracy-valid) form of the M4 comparison:
# the method (Algorithm B) runs for the `adversarial` arm; dropout is OFF so
# `--lambda 0` reproduces `baseline` bit-for-bit (tests/test_degeneracy.py).
# The paper's exact 0.94%->0.84% magnitudes additionally need dropout ON +
# early stopping; that higher-fidelity arm lives in experiments/m4_adversarial.py
# with committed results in results/m4_adversarial.json.
#
# Runtime: ~1-2 min total on this CPU (5000 SGD steps per arm). The broader
# milestone experiments (M1-M9, E1) are NOT run here — they live in
# experiments/*.py with committed results/*.json (see README.md).
set -euo pipefail

cd "$(dirname "$0")"

# PyTorch's default thread pool over-spawns on multi-core boxes and contends
# (~7x slowdown observed at 1500 steps). 4 threads is the sweet spot here.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Activate the pinned venv if present, else rely on the caller's environment.
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[run_all_arms] arm: baseline (clean maxout training)" >&2
python run_experiment.py --baseline --steps 5000 --seed 0 --units 240 \
  --pieces 5 --batch-size 100 --lr 0.1 --alpha 0.5

echo "[run_all_arms] arm: adversarial (FGSM adversarial training, eps=0.25, alpha=0.5)" >&2
python run_experiment.py --lambda 0.25 --steps 5000 --seed 0 --units 240 \
  --pieces 5 --batch-size 100 --lr 0.1 --alpha 0.5
