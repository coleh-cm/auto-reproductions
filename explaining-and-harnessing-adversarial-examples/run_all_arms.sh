#!/usr/bin/env bash
# run_all_arms.sh — run EVERY arm of claims.json at EVERY seed, write measured.json.
#
# Paper: Goodfellow, Shlens, Szegedy, "Explaining and Harnessing Adversarial
# Examples" (ICLR 2015, arXiv:1412.6572v3). The arms and metrics are claims.json's
# (13 arms: M1-M9, E1, the L1 control; m5 contributes two arms from one command,
# m7 contributes two). Seeds are claims.json's: [0, 1, 2].
#
# This delegates to make_measured.py, which:
#   * runs each arm's command_per_seed at each seed (writing per-seed result files
#     via --out, so concurrent runs never clobber each other);
#   * resolves every metric from each result via the claims.json pointer and
#     writes measured.json as {arm: {seed: {metric: value}}};
#   * prints exactly one `FINAL <arm>=<headline value>` line per arm to STDOUT
#     (the value a reader sees in the log; BLOCKED if the environment cannot
#     produce it). Progress goes to STDERR only, so stdout is exactly the 13
#     gate lines, paired to arms by name.
#
# Sub-scale: m5's paper-full config (1600 units / patience 100 / 5 seeds,
# tex:497-512) is infeasible on this CPU; make_measured.py appends
# `--units 240 --epochs 12` to the m5 command (the reproduction's established
# sub-scale, recorded in measured.json['_meta']['subscale_overrides'] and
# REPRODUCTION.md). Every other arm is already sub-scale (DEFAULT_UNITS=240).
# The m5 headline magnitude (0.782%) is rated compute_invariance=low in
# claims.json; the HIGH m5 claim is the DIRECTION (c11, advtrain <= baseline),
# which the sub-scale reproduces. Real MNIST throughout (no synthetic fallback).
#
# Runtime: ~15-20 min on this CPU (m5 and e1 are the long poles, ~4 min/seed).
# For a fast path-prover see smoke.sh (never evidence about the paper).
set -euo pipefail

cd "$(dirname "$0")"

# PyTorch's default thread pool over-spawns on multi-core boxes and contends.
# make_measured.py runs up to --jobs arms concurrently, each with --omp threads.
# 5 jobs x 2 threads = 10 threads on this 16-core box is the sweet spot.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Run all arms x all seeds, write measured.json, print one FINAL line per arm.
python make_measured.py --jobs 5 --omp 2
