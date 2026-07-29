#!/usr/bin/env bash
# Run every arm at the paper's full configuration on BOTH datasets (D1 synthetic
# + D2 ACS Income) and print one FINAL line per (dataset, arm).  The gate checks
# the acs_income_* lines for the T1 iteration ordering (SPEC §5, T1).
#
# Each arm prints exactly:   FINAL <dataset>_<arm>=<value>
# where <value> = the arm's own outer iterations to reach 1% relative
# worst-group suboptimality (F(x)-OPT)/OPT <= 0.01, or `NR` if not reached.
#
# Usage: ./run_all_arms.sh [MAXOUTER] [TIME]
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
MAXOUTER="${1:-300}"
TIME="${2:-120}"
DATASETS="${DATASETS:-acs_income synthetic}"

# arm name -> geometry flag (None for non-ball arms)
run_arm () {
  local arm="$1" geom="$2" ds="$3"
  if [ "$geom" = "none" ]; then
    $PY run_arm.py --arm "$arm" --dataset "$ds" --max-outer "$MAXOUTER" --time-budget "$TIME"
  else
    $PY run_arm.py --arm "$arm" --geometry "$geom" --dataset "$ds" --max-outer "$MAXOUTER" --time-budget "$TIME"
  fi
}

# Arms the paper compares (§8.1.2): first-order, IPM, ball-oracle (ours), OPT ref.
# Each entry is "<run_arm.py --arm value>:<geometry flag or none>".
declare -a ARMS=(
  "subgradient:none"
  "smoothed_gd:none"
  "smoothed_hb:none"
  "smoothed_nesterov:none"
  "ipm:none"
  "ball_oracle:euclidean"
  "ball_oracle:lewis"
  "opt_reference:none"
)

for ds in $DATASETS; do
  for entry in "${ARMS[@]}"; do
    arm="${entry%%:*}"; geom="${entry##*:}"
    run_arm "$arm" "$geom" "$ds" || echo "FINAL ${ds}_${arm}=ERR"
  done
done
