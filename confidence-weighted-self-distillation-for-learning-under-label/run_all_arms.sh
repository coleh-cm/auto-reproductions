#!/usr/bin/env bash
# Run every arm at the paper's full configuration at every seed, write
# measured.json, and print one FINAL line per arm-seed.
#
# Arms and seeds are read from claims.json (the source of truth for what the
# numbers gate evaluates). Each arm prints exactly one line
#   FINAL <arm name>=<value>      (claims.json's arm names)
# or   FINAL <arm name>=BLOCKED   (if the run failed or output did not parse)
# to stdout; that line is what a reader sees in the log. run_experiment.py
# itself also prints its own 'FINAL accuracy=<float>' line, which is captured
# and used as the source of the value.
#
# Usage: ./run_all_arms.sh
# Output: measured.json is (re)written at the repo root; FINAL lines on stdout.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Pick a Python: prefer the venv, fall back to system python3.
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

# Read arms (names, in claims.json declaration order) and seeds, one per line.
mapfile -t ARMS  < <($PY -c "import json;print(*json.load(open('claims.json'))['arms'].keys(),sep='\n')")
mapfile -t SEEDS < <($PY -c "import json;print(*json.load(open('claims.json'))['seeds'],sep='\n')")
if [ "${#ARMS[@]}" -eq 0 ] || [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "ERROR: could not read arms/seeds from claims.json" >&2
  exit 2
fi

# lambda value for an arm, read from claims.json config.
lambda_for_arm() {
  $PY -c "import json,sys;print(json.load(open('claims.json'))['arms'][sys.argv[1]]['config']['lambda'])" "$1"
}

# Run one arm at one seed. Prints 'FINAL <arm>=<value>' (or =BLOCKED) on its
# first stdout line and the bare value (or BLOCKED) on the second, so the
# caller can collect both.
run_one() {
  local arm="$1" seed="$2" lam
  if ! lam="$(lambda_for_arm "$arm")"; then
    echo "FINAL ${arm}=BLOCKED"; echo "BLOCKED"; return
  fi
  local out rc val
  out="$($PY run_experiment.py --lambda "$lam" --seed "$seed" 2>/dev/null)"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FINAL ${arm}=BLOCKED"; echo "BLOCKED"; return
  fi
  val="$(printf '%s\n' "$out" | grep -E '^FINAL accuracy=' | head -1 | sed 's/^FINAL accuracy=//')"
  if [ -z "$val" ]; then
    echo "FINAL ${arm}=BLOCKED"; echo "BLOCKED"; return
  fi
  echo "FINAL ${arm}=${val}"; echo "$val"
}

# Collect results, print FINAL lines.
RESULTS=/tmp/cwsd_results.txt
: > "$RESULTS"
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    { read -r line; read -r val; } < <(run_one "$arm" "$seed")
    echo "$line"
    printf '%s\t%s\t%s\n' "$arm" "$seed" "$val" >> "$RESULTS"
  done
done

# Write measured.json from the collected results.
$PY - "$RESULTS" <<'PY'
import json, sys
results = []
with open(sys.argv[1]) as f:
    for ln in f:
        arm, seed, val = ln.rstrip("\n").split("\t")
        results.append((arm, int(seed), val))
c = json.load(open("claims.json"))
arms = list(c["arms"].keys())
seeds = [int(s) for s in c["seeds"]]
# measured.json shape, exactly as the numbers gate consumes it:
#   {arm: {seed: {metric: value}}}
# at the top level -- NO outer "arms" wrapper, NO "_comment" key. The gate
# iterates the top-level dict treating each key as an arm and each value as
# {seed: {metric: value}}; a top-level string such as "_comment" makes it raise
# AttributeError: 'str' object has no attribute 'get'. Any documentation of the
# shape belongs in REPRODUCTION.md, not in the JSON.
out = {}
for arm in arms:
    out[arm] = {}
    for seed in seeds:
        v = [val for (a, s, val) in results if a == arm and s == seed][0]
        if v == "BLOCKED":
            out[arm][str(seed)] = {"accuracy": "BLOCKED"}
        else:
            out[arm][str(seed)] = {"accuracy": float(v)}
json.dump(out, open("measured.json", "w"), indent=2)
print("wrote measured.json", file=__import__("sys").stderr)
PY
