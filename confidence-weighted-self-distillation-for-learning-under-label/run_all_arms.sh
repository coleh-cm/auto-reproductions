#!/usr/bin/env bash
# Run every arm at the paper's full configuration at every seed, write
# measured.json, and print one FINAL line per arm-seed.
#
# Arms/seeds/metrics come from claims.json (the source of truth the numbers
# gate evaluates). Each arm prints exactly one line
#   FINAL <arm name>=<value>      (the arm's headline metric = accuracy)
# or   FINAL <arm name>=BLOCKED   (run failed / output did not parse)
# to stdout; that line is what a reader sees in the log. run_experiment.py
# itself prints its own 'FINAL accuracy=<float>' line; run_all_arms.sh
# additionally reads the --metrics-out JSON to collect EVERY metric
# (accuracy + the structural invariants of Eqs. 1-4) into measured.json, so
# the gate can adjudicate the structural claims as well as the accuracy ones.
#
# measured.json shape (exactly as the numbers gate consumes it):
#   {
#     "_meta": { "schema": ..., "blocked_sentinel": "BLOCKED",
#                "seeds": [...], "headline_metric": {<arm>: <metric>} },
#     "<arm>": { "<seed>": { "<metric>": <value>, ... }, ... }, ...
#   }
# The top level is keyed by arm (the _meta key is reserved and ignored by the
# gate). Any documentation of the shape belongs in REPRODUCTION.md, not in the
# JSON. A run that fails, prints no FINAL accuracy= line, or writes no metrics
# JSON is BLOCKED (every declared metric for that arm-seed set to "BLOCKED");
# a number is never fabricated.
#
# Usage: ./run_all_arms.sh
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

# Run one arm at one seed. Prints two lines:
#   line1: "FINAL <arm>=<accuracy>" (or "FINAL <arm>=BLOCKED")
#   line2: the path to the --metrics-out JSON (or empty if BLOCKED)
run_one() {
  local arm="$1" seed="$2" lam
  if ! lam="$(lambda_for_arm "$arm")"; then
    echo "FINAL ${arm}=BLOCKED"; echo ""; return
  fi
  local tmpf rc headline out
  tmpf="$(mktemp)"
  out="$($PY run_experiment.py --lambda "$lam" --seed "$seed" --metrics-out "$tmpf" 2>/dev/null)"
  rc=$?
  headline="$(printf '%s\n' "$out" | grep -E '^FINAL accuracy=' | head -1 | sed 's/^FINAL accuracy=//')"
  if [ $rc -ne 0 ] || [ -z "$headline" ] || [ ! -s "$tmpf" ]; then
    rm -f "$tmpf"
    echo "FINAL ${arm}=BLOCKED"; echo ""; return
  fi
  echo "FINAL ${arm}=${headline}"; echo "$tmpf"
}

# Collect results, print FINAL lines, remember the per-run metrics JSON path.
RESULTS=/tmp/cwsd_metrics_paths.txt
: > "$RESULTS"
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    { read -r line; read -r tmpf; } < <(run_one "$arm" "$seed")
    echo "$line"
    printf '%s\t%s\t%s\n' "$arm" "$seed" "$tmpf" >> "$RESULTS"
  done
done

# Assemble measured.json from the per-run metrics JSON files.
$PY - "$RESULTS" <<'PY'
import json, os, sys
rows = []
with open(sys.argv[1]) as f:
    for ln in f:
        arm, seed, path = ln.rstrip("\n").split("\t")
        rows.append((arm, int(seed), path))
c = json.load(open("claims.json"))
arms = list(c["arms"].keys())
seeds = [int(s) for s in c["seeds"]]
out = {
    "_meta": {
        "schema": "measured.<arm>.<seed>.<metric>; arm keys are exactly "
                  "claims.json['arms']; this _meta key is reserved and "
                  "ignored by the gate.",
        "blocked_sentinel": "BLOCKED",
        "seeds": seeds,
        "headline_metric": {
            a: next(iter(spec.get("metrics", {})))
            for a, spec in c["arms"].items()
        },
    }
}
for arm in arms:
    out[arm] = {}
    declared = list(c["arms"][arm].get("metrics", {}))
    for seed in seeds:
        path = [p for (a, s, p) in rows if a == arm and s == seed]
        if not path or not path[0] or not os.path.exists(path[0]):
            # whole run blocked: mark every declared metric BLOCKED, never
            # fabricate a number (a blocked run is indistinguishable from the
            # method never having been applied).
            out[arm][str(seed)] = {m: "BLOCKED" for m in declared}
            continue
        with open(path[0]) as f:
            mj = json.load(f)
        # only keep the metrics this arm declares; if a declared metric is
        # missing from the run's JSON, mark it BLOCKED rather than omitting it.
        out[arm][str(seed)] = {m: mj.get(m, "BLOCKED") for m in declared}
        os.remove(path[0])
json.dump(out, open("measured.json", "w"), indent=2)
print("wrote measured.json", file=sys.stderr)
PY
