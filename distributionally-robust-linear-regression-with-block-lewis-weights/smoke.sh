#!/usr/bin/env bash
# Smoke test: the SAME code path at a size that finishes in a couple of minutes,
# printing one FINAL line.  This proves the path runs end-to-end; its output is
# NOT evidence about the paper (small problem, tiny grids, few outer steps).
# Real runs use run_all_arms.sh at the paper's full configuration.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"

# Tiny synthetic problem, all arms, very small grids, 6 outer iterations.
$PY - <<'PYEOF'
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))
import numpy as np
from gdr.data_synthetic import make_synthetic
from gdr.problem import max_loss
from gdr.reference import solve_opt
from gdr.runner import run_arm, time_to_gap

prob = make_synthetic(d=5, m=10, n_adv=2, n_per_group=15, seed=1, E_ADV=1e3, DIST=3.0)
xopt, OPT = solve_opt(prob)
x0 = np.linalg.lstsq(prob["A"], prob["b"], rcond=None)[0]
cfgs = {
  "subgradient": {"lr_grid":[1e-2,1e-1], "schedule_grid":["const"]},
  "smoothed_hb": {"lr_grid":[1e-2,1e-1], "momentum_grid":[0.9], "beta_grid":[1e-1], "delta_grid":[1e-1]},
  "ipm": {"mu0_grid":[10.0], "theta_grid":[0.5]},
  "ball_oracle_lewis": {"geometry":"lewis","r0_grid":[10.0],"shrink_grid":[0.5],"beta_grid":[1e-1],"delta_grid":[1e-1]},
}
last = None
for arm, cfg in cfgs.items():
    cfg = dict(cfg); cfg["opt"] = OPT
    geom = cfg.pop("geometry", None)
    a = "ball_oracle" if arm.startswith("ball_oracle") else arm
    h = run_arm(a, cfg, prob, x0, OPT, max_outer=6, time_budget=30.0)
    it, _ = time_to_gap(h, rel_gap=0.05)
    last = f"{arm}: gap={h['gap'][-1]:.3f} iters_to_5%={it}"
    print(f"  smoke {last}", file=sys.stderr)
# one FINAL line for the gate to find
print(f"FINAL smoke=ok")
PYEOF
