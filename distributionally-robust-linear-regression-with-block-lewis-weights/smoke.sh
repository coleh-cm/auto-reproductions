#!/usr/bin/env bash
# Smoke test: the SAME code path at a size that finishes in a couple of minutes,
# printing one FINAL line.  This proves the path runs end-to-end; its output is
# NOT evidence about the paper (small problem, tiny grids, few outer steps).
# Real runs use run_all_arms.sh at the paper's full configuration.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"

# Tiny synthetic problem, all 7 arms, very small grids, 12 outer iterations.
$PY - <<'PYEOF'
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))
import numpy as np
from gdr.data_synthetic import make_synthetic
from gdr.problem import max_loss
from gdr.reference import solve_opt
from gdr.runner import run_arm, time_to_gap
from run_arm import normalize_problem, erm_warm_start

# Tiny synthetic problem, all 7 arms, very small grids, 12 outer iterations.
# IMPORTANT: this exercises the SAME code path as run_arm.py -- in particular
# the OPT=1 normalization (run_arm.normalize_problem, U15) that every real run
# applies.  Without it the loss scale is O(E_ADV) and the subgradient's fixed
# step sizes diverge (a 1e24 "gap" that is not a measurement); with it every
# arm sees the O(1) loss scale the real runs use, so the smoke gap is finite
# and meaningful (still NOT paper evidence -- small problem, tiny grids).
#
# All SEVEN paper arms (§8.1.2) are exercised so the smoke proves *every* arm
# code path runs end-to-end -- not just a subset.  (The earlier smoke covered
# only 4/7; smoothed_gd / smoothed_nesterov / ball_oracle_euclidean were never
# run here, so a latent runtime bug in them would have passed the gate.)  At
# this size the E11 reset (Σw_i >= m) fires, so ball_oracle_euclidean and
# ball_oracle_lewis are bit-identical -- the degeneracy the test suite checks.
#
# Expected smoke behavior (consistent with the paper, NOT paper evidence):
# the four first-order arms (subgradient, smoothed_gd/hb/nesterov) make real
# monotone progress (gap init->final shrinks) but PLATEAU well above 5%, so
# their iters_to_5% is None.  This is exactly the paper's §8 T4 finding --
# "first-order methods' plateau" on the heterogeneous instance, while the
# second-order arms (IPM, ball-oracle) converge fast.  Printing init->final
# (not just final) makes the progress visible so a "None" cannot be misread
# as a broken arm: a broken arm would show init == final.
prob = make_synthetic(d=5, m=10, n_adv=2, n_per_group=15, seed=1, E_ADV=1e3, DIST=3.0)
xopt, OPT = solve_opt(prob)
prob, scale = normalize_problem(prob, OPT)   # OPT == 1 after this (U15)
opt_norm = 1.0
x0 = erm_warm_start(prob)
cfgs = {
  "subgradient": {"lr_grid":[1e-3,1e-2], "schedule_grid":["const","1/sqrt_t"]},
  "smoothed_gd": {"lr_grid":[1e-2,1e-1], "beta_grid":[1e-1], "delta_grid":[1e-1]},
  "smoothed_hb": {"lr_grid":[1e-2,1e-1], "momentum_grid":[0.9], "beta_grid":[1e-1], "delta_grid":[1e-1]},
  "smoothed_nesterov": {"lr_grid":[1e-2,1e-1], "momentum_grid":[0.9], "beta_grid":[1e-1], "delta_grid":[1e-1]},
  "ipm": {"mu0_grid":[10.0], "theta_grid":[0.5]},
  "ball_oracle_euclidean": {"geometry":"euclidean","r0_grid":[10.0],"shrink_grid":[0.5],"beta_grid":[1e-1],"delta_grid":[1e-1]},
  "ball_oracle_lewis": {"geometry":"lewis","r0_grid":[10.0],"shrink_grid":[0.5],"beta_grid":[1e-1],"delta_grid":[1e-1]},
}
last = None
for arm, cfg in cfgs.items():
    cfg = dict(cfg); cfg["opt"] = opt_norm
    geom = cfg.pop("geometry", None)
    a = "ball_oracle" if arm.startswith("ball_oracle") else arm
    h = run_arm(a, cfg, prob, x0, opt_norm, max_outer=12, time_budget=30.0)
    it, _ = time_to_gap(h, rel_gap=0.05)
    g = h["gap"]
    last = f"{arm}: gap={g[0]:.3f}->{g[-1]:.3f} iters_to_5%={it}"
    print(f"  smoke {last}", file=sys.stderr)
# one FINAL line for the gate to find
print(f"FINAL smoke=ok")
PYEOF
