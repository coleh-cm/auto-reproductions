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
# run here, so a latent runtime bug in them would have passed the gate.)  The
# Lewis arm genuinely enters the block-Lewis code path (block_lewis_weights,
# should_reset_W, geometry_M are all called); at this size sum(w)=9 < m=10 so
# the E11 reset does NOT fire and the Lewis geometry M=A^T W A is the one used.
# Both ball arms nonetheless produce the same trajectory here because the
# trust-region radius r0=10 is large enough that the M-norm constraint never
# binds, so the Newton step -H^{-1} g is geometry-independent in early iters.
# (The bit-identical Lewis==Euclidean degeneracy is checked in tests/ on a
# problem where the reset DOES fire.)
#
# Expected smoke behavior (consistent with the paper, NOT paper evidence):
# the four first-order arms (subgradient, smoothed_gd/hb/nesterov) make real
# monotone progress (gap init->final shrinks) but PLATEAU well above 5%, so
# their iters_to_5% is None.  This is exactly the paper's §8 T4 finding --
# "first-order methods' plateau" on the heterogeneous instance, while the
# second-order arms (IPM, ball-oracle) converge fast.
#
# MACHINE-CHECKED progress (closes the round-7 gap).  Printing `gap=init->final`
# alone is only a human-readable diagnostic: a NO-OP arm that returns the warm
# start x0 (final == init) would still pass a gate that only checks
# `FINAL smoke=ok`, reading identically to a genuine plateau.  So the gate now
# ASSERTS, for every arm, that (a) every recorded gap is finite and (b) the
# final gap is STRICTLY below the initial gap.  A no-op/broken arm fails
# (final == init); a divergent arm fails (non-finite / increasing).  Only if
# all 7 arms pass does the script print `FINAL smoke=ok`; otherwise it prints
# `FINAL smoke=FAIL` with the offending arm and exits non-zero.  This is the
# same distinction `tests/test_invariants.py::test_first_order_arm_makes_progress`
# enforces, but at the GATE level -- so the smoke is real evidence every arm
# optimizes, not just that it did not crash.  (Still NOT paper evidence: tiny
# problem, tiny grids, 12 iters; the assertion is about the code path, not the
# paper's numbers.)
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
results = []   # (arm, g_init, g_final, iters_to_5%, ok)
for arm, cfg in cfgs.items():
    cfg = dict(cfg); cfg["opt"] = opt_norm
    # keep cfg["geometry"] in cfg so the ball_oracle solver reads it (the
    # solver module is always "ball_oracle"; geometry selects euclidean vs
    # lewis inside _geometry).  Popping it made both ball_oracle_* smoke
    # entries run the euclidean path, so the Lewis code path was never
    # exercised here -- contradicting the "all seven arms" claim above.
    a = "ball_oracle" if arm.startswith("ball_oracle") else arm
    h = run_arm(a, cfg, prob, x0, opt_norm, max_outer=12, time_budget=30.0)
    it, _ = time_to_gap(h, rel_gap=0.05)
    g = h["gap"]
    g0, gN = float(g[0]), float(g[-1])
    # machine-check the arm actually optimizes (not a no-op / broken arm):
    # every gap finite, and the final gap strictly below the initial gap.
    finite = all(np.isfinite(v) for v in g)
    progressed = finite and (gN < g0 - 1e-6)
    ok = bool(finite and progressed)
    results.append((arm, g0, gN, it, ok))
    last = f"{arm}: gap={g0:.3f}->{gN:.3f} iters_to_5%={it}"
    print(f"  smoke {last} progress={'YES' if ok else 'NO'}", file=sys.stderr)

# Gate: ALL arms must make strict, finite progress.  A no-op arm (final ==
# init) or a divergent arm (non-finite / increasing) fails the smoke -- this
# is the machine-checked invariant that `FINAL smoke=ok` now means something,
# not just "did not crash".  (Tests/test_invariants.py enforces the same per
# first-order arm with wider grids/iters; here it is checked at the gate.)
bad = [(arm, g0, gN) for (arm, g0, gN, _it, ok) in results if not ok]
if bad:
    detail = "; ".join(f"{a}: {gi:.4g}->{gf:.4g}" for a, gi, gf in bad)
    print(f"FINAL smoke=FAIL (no progress / non-finite: {detail})")
    sys.exit(1)
# one FINAL line for the gate to find
print(f"FINAL smoke=ok")
PYEOF
