"""ACS Income loader for the GDR linear-regression reproduction.

Implements the ACS Income instance of `paper/experiments.tex:148` (the
"Real-world Experiment: ACS Income" subsection) and SPEC section 6 item 12.

What this module produces
------------------------
A :class:`gdr.types.GroupProblem` with

* ``m = 51`` groups (the 50 US states + Puerto Rico), grouped by **region of
  residence** = the state whose 2018 1-Year ACS PUMS person file the record came
  from (`paper/experiments.tex:148`),
* ``per_group = 200`` individuals per region subsampled with an explicit
  ``numpy.random.Generator(seed)`` -> ``n = 51 * 200 = 10200`` rows
  (`paper/experiments.tex:148`),
* ``d = 10`` features = ``folktables.ACSIncome.features`` used **as numeric
  codes** (no one-hot), standardized per-feature with a **population z-score**
  (mean 0, std 1) computed over the full filtered population *before* the
  per-region subsample (SPEC section 6 item 12 [reconstructed]),
* target ``b = target_scale * log1p(PINCP)`` (the RAW log-income target -- see
  the empirical resolution below).  We deliberately do **not** use
  ``ACSIncome.target_transform`` (which binarizes income); the paper predicts
  *log personal income* (`paper/experiments.tex:148`).

Target-scale resolution (SPEC section 6 item 12)
------------------------------------------------
The paper reports ERM average MSE ``108.2``, worst ``138.1`` (California),
spread ``sigma = 11.8`` (`paper/experiments.tex:189`).  The SPEC/prompt
anticipated that the raw log-income MSE would be ``~0.6-0.7`` and so would need
a scalar ``target_scale`` (with the target population-z-scored) to reach ~108.

Empirically, under the *frozen* no-intercept ``d=10`` contract, the RAW
(unstandardized) ``log1p(PINCP)`` target already reproduces the paper's
statistics almost exactly: the global mean of log-income (~10.6) cannot be fit
by a no-intercept model on centered features, so the ERM MSE is dominated by
that squared constant offset (~112) and the worst-hit groups are exactly the
high-income states the paper names.  Standardizing the target instead removes
that offset and makes *Puerto Rico* the worst group (ratio worst/avg 1.51 vs
the paper's 1.276), which does not match.  We therefore do **not** standardize
the target; ``target_scale`` is applied to the raw log-income target.

Because every group loss is a squared residual, ``MSE(s) = s**2 * MSE(1)``
exactly (the ERM argmin scales linearly with the target).  We compute the ERM
average MSE once at ``target_scale = 1`` and, for each candidate in
``ACS_TARGET_SCALE_CANDIDATES`` (which includes {1, 10, 13, 100, ...}),
evaluate ``scale**2 * base_avg`` and pick the value whose ERM average MSE is
closest to ``108.2``.  The headline iteration metric (iterations to 1 %
relative gap) is scale-invariant, so an imperfect match does not block the run.

Empirical result on the 2018 1-Year ACS (this run, seed=0, per_group=200):
    target_scale        = 1           (auto; raw log-income needs no scaling)
    ERM avg MSE         = 104.9       (paper 108.2;  ~3% low -- subsample-seed
                                       noise, paper's seed unreported)
    ERM worst MSE       = 135.1       (paper 138.1; worst group = California)
    ERM group-loss std  = 11.6        (paper 11.8)
    worst-hit states    = CA, HI, FL, NJ, NY, NV  (paper: "California; New
                                       York, Hawaii, Nevada, Florida, New
                                       Jersey", `paper/experiments.tex:189`)
    worst/avg ratio     = 1.29        (paper 138.1/108.2 = 1.276)

All methods are warm-started at the ERM on ACS (`paper/experiments.tex:148`).
"""

from __future__ import annotations

import os
import sys

# When run as a script (`.venv/bin/python gdr/data_acs.py`) the repo root is not
# on sys.path; insert it so the sibling `from gdr.types import ...` resolves.
if __package__ in (None, ""):  # pragma: no cover - script-mode bootstrap
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import numpy as np
import pandas as pd

from folktables import ACSDataSource, ACSIncome

from gdr.types import GroupProblem

__all__ = [
    "STATE_NAMES",
    "ACS_FEATURES",
    "ACS_TARGET_SCALE_CANDIDATES",
    "ACS_TARGET_REFERENCE",
    "make_acs",
]

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

# 50 states + Puerto Rico in folktables.state_list order; group index i maps to
# STATE_NAMES[i].  (`paper/experiments.tex:148`; SPEC section 6 item 12.)
# Imported from folktables so the order is guaranteed to match the data files.
try:  # pragma: no cover - import path depends on folktables version
    from folktables.load_acs import state_list as STATE_NAMES
except Exception:  # pragma: no cover
    from folktables import state_list as STATE_NAMES  # type: ignore

assert len(STATE_NAMES) == 51, f"expected 51 states+PR, got {len(STATE_NAMES)}"

# The 10 ACSIncome features used as NUMERIC (not one-hot) -> d=10.
# (`paper/experiments.tex:148`; SPEC section 6 item 12.)
ACS_FEATURES = list(ACSIncome.features)
assert len(ACS_FEATURES) == 10

# Candidate target scales swept to match the paper's ERM average MSE of 108.2
# (`paper/experiments.tex:189`).  Includes the SPEC-named {1, 10, 13, 100} plus
# neighbours so the closest match is found cleanly.
ACS_TARGET_SCALE_CANDIDATES = [1, 5, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 50, 100]

# Paper reference statistics for the target-scale match (paper/experiments.tex:189).
ACS_TARGET_REFERENCE = {"avg": 108.2, "worst": 138.1, "std": 11.8}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _adult_filter(df: pd.DataFrame) -> pd.DataFrame:
    """folktables ``adult_filter`` -> employed US adults.

    Mirrors ``folktables.acs.adult_filter`` (AGEP>16, PINCP>100, WKHP>0,
    PWGTP>=1).  The paper restricts to "employed US adults"
    (`paper/experiments.tex:148`).
    """
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    return df


def _load_state_frames(
    year: int | str,
    horizon: str,
    cache_dir: str,
    download: bool,
) -> list[pd.DataFrame]:
    """Load + adult-filter the 51 state person files, in STATE_NAMES order.

    Each returned DataFrame carries only the 10 features + ``PINCP`` and is the
    filtered population for one region.  Region index = position in the list.
    """
    frames: list[pd.DataFrame] = []
    cols = ACS_FEATURES + ["PINCP"]
    ds = ACSDataSource(
        survey_year=str(year), horizon=horizon, survey="person", root_dir=cache_dir
    )
    for state in STATE_NAMES:
        # density=1.0 keeps every row (folktables subsamples via density; we do
        # our own deterministic subsample with a numpy Generator below).
        df = ds.get_data(states=[state], density=1.0, random_seed=0, download=download)
        df = _adult_filter(df)
        # Keep only the columns we need; drop rows with any NaN in features or
        # PINCP so the regression target is always defined (NaN handling is
        # unstated in the paper; dropping is the clean, non-arbitrary choice).
        df = df[cols].copy()
        df = df.dropna(subset=cols).reset_index(drop=True)
        if df.shape[0] == 0:
            raise RuntimeError(
                f"state {state}: empty after adult filter + dropna (degenerate)"
            )
        frames.append(df)
    return frames


def _stack_population(
    frames: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack the per-state frames into (X [N,10], y_raw [N], group_id [N]).

    Rows are contiguous by group (group_id nondecreasing), matching the
    :class:`GroupProblem` contract.
    """
    Xs, ys, gids = [], [], []
    for i, df in enumerate(frames):
        X = df[ACS_FEATURES].to_numpy(dtype=np.float64)  # [n_i, 10]
        y = df["PINCP"].to_numpy(dtype=np.float64)  # [n_i]
        gid = np.full(X.shape[0], i, dtype=np.int32)  # [n_i]
        Xs.append(X)
        ys.append(y)
        gids.append(gid)
    X = np.concatenate(Xs, axis=0)  # [N, 10]
    y = np.concatenate(ys, axis=0)  # [N]
    group_id = np.concatenate(gids, axis=0)  # [N]
    assert X.ndim == 2 and X.shape[1] == 10
    assert group_id.ndim == 1 and X.shape[0] == y.shape[0] == group_id.shape[0]
    # group_id must be nondecreasing (contiguous) for the GroupProblem contract.
    assert np.all(np.diff(group_id) >= 0)
    return X, y, group_id


def _population_zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population z-score: subtract mean, divide by std (ddof=0) per column.

    Returns (Z, mean, std) with std floor of 1.0 to avoid divide-by-zero on a
    constant column.
    """
    mean = X.mean(axis=0)  # [d]
    std = X.std(axis=0)  # [d], population std (ddof=0)
    std_safe = np.where(std < 1e-12, 1.0, std)  # [d]
    Z = (X - mean) / std_safe  # [N, d]
    return Z, mean, std_safe


def _subsample_per_group(
    X: np.ndarray,
    y: np.ndarray,
    group_id: np.ndarray,
    per_group: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick ``per_group`` rows per group without replacement, contiguously.

    Uses an explicit :class:`numpy.random.Generator` (SPEC determinism
    contract).  Raises if any group has fewer than ``per_group`` rows (a
    degenerate result that must not silently pass).
    """
    sizes = np.bincount(group_id, minlength=int(group_id.max()) + 1)  # [m]
    Xs, ys, gids = [], [], []
    for i, n_i in enumerate(sizes):
        if n_i < per_group:
            raise RuntimeError(
                f"group {i} ({STATE_NAMES[i]}) has only {n_i} filtered rows, "
                f"need {per_group} (degenerate subsample)"
            )
        idx = rng.choice(int(n_i), size=per_group, replace=False)  # [per_group]
        mask = group_id == i
        rows = np.where(mask)[0]  # [n_i], contiguous in i
        chosen = rows[idx]  # [per_group]
        Xs.append(X[chosen])
        ys.append(y[chosen])
        gids.append(np.full(per_group, i, dtype=np.int32))
    Xs_arr = np.concatenate(Xs, axis=0)  # [m*per_group, 10]
    ys_arr = np.concatenate(ys, axis=0)  # [m*per_group]
    gids_arr = np.concatenate(gids, axis=0)  # [m*per_group]
    return Xs_arr, ys_arr, gids_arr


def _erm_stats(problem: GroupProblem) -> dict[str, float]:
    """ERM average / worst / std of group losses (E1/E2/E3).

    (`paper/experiments.tex:189`).  Uses :meth:`GroupProblem.erm` (E3) and
    :meth:`GroupProblem.group_losses` (E1).
    """
    x_erm = problem.erm()  # [d], E3 (paper/intro.tex:5)
    losses = problem.group_losses(x_erm)  # [m], E1 (paper/experiments.tex:9)
    avg = float(np.mean(losses))
    worst = float(np.max(losses))
    std = float(np.std(losses, ddof=0))
    worst_idx = int(np.argmax(losses))
    return {
        "avg": avg,
        "worst": worst,
        "std": std,
        "worst_idx": worst_idx,
        "worst_state": STATE_NAMES[worst_idx],
    }


def _choose_target_scale(base_avg: float) -> tuple[float, float]:
    """Pick the candidate ``target_scale`` whose ``scale**2 * base_avg`` is
    closest to the paper's 108.2 (`paper/experiments.tex:189`).

    Returns (chosen_scale, achieved_avg_mse).  Raises if ``base_avg`` is not
    positive (a degenerate / failed ERM).
    """
    if not np.isfinite(base_avg) or base_avg <= 0:
        raise RuntimeError(
            f"degenerate base ERM avg MSE = {base_avg}; cannot pick target_scale"
        )
    target = ACS_TARGET_REFERENCE["avg"]
    best_scale, best_avg, best_err = None, None, np.inf
    for s in ACS_TARGET_SCALE_CANDIDATES:
        achieved = float(s) ** 2 * base_avg
        err = abs(achieved - target)
        if err < best_err:
            best_err, best_scale, best_avg = err, float(s), achieved
    if best_scale is None:
        raise RuntimeError("no target_scale candidate evaluated (empty grid)")
    return best_scale, best_avg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_acs(
    year: int = 2018,
    horizon: str = "1-Year",
    per_group: int = 200,
    seed: int = 0,
    target_scale: float | None = None,
    cache_dir: str = "data",
) -> GroupProblem:
    """Build the ACS Income :class:`GroupProblem`.

    Parameters
    ----------
    year, horizon : folktables ACS year/horizon (`paper/experiments.tex:148`;
        SPEC section 6 item 12 assumes 2018 / '1-Year').
    per_group : individuals per region (paper uses 200 -> n=10200).
    seed : integer seed for the per-region subsample ``Generator`` (SPEC
        determinism contract; SPEC section 6 item 12).
    target_scale : scalar multiplier on the standardized log-income target.
        If ``None``, auto-select from :data:`ACS_TARGET_SCALE_CANDIDATES` the
        value whose ERM average MSE is closest to 108.2
        (`paper/experiments.tex:189`).
    cache_dir : folktables cache root (census.gov download).

    Returns
    -------
    GroupProblem with A [n=per_group*51, d=10] standardized features, b the
    (optionally scaled) raw-log-income target, group_id 0..50.

    Notes
    -----
    The chosen ``target_scale`` and the achieved ERM avg/worst/std are attached
    to the returned object as ``problem.acs_meta`` (a dict) for the harness.
    Census data is downloaded by folktables on first use and cached under
    ``cache_dir``.  See the module docstring for the empirical target-scale
    resolution (raw log-income target, target_scale ~ 1).
    """
    rng = np.random.default_rng(seed)  # SPEC determinism contract

    frames = _load_state_frames(year, horizon, cache_dir, download=True)
    if len(frames) != 51:
        raise RuntimeError(f"expected 51 state frames, got {len(frames)}")

    X_all, y_raw_all, group_id_all = _stack_population(frames)  # [N,10],[N],[N]

    # Standardize FEATURES with the population z-score BEFORE subsample
    # (SPEC section 6 item 12 [reconstructed]).  The GroupProblem contract is
    # d=10 with NO intercept column, so the design matrix is the centered
    # features alone.
    X_std, _feat_mean, _feat_std = _population_zscore(X_all)  # [N,10]

    # Target = log1p(PINCP) (the paper predicts *log personal income*,
    # `paper/experiments.tex:148`).  We do NOT use ACSIncome.target_transform
    # (which binarizes income).
    #
    # EMPIRICAL TARGET-STANDARDIZATION DECISION (SPEC section 6 item 12):
    # The SPEC/prompt assumed the target would be population-z-scored and then
    # rescaled by `target_scale` to lift the MSE from ~0.6-0.7 to ~108.  But
    # under the *frozen* no-intercept d=10 contract, an UNSTANDARDIZED raw
    # log-income target already reproduces the paper's reported statistics
    # (`paper/experiments.tex:189`) almost exactly, because the global mean of
    # log-income (~10.6) cannot be fit by a no-intercept model on centered
    # features, so the ERM MSE is dominated by that constant offset (~112) and
    # the worst-hit groups are the high-income states the paper names.  We
    # verified empirically on the 2018 1-Year ACS:
    #   * raw log-income, no-intercept  -> avg 104.9, worst 135.1 (California),
    #     std 11.6; worst states {CA,HI,FL,NJ,NY,NV} -- matches the paper's
    #     "California; New York, Hawaii, Nevada, Florida, New Jersey"
    #     (`paper/experiments.tex:189`); target_scale ~ 1.
    #   * standardized log-income      -> avg 0.51, worst 0.77 (Puerto Rico!),
    #     std 0.088; does NOT match the paper (PR is worst, ratio 1.51 vs 1.276).
    # We therefore do NOT standardize the target; `target_scale` is applied to
    # the RAW log-income target and auto-selected to ~1.
    y_target = np.log1p(np.clip(y_raw_all, 0.0, None))  # [N], raw log1p(PINCP)

    # Per-region subsample (deterministic via rng).
    Xs, ys, gids = _subsample_per_group(
        X_std, y_target, group_id_all, per_group, rng
    )  # [m*pg,10],[m*pg],[m*pg]

    # Build the problem at scale=1 to compute base ERM stats.
    problem_base = GroupProblem(A=Xs, b=ys, group_id=gids)
    base_stats = _erm_stats(problem_base)  # uses E3 ERM, E1 losses

    if target_scale is None:
        chosen_scale, achieved_avg = _choose_target_scale(base_stats["avg"])
    else:
        chosen_scale = float(target_scale)
        achieved_avg = chosen_scale ** 2 * base_stats["avg"]

    # Scale the target (MSE scales as scale**2).  Rebuild with scaled b.
    b_scaled = chosen_scale * ys  # [m*pg]
    problem = GroupProblem(A=Xs, b=b_scaled, group_id=gids)
    final_stats = _erm_stats(problem)
    # Sanity: final_stats['avg'] must equal achieved_avg (within fp tol).
    if not np.isclose(final_stats["avg"], achieved_avg, rtol=1e-8, atol=1e-8):
        raise RuntimeError(
            f"target_scale invariant broken: expected avg {achieved_avg}, "
            f"got {final_stats['avg']}"
        )

    meta = {
        "year": int(year),
        "horizon": horizon,
        "per_group": int(per_group),
        "seed": int(seed),
        "target_scale": float(chosen_scale),
        "base_avg_mse": float(base_stats["avg"]),
        "base_worst_mse": float(base_stats["worst"]),
        "base_std": float(base_stats["std"]),
        "erm_avg_mse": float(final_stats["avg"]),
        "erm_worst_mse": float(final_stats["worst"]),
        "erm_std": float(final_stats["std"]),
        "erm_worst_state": final_stats["worst_state"],
        "erm_worst_idx": int(final_stats["worst_idx"]),
        "n": int(problem.n),
        "m": int(problem.m),
        "d": int(problem.d),
        "reference_avg_mse": ACS_TARGET_REFERENCE["avg"],
        "reference_worst_mse": ACS_TARGET_REFERENCE["worst"],
        "reference_std": ACS_TARGET_REFERENCE["std"],
        "target_scale_auto": target_scale is None,
    }
    # Attach metadata for the harness (GroupProblem is frozen, so use object).
    object.__setattr__(problem, "acs_meta", meta)
    return problem


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _build_tiny_problem() -> GroupProblem:
    """A tiny inline GroupProblem (no ACS download) to exercise the helpers."""
    rng = np.random.default_rng(0)
    m, d, pg = 4, 3, 30
    A = rng.standard_normal((m * pg, d)).astype(np.float64)
    true_x = rng.standard_normal(d)
    b = A @ true_x + 0.1 * rng.standard_normal(m * pg)
    group_id = np.repeat(np.arange(m, dtype=np.int32), pg)
    return GroupProblem(A=A, b=b, group_id=group_id)


def _self_test() -> bool:
    """Run the module's self-tests.  Returns True iff all PASS."""
    ok = True
    msgs = []

    # 1. STATE_NAMES is 51 and matches folktables.
    if len(STATE_NAMES) != 51:
        msgs.append(f"FAIL STATE_NAMES len={len(STATE_NAMES)} != 51")
        ok = False
    else:
        msgs.append("PASS STATE_NAMES has 51 entries")
    # PR is last and CA present (paper: worst group is California).
    if STATE_NAMES[-1] != "PR":
        msgs.append(f"FAIL STATE_NAMES[-1]={STATE_NAMES[-1]} != PR")
        ok = False
    if "CA" not in STATE_NAMES:
        msgs.append("FAIL CA not in STATE_NAMES")
        ok = False

    # 2. ACS_FEATURES has 10 numeric features.
    if len(ACS_FEATURES) != 10:
        msgs.append(f"FAIL ACS_FEATURES len={len(ACS_FEATURES)} != 10")
        ok = False
    else:
        msgs.append(f"PASS ACS_FEATURES = {ACS_FEATURES}")

    # 3. Helpers on a tiny synthetic problem.
    prob = _build_tiny_problem()
    if prob.m != 4 or prob.n != 120 or prob.d != 3:
        msgs.append(
            f"FAIL tiny problem m/n/d = {prob.m}/{prob.n}/{prob.d}"
        )
        ok = False
    else:
        msgs.append("PASS tiny problem m=4 n=120 d=3")

    x_erm = prob.erm()  # [d]
    if x_erm.shape != (3,):
        msgs.append(f"FAIL ERM shape {x_erm.shape}")
        ok = False
    else:
        msgs.append("PASS ERM shape (3,)")

    losses = prob.group_losses(x_erm)  # [m]
    if losses.shape != (4,) or not np.all(np.isfinite(losses)):
        msgs.append(f"FAIL group_losses {losses.shape}")
        ok = False
    else:
        msgs.append(f"PASS group_losses shape (4,) finite")

    stats = _erm_stats(prob)
    if not (stats["avg"] > 0 and np.isfinite(stats["avg"])):
        msgs.append(f"FAIL ERM stats avg={stats['avg']}")
        ok = False
    else:
        msgs.append(
            f"PASS ERM stats avg={stats['avg']:.4g} worst={stats['worst']:.4g}"
        )

    # 4. _choose_target_scale picks something finite & positive.
    s, a = _choose_target_scale(stats["avg"])
    if not (np.isfinite(s) and s > 0 and np.isfinite(a) and a > 0):
        msgs.append(f"FAIL choose_target_scale s={s} a={a}")
        ok = False
    else:
        msgs.append(
            f"PASS choose_target_scale -> scale={s} "
            f"(avg {a:.4g} = {s}**2*{stats['avg']:.4g})"
        )

    # 5. _subsample_per_group determinism + shape.
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    X = np.arange(4 * 30 * 2, dtype=np.float64).reshape(4 * 30, 2)
    y = np.arange(4 * 30, dtype=np.float64)
    gid = np.repeat(np.arange(4, dtype=np.int32), 30)
    Xa, ya, ga = _subsample_per_group(X, y, gid, 10, rng1)
    Xb, yb, gb = _subsample_per_group(X, y, gid, 10, rng2)
    if not (np.array_equal(Xa, Xb) and np.array_equal(ga, gb)):
        msgs.append("FAIL subsample not deterministic for same seed")
        ok = False
    elif ga.shape != (40,) or Xa.shape != (40, 2):
        msgs.append(f"FAIL subsample shape {ga.shape} {Xa.shape}")
        ok = False
    else:
        msgs.append("PASS subsample deterministic + shape (40, 2)")

    # 6. _population_zscore produces mean~0 std~1.
    Z, mu, sd = _population_zscore(np.random.default_rng(1).standard_normal((1000, 3)))
    if not np.allclose(Z.mean(axis=0), 0, atol=1e-9) or not np.allclose(
        Z.std(axis=0), 1, atol=1e-9
    ):
        msgs.append("FAIL zscore mean/std not 0/1")
        ok = False
    else:
        msgs.append("PASS population zscore -> mean 0 std 1")

    # 7. End-to-end make_acs on real data (only if all 51 state files cached).
    cache = "data"
    base = os.path.join(cache, str(2018), "1-Year")
    from folktables.load_acs import _STATE_CODES  # type: ignore
    have_all = all(
        os.path.isfile(os.path.join(base, f"psam_p{code}.csv"))
        for code in _STATE_CODES.values()
    )
    if have_all:
        try:
            prob_acs = make_acs(year=2018, horizon="1-Year", per_group=200, seed=0)
            meta = getattr(prob_acs, "acs_meta", {})
            if prob_acs.m != 51 or prob_acs.n != 10200 or prob_acs.d != 10:
                msgs.append(
                    f"FAIL make_acs m/n/d = {prob_acs.m}/{prob_acs.n}/{prob_acs.d}"
                )
                ok = False
            else:
                msgs.append(
                    f"PASS make_acs m=51 n=10200 d=10 | "
                    f"target_scale={meta.get('target_scale')} "
                    f"ERM avg={meta.get('erm_avg_mse'):.3f} "
                    f"worst={meta.get('erm_worst_mse'):.3f} "
                    f"std={meta.get('erm_std'):.3f} "
                    f"worst_state={meta.get('erm_worst_state')}"
                )
                # Honest reporting of the scale match.
                ref = ACS_TARGET_REFERENCE
                avg_err = abs(meta.get("erm_avg_mse", 0.0) - ref["avg"])
                msgs.append(
                    f"     paper ref: avg={ref['avg']} worst={ref['worst']} "
                    f"std={ref['std']} | |avg_err|={avg_err:.3f}"
                )
        except Exception as e:  # pragma: no cover
            msgs.append(f"FAIL make_acs raised: {e!r}")
            ok = False
    else:
        msgs.append(
            "SKIP make_acs end-to-end (not all 51 state files cached yet)"
        )

    for m in msgs:
        print(m)
    return ok


if __name__ == "__main__":
    _passed = _self_test()
    print("PASS" if _passed else "FAIL")
    raise SystemExit(0 if _passed else 1)
