"""Data unit for the GDR least-squares reproduction (SPEC sec3 D1/D2, sec7 U1/U4/U7/U18).

Sources: D1 synthetic (make_synthetic, experiments.tex:4-38), D2 ACS Income
(load_acs_income, experiments.tex:145-152), a smoke fixture (make_smoke), and
the frozen dispatcher load_problem (SPEC sec4).

Folding (body.tex:27): A_{S_i} <- A_{S_i}/sqrt(n_i), b_{S_i} <- b_{S_i}/sqrt(n_i)
applied at load time; after folding ell_i(x) = ||(folded A_{S_i})x - (folded b_{S_i})||^2
== (1/n_i)||orig residual||^2.  All arrays np.float64; pseudoinverse convention
(body.tex:27) via np.linalg.pinv.  Only numpy is imported at top level;
pandas/folktables are imported lazily inside load_acs_income (no network at import).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .problem import Problem  # noqa: F401  (re-export frozen Problem TypedDict)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_ACS_CACHE = os.path.join(_DATA_DIR, "acs_income_folded.npz")
# Bumped on layout/semantics change so a stale (e.g. POBP-grouped, or seed-blind)
# cache is rebuilt rather than silently reused; load_acs_income also checks
# meta['group_by'] and meta['seed'].  v3: added seed validation in _load_cache.
_ACS_CACHE_SCHEMA = 3


def fold_group(A_i: np.ndarray, b_i: np.ndarray, n_i: int):
    """Fold 1/sqrt(n_i) into the data (body.tex:27).  A_i:[n_i,d] b_i:[n_i] -> folded."""
    A_i = np.asarray(A_i, dtype=np.float64)          # [n_i, d]
    b_i = np.asarray(b_i, dtype=np.float64)          # [n_i]
    s = np.sqrt(np.float64(int(n_i)))                # scalar > 0
    return A_i / s, b_i / s                          # ([n_i, d], [n_i])


def _build_folded_problem(name, orig_A_blocks, orig_b_blocks, n_i_pre, meta):
    """Stack per-group *original* blocks into a folded Problem via fold_group.

    orig_A_blocks: list of m [n_i,d] float64 (pre-fold, natural).  offsets are
    the contiguous cumsum group row ranges.  Returns a Problem (SPEC sec4).
    """
    m = len(orig_A_blocks)
    if m == 0:
        raise ValueError("need at least one group")
    d = int(orig_A_blocks[0].shape[1])
    n_i = np.array([int(bl.shape[0]) for bl in orig_A_blocks], dtype=np.int64)  # [m]
    if len(n_i_pre) != m:
        raise ValueError("n_i_pre length mismatch")
    folded_A, folded_b = [], []
    for i in range(m):
        Af, bf = fold_group(orig_A_blocks[i], orig_b_blocks[i], int(n_i[i]))   # ([n_i,d],[n_i])
        folded_A.append(Af)
        folded_b.append(bf)
    A = np.vstack(folded_A).astype(np.float64)        # [n, d]
    b = np.concatenate(folded_b).astype(np.float64)  # [n]
    offsets = np.zeros(m + 1, dtype=np.int64)         # [m+1]
    offsets[1:] = np.cumsum(n_i)                      # contiguous group row ranges
    return Problem(A=A, b=b, offsets=offsets, name=name, m=m, d=d,
                   n_i=np.asarray(n_i_pre, dtype=np.int64), meta=meta)


# --- D1 synthetic (experiments.tex:4-38) -----------------------------------
# Paper states only: d=10, m=100, 5 adversarial, cond(A^T A)~1e5 (experiments.tex:38);
# shared orthonormal eigenbasis; normal groups moderate cond, aligned geometry,
# optima near a common center, independent noise; adversarial groups one
# extremely-large-curvature direction (distinct per group), optima far along it,
# very small noise (experiments.tex:19-25).  All concrete numbers UNSTATED (U1;
# notebook absent from arXiv tarball).  Defaults below match
# arms.json _dataset_overrides.synthetic; all are keyword-overridable.
#   E_LO=0.01, E_HI=1.0, E_ADV=1e6, DIST=5.0, n_per_group=50, seed=0,
#   sig_x=0.3, sig=0.5, sig_adv=0.02  (sig* not in arms.json -> sibling choice).
_SYN_SEED_DEFAULT = 0
_SYN_M_DEFAULT = 100
_SYN_D_DEFAULT = 10
_SYN_N_ADV_DEFAULT = 5
_SYN_N_PER_GROUP_DEFAULT = 50
_SYN_E_LO_DEFAULT = 0.01
_SYN_E_HI_DEFAULT = 1.0
_SYN_E_ADV_DEFAULT = 1.0e6
_SYN_DIST_DEFAULT = 5.0
_SYN_SIG_X_DEFAULT = 0.3
_SYN_SIG_DEFAULT = 0.5
_SYN_SIG_ADV_DEFAULT = 0.02


def make_synthetic(seed=_SYN_SEED_DEFAULT, m=_SYN_M_DEFAULT, d=_SYN_D_DEFAULT,
                   n_adv=_SYN_N_ADV_DEFAULT, n_i=None, *,
                   n_per_group=_SYN_N_PER_GROUP_DEFAULT, E_LO=_SYN_E_LO_DEFAULT,
                   E_HI=_SYN_E_HI_DEFAULT, E_ADV=_SYN_E_ADV_DEFAULT,
                   DIST=_SYN_DIST_DEFAULT, sig_x=_SYN_SIG_X_DEFAULT,
                   sig=_SYN_SIG_DEFAULT, sig_adv=_SYN_SIG_ADV_DEFAULT,
                   center=None):
    """Construct D1 (experiments.tex:4-38).  Positional (seed,m,d,n_adv,n_i) is
    the frozen interface (SPEC sec4); keyword-only overrides carry the arms.json
    dataset_overrides grid (U1).  Defaults match arms.json _dataset_overrides.synthetic.

    Shared orthonormal basis U=QR(randn); normal Hessians U diag(e) U^T share
    eigenvectors (experiments.tex:19).  Adversarial groups add a rank-1 spike
    E_ADV*outer(v_k,v_k) with v_k at distinct spread angles in span(U[:,0],U[:,1])
    (CONFLICT) and optima center+DIST*v_k (far), very small noise sig_adv.

    DEPARTURE from literal 'shares eigenvectors' (experiments.tex:19): v_k is a
    combination of two U columns, so the spike rotates the adversarial top
    eigenvector off U.  NECESSARY: with the spike pinned to a single shared
    eigenvector (distinct U column per adversarial group) the huge adversarial
    curvature dominates the ERM gradient and ERM fits every adversarial group on
    its own independent axis, leaving the ERM worst group to be a normal group
    ~= the robust-opt band, i.e. ~no gap, contradicting experiments.tex:38.
    Normal groups keep shared U exactly.  Matches canonical
    gdr.data_synthetic.make_synthetic (byte-identical under arms.json params).
    See meta['deviation_note'].  Returns a folded Problem.
    """
    rng = np.random.default_rng(seed)                     # deterministic (U7)
    if center is None:
        center = np.zeros(d, dtype=np.float64)            # [d]
    else:
        center = np.asarray(center, dtype=np.float64).reshape(-1)  # [d]
    # resolve per-group pre-fold row counts n_i_arr [m]
    if n_i is None:
        n_i_arr = np.full(m, int(n_per_group), dtype=np.int64)     # [m]
    else:
        n_i_arr = np.asarray(n_i, dtype=np.int64).reshape(-1)      # [m] or [1]
        if n_i_arr.shape[0] == 1:
            n_i_arr = np.full(m, int(n_i_arr[0]), dtype=np.int64)  # [m]
        if n_i_arr.shape[0] != m:
            raise ValueError(f"n_i must have length m={m} or 1, got {n_i_arr.shape[0]}")
    if n_adv < 0 or n_adv > m:
        raise ValueError(f"n_adv must satisfy 0 <= n_adv <= m={m}")

    U, _ = np.linalg.qr(rng.standard_normal((d, d)))     # [d, d] orthonormal cols
    if n_adv > 0:
        ang = np.linspace(0.0, np.pi, n_adv, endpoint=False) + rng.uniform(0.0, 0.3)
        adv_dirs = [U[:, 0] * np.cos(a) + U[:, 1] * np.sin(a) for a in ang]  # list of [d]
    else:
        adv_dirs = []

    log_E_LO, log_E_HI = np.log(E_LO), np.log(E_HI)
    orig_A_blocks, orig_b_blocks = [], []
    n_adv_used = 0
    for i in range(m):
        ni = int(n_i_arr[i])                              # pre-fold rows of group i
        if ni < 1:
            raise ValueError(f"group {i} has n_i={ni} < 1")
        is_adv = i < n_adv                                # first n_adv groups adversarial
        if is_adv:
            v = adv_dirs[n_adv_used]                      # [d] unit sharp direction
            n_adv_used += 1
            eigs = np.exp(rng.uniform(log_E_LO, log_E_HI, size=d))   # [d] moderate base
            Sigma = (U * eigs) @ U.T + E_ADV * np.outer(v, v)        # [d, d] SPD
            x_star = center + DIST * v                     # [d] far along sharp direction
            noise_sd = sig_adv
        else:
            eigs = np.exp(rng.uniform(log_E_LO, log_E_HI, size=d))   # [d] moderate
            Sigma = (U * eigs) @ U.T                        # [d, d] SPD, shares U
            x_star = center + sig_x * rng.standard_normal(d)  # [d] near center
            noise_sd = sig
        w, Ve = np.linalg.eigh(Sigma)                      # w [d], Ve [d, d]
        w = np.clip(w, 0.0, None)                          # guard tiny negatives
        Gsqrt = (Ve * np.sqrt(w)) @ Ve.T                  # [d, d] = Sigma^{1/2}
        # orig rows ~ N(0,Sigma): Z @ Sigma^{1/2 T}; natural sample Gram (no
        # orthonormal-row round-trip); fold_group divides by sqrt(n_i).
        Z = rng.standard_normal((ni, d))                   # [n_i, d]
        orig_A_i = Z @ Gsqrt.T                            # [n_i, d] orig design (pre-fold)
        orig_noise = noise_sd * rng.standard_normal(ni)   # [n_i] orig per-row noise
        orig_b_i = orig_A_i @ x_star + orig_noise          # [n_i] orig responses (pre-fold)
        orig_A_blocks.append(orig_A_i.astype(np.float64))
        orig_b_blocks.append(orig_b_i.astype(np.float64))

    problem = _build_folded_problem("synthetic", orig_A_blocks, orig_b_blocks,
                                    n_i_arr, meta={})
    # stacked (folded) Gram eigenvalue range, computed once (cond ~1e5, experiments.tex:38)
    gram = problem["A"].T @ problem["A"]                  # [d, d]
    ev = np.linalg.eigvalsh(gram)                         # [d], ascending
    gram_eig_min = float(ev[0])
    gram_eig_max = float(ev[-1])
    gram_cond = gram_eig_max / max(gram_eig_min, 1e-300)

    problem["meta"] = {
        "source": "synthetic (D1, experiments.tex:4-38)",
        "seed": int(seed), "m": int(m), "d": int(d), "n_adv": int(n_adv),
        "n_i": n_i_arr.tolist(),
        "n_per_group": int(n_per_group) if n_i is None else None,  # U18 default
        "basis": "shared orthonormal U from QR(seed); normal Hessians = U diag(e) U^T",
        "E_LO": float(E_LO), "E_HI": float(E_HI), "E_ADV": float(E_ADV),
        "DIST": float(DIST), "sig_x": float(sig_x), "sig": float(sig),
        "sig_adv": float(sig_adv),
        "normal_eig_range": [float(E_LO), float(E_HI)],    # logU; moderate
        "normal_optima": f"center + N(0, sig_x^2 I), sig_x={sig_x}",
        "normal_noise_std": float(sig),                    # group loss at opt ~ sig^2
        "adversarial_spike_eig": float(E_ADV),              # rank-1 spike along v_k
        "adversarial_directions": "distinct spread angles in span(U[:,0],U[:,1]) (conflict)",
        "adversarial_optima": f"center + DIST * v_k, DIST={DIST}",
        "adversarial_noise_std": float(sig_adv),
        "deviation_note": (
            "Adversarial Hessians carry a rank-1 spike E_ADV*outer(v_k,v_k) with "
            "v_k in span(U[:,0],U[:,1]); this rotates the adversarial top "
            "eigenvector off the shared basis U, a literal departure from the "
            "'shares eigenvectors' phrasing of experiments.tex:19 (recorded as "
            "a paper-internal inconsistency, SPEC U19).  Necessary AND "
            "empirically verified (Round-15, CVXPY E20 epigraph QCQP, seed=0): "
            "the paper-literal shared-eigenbasis construction (spike on a "
            "distinct U column per adversarial group) gives cond(A^T A)=8.8e4 "
            "but F(ERM)/OPT=1.060 (gap 6%) with the ERM worst group = a NORMAL "
            "group, contradicting phenomenon 3 (ERM incurs substantial loss on "
            "adversarial groups, experiments.tex:33) and the 'clear gap' of "
            "experiments.tex:38; the rotated-spike construction used here gives "
            "cond=1.40e5 and F(ERM)/OPT=1.469 (gap 47%) with the ERM worst group "
            "= an adversarial group, reproducing phenomena 2,3 and the gap.  "
            "The paper's {shares eigenvectors; misaligned sharp directions; "
            "ERM-incurs-loss-on-adversarial} are mutually inconsistent under "
            "orthogonal eigenvectors.  Normal groups keep shared U exactly.  "
            "Matches canonical gdr.data_synthetic.make_synthetic and the "
            "measured gap in arms.json _dataset_overrides.synthetic."),
        "stacked_gram_eig_min": gram_eig_min,
        "stacked_gram_eig_max": gram_eig_max,
        "stacked_gram_cond": gram_cond,                     # target ~1e5 (experiments.tex:38)
    }
    return problem


def make_smoke(seed=0):
    """Tiny folded Problem for smoke.sh: m=6, d=3, ~8 rows/group; planted optimum,
    mild noise; enough to exercise solver/objective/reference in milliseconds."""
    rng = np.random.default_rng(seed)
    m, d, n_per = 6, 3, 8
    x_true = rng.standard_normal(d).astype(np.float64)    # [d] planted optimum
    orig_A_blocks, orig_b_blocks = [], []
    n_i_arr = np.full(m, n_per, dtype=np.int64)            # [m]
    for i in range(m):
        Ai = rng.standard_normal((n_per, d))               # [n_per, d] orig design
        noise = 0.1 * rng.standard_normal(n_per)           # [n_per] small noise
        bi = Ai @ x_true + noise                           # [n_per] orig responses
        orig_A_blocks.append(Ai.astype(np.float64))
        orig_b_blocks.append(bi.astype(np.float64))
    meta = {"source": "smoke fixture", "seed": int(seed), "m": m, "d": d,
            "n_i": n_i_arr.tolist(), "x_true": x_true.tolist(), "noise_std": 0.1}
    return _build_folded_problem("smoke", orig_A_blocks, orig_b_blocks, n_i_arr, meta)


# --- D2 ACS Income grouped by region of residence (experiments.tex:145-152) --
# Paper: "group the data by region of residence", all m=51 regions (50 states +
# PR), 200/region -> n=10,200 (experiments.tex:148).  Region of residence is ST
# (state FIPS), NOT POBP (place of birth): ST selects residents of the 51
# regions, POBP would select people born there (different population).  Group
# order = folktables.state_list (AL..WY, PR -> m=51).
_ST_NAMES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
    'WI', 'WY', 'PR',
]
_ST_NAME_TO_CODE = {
    'AL': 1, 'AK': 2, 'AZ': 4, 'AR': 5, 'CA': 6, 'CO': 8, 'CT': 9, 'DE': 10,
    'FL': 12, 'GA': 13, 'HI': 15, 'ID': 16, 'IL': 17, 'IN': 18, 'IA': 19,
    'KS': 20, 'KY': 21, 'LA': 22, 'ME': 23, 'MD': 24, 'MA': 25, 'MI': 26,
    'MN': 27, 'MS': 28, 'MO': 29, 'MT': 30, 'NE': 31, 'NV': 32, 'NH': 33,
    'NJ': 34, 'NM': 35, 'NY': 36, 'NC': 37, 'ND': 38, 'OH': 39, 'OK': 40,
    'OR': 41, 'PA': 42, 'RI': 44, 'SC': 45, 'SD': 46, 'TN': 47, 'TX': 48,
    'UT': 49, 'VT': 50, 'VA': 51, 'WA': 53, 'WV': 54, 'WI': 55, 'WY': 56,
    'PR': 72,
}
assert len(_ST_NAMES) == 51 and set(_ST_NAMES) == set(_ST_NAME_TO_CODE)
_ST_CODES = [_ST_NAME_TO_CODE[s] for s in _ST_NAMES]   # [51], ordered
# ACSIncome.features (smoke_imports.py; experiments.tex:148): 10 features.
# ST is the grouping variable, NOT one of the 10 features -> d=10.
_ACS_FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP",
                 "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
assert len(_ACS_FEATURES) == 10


def _save_cache(path, problem):
    """Persist a folded Problem to .npz (arrays + json meta) for fast re-run."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta_json = json.dumps(problem["meta"], default=str)
    np.savez(path, A=problem["A"], b=problem["b"], offsets=problem["offsets"],
             n_i=problem["n_i"], m=np.int64(problem["m"]), d=np.int64(problem["d"]),
             name=np.array(problem["name"]), meta_json=np.array(meta_json))


def _load_cache(path, seed=None):
    """Reload a folded Problem written by _save_cache; validates schema, ST grouping,
    and (when ``seed`` is given) the requested seed.

    The seed check is required: ACS keeps m=51, n=10200 for every seed (200/region x
    51), so a seed-blind cache silently serves one seed's data for another -- the
    same class of bug Round-3 fixed for the OPT-value cache.  A seed mismatch
    forces a rebuild.  (Rejects a stale POBP-grouped cache likewise.)"""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta_json"]))
    if int(meta.get("cache_schema", 0)) != _ACS_CACHE_SCHEMA:
        raise ValueError("cache schema mismatch; rebuild required")
    if not str(meta.get("group_by", "")).startswith("ST"):
        raise ValueError(f"cache grouped by {meta.get('group_by')!r}, not ST; rebuild required")
    if seed is not None and int(meta.get("seed", -1)) != int(seed):
        raise ValueError(f"cache seed {meta.get('seed')!r} != requested {seed}; rebuild required")
    return Problem(A=np.asarray(d["A"], dtype=np.float64),
                  b=np.asarray(d["b"], dtype=np.float64),
                  offsets=np.asarray(d["offsets"], dtype=np.int64),
                  name=str(d["name"]), m=int(d["m"]), d=int(d["d"]),
                  n_i=np.asarray(d["n_i"], dtype=np.int64), meta=meta)


def load_acs_income(seed=0, year=2018, horizon="1-Year", n_per_region=200):
    """Load D2 ACS Income (experiments.tex:145-152), or None if blocked.

    Predicts log1p(PINCP) from the 10 ACSIncome features for employed US adults,
    grouped by region of residence (ST: 50 states + PR = 51 regions).
    1. ACSDataSource.get_data(states=None, download=True, density=1.0,
       random_seed=seed): full PUMS for all 51 states (files on disk reused;
       missing downloaded).  Full density guarantees every region retains >=200
       employed rows so m=51 exactly (experiments.tex:148); a 0.1 density load
       could drop sparse regions below the per-region quota.
    2. adult_filter (AGEP>16, PINCP>100, WKHP>0, PWGTP>=1) + dropna(features,PINCP,ST).
    3. Group by ST in state_list order; subsample n_per_region WITHOUT replacement
       (numpy.default_rng(seed)); if fewer rows take all; drop any region with <10.
    4. Target log1p(PINCP) (U4). 5. Standardize 10 features globally (U4).
    6. Fold by 1/sqrt(n_i) (body.tex:27).  Cache to data/acs_income_folded.npz
    (schema/grouping/seed validated on reload -- a seed mismatch forces a
    rebuild, never silently serving another seed's data).  Returns None (prints
    a message) if the download/load fails, so ACS is a blocked result, not
    silent synthetic.  No download at import time.
    """
    if os.path.exists(_ACS_CACHE):
        try:
            return _load_cache(_ACS_CACHE, seed=seed)
        except Exception as e:  # stale/incompatible/seed-mismatched cache -> rebuild
            print(f"[gdr.data] cache rejected ({e}); rebuilding.")

    try:
        from folktables import ACSDataSource  # lazy (no network at import)
    except Exception as e:
        print(f"[gdr.data] ACS blocked: folktables not importable ({e!r}).")
        return None

    try:
        ds = ACSDataSource(survey_year=str(year), horizon=horizon,
                           survey="person", root_dir=_DATA_DIR)
        df = ds.get_data(states=None, download=True, density=1.0, random_seed=seed)
    except Exception as e:
        print(f"[gdr.data] ACS blocked: download/load failed "
              f"(year={year}, horizon={horizon}): {e!r}.")
        print("[gdr.data] Treat ACS as a blocked result; the synthetic D1 path "
              "is independent and still usable.")
        return None

    try:
        import pandas as pd  # lazy
    except Exception as e:
        print(f"[gdr.data] ACS blocked: pandas not importable ({e!r}).")
        return None

    # downselect columns (full PUMS has ~286); dedupe (AGEP/WKHP repeat).
    keep_cols = list(dict.fromkeys(_ACS_FEATURES + ["PINCP", "ST", "AGEP", "WKHP", "PWGTP"]))
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # adult_filter (acs.py): AGEP>16, PINCP>100, WKHP>0, PWGTP>=1.
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    df = df.dropna(subset=_ACS_FEATURES + ["PINCP", "ST"])

    rng = np.random.default_rng(seed)              # deterministic (U7)
    kept_names, kept_codes, sub_frames = [], [], []
    for st_name in _ST_NAMES:
        code = _ST_NAME_TO_CODE[st_name]
        g = df[df["ST"] == code]
        if len(g) < 10:                             # drop regions with <10 rows
            continue
        n = min(n_per_region, len(g))
        idx = rng.choice(len(g), size=n, replace=False)   # without replacement
        kept_names.append(st_name)
        kept_codes.append(int(code))
        sub_frames.append(g.iloc[idx].reset_index(drop=True))
    if not sub_frames:
        print("[gdr.data] ACS blocked: no region had >=10 rows after filtering.")
        return None
    sub = pd.concat(sub_frames, ignore_index=True)  # [n, cols]

    y = np.log1p(sub["PINCP"].to_numpy(dtype=np.float64))   # [n]
    X = sub[_ACS_FEATURES].to_numpy(dtype=np.float64)        # [n, 10]
    mu = X.mean(axis=0)                                      # [10]
    sd = X.std(axis=0)                                       # [10]
    sd = np.where(sd < 1e-12, 1.0, sd)                       # guard constant cols
    X = (X - mu) / sd                                        # [n, 10] standardized

    st = sub["ST"].to_numpy(dtype=np.int64)                  # [n]
    orig_A_blocks, orig_b_blocks, n_i_list = [], [], []
    for code, name in zip(kept_codes, kept_names):
        mask = st == code
        Xi = X[mask]                                         # [n_i, 10] orig (standardized)
        yi = y[mask]                                         # [n_i]     orig
        orig_A_blocks.append(Xi.astype(np.float64))
        orig_b_blocks.append(yi.astype(np.float64))
        n_i_list.append(int(mask.sum()))
    n_i_arr = np.array(n_i_list, dtype=np.int64)             # [m]

    meta = {
        "source": "acs_income (D2, experiments.tex:145-152)",
        "cache_schema": _ACS_CACHE_SCHEMA, "seed": int(seed), "year": int(year),
        "horizon": horizon, "features": list(_ACS_FEATURES),
        "target": "log1p(PINCP)", "target_raw": "PINCP",   # U4: chose log1p
        "standardize": "global zero-mean unit-var across all subsampled rows",  # U4
        "subsample": f"n_per_region={n_per_region} without replacement (numpy rng(seed))",
        "group_by": "ST (region of residence; 50 US states + Puerto Rico = 51 regions)",
        "group_variable": "ST",
        "filter": "adult_filter (AGEP>16, PINCP>100, WKHP>0, PWGTP>=1) + dropna(features,PINCP,ST)",
        "density": 1.0,                                     # full PUMS (no pre-subsampling)
        "m": int(len(kept_codes)), "d": int(X.shape[1]), "n_i": n_i_arr.tolist(),
        "region_codes": [int(c) for c in kept_codes], "region_names": kept_names,
        "feature_mean": mu.tolist(), "feature_std": sd.tolist(),
    }
    problem = _build_folded_problem("acs_income", orig_A_blocks, orig_b_blocks, n_i_arr, meta)
    try:
        _save_cache(_ACS_CACHE, problem)
    except Exception as e:  # pragma: no cover - cache best-effort
        print(f"[gdr.data] warning: could not write ACS cache ({e!r}).")
    return problem


def load_problem(name, seed=0):
    """Return a folded Problem by name (SPEC sec4 frozen dispatcher).
    name in {'synthetic','acs_income','smoke'}; 'acs_income' may return None if
    the network download is blocked (see load_acs_income)."""
    if name == "synthetic":
        return make_synthetic(seed=seed)
    if name == "acs_income":
        return load_acs_income(seed=seed)
    if name == "smoke":
        return make_smoke(seed=seed)
    raise ValueError(f"unknown problem name {name!r}; "
                     f"expected one of 'synthetic', 'acs_income', 'smoke'.")
