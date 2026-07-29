"""ACS Income regression task (SPEC D2, experiments.tex:145-152).

folktables ACSIncome (ding2021retiring), 2018 1-Year PUMS.  Predict log
personal income from d=10 standardized features (AGEP, COW, SCHL, MAR, OCCP,
POBP, RELP, WKHP, SEX, RAC1P) for employed US adults (adult_filter: AGEP>16,
PINCP>100, WKHP>0).  Group by state (ST), all m=51 regions (50 states + PR),
subsample 200 individuals per region -> n=10,200.  Group loss = MSE per region;
warm start = ERM; OPT via CVXPY (experiments.tex:148).

The paper does *regression* on log income, so we override ACSIncome's binary
target_transform with ``log1p(PINCP)``.  Features are z-scored globally
("standardized", experiments.tex:148).  Subsampling is without replacement per
state with a fixed seed (U4/U7: scheme/seed unstated -> our disclosed choice).

Data live in ``data/2018/1-Year/psam_p<code>.csv`` (census PUMS); we read the
CSVs directly for full control over per-state subsampling.  Folding
(1/sqrt(n_i)) is applied in make_problem, so the returned group loss is exactly
the per-state MSE.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .problem import Problem, make_problem

FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
# state FIPS -> 2-char code, matching folktables.load_acs._STATE_CODES (m=51)
_STATE_CODES = {'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
                 'CO': '08', 'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13',
                 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19',
                 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
                 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
                 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
                 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
                 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
                 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50',
                 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56',
                 'PR': '72'}
_STATE_NAMES = list(_STATE_CODES.keys())   # ordered, m=51


def _adult_filter(df: pd.DataFrame) -> pd.DataFrame:
    """folktables canonical adult_filter (employed adults): AGEP>16, PINCP>100,
    WKHP>0, PWGTP>=1  (folktables/acs.py:78-81; the paper uses the ACSIncome task,
    experiments.tex:148). The PWGTP>=1 clause was previously omitted (Round-11
    review): it drops 0 rows on the 2018 1-Year PUMS (verified across all 51
    states: 1.6645M rows pass the first three filters, 0 dropped by PWGTP>=1), so
    the committed ACS results are unchanged numerically, but the filter now
    matches folktables exactly instead of silently dropping a clause.
    """
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    return df


def make_acs_income(
    data_root: str = "data",
    year: str = "2018",
    horizon: str = "1-Year",
    per_state: int = 200,
    seed: int = 0,
    states: list[str] | None = None,
    require_all: bool = True,
) -> Problem:
    """Build the folded ACS-Income GDR problem (D2).

    ``data_root/{year}/{horizon}/psam_p<code>.csv`` must exist for each state.
    If ``require_all`` and a state file is missing, raise; otherwise skip the
    state (m shrinks).  Returns a folded Problem grouped by state in
    ``_STATE_NAMES`` order.
    """
    states = states or _STATE_NAMES
    datadir = os.path.join(data_root, year, horizon)
    rng = np.random.default_rng(seed)
    cols = FEATURES + ["PINCP", "ST", "PWGTP"]
    A_blocks = []
    b_blocks = []
    kept = []
    for st in states:
        code = _STATE_CODES[st]
        path = os.path.join(datadir, f"psam_p{code}.csv")
        if not os.path.isfile(path):
            if require_all:
                raise FileNotFoundError(f"missing ACS file for {st}: {path}")
            continue
        df = pd.read_csv(path, usecols=lambda c: c in cols)
        df = _adult_filter(df)
        # drop rows with any missing feature/target
        df = df.dropna(subset=FEATURES + ["PINCP"])
        if len(df) < per_state:
            # not enough records; use all of them (with replacement-free)
            sel = df
        else:
            sel = df.sample(n=per_state, random_state=int(rng.integers(0, 2**31 - 1)))
        X = sel[FEATURES].to_numpy(dtype=np.float64)        # [n_i, 10]
        y = np.log1p(sel["PINCP"].to_numpy(dtype=np.float64))  # [n_i] log income
        A_blocks.append(X)
        b_blocks.append(y)
        kept.append(st)

    # global standardization of features (mean/std over the whole pooled sample)
    Xall = np.vstack(A_blocks)                               # [n, 10]
    mu = Xall.mean(axis=0)
    sd = Xall.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    A_blocks = [(A - mu) / sd for A in A_blocks]

    prob = make_problem(A_blocks, b_blocks, name="acs_income",
                        n_i_pre=np.array([len(b) for b in b_blocks], dtype=np.int64),
                        meta={"seed": seed, "per_state": per_state, "year": year,
                              "states": kept, "feature_mean": mu.tolist(),
                              "feature_std": sd.tolist()})
    return prob
