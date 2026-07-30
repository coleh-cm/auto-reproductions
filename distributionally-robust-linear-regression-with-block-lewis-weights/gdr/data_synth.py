"""Synthetic heterogeneous group-regression generator (SPEC section 6 item 1).

Reconstructs the adversarial synthetic instance of
*Distributionally Robust Linear Regression With Block Lewis Weights*
(Manoj & Patel, 2026; arXiv 2607.00252) described qualitatively in
``paper/experiments.tex:19-38``.  **The notebook that pinned the generator
down was never published** (the arXiv source bundle contains no notebook,
see SPEC section 0), so every numeric constant below is a
``[reconstructed]`` decision (SPEC section 6 item 1) tuned to hit the four
numeric anchors the paper does state:

    d = 10,  m = 100,  5 adversarial groups,  kappa(stacked Gram) ~ 1e5,
    and a visible ERM-vs-robust-optimum gap            (paper/experiments.tex:38)

Construction (RECONSTRUCTED; recorded here because the notebook was never
published):

  * A shared orthonormal eigenbasis ``Q in R^{d x d}`` drawn Haar-measure
    from a seeded ``numpy.random.Generator`` (QR of a standard-normal matrix,
    ``paper/experiments.tex:19`` -- "shared orthonormal coordinate system").
  * Every group's Hessian is diagonal-on-Q, i.e. its curvature spectrum is
    ``lambda_i in R^d`` realised in the Q basis (``paper/experiments.tex:19``:
    "Hessian shares eigenvectors ... eigenvalues vary across groups").
  * Normal groups (``m - n_adversarial`` of them): eigenvalues log-uniform in
    ``[lo, hi]`` (default ``[1e-1, 1e1]``, a "moderate condition number",
    ``paper/experiments.tex:22``); group optimum ``x_i*`` sampled tightly
    around the common population centre ``x_pop = 0`` (``paper/experiments.tex:22``:
    "concentrated around a common center"); per-group design
    ``A_{S_i} = sqrt(n_i) * G_i @ diag(sqrt(lambda_i)) @ Q`` with
    ``G_i in R^{n_i x d}`` iid standard normal ("rows = n_i standard normal
    covariates scaled", SPEC section 6 item 1); response
    ``b_{S_i} = A_{S_i} x_i* + eps``, ``eps ~ N(0, sigma_n^2 I)``,
    ``sigma_n`` moderate (``paper/experiments.tex:22``).
  * Adversarial groups (``n_adversarial`` of them): one eigen-direction per
    group has *extreme* curvature ``L_adv`` (the distinct sharp direction per
    adversarial group, ``paper/experiments.tex:25``); the remaining
    directions stay in the moderate ``[lo, hi]`` range; the group optimum lies
    FAR from ``x_pop`` along its sharp direction (offset magnitude
    ``offset``, ``paper/experiments.tex:25``); the noise is very small
    (``sigma_adv``), "sharply concentrated around their optima"
    (``paper/experiments.tex:25``).
  * Per-group sizes ``n_i = n_per_group`` (SPEC section 6 item 2,
    ``[reconstructed]``); total ``n = m * n_per_group`` (default 5000).

Condition-number calibration (the one anchor that is not determined by the
qualitative recipe).  Because all group Hessians share the eigenbasis ``Q``,
the stacked Gram ``A^T A`` is (in expectation, and up to the Wishart
fluctuation of the ``G_i``) diagonal in ``Q`` with entries
``n_i^2 * (sum_i lambda_i_j)``.  Holding every random draw fixed and varying
only the adversarial sharp eigenvalue ``L_adv`` moves the largest stacked
eigenvalue (the adversarial sharp direction) while leaving the smallest
eigenvalue (a normal-only direction) essentially untouched, so the
condition number is (to leading order) affine in ``L_adv``:

    kappa(L_adv) = (base + n_per^2 * L_adv) / g_min          (reconstructed)

We therefore build once with a trial ``L_adv = 1`` to read off ``g_min`` and
``base``, solve the affine equation for the ``L_adv`` that yields
``cond_target``, rebuild, and take one Newton-style correction using the
realised ``g_min`` so the *realised* condition number lands within a few
percent of ``cond_target`` (always within one order of magnitude, per
SPEC section 6 item 1 / ``paper/experiments.tex:38``).

The adversarial extreme curvature that this calibration produces is large
(on the order of ``1e7`` in the eigenvalue, i.e. ``~1e3.5`` on the
singular-value scale) -- the "e.g. 1e3 to 1e4" range in the task prompt is
only a qualitative guide; the spec mandates *tuning the adversarial
curvature magnitude so cond(A^T A) lands near 1e5*, which is what this
routine does.

Determinism: the single source of randomness is a
``numpy.random.Generator`` built from the integer ``seed`` argument; no
global ``numpy.random`` is touched, and the calibration rebuilds reuse the
exact same draws (only the sharp eigenvalue changes between rebuilds), so
the returned problem is a pure function of ``seed``.

Only ``numpy`` is required for generation; the ``__main__`` self-test also
uses ``cvxpy`` to compute the robust optimum (E4, ``paper/experiments.tex:44-50``)
so the ERM-gap anchor can be checked.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow this file to be run directly as `python gdr/data_synth.py` (hard rule
# 6): when executed as a script the script's own dir is on sys.path, not the
# repo root, so the package-qualified import below would fail.  Put the repo
# root (parent of this file's dir) on sys.path when `gdr` is not importable.
try:
    from gdr.types import GroupProblem
except ImportError:  # pragma: no cover - depends on invocation style
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from gdr.types import GroupProblem

__all__ = ["make_synth"]


# --------------------------------------------------------------------------
# Internal reconstruction constants (SPEC section 6 item 1; [reconstructed]).
# Exposed as keyword-only overrides so the four hard anchors are tunable,
# but the public signature of ``make_synth`` is exactly the SPEC contract.
# --------------------------------------------------------------------------
_DEFAULT_OFF = 7.0          # adversarial optimum offset along sharp dir (5-10)
_DEFAULT_NORMAL_SPREAD = 0.2  # std of normal optima around x_pop (concentrated)
_DEFAULT_SIGMA_N = 0.5       # normal-group noise std (moderate)
_DEFAULT_SIGMA_ADV = 0.05    # adversarial-group noise std (very small)
_DEFAULT_LO = 1e-1           # moderate eigenvalue range lower bound
_DEFAULT_HI = 1e1            # moderate eigenvalue range upper bound


def _haar_basis(d: int, rng: np.random.Generator) -> np.ndarray:
    """Return a Haar-distributed orthonormal ``Q in R^{d x d}`` (seeded).

    Constructed as the Q factor of a ``d x d`` standard-normal matrix, which
    is uniformly distributed over the orthogonal group (Stewart / Mezzadri).
    This is the "shared orthonormal coordinate system" of
    `paper/experiments.tex:19`.
    """
    G = rng.standard_normal((d, d))              # [d, d] standard normal
    Q, R = np.linalg.qr(G)                         # [d, d] orthonormal, R upper-tri
    # Enforce a deterministic positive diagonal on R so the Haar draw is a
    # pure function of the seed (numpy's QR signs are deterministic for a
    # fixed matrix; this normalises them regardless of the BLAS backend).
    sgn = np.sign(np.diagonal(R))                 # [d]
    Q = Q * sgn[None, :]                           # [d, d]
    assert Q.shape == (d, d), f"Q shape {Q.shape}, expected ({d},{d})"
    assert np.allclose(Q.T @ Q, np.eye(d)), "Q is not orthonormal"
    return Q


def _draw_raw(
    m: int,
    d: int,
    n_adv: int,
    n_per: int,
    lo: float,
    hi: float,
    offset: float,
    normal_spread: float,
    sigma_n: float,
    sigma_adv: float,
    seed: int,
) -> dict:
    """Draw every random object the generator needs, *except* the adversarial
    sharp eigenvalue ``L_adv`` (which is applied later by ``_build`` so the
    condition-number calibration can vary it without re-rolling the rng).

    All returned arrays carry explicit shapes in comments.
    """
    rng = np.random.default_rng(seed)
    Q = _haar_basis(d, rng)                                  # [d, d]

    x_pop = np.zeros(d, dtype=np.float64)                    # [d]
    lambdas = np.empty((m, d), dtype=np.float64)             # [m, d] moderate spectra
    xstar_q = np.empty((m, d), dtype=np.float64)              # [m, d] optima in Q basis
    sharp_idx = np.arange(n_adv, dtype=np.int64)             # [n_adv] distinct dirs

    if n_adv > d:
        raise ValueError(
            f"need n_adversarial ({n_adv}) <= d ({d}) for distinct sharp directions"
        )

    # adversarial sharp-direction signs (distinct direction per group)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_adv)    # [n_adv]

    log_lo, log_hi = np.log(lo), np.log(hi)
    for i in range(m):
        # moderate spectrum, log-uniform in [lo, hi] per coordinate
        lam = rng.uniform(log_lo, log_hi, size=d)            # [d] (log space)
        if i < n_adv:
            k = int(sharp_idx[i])
            lam[k] = 0.0  # placeholder log(L_adv)=log(1); overwritten in _build
            xstar_q[i] = x_pop.copy()                        # [d]
            xstar_q[i, k] = offset * signs[i]                # far along sharp dir
        else:
            # normal optima: concentrated around x_pop (paper/experiments.tex:22)
            xstar_q[i] = x_pop + normal_spread * rng.standard_normal(d)  # [d]
        lambdas[i] = np.exp(lam)                             # [d]

    # per-group covariate matrices and noise (drawn once, reused across L_adv)
    Gis = [rng.standard_normal((n_per, d)) for _ in range(m)]   # list of [n_per, d]
    noises = [
        (sigma_adv if i < n_adv else sigma_n) * rng.standard_normal(n_per)  # [n_per]
        for i in range(m)
    ]
    return {
        "Q": Q,                       # [d, d]
        "lambdas": lambdas,           # [m, d] (sharp entries are placeholder 1.0)
        "xstar_q": xstar_q,           # [m, d]
        "sharp_idx": sharp_idx,       # [n_adv]
        "Gis": Gis,                   # list[m] of [n_per, d]
        "noises": noises,             # list[m] of [n_per]
        "n_per": n_per,
    }


def _build(raw: dict, L_adv: float) -> GroupProblem:
    """Assemble a `GroupProblem` from the fixed random draws in ``raw`` and a
    chosen adversarial sharp eigenvalue ``L_adv``.

    Per-group design (SPEC section 6 item 1, ``paper/experiments.tex:19``)::

        A_{S_i} = sqrt(n_i) * G_i @ diag(sqrt(lambda_i)) @ Q      # [n_i, d]
        b_{S_i} = A_{S_i} x_i* + eps                              # [n_i]

    with ``x_i*`` the ambient optimum ``Q @ xstar_q[i]``.
    """
    Q = raw["Q"]                                  # [d, d]
    lambdas = raw["lambdas"].copy()               # [m, d]
    xstar_q = raw["xstar_q"]                      # [m, d]
    sharp_idx = raw["sharp_idx"]                  # [n_adv]
    Gis = raw["Gis"]                              # list[m] of [n_per, d]
    noises = raw["noises"]                        # list[m] of [n_per]
    n_per = raw["n_per"]
    m, d = lambdas.shape

    # apply the chosen adversarial sharp curvature
    for i in range(sharp_idx.shape[0]):
        lambdas[i, sharp_idx[i]] = L_adv

    Alist = []   # list[m] of [n_per, d]
    blist = []   # list[m] of [n_per]
    gid = []     # list[m] of [n_per] int32
    for i in range(m):
        Di = np.sqrt(lambdas[i])                  # [d] per-coordinate scales
        Gi = Gis[i]                               # [n_per, d]
        # A_{S_i} = sqrt(n_i) * (Gi * Di[None,:]) @ Q   ->  [n_per, d]
        Ai = np.sqrt(n_per) * (Gi * Di[None, :]) @ Q
        assert Ai.shape == (n_per, d), f"A_{i} shape {Ai.shape}, expected ({n_per},{d})"
        xi_amb = Q @ xstar_q[i]                   # [d] ambient optimum
        bi = Ai @ xi_amb + noises[i]              # [n_per]
        Alist.append(Ai)
        blist.append(bi)
        gid.append(np.full(n_per, i, dtype=np.int32))

    A = np.vstack(Alist).astype(np.float64)       # [n, d]   n = m * n_per
    b = np.concatenate(blist).astype(np.float64)  # [n]
    group_id = np.concatenate(gid).astype(np.int32)  # [n]
    assert A.shape[0] == m * n_per, "stacked design row count mismatch"
    return GroupProblem(A, b, group_id)


def _stacked_gram_cond(gp: GroupProblem) -> tuple[float, float]:
    """Return ``(cond, g_min)`` of the stacked Gram ``A^T A`` (``[d, d]``).

    ``cond = lambda_max / lambda_min`` is the condition number anchored at
    ~1e5 by ``paper/experiments.tex:38``.  Eigenvalues via the symmetric
    solver; ``A^T A`` is symmetric PSD by construction.
    """
    G = gp.A.T @ gp.A                              # [d, d] = [d, n] @ [n, d]
    assert G.shape == (gp.d, gp.d), f"Gram shape {G.shape}"
    ev = np.linalg.eigvalsh(G)                     # [d] ascending eigenvalues
    g_min = float(ev[0])
    g_max = float(ev[-1])
    if g_min <= 0.0:
        raise ValueError(f"stacked Gram is not PD (g_min={g_min}); degenerate design")
    return g_max / g_min, g_min


def _calibrate_sharp_curvature(
    raw: dict, cond_target: float, n_per: int
) -> tuple[GroupProblem, float, float]:
    """Solve the affine condition-number model for ``L_adv`` and return the
    built `GroupProblem`, the chosen ``L_adv``, and the *realised* condition
    number.

    Model (reconstructed; see module docstring)::

        kappa(L) = (base + n_per^2 * L) / g_min

    where ``g_min`` is the smallest stacked-Gram eigenvalue (a normal-only
    direction, independent of ``L_adv``) and ``base`` is the non-sharp part
    of the largest (adversarial sharp) eigenvalue.  A trial build with
    ``L = 1`` reads off ``base`` and ``g_min``; one Newton-style correction
    using the realised ``g_min`` tightens the fit so the realised condition
    number lands within a few percent of ``cond_target``.
    """
    n_per2 = float(n_per) ** 2

    # --- trial build, read affine model ----------------------------------
    L0 = 1.0
    gp0 = _build(raw, L0)
    cond0, g_min0 = _stacked_gram_cond(gp0)
    g_max0 = cond0 * g_min0
    base = g_max0 - n_per2 * L0                     # non-sharp part of the max eig

    # first estimate of L_adv from the affine model
    L1 = (cond_target * g_min0 - base) / n_per2
    if not np.isfinite(L1) or L1 <= 0.0:
        raise ValueError(
            f"condition-number calibration degenerate: L_adv={L1} "
            f"(cond0={cond0}, g_min0={g_min0}, base={base})"
        )

    # --- one correction using the realised g_min at L1 -------------------
    gp1 = _build(raw, L1)
    cond1, g_min1 = _stacked_gram_cond(gp1)
    g_max1 = cond1 * g_min1
    base1 = g_max1 - n_per2 * L1
    L2 = (cond_target * g_min1 - base1) / n_per2
    if not np.isfinite(L2) or L2 <= 0.0:
        # fall back to the first estimate if the correction misbehaves
        L2 = L1

    gp = _build(raw, L2)
    cond, _ = _stacked_gram_cond(gp)
    return gp, float(L2), float(cond)


def make_synth(
    m: int = 100,
    d: int = 10,
    n_adversarial: int = 5,
    n_per_group: int = 50,
    cond_target: float = 1e5,
    seed: int = 0,
    *,
    offset: float = _DEFAULT_OFF,
    normal_spread: float = _DEFAULT_NORMAL_SPREAD,
    sigma_n: float = _DEFAULT_SIGMA_N,
    sigma_adv: float = _DEFAULT_SIGMA_ADV,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> GroupProblem:
    """Generate the synthetic heterogeneous group-regression instance.

    Parameters
    ----------
    m : int
        Number of groups (default 100, `paper/experiments.tex:38`).
    d : int
        Ambient dimension (default 10, `paper/experiments.tex:38`).
    n_adversarial : int
        Number of adversarial (high-curvature, far-optimum) groups
        (default 5, `paper/experiments.tex:38`).  Must be ``<= d`` so each
        has a distinct sharp eigen-direction.
    n_per_group : int
        Rows per group ``n_i`` (default 50, SPEC section 6 item 2,
        ``[reconstructed]``); total rows ``n = m * n_per_group``.
    cond_target : float
        Target condition number of the stacked Gram ``A^T A`` (default
        ``1e5``, `paper/experiments.tex:38`).  The realised condition number
        lands within one order of magnitude of this.
    seed : int
        Integer seed for the single `numpy.random.Generator` used.

    Keyword-only (reconstruction constants, SPEC section 6 item 1) :
        offset, normal_spread, sigma_n, sigma_adv, lo, hi.

    Returns
    -------
    GroupProblem
        A grouped least-squares problem with ``m`` contiguous groups of
        ``n_per_group`` rows each, shared-Q curvature structure, ``n_adv``
        adversarial groups, and stacked-Gram condition number near
        ``cond_target``.

    Raises
    ------
    ValueError
        On degenerate input (``n_adversarial > d``, non-positive sizes, a
        non-PD stacked Gram, or a calibration that cannot reach the target).
    """
    # --- input validation ------------------------------------------------
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    if n_adversarial < 0 or n_adversarial > m:
        raise ValueError(f"n_adversarial must be in [0, m]=[0,{m}], got {n_adversarial}")
    if n_adversarial > d:
        raise ValueError(
            f"n_adversarial ({n_adversarial}) must be <= d ({d}) for distinct sharp dirs"
        )
    if n_per_group < 1:
        raise ValueError(f"n_per_group must be >= 1, got {n_per_group}")
    if not np.isfinite(cond_target) or cond_target <= 1.0:
        raise ValueError(f"cond_target must be > 1, got {cond_target}")
    if not np.isscalar(seed) or int(seed) < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    if lo <= 0 or hi <= lo:
        raise ValueError(f"need 0 < lo < hi, got lo={lo} hi={hi}")
    if offset <= 0:
        raise ValueError(f"offset must be > 0, got {offset}")
    if normal_spread < 0 or sigma_n < 0 or sigma_adv < 0:
        raise ValueError("normal_spread / sigma_n / sigma_adv must be >= 0")

    raw = _draw_raw(
        m=m,
        d=d,
        n_adv=n_adversarial,
        n_per=n_per_group,
        lo=lo,
        hi=hi,
        offset=offset,
        normal_spread=normal_spread,
        sigma_n=sigma_n,
        sigma_adv=sigma_adv,
        seed=int(seed),
    )
    gp, L_adv, cond = _calibrate_sharp_curvature(raw, cond_target, n_per_group)

    # The realised condition number must land within one order of magnitude
    # of the target (SPEC section 6 item 1 / paper/experiments.tex:38); a
    # construction that misses that anchor has failed and must not silently
    # return a degenerate-looking problem.
    lo_band, hi_band = cond_target / 10.0, cond_target * 10.0
    if not (lo_band <= cond <= hi_band):
        raise ValueError(
            f"realised cond {cond:.3e} outside the one-order-of-magnitude band "
            f"[{lo_band:.1e}, {hi_band:.1e}] around target {cond_target:.1e}"
        )

    return gp


# --------------------------------------------------------------------------
# Self-test (hard rule 6): builds a tiny problem inline, computes the robust
# optimum via cvxpy (E4), and asserts the four anchors hold.  Does not
# depend on sibling modules.
# --------------------------------------------------------------------------
def _reference_optimum_epigraph(gp: GroupProblem) -> tuple[np.ndarray, float]:
    """Robust optimum (E4) via the cvxpy epigraph program.

    ``min_{x,t} t  s.t.  (1/n_i)||A_{S_i} x - b_{S_i}||^2 <= t  for all i``
    (`paper/experiments.tex:44-50`).  The problem is internally rescaled so
    the solver sees O(1) numbers (the sharp-curvature constraints make the
    raw problem ill-scaled at cond ~1e5).  Hard rule 3: a failed solve raises
    rather than returning a negative verdict.
    """
    import cvxpy as cp

    d = gp.d
    x = cp.Variable(d)                  # [d]
    t = cp.Variable()                   # scalar epigraph
    cons = [t >= 0]

    # rescale so the solver operates on O(1) magnitudes (cond is invariant)
    scale = float(np.sqrt(np.mean(gp.A ** 2)))
    if scale <= 0:
        raise ValueError("design matrix is all-zero; degenerate problem")
    A = gp.A / scale                    # [n, d]
    b = gp.b / scale                    # [n]
    for i in range(gp.m):
        s, e = gp.slices[i]
        Ai = A[s:e]                     # [n_i, d]
        bi = b[s:e]                     # [n_i]
        ni = float(e - s)
        # E1 group-loss epigraph constraint (paper/experiments.tex:9,44-50)
        cons.append(t >= (1.0 / ni) * cp.sum_squares(Ai @ x - bi))
    prob = cp.Problem(cp.Minimize(t), cons)

    # try solvers in order of accuracy; accept only a verifiably feasible solve
    attempts = [
        (cp.CLARABEL, dict(max_iter=1000, tol_gap_abs=1e-9, tol_gap_rel=1e-9,
                           tol_feas=1e-9, tol_infeas_abs=1e-9, tol_infeas_rel=1e-9)),
        (cp.SCS, dict(max_iters=100000, eps=1e-9)),
        (cp.CLARABEL, dict()),
    ]
    last_status = None
    for solver, kw in attempts:
        try:
            prob.solve(solver=solver, **kw)
        except Exception as ex:  # solver raised; try the next
            last_status = f"{solver}: {ex}"
            continue
        last_status = str(prob.status)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            continue
        if x.value is None or prob.value is None:
            continue
        x_star = np.asarray(x.value, dtype=np.float64).ravel()  # [d]
        opt_val = float(prob.value) * (scale ** 2)              # back to original scale
        # verify feasibility on the ORIGINAL problem (hard rule 3): the
        # returned x must actually achieve the claimed worst-group loss.
        worst = gp.worst_loss(x_star)
        if abs(worst - opt_val) <= 1e-3 * max(1.0, abs(opt_val)) + 1e-3:
            assert x_star.shape == (d,)
            return x_star, opt_val

    raise RuntimeError(
        f"cvxpy epigraph solve failed to produce a feasible solution "
        f"(last status: {last_status}); cannot trust the robust optimum"
    )


def _self_test(seed: int = 0) -> None:
    """Construct the default instance and check the four paper anchors."""
    gp = make_synth(seed=seed)

    # structural anchors: m=100, d=10, 5 adversarial, n=5000 (experiments.tex:38)
    assert gp.m == 100, f"m={gp.m}, expected 100"
    assert gp.d == 10, f"d={gp.d}, expected 10"
    assert gp.n == 5000, f"n={gp.n}, expected 5000"
    assert int(gp.sizes.sum()) == gp.n, "sizes sum != n"
    assert (gp.sizes == 50).all(), f"per-group sizes not all 50: {gp.sizes}"
    # 5 adversarial groups = the first 5 (group_id contiguous, nondecreasing)
    assert (np.diff(gp.group_id) >= 0).all(), "group_id not nondecreasing"
    # eigenbasis / curvature anchor: kappa(A^T A) ~ 1e5 within one order of mag
    cond, g_min = _stacked_gram_cond(gp)
    assert 1e4 <= cond <= 1e6, f"cond {cond:.3e} outside [1e4, 1e6]"

    # ERM (E3) and robust optimum (E4); F(ERM) > OPT by a visible margin
    x_erm = gp.erm()                                  # [d]  (E3, intro.tex:5)
    assert x_erm.shape == (gp.d,)
    F_erm = gp.worst_loss(x_erm)                       # E2 (experiments.tex:13)
    x_opt, OPT = _reference_optimum_epigraph(gp)       # E4 (experiments.tex:44-50)
    assert x_opt.shape == (gp.d,)
    # sanity: OPT is a lower bound on any feasible worst-group loss
    assert OPT <= F_erm + 1e-6 * max(1.0, F_erm), "OPT > F(ERM) (infeasible)"

    gap = F_erm - OPT
    ratio = F_erm / OPT if OPT > 0 else np.inf
    # "visible margin": require at least a 5% relative gap.  At the default
    # seed this is ~11%, comfortably above the bar.
    assert ratio > 1.05, (
        f"ERM-vs-OPT gap too small: F(ERM)={F_erm:.4g}, OPT={OPT:.4g}, ratio={ratio:.4f}"
    )

    print(
        f"cond(A^T A) = {cond:.4e}  (target 1e5, g_min={g_min:.3e})\n"
        f"n={gp.n}  m={gp.m}  d={gp.d}  n_per_group={gp.sizes[0]}\n"
        f"F(ERM)     = {F_erm:.4g}\n"
        f"OPT (E4)   = {OPT:.4g}\n"
        f"gap        = {gap:.4g}   (ratio F(ERM)/OPT = {ratio:.4f})\n"
        f"PASS"
    )


if __name__ == "__main__":
    _self_test()
