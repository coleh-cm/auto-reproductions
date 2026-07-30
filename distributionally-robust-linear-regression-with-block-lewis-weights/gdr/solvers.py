"""Optimizer arms + the CVXPY reference optimum (E4, arms 3.0-3.4).

Every arm returns a :class:`History` whose ``x`` records one iterate per *natural
outer iteration* (``paper/experiments.tex:96-104``):

* subgradient / smoothed gradient : one full (sub)gradient update = one iteration
* IPM                              : one outer Newton step of the barrier = one iteration
* ball-oracle                      : one trust-region Newton solve = one iteration

``x[0]`` is the warm start, so the gap curve has ``T+1`` points for ``T`` outer
iterations.  ``wall[t]`` is the cumulative wall-clock at iterate ``t``.

All arms ``raise`` on degenerate inputs (empty budget, zero-length solve, a CVXPY
failure) instead of returning an empty History -- a fit that did nothing must
fail loudly (SPEC: "no success path may report OK on an empty result").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .types import GroupProblem
from .objectives import SmoothObjective, make_smoothed, make_regularized
from . import lewis

__all__ = [
    "History",
    "reference_optimum",
    "solve_subgradient",
    "solve_smooth",
    "solve_ipm",
    "solve_ball_oracle",
]


@dataclass
class History:
    """Iterate trace + wall-clock per natural outer iteration.

    ``x[0]`` is the warm start; ``x[t]`` for ``t>=1`` is after the t-th outer
    iteration.  ``wall`` is cumulative seconds from ``time.perf_counter``.
    """

    x: list[np.ndarray] = field(default_factory=list)
    wall: list[float] = field(default_factory=list)

    def append(self, x: np.ndarray, wall: float) -> None:
        self.x.append(np.asarray(x, dtype=np.float64).ravel())
        self.wall.append(float(wall))

    def worst_losses(self, problem: GroupProblem) -> np.ndarray:
        return np.array([problem.worst_loss(xt) for xt in self.x], dtype=np.float64)


# ------------------------------------------------------------------------------------------------
# Reference arm: CVXPY epigraph QCQP  (E4)
# ------------------------------------------------------------------------------------------------
def reference_optimum(problem: GroupProblem, solver_name: str | None = None) -> tuple[np.ndarray, float]:
    """E4: solve  min_{x,t} t  s.t.  (1/n_i)||A_{S_i} x - b_{S_i}||^2 <= t  for all i
    via CVXPY (``paper/experiments.tex:44-50``).  Returns ``(x_star, OPT)``.

    Raises if the solve fails or returns a non-finite status -- a grader that
    cannot run must raise, never silently return a wrong value.
    """
    import cvxpy as cp

    d = problem.d
    x = cp.Variable(d)
    t = cp.Variable()
    cons = []
    for i, (s, e) in enumerate(problem.slices):
        Ai = problem.A[s:e]
        bi = problem.b[s:e]
        ni = float(problem.sizes[i])
        # (1/n_i) ||A_i x - b_i||_2^2 <= t
        cons.append(cp.sum_squares(Ai @ x - bi) / ni <= t)
    prob = cp.Problem(cp.Minimize(t), cons)
    kwargs = {}
    if solver_name is not None:
        kwargs["solver"] = solver_name
    try:
        prob.solve(**kwargs)
    except Exception:
        # try the default conic solvers in turn, but raise if all fail
        for sn in ("CLARABEL", "SCS", "ECOS"):
            try:
                prob.solve(solver=getattr(cp, sn))
                break
            except Exception:
                continue
        else:
            raise RuntimeError("reference_optimum: all CVXPY solvers failed")
    if prob.status not in ("optimal", "optimal_inaccurate") or x.value is None or t.value is None:
        raise RuntimeError(f"reference_optimum: CVXPY status {prob.status!r} -- refusing to return a bad optimum")
    xstar = np.asarray(x.value, dtype=np.float64).ravel()
    opt = float(t.value)
    if not np.isfinite(opt):
        raise RuntimeError(f"reference_optimum: non-finite OPT {opt}")
    return xstar, opt


# ------------------------------------------------------------------------------------------------
# Subgradient arm (3.1)
# ------------------------------------------------------------------------------------------------
def solve_subgradient(problem: GroupProblem, x0: np.ndarray, cfg: dict, budget: int) -> History:
    """Subgradient descent on the non-smooth max-loss (``paper/experiments.tex:57-58``).

    Step: pick ``i* = argmax_i ell_i(x)`` (lowest index on ties, SPEC section 6
    item 18), subgradient ``g = (2/n_{i*}) A_{S_i*}^T (A_{S_i*} x - b_{S_i*})``
    (gradient of the active group loss), update ``x <- x - eta_t g``.

    ``cfg["schedule"]`` selects ``"fixed"`` (eta_t = step) or ``"inv_sqrt"``
    (eta_t = step / sqrt(t+1)); ``cfg["step"]`` is the base step.  The paper runs
    both and reports the best, so this function runs whichever schedule is
    configured and the harness picks the lower-worst-loss variant.
    """
    if budget <= 0:
        raise ValueError("subgradient budget must be > 0")
    step = float(cfg["step"])
    schedule = cfg.get("schedule", "fixed")
    if step is None or step <= 0:
        raise ValueError(f"subgradient step must be > 0, got {step}")
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    sizes = problem.sizes.astype(np.float64)
    h = History()
    t0 = time.perf_counter()
    h.append(x, 0.0)
    for t in range(budget):
        losses = problem.group_losses(x)
        i_star = int(np.argmax(losses))
        s, e = problem.slices[i_star]
        Ai = problem.A[s:e]
        ri = Ai @ x - problem.b[s:e]
        g = (2.0 / sizes[i_star]) * (Ai.T @ ri)
        eta = step if schedule == "fixed" else step / np.sqrt(t + 1.0)
        x = x - eta * g
        h.append(x, time.perf_counter() - t0)
    if len(h.x) <= 1:
        raise RuntimeError("subgradient produced no iterates")
    return h


# ------------------------------------------------------------------------------------------------
# Smoothed gradient arms (3.2): GD / Heavy-Ball / Nesterov on ftilde
# ------------------------------------------------------------------------------------------------
def solve_smooth(problem: GroupProblem, x0: np.ndarray, cfg: dict, budget: int) -> History:
    """Smoothed gradient methods on f~_{beta,delta} (``paper/experiments.tex:60-66``).

    ``cfg["method"]`` in {``"gd"``, ``"heavy_ball"``, ``"nesterov"``}; smoothing
    params ``beta``, ``delta`` and the step ``step`` (and ``momentum`` for
    Heavy-Ball) are tuned externally.
    """
    if budget <= 0:
        raise ValueError("smooth budget must be > 0")
    method = cfg["method"]
    beta = float(cfg["beta"]); delta = float(cfg["delta"]); step = float(cfg["step"])
    if beta <= 0 or delta <= 0 or step <= 0:
        raise ValueError(f"bad smooth cfg: beta={beta} delta={delta} step={step}")
    sm = make_smoothed(problem, beta, delta)
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    h = History()
    t0 = time.perf_counter()
    h.append(x, 0.0)
    y = x.copy()                     # Nesterov momentum point
    x_prev = x.copy()                # Heavy-Ball previous iterate
    for t in range(budget):
        if method == "gd":
            g = sm.grad(x)
            x = x - step * g
        elif method == "heavy_ball":
            mu = float(cfg["momentum"])
            g = sm.grad(x)
            d = x - x_prev
            x_new = x - step * g + mu * d
            x_prev = x
            x = x_new
        elif method == "nesterov":
            # standard accelerated gradient on a smooth convex f
            mom = float(cfg.get("momentum", (t / (t + 3.0))))     # default Nesterov schedule
            y = x + mom * (x - x_prev) if t > 0 else x
            g = sm.grad(y)
            x_new = y - step * g
            x_prev = x
            x = x_new
        else:
            raise ValueError(f"unknown smooth method {method!r}")
        h.append(x, time.perf_counter() - t0)
    if len(h.x) <= 1:
        raise RuntimeError("smooth produced no iterates")
    return h


# ------------------------------------------------------------------------------------------------
# IPM arm (3.3): log-barrier interior point on the epigraph reformulation
# ------------------------------------------------------------------------------------------------
def solve_ipm(problem: GroupProblem, x0: np.ndarray, cfg: dict, budget: int) -> History:
    """Log-barrier IPM on  min_{x,t} tau*t - sum_i log(t - ell_i(x))
    (``paper/experiments.tex:68-69``, Boyd-Vandenberghe section 6.4).

    Each *outer* iteration: one Newton step on the centring problem at the
    current barrier parameter ``tau``, then ``tau`` is grown by ``growth``.
    Inner damping is a Levenberg-style ``lambda * I`` on the KKT Hessian.  All
    barrier internals are reconstructed (SPEC section 6 item 10): defaults are
    barrier0 = 1, growth = 10, inner cap = 1 Newton step per outer iteration
    (so "one iteration = one outer Newton step", matching the paper's accounting).
    """
    if budget <= 0:
        raise ValueError("ipm budget must be > 0")
    A = problem.A
    b = problem.b
    sizes = problem.sizes.astype(np.float64)
    slices = problem.slices
    d = problem.d
    m = problem.m
    lam = float(cfg.get("damping", 1e-8))
    # barrier parameter schedule: short-step growth 1 + growth/sqrt(m) (default
    # growth coefficient 1.0 -> the textbook 1+1/sqrt(m) short step).  ``barrier0``
    # is the initial tau; a near-central start (m/barrier0 ~ gap0) minimises the
    # centring cost.  All internals reconstructed (SPEC section 6 item 10).
    tau = float(cfg.get("barrier0", 1.0))
    growth_coef = float(cfg.get("growth", 1.0))
    centring_tol = float(cfg.get("centring_tol", 0.5))     # Newton decrement threshold

    def grad_hess(xv, tv, tau_v):
        g = np.zeros(d + 1); H = np.zeros((d + 1, d + 1))
        g[d] = tau_v
        for i, (s, e) in enumerate(slices):
            Ai = A[s:e]; bi = b[s:e]; ni = sizes[i]
            ri = Ai @ xv - bi
            li = float(ri @ ri) / ni
            gap = tv - li
            if gap <= 1e-12:
                return None
            inv = 1.0 / gap
            ge = (2.0 / ni) * (Ai.T @ ri)
            g[:d] += inv * ge
            g[d] -= inv
            He = (2.0 / ni) * (Ai.T @ Ai)
            H[:d, :d] += (inv * inv) * np.outer(ge, ge) + inv * He
            H[:d, d] += -inv * inv * ge
            H[d, :d] += -inv * inv * ge
            H[d, d] += inv * inv
        return g, H

    def barrier_obj(xv, tv, tau_v):
        L = problem.group_losses(xv)
        gg = tv - L
        if np.any(gg <= 0):
            return float("inf")
        return tau_v * tv - float(np.sum(np.log(gg)))

    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    losses = problem.group_losses(x)
    t = float(np.max(losses)) * (1.0 + 1e-3) + 1e-3   # strictly feasible start
    h = History()
    t0 = time.perf_counter()
    h.append(x, 0.0)
    for _ in range(budget):
        gh = grad_hess(x, t, tau)
        if gh is None:                       # lost feasibility -> restore
            t = float(np.max(problem.group_losses(x))) + 1.0
            gh = grad_hess(x, t, tau)
            if gh is None:
                break
        g, H = gh
        Hd = H + lam * np.eye(d + 1)
        try:
            dz = np.linalg.solve(Hd, -g)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(Hd, -g, rcond=None)[0]
        if not np.all(np.isfinite(dz)):
            tau *= max(1.0 + growth_coef / np.sqrt(m), 1.01)
            h.append(x, time.perf_counter() - t0)
            continue
        decrement = float(-g @ dz)            # Newton decrement squared
        # centred for this tau -> increase barrier parameter (counts as one iter)
        if decrement < centring_tol:
            tau *= max(1.0 + growth_coef / np.sqrt(m), 1.01)
            h.append(x, time.perf_counter() - t0)
            continue
        # damped Newton step with feasibility-preserving Armijo backtracking
        fcur = barrier_obj(x, t, tau)
        a = 1.0
        descent = float(g @ dz)
        accepted = False
        for _bt in range(60):
            xn = x + a * dz[:d]; tn = t + a * dz[d]
            ln = problem.group_losses(xn)
            if tn > np.max(ln) + 1e-12 and np.all(np.isfinite(ln)):
                fn = barrier_obj(xn, tn, tau)
                if np.isfinite(fn) and fn <= fcur + 1e-4 * a * descent:
                    accepted = True; break
            a *= 0.5
        if not accepted:
            # cannot reduce at this tau -> nudge tau up and retry next iter
            tau *= max(1.0 + growth_coef / np.sqrt(m), 1.01)
            h.append(x, time.perf_counter() - t0)
            continue
        x = xn; t = tn
        h.append(x, time.perf_counter() - t0)
    if len(h.x) <= 1:
        raise RuntimeError("ipm produced no iterates")
    return h


# ------------------------------------------------------------------------------------------------
# Ball-oracle arms (3.4): trust-region damped-Newton on ftilde (no acceleration)
# ------------------------------------------------------------------------------------------------
def _solve_trust_region(
    sm: SmoothObjective,
    M: np.ndarray,
    q: np.ndarray,
    radius: float,
    inner_iters: int,
    x_scale: float = 1.0,
) -> np.ndarray:
    """One trust-region solve:  min f~(x)  s.t.  ||x - q||_M <= radius.

    Damped Newton with M-ellipsoid projection (SPEC section 6 item 7,
    reconstructed): at each inner step form the Newton system
    ``(H + nu * M) d = -g`` with a Levenberg damping ``nu``, project the step to
    the trust region (scale to the boundary if it exceeds ``radius`` in the
    M-norm), then accept by an Armijo backtracking line search on f~.  The
    regularised surrogate fhat (T2) is just an ``sm`` whose Hessian already
    contains the regulariser, so the same solver handles both.
    """
    x = np.asarray(q, dtype=np.float64).ravel().copy()
    d = x.shape[0]
    # Cholesky of M (M is PSD); fall back to eig-decomp if needed
    try:
        Lm = np.linalg.cholesky(M + 1e-12 * np.eye(d))
    except np.linalg.LinAlgError:
        Lm = None
    Minv = np.linalg.pinv(M)

    def mnorm(v: np.ndarray) -> float:
        return float(np.sqrt(max(v @ (M @ v), 0.0)))

    f0 = sm.value(x)
    for _ in range(inner_iters):
        g = sm.grad(x)
        H = sm.hess(x)
        # damping: ensure H + nu M is PD
        nu = 1e-8
        for _ in range(10):
            try:
                dstep = np.linalg.solve(H + nu * M, -g)
                if np.all(np.isfinite(dstep)):
                    break
            except np.linalg.LinAlgError:
                pass
            nu *= 10.0
        else:
            dstep = -Minv @ g
        # trust-region projection: scale to boundary along Newton dir if outside
        nd = mnorm(dstep)
        if nd > radius and nd > 0:
            dstep = dstep * (radius / nd)
        # Armijo backtracking
        a = 1.0
        fnew = sm.value(x + a * dstep)
        descent = float(g @ dstep)
        while a > 1e-10 and (not np.isfinite(fnew) or fnew > f0 + 1e-4 * a * descent):
            a *= 0.5
            fnew = sm.value(x + a * dstep)
        if a <= 1e-10:
            break                      # no progress -> subproblem converged
        x = x + a * dstep
        f0 = fnew
    return x


def solve_ball_oracle(problem: GroupProblem, x0: np.ndarray, cfg: dict, budget: int) -> History:
    """Trust-region ball-oracle on the smoothed surrogate (``paper/experiments.tex:71-78``).

    One outer iteration = one trust-region Newton solve; the center moves to
    the solve's output and the radius is optionally shrunk (``cfg["decay"]``).
    Geometry is ``naive`` (M = A^T A) or ``lewis`` (M = A^T W A from block Lewis
    weights on the augmented matrix, with the W=I reset of Algorithm 1 line 3).
    No acceleration (``paper/experiments.tex:78``).
    """
    if budget <= 0:
        raise ValueError("ball_oracle budget must be > 0")
    geometry = cfg["geometry"]
    beta = float(cfg["beta"]); delta = float(cfg["delta"])
    radius0 = float(cfg["radius0"]); decay = float(cfg.get("decay", 1.0))
    inner_iters = int(cfg.get("inner_iters", 20))
    reg_on = bool(cfg.get("reg_on", False))
    radius = radius0

    # geometry matrix M
    w_override = cfg.get("weights_override", None)
    if geometry == "naive":
        M = lewis.geometry_matrix(problem, w=None)
        sqrt_w = np.ones(problem.n)
    elif geometry == "lewis":
        if w_override is not None:
            w = np.asarray(w_override, dtype=np.float64).ravel()
        else:
            w = lewis.block_lewis_weights(problem, p=cfg.get("lewis_p", None),
                                          n_iters=cfg.get("lewis_iters"))
        w = lewis.reset_to_identity(w, problem.m)          # Algorithm 1 line 3 reset
        M = lewis.geometry_matrix(problem, w=w, p=cfg.get("lewis_p", None))
        sqrt_w = np.sqrt(lewis.expand_weights(w, problem))
    else:
        raise ValueError(f"unknown geometry {geometry!r}")

    # surrogate (optionally regularised, T2)
    if reg_on:
        min_rank_m = min(int(np.linalg.matrix_rank(problem.A)), problem.m)
        if min_rank_m <= 0:
            raise ValueError("rank(A)=0; cannot form the T2 regulariser")
        coef = float(cfg.get("reg_coef", beta / (1000.0 * min_rank_m)))
        sm = make_regularized(problem, beta, delta, sqrt_w, x0, coef)
    else:
        sm = make_smoothed(problem, beta, delta)

    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    h = History()
    t0 = time.perf_counter()
    h.append(x, 0.0)
    q = x.copy()
    for _ in range(budget):
        q = _solve_trust_region(sm, M, q, radius, inner_iters)
        h.append(q, time.perf_counter() - t0)
        radius *= decay
    if len(h.x) <= 1:
        raise RuntimeError("ball_oracle produced no iterates")
    return h
