"""Log-barrier interior-point method on the epigraph form (E21).

Paper: IPM baseline (E21, experiments.tex:68-69).  We solve the epigraph
reformulation of the worst-group objective

    min_{x,t}  t   s.t.  l_i(x) = ||A_{S_i} x - b_{S_i}||^2 <= t  for all i     (E20)

via the log barrier.  We use the standard *decreasing-μ* central path of
BV04 §11.2 (cited at intro.tex:17),

    Phi_mu(x, t) = t - mu * sum_i ln(t - l_i(x))                                 (E21)

with barrier parameter path mu_k = mu0 * theta^k, 0 < theta < 1.  (SPEC's prose
writes "mu*t - sum ln"; with *decreasing* mu that form diverges — t -> +inf —
so we adopt the equivalent convergent standard form where mu weights the
barrier, not the objective; this is the textbook decreasing-mu central path
and matches the paper's empirical result that the IPM converges rapidly,
experiments.tex:107.)  Centering is by damped Newton; **one outer iteration =
one barrier Newton step on (x, t)** (experiments.tex:100), after which mu is
reduced by theta.  We start strictly feasible at t0 = max_loss(x0) + 1 so every
gap_i = t0 - l_i(x0) >= 1 > 0.

Gradient/Hessian of Phi_mu w.r.t. z = (x [d], t [1]).  Let l_i = ||r_i||^2,
g_i = 2 A_{S_i}^T r_i  [d] (gradient of l_i), H_i = 2 A_{S_i}^T A_{S_i}  [d,d]
(Hessian of l_i, constant in x), gap_i = t - l_i > 0.  Then

    grad_x = mu * sum_i g_i / gap_i                          [d]
    grad_t = 1 - mu * sum_i 1 / gap_i                        scalar
    H_xx   = mu * sum_i ( H_i / gap_i + g_i g_i^T / gap_i^2 )  [d, d]
    H_xt   = -mu * sum_i g_i / gap_i^2                       [d]
    H_tt   = mu * sum_i 1 / gap_i^2                          scalar

(d is tiny, so we form the full (d+1)x(d+1) Hessian and do one linear solve
per Newton step; SPEC sec4 note U13.)  Best-of-grid over (mu0, theta); keep
the config with the lowest final F(x) (experiments.tex:92).  The reported gap
is ``max_loss(problem, x) - opt`` (squared scale).
"""

from __future__ import annotations

import time
import numpy as np

from gdr.solvers import (
    DEFAULTS, as_list, get_opt, gap_now, seed_history, solve_pd,
    max_loss, group_losses,
)


def _group_cache(problem):
    """Precompute constant per-group design Gram matrices.

    Returns a list of (Ai [n_i,d], AtA_i [d,d]) for i = 0..m-1.  H_i = 2*AtA_i.
    """
    A = problem["A"]                       # [n, d]
    offsets = problem["offsets"]           # [m+1] int64
    m = problem["m"]
    cache = []
    for i in range(m):
        s, e = int(offsets[i]), int(offsets[i + 1])
        Ai = A[s:e, :]                      # [n_i, d]
        AtA_i = Ai.T @ Ai                   # [d, d]
        cache.append((Ai, AtA_i))
    return cache


def _newton_step(problem, cache, x, t, mu, inner_tol):
    """One damped Newton step on Phi_mu at z = (x, t).

    Returns (x_new, t_new, decrement, accepted) where ``decrement`` is
    sqrt(grad^T H^{-1} grad) (the Newton decrement) and ``accepted`` is False
    if the step was a no-op (decrement below inner_tol).  Maintains strict
    feasibility (gap_i > 0) via backtracking.
    """
    b = problem["b"]                       # [n]
    offsets = problem["offsets"]           # [m+1] int64
    m = problem["m"]
    d = problem["d"]

    # Per-group quantities at the current (x, t).
    r = problem["A"] @ x - b               # [n]
    g_i = np.empty((m, d), dtype=np.float64)  # [m, d]  grad of l_i
    l_i = np.empty(m, dtype=np.float64)     # [m]     l_i = ||r_i||^2
    for i in range(m):
        s, e = int(offsets[i]), int(offsets[i + 1])
        ri = r[s:e]                         # [n_i]
        Ai = cache[i][0]                   # [n_i, d]
        l_i[i] = float(ri @ ri)            # scalar
        g_i[i] = 2.0 * (Ai.T @ ri)         # [d]
    gap = t - l_i                          # [m]  gap_i = t - l_i > 0
    # Guard: if any gap_i <= 0 we are infeasible; clip to a tiny positive floor
    # so the Newton system is still solvable (should not happen given the
    # backtracking line search, but defends against round-off).
    gap = np.maximum(gap, 1e-12)           # [m]

    inv_gap = 1.0 / gap                    # [m]
    inv_gap2 = inv_gap ** 2                # [m]

    # Gradient [d+1].  (Form B: Phi = t - mu * sum ln(t - l_i).)
    grad_x = mu * (g_i * inv_gap[:, None]).sum(axis=0)   # [d]
    grad_t = 1.0 - mu * float(inv_gap.sum())             # scalar
    grad = np.empty(d + 1, dtype=np.float64)             # [d+1]
    grad[:d] = grad_x
    grad[d] = grad_t

    # Hessian [d+1, d+1].
    H = np.zeros((d + 1, d + 1), dtype=np.float64)
    # H_xx = mu * sum_i ( H_i/gap_i + g_i g_i^T / gap_i^2 )
    for i in range(m):
        AtA_i = cache[i][1]                # [d, d]
        H[:d, :d] += mu * 2.0 * AtA_i * inv_gap[i]              # mu * H_i / gap_i
        H[:d, :d] += mu * inv_gap2[i] * np.outer(g_i[i], g_i[i])  # mu * g_i g_i^T / gap_i^2
    # H_xt = -mu * sum_i g_i / gap_i^2
    H[:d, d] = -mu * (g_i * inv_gap2[:, None]).sum(axis=0)   # [d]
    H[d, :d] = H[:d, d]
    # H_tt = mu * sum_i 1 / gap_i^2
    H[d, d] = mu * float(inv_gap2.sum())                     # scalar
    # Tiny regularization for definiteness (rank-deficiency; body.tex:27).
    H += 1e-12 * np.eye(d + 1, dtype=np.float64)

    dz = -solve_pd(H, grad)                 # [d+1]  Newton direction
    decrement = float(np.sqrt(max(0.0, grad @ (-dz))))  # sqrt(grad^T H^{-1} grad)
    if decrement <= inner_tol:
        return x.copy(), t, decrement, False  # centering essentially done

    # Damped Newton backtracking to keep gap_i = t - l_i > 0.
    dx = dz[:d]                             # [d]
    dt = dz[d]                              # scalar
    alpha = 1.0
    for _ in range(60):
        x_new = x + alpha * dx             # [d]
        t_new = t + alpha * dt             # scalar
        if t_new - float(group_losses(problem, x_new).max()) > 0.0:
            break
        alpha *= 0.5
    else:
        # Could not find a feasible step; keep the point, signal accepted=False.
        return x.copy(), t, decrement, False
    return x_new, float(t_new), decrement, True


def _run_single(problem, x0, max_outer, deadline, mu0, theta, inner_tol, opt):
    """One (mu0, theta) run from x0.  Returns (history, final F).

    Each OUTER iteration = one barrier-parameter reduction mu <- theta*mu,
    preceded by damped-Newton CENTERING on Phi_mu (inner Newton steps until the
    Newton decrement <= inner_tol or the inner cap).  This is the standard
    decreasing-mu central path (BV04 §11.2): one outer iteration is one mu
    reduction, and the centering inside it is what lets a handful of outer
    iterations reach high accuracy (the paper reports IPM = 8 outer iterations
    to 1% on ACS, experiments.tex:107,180).  The reported gap is
    ``max_loss(problem, x) - opt`` after each outer iteration.
    """
    cache = _group_cache(problem)          # list of (Ai, AtA_i)
    h = seed_history(problem, x0, opt)     # iter 0 recorded
    x = h["x"].copy()                      # [d]
    # Start strictly feasible with a margin ~ mu0 so the first center sits near
    # the central-path point for mu0 (margin ~ mu balances the objective t and
    # the barrier; SPEC E21 guard).  mu0 is already scaled by L0 in run().
    t = float(max_loss(problem, x)) + max(float(mu0), 1.0)  # strictly feasible t0
    mu = float(mu0)                         # barrier parameter
    t0 = time.perf_counter()
    INNER_CAP = 50                         # centering Newton cap per outer iter
    for k in range(1, max_outer + 1):
        if h["gap"][-1] <= 0.0:
            break
        if time.perf_counter() > deadline:
            break
        # center on Phi_mu: inner damped-Newton until decrement <= inner_tol
        for _ in range(INNER_CAP):
            x, t, dec, acc = _newton_step(problem, cache, x, t, mu, inner_tol)
            if (not acc) or dec <= inner_tol:
                break
            if time.perf_counter() > deadline:
                break
        mu *= theta                        # reduce barrier parameter (one outer iter)
        h["iter"].append(k)
        h["gap"].append(gap_now(problem, x, opt))
        t_elapsed = time.perf_counter() - t0
        h["time"].append(t_elapsed)
        if h["gap"][-1] <= 0.0:
            break
        if t_elapsed > deadline - t0:
            break
    h["x"] = x
    return h, float(max_loss(problem, x))


def run(problem, cfg, x0, max_outer, time_budget):
    """Best-of-grid log-barrier IPM.  See module docstring."""
    cfg = cfg or {}
    opt = get_opt(problem, cfg)
    x0 = np.asarray(x0, dtype=np.float64)          # [d]
    mu0_grid = as_list(cfg.get("mu0_grid"), DEFAULTS["mu0_grid"])
    theta_grid = as_list(cfg.get("theta_grid"), DEFAULTS["theta_grid"])
    inner_tol = cfg.get("inner_tol", DEFAULTS["inner_tol"])
    if isinstance(inner_tol, (list, tuple, np.ndarray)):
        inner_tol_grid = [float(v) for v in inner_tol]
    else:
        inner_tol_grid = [float(inner_tol)]

    # Scale the barrier parameter by the initial loss magnitude L0 = F(x0).
    # The log barrier Phi = t - mu*sum ln(t-l_i) only has a meaningful center
    # when mu is comparable to the objective scale t (~1e7 on the raw data);
    # with mu=1 (unscaled) the barrier is negligible, centering pins t at the
    # feasibility boundary and x never rebalances.  Scaling mu by L0 is exactly
    # equivalent to the paper's WLOG OPT=1 rescaling (U15): the argmin of
    # min_x max_i ||A_i x - b_i||^2 is invariant under (A,b) -> (A,b)/sqrt(L0),
    # and on the rescaled problem mu'=mu/L0 ~ O(1).  This yields the paper's
    # ~8-outer-iteration convergence to 1% (experiments.tex:180).
    L0 = float(max_loss(problem, x0))
    if not np.isfinite(L0) or L0 <= 0.0:
        L0 = 1.0

    t_start = time.perf_counter()
    deadline = t_start + float(time_budget)
    best = None
    best_final = np.inf
    for mu0 in mu0_grid:
        for theta in theta_grid:
            for inner_tol in inner_tol_grid:
                if time.perf_counter() > deadline:
                    break
                h, final = _run_single(problem, x0, max_outer, deadline,
                                       float(mu0) * L0, float(theta), inner_tol, opt)
                if final < best_final:
                    best_final = final
                    best = h
            if time.perf_counter() > deadline:
                break
        if time.perf_counter() > deadline:
            break
    if best is None:
        best = seed_history(problem, x0, opt)
    return best
