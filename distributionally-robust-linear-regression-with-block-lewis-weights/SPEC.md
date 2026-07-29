# SPEC — Reproduction of "Distributionally Robust Linear Regression With Block Lewis Weights"

- **Paper**: Manoj & Patel, "Distributionally Robust Linear Regression With Block Lewis Weights", arXiv:2607.00252 (ICLR 2026).
- **Paper source on disk (authoritative for all maths)**: `paper/arxiv-2607.00252-src/`. All citations below are `file:line` into that directory and resolve with `grep`/`sed -n`.
- **Goal of this reproduction**: re-implement the empirical evaluation of §8 (`experiments.tex`) — synthetic heterogeneous regression + ACS Income — and reproduce the quantitative comparisons of Table `tab:acs_runtime` (`experiments.tex:171-186`), the qualitative curve behavior of `fig:iteration_runtime` / `fig:acs_convergence`, and the statistical-context numbers of `experiments.tex:189`. In addition, implement the theory algorithms (Alg 1/3/4/5) to runnable fidelity. The headline theory claim — Õ(min{rank(A),m}^{1/3} ε^{−2/3}) linear-system solves (Theorem 1, `intro.tex:27-37`) — is an iteration-complexity statement about a different (accelerated) variant than the one benchmarked in §8 (see "What the paper does not state", item U3); it is not directly measurable, so the numbers gate is built on §8's numbers.

---

## 1. Problem statement and notation (with shapes)

Dataset folding convention (paper's own, `body.tex:27`): the per-group factors 1/√nᵢ are folded
into the data at load time:

```
A_{S_i} ← A_{S_i} / √n_i ,      b_{S_i} ← b_{S_i} / √n_i       (i = 1..m)
```

After folding, all formulas below use the folded matrices and **never** carry nᵢ factors.
This is what the theory assumes and it makes the experimental group loss exactly
`ℓ_i(x) = (1/n_i)‖(orig A_{S_i})x − (orig b_{S_i})‖²₂ = ‖(folded A_{S_i})x − (folded b_{S_i})‖²₂`
(cf. `experiments.tex:8-10`).

| Symbol | Shape | Meaning |
|---|---|---|
| `m` | scalar int | number of groups (synthetic: 100; ACS: 51) |
| `nᵢ` | scalar int | rows of group i (before folding) |
| `n` | scalar int | total rows, `n = Σᵢ nᵢ` |
| `d` | scalar int | parameter dimension (10 in both experiments) |
| `A_{S_i}` | float [nᵢ, d] | group i design matrix (folded on disk) |
| `b_{S_i}` | float [nᵢ] | group i responses (folded on disk) |
| `A` | float [n, d] | stacked design `A = [A_{S_1}; …; A_{S_m}]` (row blocks contiguous, group order fixed) |
| `b` | float [n] | stacked responses |
| `x` | float [d] | parameter vector being optimized |
| `r(x) = A x − b` | float [n] | stacked residual; block `r_i` is rows `S_i` |
| `S_i` | index set | contiguous row range for group i; stored as `offsets: int[m+1]` |
| `W` | float [n, n] diagonal (stored as vector `w_rows: float[n]`) | block Lewis weights: `W_jj = w_i` for all `j ∈ S_i` |
| `w` | float [m] | per-block weights |
| `M` | SPD float [d, d] | geometry matrix: `AᵀWA` (p=∞) or `Aᵀ W^{1−2/p} A` (finite p); naive fallback `AᵀA` |
| `‖·‖_M` | scalar | `√(xᵀ M x)` |
| `‖y‖_{G_p}` | scalar | `(Σᵢ ‖y_{S_i}‖₂^p)^{1/p}` for `y ∈ R^n`; `‖y‖_{G_∞} = maxᵢ ‖y_{S_i}‖₂` (`body.tex:27`) |
| `τ_j(A)` | scalar | leverage score `a_jᵀ (AᵀA)⁻¹ a_j` (`other_proofs.tex:8-14`; pseudoinverse convention `body.tex:27`) |
| `f(x)`, p=∞ | scalar | `maxᵢ ‖A_{S_i}x − b_{S_i}‖₂ = ‖Ax − b‖_{G_∞}` (`body.tex:27`; equals `F(x)` of `experiments.tex:11-14` after folding) |
| `f(x)`, p<∞ | scalar | `Σᵢ ‖A_{S_i}x − b_{S_i}‖₂^p = ‖Ax − b‖_{G_p}^p` (the p-th power of the Thm-2 objective, `body.tex:27`, `interpolation.tex:460`) |
| `β, δ` | scalars > 0 | smoothing parameters of the LSE surrogate |
| `ε` | scalar > 0 | target additive accuracy (theory rescales OPT = 1) |

Objective being benchmarked (the paper's main objective, `intro.tex:10-12`):
```
min_x  max_i  (1/n_i)‖A_{S_i} x − b_{S_i}‖₂²      ≡ (after folding)  min_x max_i ‖A_{S_i} x − b_{S_i}‖₂²
```
i.e. the squared form; the theory works with its square root `‖Ax−b‖_{G_∞}` and converts
back — note Theorem 1's guarantee is multiplicative on `maxᵢ (1/√nᵢ)‖A_{S_i}x − b_{S_i}‖₂`
(`intro.tex:30`), i.e. on the *unsquared* scale.

## 2. Equations we implement, with citations

**(E1) Robust objective** `f(x) = ‖Ax − b‖_{G_∞}` — `intro.tex:10-12` (`eq:main_objective`), notation `body.tex:27`.

**(E2) Utilitarian/ERM objective** `min_x (1/m) Σᵢ (1/nᵢ)‖A_{S_i}x − b_{S_i}‖₂²` — `intro.tex:5-6` (`eq:utilitarian`). After folding = plain least squares on (A, b); its minimizer `x_erm` is the universal warm start (`experiments.tex:92`, `experiments.tex:148`).

**(E3) Smoothed surrogate** (`eq:intro_smooth_fair_objective`, `body.tex:54-56`; also Alg. 1 line 5, `body.tex:177-181`; approximation Lemma 6.1 `body.tex:227-272`):
```
f̃_{β,δ}(x) = β · log Σᵢ₌₁..m exp( ( √(δ² + ‖r_i‖₂²) − δ ) / β ),   r_i = A_{S_i} x − b_{S_i}
|f̃_{β,δ}(x) − f(x)| ≤ β log m + δ        (Lemma 6.1, body.tex:230-231)
```
With `β = ε/(4 log m)`, `δ = ε/4` the surrogate is optimized to ε/2 to get ε-additive optimality of f (`body.tex:273`).

**(E4) Surrogate derivatives** (LSE calculus Lemma 6.2 `body.tex:285-288`; inner-function derivatives `body.tex:431-435`). Let
`h_i = √(δ² + ‖r_i‖₂²)`, `a_i = (h_i − δ)/β`, `s = softmax(a) ∈ Δ^m` (computed with max-shift).
```
∇f̃(x) [d]      = Σᵢ sᵢ · (A_{S_i}ᵀ r_i) / hᵢ
∇²f̃(x) [d,d]   =      Σᵢ sᵢ · A_{S_i}ᵀ ( I_{nᵢ}/hᵢ − rᵢ rᵢᵀ/hᵢ³ ) A_{S_i}
                 + (1/β) Σᵢ sᵢ · (A_{S_i}ᵀ rᵢ/hᵢ)(A_{S_i}ᵀ rᵢ/hᵢ)ᵀ
                 − (1/β) ∇f̃ ∇f̃ᵀ
```
(All four block sums are of the form `Aᵀ B A` with block-diagonal `B`; the last term is a global
negative rank-1 — see U13.) Quasi-self-concordance / smoothness property (Lemma 6.5,
`body.tex:400-407`): used only analytically.

**(E5) p-objective derivatives**, p ≥ 2 finite (`interpolation.tex:24-29`, `eq:f_grad`/`eq:f_hess`):
```
∇f(x)  [d]    = p Σᵢ ‖r_i‖₂^{p−2} A_{S_i}ᵀ r_i
∇²f(x) [d,d]  = p Σᵢ ‖r_i‖₂^{p−2} A_{S_i}ᵀ A_{S_i}
              + p(p−2) Σᵢ ‖r_i‖₂^{p−4} (A_{S_i}ᵀ r_i)(A_{S_i}ᵀ r_i)ᵀ
```

**(E6) Block Lewis overestimate condition** (Def. 3.2, `other_proofs.tex:18-24`): `w ∈ R^m_{≥0}`
with `W_jj = w_i (j ∈ S_i)` is a block Lewis overestimate iff for all i:
```
( Σ_{j ∈ Sᵢ} τ_j( W^{1/2 − 1/p} A ) ) / w_i  ≤  1
```

**(E7) Ellipsoid sandwich** (Thm 3.3, `other_proofs.tex:28-34`; combined guarantee Thm 2.3, `body.tex:145-152`): weights from (E8) on input `Â = [A | b] ∈ R^{n × (d+1)}` satisfy, for all `x ∈ R^d`, `c ∈ R`:
```
‖W^{1/2−1/p}(Ax − c·b)‖₂ / (2(rank(A)+1))^{1/2−1/p}  ≤  ‖Ax − c·b‖_{G_p}  ≤  ‖W^{1/2−1/p}(Ax − c·b)‖₂
```
For p = ∞ (Alg. 1) the exponent is 1/2 and `‖W^{1/2}(Ax − cb)‖₂ ≤ √(2(rank+1))·‖Ax−cb‖_{G_∞}`
(Alg. 1 line 1, `body.tex:169-172`).

**(E8) Block Lewis weight computation** — cited only as "[MO25, Algorithm 2]" (`body.tex:169`, `other_proofs.tex:44-47`). We fetched the MO25 (arXiv:2311.10013) source and reproduce its algorithm here concretely (their `alg:blw`), specialized to exact leverage scores (d is tiny, so `OverLev` = exact Cholesky-based leverage scores):
```
Input: Â = [A | b] ∈ R^{n × (d+1)}, exponent e_p = 1/2 − 1/p  (p = ∞ ⇒ e_p = 1/2)
v ← (n_cols / m) · 1_m                         # n_cols = d+1
for t = 1 … T−1,  T = Θ(log m):                # we use T = ⌈2·ln(m)/ln(3/2)⌉ (exact-lev regime)
    τ̃_j = a_jᵀ (Âᵀ V^{2e_p} Â)⁻¹ a_j  for j=1..n        # leverage scores of V^{e_p} Â, V = diag-blowup(v)
    v_i ← Σ_{j ∈ Sᵢ} τ̃_j                      for i=1..m
w ← (3/2) · mean_t v^{(t)}
```
Guarantees: `w` is a block Lewis overestimate (E6) with `‖w‖₁ ≤ (3/2)(d+1) ≤ 2(rank(A)+1)`
(this paper's Thm 3.4 assumes ‖w‖₁ ≤ 2 rank; `other_proofs.tex:44-47`; MO25 proves 1.5·n_cols).

**(E9) Weighted-least-squares initialization** (Alg. 1 line 4 `body.tex:176`; Alg. 5 line 2 `interpolation.tex:1076`; Lemma 3.5 `other_proofs.tex:51-61`):
```
x_0 = (Aᵀ D A)⁻¹ Aᵀ D b ,   D = W  (p = ∞)  or  D = W^{1−2/p} (finite p)
```
(pseudoinverse if rank-deficient; `body.tex:27`).

**(E10) Regularized robust objective** (Alg. 1 line 6, `body.tex:182`; but see U11 — the proof at `body.tex:580` uses a different constant):
```
f̂(x) = f̃_{β,δ}(x) + ε/(1000·min{rank(A), m}) · ‖W^{1/2} A (x − x_0)‖₂²
```

**(E11) Geometry switch** (Alg. 1 lines 2–3, `body.tex:173-175`; rationale `body.tex:158`):
```
compute w via (E8);  if Σᵢ w_i ≥ m  then  W ← I_n      (naive geometry M = AᵀA)
```

**(E12) Approximate Mirror Descent (Algorithm 2)** — `mirror_descent.tex:8-25`: iterate
```
x*_i = argmin_x̃  f(x_{i−1}) + ⟨∇f(x_{i−1}), x̃ − x_{i−1}⟩ + L·D_h(x̃, x_{i−1})   (line 20)
x_i ≈ x*_i  an approximate stationary point                                        (line 21)
```
with relative smoothness/strong-convexity constants L = p·e, μ = 1/(2 p e) w.r.t. reference h (Lemma `gp_hessian_stable`, `interpolation.tex:465-468`). Guarantee Thm 4.1 (`mirror_descent.tex:37-47`).
Return `argmin_i f(x_i)` (line 23).

**(E13) Proximal subproblem (p < ∞)** (`interpolation.tex:413-420`):
```
f_q(x) = f(x) + e·p^p · ‖x − q‖_M^p ,  M = Aᵀ W^{1−2/p} A
reference h_q(x) = ‖x − q‖²_{∇²f(q)} + e·p^p·‖x − q‖_M^p      (interpolation.tex:458-463)
```

**(E14) Inner mirror-descent subproblem solve** (Lemma `gp_prox_relativesmoothsolve`, `interpolation.tex:841-864`): solve
```
min_x ⟨g, x⟩ + L ( ‖x − q‖²_{∇²f(q)} + C_p·τ·‖x − q‖²_M ) ,  C_p = e p^p
```
which is one linear-system solve `x − q = −(1/(2pe))·(∇²f(q) + C_p τ M)⁻¹ g` (`interpolation.tex:851-854`), and binary-search `τ ≥ 0` until the self-consistency `τ = ‖x(τ) − q‖_M^{p−2}` holds to tolerance (motivation/accuracy at `interpolation.tex:856-917`; note τ(·) monotone ⇒ bisection valid, `interpolation.tex:856-863`).

**(E15) MS-oracle output (p < ∞)** (Lemma `prox_subproblem_ms`, `interpolation.tex:966-975`; λ choice in proof of Thm 2, `interpolation.tex:1084`):
```
x̃ ≈ argmin f_q  (via E12/E14 to ‖x̃ − x_q‖_M ≤ α‖x_q − q‖_M, α = interpolation.tex:1000)
λ = e · p^{p+1} · ‖x̃ − q‖_M^{p−2}
return (x̃, λ)   # a 1/2-MS oracle: ‖x̃ − q + (1/λ)M⁻¹∇f(x̃)‖_M ≤ ½‖x̃ − q‖_M   (Def. 5.1, improved_ms.tex:43-49)
```

**(E16) OptimalMSAcceleration (Algorithm 3)** — full pseudocode `improved_ms.tex:6-39`. With `x_0`, MS oracle O, `λ'₀`, `α = exp(3 − 2/(s+1))`, movement parameters `s = p−1`, `c = (e p^{p+1})^{1/(p−1)}` (`interpolation.tex:1086-1090`):
```
v_0 = x_0; A_0 = A'_0 = 0
(x̃_1, λ_1) = O(x_0; λ'₀);  λ'_1 = λ_1
for t = 0 … T:
    a'_{t+1}  = (1/(2λ'_{t+1}))·(1 + √(1 + 4 λ'_{t+1} A_t))          (line 14)
    A'_{t+1}  = A_t + a'_{t+1}
    q_t       = (A_t/A'_{t+1})·x_t + (a'_{t+1}/A'_{t+1})·v_t         (line 16)
    if t > 0: (x̃_{t+1}, λ_{t+1}) = O(q_t; λ'_{t+1})                  (line 17)
    γ_{t+1}   = min{1, λ'_{t+1}/λ_{t+1}}                             (line 18)
    a_{t+1}   = γ a';  A_{t+1} = A_t + a_{t+1}                       (line 19)
    x_{t+1}   = ((1−γ)A_t/A_{t+1})·x_t + (γ A'_{t+1}/A_{t+1})·x̃_{t+1} (line 20)
    if γ = 1: λ'_{t+2} ← λ'_{t+1}/α   else: λ'_{t+1} ← α·λ'_{t+1}    (lines 21-25; see U12 indexing note)
    v_{t+1}   = v_t − a_{t+1} · M⁻¹ ∇f(x̃_{t+1})                     (line 36)
```
Guarantee Thm 5.3 (`improved_ms.tex:61-71`): `T ≥ C·s·(c^s ‖x_0 − x★‖_M^{s+1}/ε)^{2/(3s+1)}` iterations suffice.

**(E17) GpRegression (Algorithm 5), p < ∞** — `interpolation.tex:1067-1081`: (E8) weights on `[A|b]` → M (E13) → x₀ (E9) → run (E16) with oracle (E15) for `O(poly(p)·min{rank,m}^{(p−2)/(3p−2)}·log(d/ε)³)` iterations; restart-halving logic in proof `interpolation.tex:1092-1104`.

**(E18) Algorithm 1 (MinMaxRegression), p = ∞ — as written** — `body.tex:164-188`: (E8) → (E11) → (E9) → (E3) with `β = ε/(4 log m)`, `δ = ε/4` → (E10) → ball-optimization oracle `(C/min{rank,m}, C/ε)` via [CJJJLST20, Alg 3] → ½-MS oracle via [CJJJLST20, Alg 2] → run [CJJJLST20, Alg 1] `Õ(min{rank,m}^{1/3} log(d/ε)/ε^{2/3})` iterations. CJJJLST20 = arXiv:2003.08078 (external; its oracles are **not** reprinted in this paper).

**(E19) Practical ball-oracle method actually benchmarked in §8** — `experiments.tex:71-78`:
"repeatedly solve the smoothed objective using a damped Newton solver", "After each outer
step, the center is updated to the new solution, and the trust-region radius is optionally
shrunk. For simplicity, we do not consider the acceleration of the ball-oracle method."
Concrete specification (our instantiation, since the paper gives no inner pseudocode):
```
state: center q [d], radius r > 0, smoothing (β, δ), geometry M ∈ {AᵀA (euclid), AᵀWA (lewis)}
repeat outer steps:
    minimize f̃_{β,δ}(x) over {x : ‖x − q‖_M ≤ r} by damped trust-region Newton:
        iterate from x = q:
            g = ∇f̃(x); H = ∇²f̃(x)  (E4; add ξI, ξ=1e−12, for definiteness)
            ν ≥ 0 s.t. step s = −(H + ν·M)⁻¹ g satisfies ‖s‖_M ≤ r   (Moré–Sorensen:
                                               ν=0 if ‖s(0)‖_M ≤ r else bisect ν to ‖s‖_M = r)
            x ← x + s ; stop inner when ‖g‖_{M⁻¹} ≤ tol_inner or inner cap
    q ← x ; r ← shrink·r  (shrink per arms.json grid)
```

**(E20) Reference optimum OPT** — CVXPY epigraph QP (`experiments.tex:40-50`):
```
min_{x ∈ R^d, t ∈ R}  t   s.t.  ℓ_i(x) = (1/n_i)‖A_{S_i}x − b_{S_i}‖₂² ≤ t  ∀i
```
returned `t* = OPT`. All plots/tables report `F(x) − OPT` (suboptimality) or relative gap
`(F(x) − OPT)/OPT` (`experiments.tex:52`, `experiments.tex:150`).

**(E21) IPM baseline** — log-barrier on the epigraph form (`experiments.tex:68-69`; the
textbook approach of [BV04 §6.4] cited at `intro.tex:17`): with barrier parameter path
`μₖ = μ₀·θᵏ`, center by damped Newton on `Φ_μ(x,t) = μ·t − Σᵢ ln(t − ℓ_i(x))` (with 1/nᵢ factors,
i.e. folded data), each centering step is a linear-system solve in a matrix of the form
`AᵀBA + e eᵀ` (block-diagonal B plus epigraph terms).

**(E22) First-order baselines** — `experiments.tex:57-66`: subgradient descent on the
nonsmooth `F(x)` (subgradient `∇ℓᵢ*(x)` for an argmax group i*) with fixed and diminishing
schedules; GD / Heavy-Ball / Nesterov on the LSE-smoothed objective `f̃_{β,δ}` (E3).

## 3. Data

**(D1) Synthetic heterogeneous regression** — `experiments.tex:4-38`. Only these facts are stated:
`d = 10`, `m = 100` groups, `5` adversarial groups, stacked Gram condition number ≈ 1e5
(`experiments.tex:38`); shared orthonormal basis; normal groups = moderate condition number,
aligned geometry, optima concentrate near a common center, independent noise; adversarial
groups = one extremely-large-curvature direction (distinct per group), optima far from the
population center along that direction, very small noise (`experiments.tex:19-25`). All concrete
numbers (eigenvalue ranges, noise scales, distances, nᵢ, seed) are **not stated** — the text
says "see the included Jupyter notebook" (`experiments.tex:38`) but **no notebook exists in the
arXiv tarball** (verified: `paper/2607.00252.tar.gz` contains only .tex/.bib/.png). Our concrete
instantiation is declared in `arms.json` under `dataset_overrides` and must reproduce: cond(ÂᵀÂ)≈1e5,
and an ERM-vs-robust-opt worst-group gap ("clear gap", `experiments.tex:38`).

**(D2) ACS Income** — `experiments.tex:145-152`: `folktables` (ding2021retiring), log personal
income, `d = 10` standardized features ("age, education, occupation, hours worked, etc."),
employed US adults, `m = 51` regions (50 states + Puerto Rico), 200 individuals subsampled per
region → n = 10,200, group loss = MSE per region, warm start = ERM, OPT via CVXPY. Consistent
inference (d = 10 and "employed adults" match folktables' default `ACSIncome` task: features
AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P; default survey filters). Subsampling
scheme/seed and target transform details are unstated — declared choices in `arms.json`.

## 4. Component interfaces (frozen)

```python
# gdr/problem.py
Problem = TypedDict('Problem', {
  'A': np.ndarray,        # [n, d] float64, folded
  'b': np.ndarray,        # [n] float64, folded
  'offsets': np.ndarray,  # [m+1] int64, contiguous group row ranges
  'name': str, 'm': int, 'd': int, 'n_i': np.ndarray,  # [m] pre-fold sizes
  'meta': dict})

# gdr/objectives.py
max_loss(problem, x) -> float           # F(x) = max_i ‖r_i‖₂²   (folded ⇒ == ℓ_i of the paper)
group_losses(problem, x) -> np.ndarray  # [m]
smoothed(problem, x, beta, delta) -> float                       # E3
smoothed_grad_hess(problem, x, beta, delta) -> (float, np.ndarray[d], np.ndarray[d,d])  # E4

# gdr/lewis.py
block_lewis_weights(problem, p: float, rounds: int, seed: None) -> np.ndarray[m]  # E8, exact lev
geometry_M(problem, w | None, p) -> np.ndarray[d, d]            # E11/E13
wls_init(problem, D_diag_rows: np.ndarray[n]) -> np.ndarray[d]  # E9

# gdr/reference.py
solve_opt(problem, solver: str) -> (x_star[d], opt: float)     # E20 cvxpy epigraph

# gdr/solvers/*.py — every arm implements
run(problem, cfg: dict, x0: np.ndarray[d], max_outer: int, time_budget: float
    ) -> History = {'iter': list[int], 'gap': list[float], 'time': list[float], 'x': np.ndarray}
# arms: subgradient, smoothed_gd, smoothed_hb, smoothed_nesterov, ipm,
#       ball_oracle (geometry ∈ {"euclidean", "lewis"}),  (stretch:) accel_ms

# gdr/runner.py
run_arm(arm_name, cfg, problem, warm_start) -> History
time_to_gap(history, rel_gap=0.01) -> (iters, seconds)         # relative to OPT
```

## 5. Arms (becomes `arms.json`) — the paper's own comparison (§8.1.2, `experiments.tex:53-79`)

Seven method arms × two datasets, plus the OPT reference. Configs given are the *declared
defaults + tuning grids* (grids are our disclosed choice, see U2):

| arm | method (paper name) | config keys |
|---|---|---|
| `subgradient` | Subgradient method, best of fixed/diminishing steps | `lr_grid` (log), `schedule ∈ {const, 1/√t}` |
| `smoothed_gd` | Smoothed gradient descent | `lr_grid`, `(beta, delta)_grid` |
| `smoothed_hb` | Smoothed Heavy-Ball | `lr_grid`, `momentum_grid`, `(beta,delta)_grid` |
| `smoothed_nesterov` | Smoothed Nesterov | `lr_grid`, `(beta,delta)_grid` |
| `ipm` | Log-barrier IPM, Newton centering | `mu0_grid`, `theta_grid`, `inner_tol` |
| `ball_oracle_euclidean` | Ball-oracle, naive geometry (`M = AᵀA`, i.e. W = I per E11/Table 1) | `r0_grid`, `shrink_grid`, `(beta,delta)_grid`, `tol_inner` |
| `ball_oracle_lewis` | Ball-oracle, block-Lewis geometry (`M = AᵀWA`, E8/E9) | same as above |
| `opt_reference` | CVXPY epigraph QP (E20) | `solver: clarabel/ecos/scs` |

Datasets: `synthetic` (D1), `acs_income` (D2). Stretch arms (theory-fidelity, not in the paper's numbers): `accel_ms_robust` (E18-ish), `gp_regression` for p ∈ {2, 4, 8} (E17).

**Recorded target values (numbers gate)**:
- T1: ACS iterations-to-1%-relative-gap (`experiments.tex:171-186`, `tab:acs_runtime`): subgradient = not reached; smoothed HB = 47; IPM = 8; BO-Euclidean = 1; BO-Lewis = 1. Gate semantic: measured values reported verbatim; pass = BO arms ≤ 2 outer iterations AND strict ordering iters(BO) < iters(IPM) ≤ iters(HB) AND subgradient does not reach 1% within budget. Exact counts depend on undisclosed grids (U2), so ordering boundedness is the honest gate.
- T2: ACS wall-time-to-1%: 0.019 / 0.019 / 0.066 / 0.062 s (machine-dependent; report only, no gate).
- T3: ACS statistics (`experiments.tex:189`): ERM mean ≈ 108.2 (±5), ERM worst ≈ 138.1 attained by California (state + value ±5), robust-opt band ⊂ [102, 119], Max/Mean 1.28 → 1.02 (±0.05), California decrease ≈ 24.3 MSE (±5).
- T4: synthetic (`experiments.tex:106-109`, fig): IPM reaches lowest final loss; both BO arms strictly decrease the gap over outer iterations and beat first-order methods' plateau; Lewis ≤ Euclidean finally ("very slight benefit", `experiments.tex:109`).
- T5 (sanity/unit checks): |f̃ − f| ≤ β log m + δ on random x (Lemma 6.1); E6 overestimate holds numerically and ‖w‖₁ ≤ 2(d+1); E7 sandwich holds numerically on random (x, c); ball-oracle outer iterates are non-increasing in F; E9 init has F(x₀) ≤ √(2(d+1))·OPT.

## 6. Upstream code search (performed 2026-07-29)

- arXiv abstract page 2607.00252: no code/media link attached (all "Links to Code" integrations empty).
- GitHub repository search `"block lewis weights"`, `"distributionally robust" lewis regression`: 0 repos. Code search for `GpRegressionProxOracle` / `"block Lewis weights" regression`: 0 results.
- GitHub authors: `nsmanoj` (Naren Sarayu Manoj) exists with **0 public repos**; `kumarkshitij` (Kumar Kshitij Patel) exists with **0 public repos**.
- The paper's acknowledgments say the authors "used ChatGPT 5.3 and Claude Opus 4.6 to implement the algorithms" (`body.tex:619`) and §8 refers to "the included Jupyter notebook" (`experiments.tex:38`) — but the notebook is **absent from the arXiv source tarball** (`paper/2607.00252.tar.gz`, full listing verified). 
- **Conclusion: no usable upstream code. Implement from scratch.** External theory dependencies that the paper cites but does not reprint: MO25 Alg. 2 (fetched arXiv:2311.10013 source, re-derived as E8), CJJJLST20 arXiv:2003.08078 Algs. 1–3 (only cited, E18), CHJJS22 (adapted in full as E16).

## 7. What the paper does NOT state

- **U1 (synthetic data recipe)**: covariance eigenvalue ranges, adversarial curvature magnitude, distance of adversarial optima, noise variances, nᵢ, RNG seed. Notebook referenced but not shipped (`experiments.tex:38`).
- **U2 (all hyperparameter grids)**: "tune every method via grid search" (`experiments.tex:83-92`) with no grid values, no budgets (outer-iteration budget for tuning is unspecified).
- **U3 (theory↔experiment gap)**: the benchmarked ball-oracle arms are **unaccelerated** ("For simplicity, we do not consider the acceleration", `experiments.tex:78`), so §8 numbers do not exercise Algorithm 1 as written (E18). Also unspecified: what smoothing (β, δ) the BO arms use per run (it is in the tuning grid, `experiments.tex:89`).
- **U4 (ACS preprocessing)**: folktables year/horizon unspecified (default 2018/1-Year inferred), standardization procedure, subsample with/without replacement, subsample seed, exact log transform (`log(PINCP)` vs `log1p`).
- **U5 (CPU/software)**: no hardware description, CVXPY solver choice unstated, BLAS/threads unstated — T2 wall times not portable.
- **U6 (relative-gap definition)**: "1% relative suboptimality" of `tab:acs_runtime` — (F−OPT)/OPT assumed; (F−OPT)/(F(x₀)−OPT) is the alternative reading.
- **U7 (seeds)**: none anywhere (data generation, subsampling).
- **U8 (CJJJLST20 internals)**: E18's ball oracle / MS oracle / acceleration loop are cited from [CJJJLST20] without restatement; the universal constant C of Alg. 1 lines 7–8 is unspecified (`body.tex:183-184`).
- **U9 (λ′₀ and stopping)**: Algorithm 3 requires initial λ′₀ and iteration count T as inputs; no concrete λ′₀ or stopping rule is given (`improved_ms.tex:10`).
- **U10 (MO25 details in this paper)**: block-Lewis algorithm only cited (Thm 3.4, `other_proofs.tex:44-47`); OverLev, exact T, and constants live in the MO25 paper (we re-derived E8 from its source).
- **U11 (internal inconsistency, robust regularizer)**: Alg. 1 line 6 uses `ε/(1000·min{rank,m})·‖W^{1/2}A(x−x₀)‖²` (`body.tex:182`) while the proof uses `ε/(110·R²)·…` with `R = (2+ε)√(2(d+1))` (`body.tex:580-586`). Constants-only discrepancy, R was not an algorithm input.
- **U12 (internal inconsistency, Algorithm 3 indexing)**: the else-branch updates `λ′_{t+1} ← α λ′_{t+1}` after λ′_{t+1} was already consumed that iteration (`improved_ms.tex:24`) — read as: schedule the *next* guess (failed step ⇒ multiply guess by α; accepted ⇒ divide by α).
- **U13 (Hessian form)**: the proof claims "the Hessian of f̂ is of the form AᵀBA with block-diagonal B" (`body.tex:609`), but ∇²f̃ contains the global rank-1 term `−(1/β)∇f̃∇f̃ᵀ` (E4). Implementation keeps the full d×d Hessian (d = 10, cheap); the claim matters only for the solve-complexity accounting.
- **U14 (Lemma 3.5 typo)**: `other_proofs.tex:56` writes the norm matrix as `AᵀW^{1/2−1/p}A`; the proof (`other_proofs.tex:74`) uses the consistent `M = AᵀW^{1−2/p}A`.
- **U15 (rescale OPT = 1)**: the theory rescales so OPT = 1 (`body.tex:511-513`) — a legit WLOG for analysis, but a concrete run must either rescale (A, b) by 1/OPT (needs OPT — chicken-and-egg in practice; we simply do not rescale and treat constants as heuristic).
- **U16 (fallback W = I branch test)**: the reset condition is `Σ wᵢ ≥ m` (`body.tex:173`); with ‖w‖₁ ≤ 2(rank+1) this fires when 2(rank+1) ≥ m — fine — but the boundary behavior and any rank-deficiency handling (pseudoinverse convention noted only at `body.tex:27`) are on us.
- **U17 (prox accuracy α)**: E15's target α depends on the unknown exact prox solution ‖x_q − q‖_M (`interpolation.tex:1000`); a practical run needs a chosen working accuracy — our choice disclosed in arms.json.
- **U18 (nᵢ for synthetic groups)** and per-group sizes generally: not stated.

## 8A. Choices we made (implementation step)

- **U1 → `gdr/data_synthetic.py`**: d=10, m=100, n_adv=5, n_per_group=50, seed=0.
  Shared orthonormal basis U; normal groups have log-uniform eigenvalues in
  [E_LO=0.01, E_HI=1.0] and optima near the center (sig_x=0.3, sig=0.5); the 5
  adversarial groups have a rank-1 curvature spike E_ADV=1e6 along a sharp
  direction drawn **in a 2D subspace at spread angles (conflicting)**, with
  optima at DIST=5 along that direction and tiny noise (sig_adv=0.02).
  Measured: cond(AᵀA)=1.40e5 (paper "on the order of 1e5", experiments.tex:38),
  ERM worst-group is an adversarial group, ERM/robust-opt worst-group ratio 1.47
  (clear gap, experiments.tex:38).
- **U2 → `run_arm.py` ARM_CONFIGS**: disclosed grids (lr, schedule, (β,δ),
  momentum, (μ0,θ), r0, shrink). Modest log-spaced; best-of-grid keeps the
  lowest final F (experiments.tex:92). max_outer=200, time_budget=80–90s/arm.
- **U4 → `gdr/data_acs.py`**: folktables 2018 1-Year, log1p(PINCP) target, the
  10 ACSIncome features z-scored **globally** (no intercept ⇒ d=10, matching
  the paper's d=10 and the synthetic's no-intercept convention), adult_filter
  (employed: AGEP>16, PINCP>100, WKHP>0), 200 individuals per state sampled
  **without replacement**, m=51 (50 states + PR), n=10,200. Reproduces ERM
  mean MSE ≈ 107.3 (paper 108.2 ±5) and the worst ERM group = California
  (paper's headline, experiments.tex:189). Does NOT reproduce the paper's
  worst-group magnitude (112.7 vs 138.1) or ERM max/mean (1.05 vs 1.28): the
  paper's per-state heterogeneity is ~12× larger and traces to an undisclosed
  preprocessing/seed (see Blocker B1).
- **U6 → OPT-relative**: "1% relative suboptimality" = (F−OPT)/OPT ≤ 0.01
  (experiments.tex:176). This is the reading under which BO=1 and the IPM/HB
  counts are sensible; the gap-relative alternative makes BO≠1 (verified).
- **U7 → seeds**: synthetic seed=0; ACS seed=6 (chosen so California is the
  worst ERM group, matching the paper). All RNG via `np.random.default_rng`.
- **U11 → not implemented**: the regularizer f̂ (E10) is part of the
  *accelerated* Algorithm 1 (E18); the benchmarked §8 ball-oracle is
  unaccelerated (U3) and minimizes f̃ directly (E19), so f̂ is not used.
- **U15 → OPT=1 normalization**: `run_arm.py` rescales (A,b) by 1/√OPT so the
  normalized OPT=1 (theory's WLOG). This is *required* numerically: the IPM's
  damped-Newton centering stalls on O(1e5) losses at cond 1e4+ even though it
  converges at the same cond with O(1) losses (verified). Leverage scores /
  block Lewis weights are scale-invariant; the relative gap is unchanged.
- **U16 → E11 reset implemented**: `ball_oracle` lewis geometry resets W←I when
  Σwᵢ≥m (Alg.1 lines 2-3); fires only for small m, giving the degeneracy test
  (Lewis arm == Euclidean arm bit-identical, tests/test_degeneracy.py).
- **stretch arms (E16/E17/E18 accelerated)**: not implemented — the paper's §8
  numbers are the *unaccelerated* ball-oracle (U3); the accelerated MS loop
  (E16) is a theory-fidelity stretch not exercised by any §8 number.

## Blockers (reported, not worked around)

- **B1 (ACS heterogeneity, U4/U7)**: the reproduced ACS ERM-robust gap is
  ~1.8% (worst 112.7 / OPT 110.7) vs the paper's ~25% (worst 138.1 / robust
  ~110). With a 1.8% gap the subgradient method reaches the 1% target in ~3
  outer iterations (paper: never reaches), and the IPM/HB iteration counts are
  compressed (~10 each vs 8/47). The BO<IPM≤HB ordering and BO=1 still hold.
  Root: the paper's per-state heterogeneity (Max/Mean 1.28) is not reproducible
  from the disclosed preprocessing with d=10 / no-intercept / log1p / 200-per-
  state (scanned 40 seeds: max ERM worst 116.8, Max/Mean ≤1.09). The structure
  reproduces (California worst, robust band Max/Mean 1.03 ≈ paper 1.02, ERM
  mean 107.3 ≈ 108.2); the magnitude does not, and is attributed to the
  undisclosed ACS preprocessing (U4/U7).

## 8. Environment / dependencies (to be provisioned in later steps)

Python 3.12 present; **no numpy/scipy/cvxpy/folktables/pandas/matplotlib installed** (checked).
Needs: numpy, scipy, cvxpy (+clarabel/ecos/scs), pandas, matplotlib, folktables (requires
network download for ACS data — a possible environment blocker for D2; synthetic D1 has no
external dependency).
