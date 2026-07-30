# SPEC: Distributionally Robust Linear Regression With Block Lewis Weights

- **Paper:** *Distributionally Robust Linear Regression With Block Lewis Weights* — Naren Sarayu Manoj (TTIC), Kumar Kshitij Patel (Yale), 2026
- **arXiv:** 2607.00252 (v1, 2026-06-30) · **paper_ref:** 0a8cf406-bd9b-4ebe-8a3f-c9e3d62a2a94 · **project_id:** d7735ece-02c4-4228-985c-00834c92b8f3
- **Authoritative source for all maths:** `paper/` = the arXiv e-print LaTeX bundle (extracted 2026-07-30). The PDF-extracted text in the task prompt is prose-only and is NOT trusted for equations. All citations below are `paper/<file>.tex:<line>` and are grep-able.
- **Auxiliary dependency source:** the Lewis-weight routine is imported from the companion paper Manoj–Ovsiankin (SODA 2025), arXiv 2311.10013 — cited as `[MO25]` in the paper (`paper/ref.bib:504`) — whose algorithm box we transcribed from its own LaTeX source, stored grep-ably at `paper/mo25_main.tex`.

## 0. Upstream code

- The paper's LaTeX source contains **no** code link (grep for `github|gitlab|code is available|repository|supplementary` over `paper/*.tex`: no matches).
- The arXiv abstract page for 2607.00252 contains **no** code link (grepped the HTML for github/gitlab URLs: none).
- GitHub repository search (2026-07-30, api.github.com): `"Distributionally Robust Linear Regression" in:name,description` → 0 relevant repos; `block lewis weights in:name,description` → 1 unrelated hit (a notes file); `lewis weights manoj in:name,description,readme` → 0. [MO25/2311.10013] likewise ships no code.
- The experiments section defers data-generation details to "the included Jupyter notebook" (`paper/experiments.tex:38`), but **no notebook is present in the arXiv source bundle** (bundle contents: only `*.tex`, `figures/`, `plots/` — see `paper/00README.json` for the compile manifest).
- **Verdict: no upstream code exists. We implement from scratch.** The synthetic instance must be *reconstructed* from a qualitative description plus one numeric anchor (condition number ≈ 10^5), because the notebook that pins it down was never published.

## 1. Reproduction target and scope decision

The paper's checkable numbers live in Section 8 (Empirical Evaluation), not in the theorems. The theorems
(Thm 1 = `paper/intro.tex:27-37`, Thm 2 = `paper/intro.tex:70-81`) state asymptotic linear-solve counts with
universal constants that are never instantiated; they are not numerically checkable on a box.

Critically, the empirical system the authors actually ran is **a simplification of the theory algorithms**:
> "We implement two trust-region style methods that repeatedly solve the smoothed objective using a damped Newton solver (§2.2)… After each outer step, the center is updated to the new solution, and the trust-region radius is optionally shrunk. **For simplicity, we do not consider the acceleration of the ball-oracle method.**" (`paper/experiments.tex:72-78`)

So the **reproduction scope** is:

1. The adversarial heterogeneous **synthetic** instance (d=10, m=100, 5 adversarial, κ(Gram)≈10^5, `paper/experiments.tex:38`) — reconstructed, see §6.
2. The **ACS Income** instance (m=51 regions, d=10, 200/region → n=10,200, `paper/experiments.tex:148`).
3. The seven optimizer arms of `paper/experiments.tex:53-78` plus the CVXPY epigraph reference optimum (`paper/experiments.tex:40-52`).
4. The metrics/plots the paper reports: suboptimality vs outer-iteration, suboptimality vs wall-clock (Figure 1 style), and the ACS cost-to-1%-gap table (`paper/experiments.tex:171-187` = Table `tab:acs_runtime`).

The **theory layer** (full Algorithm 1 `paper/body.tex:164-188`, Algorithm 5 `paper/interpolation.tex:1067-1081`, the MS-acceleration driver `paper/improved_ms.tex:6-39`, inexact mirror descent `paper/mirror_descent.tex:8-25`, and the p≥2 proximal oracle `paper/interpolation.tex:425-441`) is specified in §5 as a citation map. It is **not** what the experiments ran, so it is **not** in the numbers gate; implementing it is a stretch goal after the empirical arms are green.

## 2. The problem and the evaluation elements

Groups `i ∈ [m]`, group `i` has `n_i` rows; `n = Σ_i n_i`. All losses are group-mean-squared errors.

- (E1) Group loss: `ℓ_i(x) = (1/n_i)||A_{S_i} x − b_{S_i}||₂²` — `paper/experiments.tex:9` (identical to `paper/intro.tex:11` objective).
- (E2) Worst-group objective: `F(x) = max_i ℓ_i(x)` — `paper/experiments.tex:13`.
- (E3) Utilitarian baseline / ERM: `x_ERM = argmin (1/m) Σ_i ℓ_i(x)` — `paper/intro.tex:5` (closed-form weighted least squares; minimizer of the *sum* of the n_i⁻¹-weighted residual sums iff each group is normalized, see §6 item 11).
- (E4) Robust optimum `OPT`: epigraph convex program `min_{x,t} t  s.t.  ℓ_i(x) ≤ t ∀i ∈ [m]`, each constraint a convex quadratic, solved with CVXPY — `paper/experiments.tex:44-50`. The returned `t` is `OPT` (`paper/experiments.tex:50`).
- (E5) Reported gap: `F(x_t) − OPT`, best-so-far per curve — `paper/experiments.tex:52` and `paper/experiments.tex:167`.
- (E6) "1% relative gap": used for Table `tab:acs_runtime` (`paper/experiments.tex:174-186`) — the reference point of the 1% is **never stated** (§6 item 4); we compute both (`gap/gap(x_0)` and `gap/OPT`) and flag which reproduction matches.

Notation trap (flagged for implementers): the theory text folds `1/√n_i` into the data (`paper/body.tex:27`), so theory `f(x) = max_i ||A_{S_i}x − b_{S_i}||₂` is the **square root** of the empirical `F(x)`. The argmins coincide (sqrt is monotone in each group loss). All *reported numbers* use the unsquared MSE convention (E1/E2).

## 3. The method as algorithms (the empirical arms actually run)

Inputs common to all arms: a group regression problem `(A ∈ R^{n×d}, b ∈ R^n, group_id ∈ [n]→[m])`, a warm start
`x_0` (same for every arm — `paper/experiments.tex:92`; on ACS the warm start is the ERM — `paper/experiments.tex:148`),
an outer-iteration budget (unstated, §6 item 5), and per-arm tuned hyperparameters
(grids unstated, §6 item 3). Every arm reports `F(x_t)` per natural outer iteration and wall-clock; units of "one
iteration" differ by arm (`paper/experiments.tex:96-104`).

### 3.0 Reference arm (not an optimizer): CVXPY epigraph solve of (E4)

One quadratic constraint per group; solver not named (§6 item 6). This arm produces `OPT` and `x*` used by all gap computations.

### 3.1 Arm `subgradient` (`paper/experiments.tex:57-58`)

- Step: pick `i* = argmax_i ℓ_i(x_t)`; subgradient `g = (2/n_{i*}) A_{S_{i*}}^⊤ (A_{S_{i*}} x_t − b_{S_{i*}}) ∈ R^d` (gradient of the active group loss; objective non-smooth at ties — any argmax element, `paper/intro.tex:15`).
- Update: `x_{t+1} = x_t − η g` (fixed schedule) or `x_t − η_0/√t · g` (diminishing). **Both** are run, best reported (`paper/experiments.tex:58`). η grids unstated (§6 item 3).

### 3.2 Arms `smoothed_gd`, `smoothed_heavy_ball`, `smoothed_nesterov` (`paper/experiments.tex:60-66`)

Operate on the smooth surrogate (E7):

- (E7) `f̃_{β,δ}(x) = β·log Σ_{i=1}^m exp( (√(δ² + ||A_{S_i} x − b_{S_i}||₂²) − δ) / β )` — `paper/body.tex:55` (label `eq:intro_smooth_fair_objective`). β, δ tuned (`paper/experiments.tex:86`, values unstated §6 item 3).
  - Max-norm approx error: `|f̃_{β,δ}(x) − max_i √ℓ̃_i(x)| ≤ β log m + δ` — `paper/body.tex:57,272` (Lemma 6.1; `√` on losses in the folded-normalization convention).
- Derivatives actually implemented (derived in-proof, `paper/body.tex:283-303, 431-435`): with `r_i = A_{S_i}x − b_{S_i} [n_i]`, `h_i = √(δ²+||r_i||₂²) [scalar]`, `∇s_i = A_{S_i}⊤ r_i / h_i [d]`, `∇²s_i = h_i⁻¹ A_{S_i}⊤ (I_{n_i} − r_i r_i⊤/h_i²) A_{S_i} [d×d]`, softmax weights `σ = softmax(s/β) [m]`, `s_i = h_i − δ`:
  - `∇f̃ = Σ_i σ_i ∇s_i`  (`paper/body.tex:287` chain-rule form)
  - `∇²f̃ = (1/β)( Σ_i σ_i ∇s_i∇s_i⊤ − gg⊤ ) + Σ_i σ_i ∇²s_i`,  `g = Σ_i σ_i ∇s_i`  (`paper/body.tex:288,298-300`)
- Updates: GD `x ← x − η∇f̃(x)`; Heavy-Ball `x ← x − η∇f̃(x) + μ(x_t − x_{t−1})`; Nesterov: standard accelerated gradient on `∇f̃` (Table 1 row "Nesterov acceleration on smoothened objective", `paper/intro.tex:47`). Momentum coefficients tuned, values unstated (§6 item 3).

### 3.3 Arm `ipm` (`paper/experiments.tex:68-69`)

Log-barrier interior point on the epigraph reformulation (E4), per Boyd–Vandenberghe §6.4 (`paper/intro.tex:17`):
centring problems `min_{x,t} τ·t − Σ_i log(t − ℓ_i(x))` solved by Newton steps ("a sequence of smooth
approximations using Newton steps", `paper/experiments.tex:69`). Barrier parameter, growth factor, centring
tolerances and inner iteration counts all tuned/unstated (§6 item 3; `paper/experiments.tex:88`).

### 3.4 Arms `ball_oracle_euclidean`, `ball_oracle_lewis` (`paper/experiments.tex:71-78`)

Trust-region method over the smoothed surrogate; **no acceleration** (`paper/experiments.tex:78`):

```
given: geometry matrix M (PSD, d×d), initial radius r (tuned), decay ρ (tuned, applied optionally),
       smoothing (β, δ) (tuned), warm start x_0
q ← x_0
repeat for T outer iterations:
    x⁺ ≈ argmin{ f̃_{β,δ}(x) : ||x − q||_M ≤ r }          # via a damped Newton solver
    q ← x⁺ ; optionally r ← ρ·r
return q
```

- Trust-region subproblem shape is the paper's proximal-template (E9): `min_{||x−q||_M ≤ r_q} f(x)` — `paper/body.tex:31` (`eq:gp_prox_intro`); iteration of such calls is §2.2 (`paper/body.tex:110-116`).
- Geometries: "Euclidean geometry (naive ball)" and "Lewis-weight geometry … data-dependent positive definite matrix constructed from block Lewis weights" (`paper/experiments.tex:73-76`). The paper's own Table 1 identifies the naïve geometry as `M = A⊤A` (`paper/body.tex:131`, `paper/intro.tex:50`), and the theory algorithm's Lewis geometry as `M = A⊤WA` (`paper/body.tex:511`; selection between the two by `paper/body.tex:158`, Algorithm 1 lines 169-176 `paper/body.tex:169-176`). Both balls are centered ellipsoids in R^d; each damped-Newton step solves `(∇²f̃(q') + νM) d = −∇f̃(q')` with `A⊤BA` structure (`paper/body.tex:609`).
- The "damped Newton solver" damping rule, inner stopping rule, and whether the region is enforced by projection/line-search is **not specified** (§6 item 7).
- Whether the regularized `f̂(x) = f̃_{β,δ}(x) + reg·||W^{1/2}A(x−x_0)||₂²` (`paper/body.tex:182`, Algorithm 1 line 6) is included in the empirical run is **not stated** (§6 item 8).

### 3.5 The block-Lewis-weight routine (feeds `ball_oracle_lewis`)

The paper delegates to "[MO25, Algorithm 2]" on the augmented matrix `Â = [A|b] ∈ R^{n×(d+1)}` — `paper/body.tex:169` (Algorithm 1 line 1). MO25's Algorithm 2 is `alg:blw` in our copy `paper/mo25_main.tex:1809-1823`; specialized to the paper's setting (all inner norms are ℓ2, and for the max objective `p = ∞ ⇒ exponent 1/2 − 1/p = 1/2`; cf. `paper/other_proofs.tex:20-24,28-34` with p→∞):

```
w^(0) = ((d+1)/m)·1_m                                   # paper/mo25_main.tex:1814 (their n = our rank(Â) ≤ d+1)
for t = 1..T−1:
    W^(t) = diag over rows: W_jj = w_i^(t) for j ∈ S_i  # paper/mo25_main.tex:1816
    τ̃^(t)_j = a_j⊤( Â⊤ W^(t) Â )^{−1} a_j   ∀j          # leverage scores of W^(t)^{1/2} Â, paper/mo25_main.tex:1817, exact-solve OverLev
    w_i^(t+1) = Σ_{j∈S_i} τ̃^(t)_j                        # paper/mo25_main.tex:1818
w = (3/2)·(1/T) Σ_t w^(t)                               # paper/mo25_main.tex:1820-1821
```
`T = O(log m)` (`paper/mo25_main.tex:1890-1891`); constants unstated (§6 item 9). Each iteration is one linear-system solve in `Â⊤DÂ` (`paper/other_proofs.tex:44-47`).
Guarantee consumed: `||Ax−cb||_{G,∞} ≤ ||W^{1/2}(Ax−cb)||₂ ≤ √(2(rank A + 1))·||Ax−cb||_{G,∞}` — `paper/body.tex:169-172` (and `paper/other_proofs.tex:30-34` for finite p, `paper/body.tex:147-150`);
weight-sum control `rank(A)+1 ≤ Σ_i w_i ≤ 2(rank(A)+1)` and the `W = I_n` reset rule: `paper/body.tex:173-174`.
The geometry actually used downstream is `M = A⊤WA` (note: **A**, not Â, inside M) — `paper/body.tex:511`.

## 4. Symbols, with shapes

Conventions: 0-indexed groups `i ∈ {0..m−1}` map to row slices `S_i` (contiguous, Fortran-free, row-major `A` storage).
`[m] [n] [d] [n_i]` denote array shapes.

| Symbol | Shape | Meaning / citation |
|---|---|---|
| `m` | scalar | number of groups (100 synthetic; 51 ACS) — `paper/experiments.tex:38,148` |
| `n_i`, `n = Σn_i` | scalars | rows per group; total rows (`paper/intro.tex:28`) |
| `A_{S_i}` | `[n_i, d]` | group design matrix (`paper/intro.tex:7`) |
| `b_{S_i}` | `[n_i]` | group response (`paper/intro.tex:7`) |
| `A` | `[n, d]` | stacked design, rows grouped contiguously (`paper/intro.tex:28`) |
| `b` | `[n]` | stacked response (`paper/intro.tex:28`) |
| `Â = [A|b]` | `[n, d+1]` | augmented matrix for Lewis weights (`paper/other_proofs.tex:53`) |
| `x`, `x_t`, `x*`, `x̂` | `[d]` | parameter, iterate, robust optimum, output |
| `r_i(x) = A_{S_i}x − b_{S_i}` | `[n_i]` | group residual |
| `ℓ_i(x)` | scalar | `(1/n_i)||r_i||₂²` (`paper/experiments.tex:9`) |
| `F(x)` | scalar | `max_i ℓ_i(x)` (`paper/experiments.tex:13`) |
| `||y||_{G,p}` | scalar | `(Σ_i ||y_{S_i}||₂^p)^{1/p}` for `y [n]`; `p=∞ ⇒ max_i ||y_{S_i}||₂` (`paper/body.tex:27`, `paper/body.tex:218`) |
| `β, δ` | scalars > 0 | softmax temperature; huberization (`paper/body.tex:55`). Theory values `β = ε/(4 log m)`, `δ = ε/4` (`paper/body.tex:181`); empirical values tuned |
| `s_i(x)`, `h_i(x)` | scalars | `s_i = √(δ²+||r_i||₂²) − δ`, `h_i = s_i + δ` (`paper/body.tex:55,413`) |
| `σ(x)` | `[m]`, `σ ≥ 0`, `Σσ = 1` | softmax weights over groups = normalized `exp(s_i/β)` (`paper/body.tex:285,339`) |
| `f̃_{β,δ}(x)` | scalar | smoothed surrogate (E7) |
| `f̂(x)` | scalar | regularized `f̃` (`paper/body.tex:182`); regularizer coefficient disputed — §6 item 8 |
| `w` | `[m]`, ≥ 0 | per-group block Lewis overestimates (`paper/other_proofs.tex:20`) |
| `W` | `[n, n]` diag | `W_jj = w_i` for `j ∈ S_i` (`paper/other_proofs.tex:20`) |
| `τ_j(A)` | scalar | leverage score `a_j⊤ (A⊤A)^{-1} a_j` (`paper/other_proofs.tex:11-13`) |
| `M` | `[d, d]` PSD | trust-region metric: `A⊤A` (naive) or `A⊤W^{1−2/p}A` (finite p),`A⊤WA` (p=∞) (`paper/body.tex:158,511`) |
| `||x||_M` | scalar | `√(x⊤Mx)` (`paper/body.tex:27`) |
| `ε` | scalar | target accuracy (`paper/intro.tex:28`) |
| `r`, `ρ` | scalars | trust-region radius; decay (empirical, `paper/experiments.tex:78`) |
| **Theory layer** | | |
| `x̃_{t+1}, q_t, v_t` | `[d]` | MS-acceleration prox answer, query, momentum point (`paper/improved_ms.tex:12-17`) |
| `a'_{t+1}, A_t, A'_t, a_{t+1}` | scalars ≥ 0 | coupling step sizes / masses (`paper/improved_ms.tex:14-19`) |
| `λ'_t, λ_t` | scalars > 0 | candidate / realized prox scale; `λ_t = e p^{p+1}||x̃_t−q_{t−1}||_M^{p−2}` for finite p (`paper/interpolation.tex:1084`) |
| `γ_{t+1}` | scalar ∈ (0,1] | `min{1, λ'/λ}` damping (`paper/improved_ms.tex:18`) |
| `α` | scalar > 1 | `exp(3 − 2/(s+1))` multiplicative λ adjustment (`paper/improved_ms.tex:63`) |
| `σ` (MS param) | scalar ∈ [0,0.99) | MS-oracle accuracy, = 1/2 in both uses (`paper/improved_ms.tex:43-49,63`) |
| `C_p = e·p^p` | scalar | prox regularizer coefficient (`paper/interpolation.tex:459`) |
| `f_q(x̃)`, `h_q(x̃)` | scalars | prox objective `f + C_p||x̃−q||_M^p`; reference `||x̃−q||²_{∇²f(q)} + C_p||x̃−q||_M^p` (`paper/interpolation.tex:430-437`) |
| `D_h(x,y)` | scalar | Bregman divergence of reference `h` (`paper/mirror_descent.tex:13-17`) |
| `L, μ` | scalars | relative smoothness `pe` / strong convexity `1/(2pe)` of `f_q` in `h_q` (`paper/interpolation.tex:946-949`) |
| `τ` | scalar ≥ 0 | binary-searched quadratic regularization weight for inner subproblem, `τ* = ||z−q||_M^{p−2}` (`paper/interpolation.tex:843-855`) |

## 5. Equations to implement, each with a citation

Empirical arms (numbers-gate scope):

| ID | Equation | Citation (grep target) |
|---|---|---|
| E1 | `ℓ_i(x) = (1/n_i)||A_{S_i}x − b_{S_i}||₂²` | `paper/experiments.tex:9` |
| E2 | `F(x) = max_i ℓ_i(x)` | `paper/experiments.tex:13`; theorem form `paper/intro.tex:30` |
| E3 | `x_ERM = argmin_x (1/m)Σ_i (1/n_i)||A_{S_i}x−b_{S_i}||₂²` | `paper/intro.tex:5` |
| E4 | `OPT = min_{x,t} t s.t. ℓ_i(x) ≤ t ∀i` (CVXPY epigraph) | `paper/experiments.tex:44-50` |
| E5 | gap curve `F(x_t) − OPT`, best-so-far | `paper/experiments.tex:52,167` |
| E7 | `f̃_{β,δ}(x) = β log Σ_i exp((√(δ²+||r_i||₂²)−δ)/β)` | `paper/body.tex:55` (also `paper/body.tex:179,224`) |
| E8 | `|f̃_{β,δ}(x) − ||Ax−b||_{G,∞}| ≤ β log m + δ` | `paper/body.tex:231` (Lemma 6.1, proof 234-272) |
| E9 | trust-region template `min_{||x−q||_M≤r_q} f(x)` | `paper/body.tex:31` |
| E10 | `∇f̃`, `∇²f̃` via softmax calculus (E10a `lse'_β`, E10b `lse''_β`; E10c `h'_i = ⟨r_i,·⟩/h_i`, `h''_i = (||d_i||²−h'_i²)/h_i`) | `paper/body.tex:287-288` and `paper/body.tex:431-435` |
| E11 | Σ σ_i ∇s_i and `∇f̃ ≤ max_i ∇s_i` sanity bound | `paper/body.tex:294` |
| E12 | block Lewis iteration (`w^(0)`, `τ̃` sweep, mean, ×3/2) | `paper/mo25_main.tex:1814-1821`; consumption `paper/body.tex:169-174` |
| E13 | ellipsoid guarantee `||Ax−cb||_{G,∞} ≤ ||W^{1/2}(Ax−cb)||₂ ≤ √(2(rankA+1))||Ax−cb||_{G,∞}` | `paper/body.tex:170-172`; proof `paper/other_proofs.tex:87-104` |
| E14 | weighted-least-squares initializer `x_0 = (A⊤WA)^{−1}A⊤Wb` | `paper/body.tex:176`; `paper/other_proofs.tex:68-71` |
| E15 | naive-geometry distortion `||Ax−b||₂/√m ≤ ||Ax−b||_{G,∞} ≤ ||Ax−b||₂` | `paper/body.tex:599-602` |

Theory layer (reference scope; cited, implementation is a stretch goal — NOT in the numbers gate):

| ID | Equation / Algorithm | Citation |
|---|---|---|
| T1 | Algorithm 1 MinMaxRegression (full accelerated pipeline: Lewis weights → W=I reset → x₀ (E14) → β=ε/4logm,δ=ε/4 → f̂ → (C/min{rank,m}, C/ε)-ball oracle via [CJJJLST20 Alg 3] → ½-MS oracle via [CJJJLST20 Alg 2] → Õ(min{rank,m}^{1/3}log(d/ε)/ε^{2/3}) iterations of [CJJJLST20 Alg 1]) | `paper/body.tex:164-188` |
| T2 | `f̂(x) = f̃_{ε/4logm,ε/4}(x) + (ε/(1000 min{rank(A),m}))||W^{1/2}A(x−x₀)||₂²` | `paper/body.tex:182`; **proof variant** `ε/(110 R²)`, `R=(2+ε)√(2(d+1))`: `paper/body.tex:580,586` |
| T3 | smoothness/QSC of f̂: `∂²f̃ ≤ (1/δ+1/β)||W^{1/2}Az||₂²`, `|∂³f̃| ≤ (16/δ+3/β)||W^{1/2}Az||₂∂²f̃` | `paper/body.tex:402-406`; components `paper/body.tex:446-468,471-479` |
| T4 | softmax–QSC composition: `|∂³ lse_β(h)| ≤ (16/β+ν)||d|| ∂² lse_β(h)` | `paper/body.tex:71-82` (Lemma 6.3); proof `paper/body.tex:332-393` |
| T5 | Hessian of f̂ is `A⊤BA` with `|S_i|×|S_i|` diagonal blocks | `paper/body.tex:609` |
| T6 | interpolation objective `min (1/m)Σ_i ((1/n_i)||r_i||²)^{p/2}` | `paper/intro.tex:66` |
| T7 | `f(x) = ||Ax−b||_{G,p}^p`, `∇f = pΣ_i ||r_i||₂^{p−2}A_i⊤r_i`, `∇²f = pΣ_i||r_i||₂^{p−2}A_i⊤A_i + p(p−2)Σ_i||r_i||₂^{p−4}(A_i⊤r_i)(A_i⊤r_i)⊤` | `paper/interpolation.tex:25-28`; bounds `paper/interpolation.tex:19` |
| T8 | strong convexity: `f(x+d) ≥ f(x) + ⟨∇f(x),d⟩ + (4/2^p)||Ad||_{G,p}^p`; component form `||v+Δ||₂^p ≥ ||v||₂^p + p||v||₂^{p−2}⟨v,Δ⟩ + (4/2^p)||Δ||₂^p` | `paper/interpolation.tex:51-61`; `paper/body.tex:98-104` |
| T9 | smoothness: `f(x)−f(x*) ≤ (p(p−1)/2) f(x)^{1−2/p}||A(x−x*)||_{G,p}²` | `paper/interpolation.tex:338-344` |
| T10 | prox subproblem `argmin f(x̃) + e p^p ||x̃−q_t||_M^p` | `paper/body.tex:92`; `paper/interpolation.tex:415` |
| T11 | Algorithm 4 GpRegressionProxOracle (ref fn `h_q`, iteration count `T ≥ C p^{O(1)} e log(dpe h_q(x̃_q)(4/(pγ))^p)`) | `paper/interpolation.tex:425-441` |
| T12 | relative smoothness/strong convexity `(1/(2pe))∇²h_q ⪯ ∇²f_q ⪯ (pe)∇²h_q` | `paper/interpolation.tex:465-468`; used at `946-949` |
| T13 | inner mirror step = quadratic solve with binary-searched τ: `z−q = −(1/(2pe))(∇²f(q)+C_pτM)^{−1}g`, `τ* = ||z−q||_M^{p−2}` | `paper/interpolation.tex:836-855,875-883` |
| T14 | ApproximateMirrorDescent (Alg 2) and its guarantee `f(x_j)−f(x*) ≤ L(1−μ/L)^T D_h(x*,x₀) + max_i ⟨Δ_i, x_i−x*⟩`, `Δ_i = ∇f(x_{i−1}) + L(∇h(x_i)−∇h(x_{i−1}))` | `paper/mirror_descent.tex:8-25`; theorem `37-47` |
| T15 | OptimalMSAcceleration (Alg 3) update rules: `a' = (1+√(1+4λ'A_t))/(2λ')`, `q_t` mix, `γ = min{1,λ'/λ}`, `v ← v − a M^{-1}∇f(x̃)` | `paper/improved_ms.tex:11-36`; MS-oracle defn `paper/improved_ms.tex:43-49`; guarantee `paper/improved_ms.tex:61-71` |
| T16 | Algorithm 5 GpRegression and halving argument `T_min ≲ p^{5/3}d^{(p−2)/(3p−2)}` | `paper/interpolation.tex:1067-1081,1092-1104` |
| T17 | block-Lewis ellipsoid guarantee for finite p | `paper/other_proofs.tex:28-34`; init lemma `paper/other_proofs.tex:51-85` |

## 6. What the paper does NOT state (register of reconstruction decisions)

Every constant that appears in an implemented expression but never receives a value in the paper is listed here.
Each item gets a **decision** in code config, labeled `[reconstructed]`.

1. **Synthetic data recipe.** Only qualitative: groups share an orthonormal eigenbasis; normal groups have "moderate condition number", optima near a common center, "independent noise"; adversarial groups have one extreme-curvature eigendirection (distinct per adversarial group), optima "far from the population center" along it, "very small" noise — `paper/experiments.tex:19-25`. No eigenvalue magnitudes, no noise σ values, no per-group `n_i`, no center norms, no basis distribution, no seed. The reference "included Jupyter notebook" (`paper/experiments.tex:38`) is absent from the arXiv bundle. Numeric anchors available: d=10, m=100, 5 adversarial, κ(stacked Gram)≈10^5, ERM-vs-OPT gap visible (`paper/experiments.tex:38`). All generation constants `[reconstructed]` and are tuned only to hit those four anchors.
2. **Per-group sizes `n_i`** for the synthetic instance: never stated. `[reconstructed]`.
3. **All hyperparameter grids and budgets, and the tie-break.** "Step sizes for gradient and subgradient methods; smoothing parameters and momentum coefficients for smoothed methods; barrier parameters and inner iteration counts for the IPM; initial trust-region radius, smoothing strength, and radius decay factors for the ball-oracle methods" (`paper/experiments.tex:81-90`) — tuned, but grids/ranges/budgets never printed. "Run a fixed number of outer iterations and select the configuration that achieves the lowest worst-group loss within this budget" (`paper/experiments.tex:92`) — the budget is unstated, and the criterion is silent on ties (two configs can achieve the same lowest loss). `[reconstructed]`: tie broken by fewest iterations to reach the tuned loss (fastest-among-equally-good).
4. **"1% relative suboptimality" definition** (Table `tab:acs_runtime`, `paper/experiments.tex:174-186`): relative to the initial (ERM) gap, or to OPT itself. Unstated; we record both and report which matches.
5. **Warm start on the synthetic instance.** `paper/experiments.tex:92` says all methods share a warm start, but its value is unstated (on ACS it is ERM, `paper/experiments.tex:148`). `[reconstructed]`: ERM, by analogy.
6. **CVXPY solver.** "calling a standard convex solver" (`paper/experiments.tex:50`) — name/tolerances unstated. `[reconstructed]`: default (CLARABEL / SCS / ECOS as available), documented in the run log.
7. **Empirical trust-region solver internals.** "damped Newton solver" (`paper/experiments.tex:72`) — damping rule, convergence tolerance, how the ellipsoid constraint `||x−q||_M ≤ r` is enforced (projection? backtracking? penalty?), and the radius shrink rule ("optionally shrunk", `paper/experiments.tex:78`) are all unstated. `[reconstructed]`: Armijo/backtracking damped Newton with boundary projection onto the M-ellipsoid, documented.
8. **f̂ regularizer in the empirical run, and its coefficient.** Algorithm 1 line 6 (`paper/body.tex:182`) uses `ε/(1000·min{rank(A),m})`; the analysis (`paper/body.tex:580`) uses `ε/(110 R²)` with `R = (2+ε)√(2(d+1))`. These disagree, and whether the experiments include either is unstated. `[reconstructed]`: expose as a boolean config, default **off** in the tuned run; when on, the Algorithm-1 coefficient `ε/(1000·min{rank(A),m})` with `ε = 4·β·log m` (the paper's coupling `β = ε/(4 log m)`, `paper/body.tex:181`).
9. **Lewis-routine constants.** `T = O(log m)` with unstated constant (`paper/mo25_main.tex:1890-1891`, `paper/other_proofs.tex:44-46`); the OverLev routine (exact vs sketched leverage scores) unstated — `[reconstructed]`: exact solve per iteration, `T = ⌈2 ln m⌉`; which `p` the empirical Lewis geometry uses is also unstated — `[reconstructed]`: p=∞ (matches Algorithm 1 line 1-6, the max objective).
10. **IPM internals.** Initial barrier parameter, growth factor, centring tolerance, inner cap (`paper/experiments.tex:88`) unstated. `[reconstructed]`.
11. **ERM definition.** Minimizer of the *group-averaged* loss (E3) vs ordinary pooled OLS differ when `n_i` vary; the paper writes the group-averaged form (`paper/intro.tex:5`) but never says which was used for the warm start. On ACS `n_i ≡ 200` so they coincide; on synthetic they may not. `[reconstructed]`: group-averaged ERM (E3).
12. **ACS details.** folktables year/horizon (2018 1-Year assumed `[reconstructed]`), the exact feature list ("age, education, occupation, hours worked, **etc.**", `paper/experiments.tex:148` — we assume folktables' default 10 ACS features), the standardization rule (population z-score assumed `[reconstructed]`), the 200-per-region subsample seed, and the **target scale**: reported losses (ERM avg 108.2, worst 138.1, `paper/experiments.tex:189`) are inconsistent with raw `np.log(income)` MSE (≈0.6-0.7) — a scaling factor (e.g. ×100 or ×1000 on log-income, or a different log) is unstated. Resolved empirically at data-pipeline time by matching the ERM statistics; the iteration-to-1% metric is scale-invariant.
13. **Seeds / repeats / variance.** No seed is reported anywhere; no experiment is repeated; all curves are single runs (presumably). Sampled elements: synthetic generation, ACS subsampling.
14. **Hardware / timing environment.** Wall-clock numbers (0.019 s etc., `paper/experiments.tex:174-184`) have no stated machine. We report our hardware.
15. **Algorithm-1 iteration count instantiation.** `Õ(min{rank,m}^{1/3}log(d/ε)/ε^{2/3})` (`paper/body.tex:185`) — log-factor and universal constant uninstantiated; irrelevant to the empirical arms (which run unaccelerated) but noted for completeness.
16. **Universal constant C** in the ball oracle radii `(C/min{rank,m}, C/ε)` (`paper/body.tex:183`); analysis also invokes `(C/√d, C/ε)` (`paper/body.tex:588`) — the two differ, another internal inconsistency.
17. **Theoretical `ε` used empirically**: the ball-oracle's "smoothing strength" is tuned (`paper/experiments.tex:89`), decoupling the empirical (β,δ) from the theory coupling β=ε/(4 log m), δ=ε/4 (`paper/body.tex:181`).
18. **Post-Newton-safety numerics**: `h_i ≥ δ > 0` keeps E10 well-defined; nothing is said about residual-flow handling of ties in argmax for the subgradient arm. `[reconstructed]`: lowest index.

## 7. Frozen component interfaces

File layout (fixed now; parallel agents implement against exactly this — **no renegotiation**):

```
gdr/
  types.py        # dataclasses + conventions (this file's contract)
  data_synth.py   # synthetic generator (§6 item 1 config)
  data_acs.py     # folktables loader (§6 item 12)
  objectives.py   # E1-E3, E7, E10, E11 + regularized variant (T2 flag)
  lewis.py        # E12-E14
  solvers.py      # arms 3.1-3.4 + reference E4
  metrics.py      # E5/E6 gap curves, cost-to-relative-gap, wall-clock
  harness.py      # tuning grids (§6 item 3), runs arms.json, writes results
arms.json         # the arm contract (§8)
SPEC.md           # this file
```

Interfaces (signatures are contractual):

```python
# types.py
@dataclass(frozen=True)
class GroupProblem:
    A: np.ndarray         # float64 [n, d], row-major, rows grouped contiguously
    b: np.ndarray         # float64 [n]
    group_id: np.ndarray  # int32  [n], values 0..m-1, nondecreasing
    @property
    def m(self) -> int: ...
    @property
    def sizes(self) -> np.ndarray: ...   # int64 [m]
    def residuals(self, x: np.ndarray) -> np.ndarray: ...        # [d] -> [n]
    def group_losses(self, x: np.ndarray) -> np.ndarray: ...     # [d] -> [m]  (E1)
    def worst_loss(self, x: np.ndarray) -> float: ...            # (E2)
    def erm(self) -> np.ndarray: ...                             # [d]  (E3, closed form)
    def augmented(self) -> tuple[np.ndarray, np.ndarray]: ...    # Â [n, d+1] as (Ahat) — used by lewis.py

# objectives.py
class SmoothObjective(Protocol):
    def value(self, x: np.ndarray) -> float: ...
    def grad(self, x: np.ndarray) -> np.ndarray: ...             # [d]
    def hess(self, x: np.ndarray) -> np.ndarray: ...             # [d, d]
def make_smoothed(problem: GroupProblem, beta: float, delta: float) -> SmoothObjective  # E7/E10
def make_regularized(problem: GroupProblem, beta: float, delta: float,
                     sqrt_w: np.ndarray, x0: np.ndarray, coef: float) -> SmoothObjective  # T2

# lewis.py
def block_lewis_weights(problem: GroupProblem, p: float | None, n_iters: int | None) -> np.ndarray
    # p=None means ∞; returns w [m] (E12); n_iters=None -> ceil(2 ln m) (§6 item 9)
def geometry_matrix(problem: GroupProblem, w: np.ndarray | None, p: float | None) -> np.ndarray
    # None -> A⊤A (naive, E15); else A⊤ W^{1-2/p} A (p=∞ -> A⊤WA) [d, d] (§3.5)

# solvers.py
@dataclass
class History:
    x: list[np.ndarray]          # iterate per natural outer iteration, x[0] = warm start
    wall: list[float]            # cumulative seconds, time.perf_counter
def solve_subgradient(problem, x0, cfg: dict, budget: int) -> History        # §3.1
def solve_smooth(problem, x0, cfg: dict, budget: int) -> History             # §3.2, cfg["method"] ∈ {gd,heavy_ball,nesterov}
def solve_ipm(problem, x0, cfg: dict, budget: int) -> History                # §3.3
def solve_ball_oracle(problem, x0, cfg: dict, budget: int) -> History        # §3.4, cfg holds geometry/radius/decay/smoothing
def reference_optimum(problem) -> tuple[np.ndarray, float]                   # E4 via cvxpy; returns (x*, OPT)

# metrics.py
def gap_curve(history, opt: float, worst_fn) -> np.ndarray       # best-so-far F(x_t) − opt (E5)
def cost_to_rel_gap(curve: np.ndarray, gap0: float, rel: float, base: str) -> int | None
    # base ∈ {"init", "opt"} (§6 item 4); None if never reached
```

Determinism contract: every stochastic component takes an explicit `numpy.random.Generator` built from the
integer seed recorded in the run config; arms report a single seed lineage per result row.

## 8. Arms (what becomes `arms.json`)

The paper's own comparison (`paper/experiments.tex:55-76`) has **seven optimizer arms plus one reference arm** —
this is the list the numbers gate checks. Paper claims that map onto arms: Table `tab:acs_runtime`
(`paper/experiments.tex:174-186`), prose ACS claims (`paper/experiments.tex:152`), statistical context
(`paper/experiments.tex:189`), qualitative synthetic claims (`paper/experiments.tex:107-109`).

| # | arm | kind | config keys | paper claim checked |
|---|---|---|---|---|
| 1 | `reference_cvxpy` | reference | solver name | OPT values; statistical-context numbers (189) |
| 2 | `subgradient` | baseline, ours=No | `step`, `schedule ∈ {fixed, inv_sqrt}`, best-of | "never reaches 1%; pinned at ERM gap" (152,186) |
| 3 | `smoothed_gd` | baseline | `beta`, `delta`, `step` | stalls far above optimum (107) |
| 4 | `smoothed_heavy_ball` | baseline | `beta`, `delta`, `step`, `momentum` | 47 it / 0.062 s to 1% (174-186) |
| 5 | `smoothed_nesterov` | baseline | `beta`, `delta`, `step` | best-tuned smoothed variant stalls (107); compare to heavy-ball |
| 6 | `ipm` | baseline | `barrier0`, `growth`, `inner_iters`, `tol` | 8 it / 0.066 s to 1% (174-186) |
| 7 | `ball_oracle_euclidean` | **ours (paper's)** | `geometry=naive`, `radius0`, `decay`, `beta`, `delta`, `reg_on` | 1 it / 0.019 s to 1% (174-186) |
| 8 | `ball_oracle_lewis` | **ours (paper's flagship)** | same + `geometry=lewis`, `lewis_p=None`, `lewis_iters=None` | 1 it / 0.019 s; "marginally faster" than euclidean (152,174-186) |

Headline gate metric: **iterations to 1% relative worst-group gap on ACS Income for `ball_oracle_lewis` — claimed 1** (and `ball_oracle_euclidean` — claimed 1). Secondary gate rows: the full `tab:acs_runtime` table (per-arm iterations + wall-clock), subgradient "not reached", and the statistical-context vector (ERM avg 108.2 / σ 11.8 / worst 138.1 (California); robust Max/Mean 1.28→1.02; CA −24.3) after resolving §6 items 4 and 12. Wall-clock rows are recorded but marked machine-dependent (informational, not pass/fail).

Synthetic-instance checks (qualitative, from `paper/experiments.tex:107-109`): subgradient plateaus; best-tuned smoothed method stalls far above OPT; both ball oracles monotonically decrease; IPM fastest to high accuracy; κ(A⊤A) ≈ 1e5 within one order of magnitude.

## 9. Environment and data plan

- Python (system), packages: `numpy scipy cvxpy pandas folktables` (matplolib for figures). `pip install` confirmed working (2026-07-30); `numpy 2.5.1` installed already.
- ACS data: downloaded by folktables from census.gov — endpoint reachable (HTTP 200, 2026-07-30). Cache under `data/` inside the repo folder.
- Determinism: every run records its full config + seed to `results/` JSON; figures regenerate from JSON.
- Compute: single machine, CPU-only, d=10 problems — cheap; CVXPY epigraph with 51 constraints is the largest solve.

## 10. Claims ledger (numbers the gate compares)

| Claim | Value | Citation |
|---|---|---|
| ACS: ball-oracle-euclid iterations to 1% gap | 1 (0.019 s) | `paper/experiments.tex:181` |
| ACS: ball-oracle-lewis iterations to 1% gap | 1 (0.019 s) | `paper/experiments.tex:182` |
| ACS: IPM | 8 (0.066 s) | `paper/experiments.tex:180` |
| ACS: smoothed heavy-ball | 47 (0.062 s) | `paper/experiments.tex:179` |
| ACS: subgradient | not reached | `paper/experiments.tex:178` |
| ACS: ball oracles < 20 ms, ~3× faster than IPM/HB | — | `paper/experiments.tex:152` |
| ACS: ERM avg MSE 108.2, σ 11.8, worst 138.1 (California) | — | `paper/experiments.tex:189` |
| ACS: robust-opt band 107–114, Max/Mean 1.28→1.02, CA −24.3 | — | `paper/experiments.tex:189` |
| Synthetic: κ(stacked Gram) ≈ 1e5; d=10, m=100, 5 adversarial | — | `paper/experiments.tex:38` |
| Synthetic qualitative: FO stall, ball oracles decrease, IPM fastest | — | `paper/experiments.tex:107-109` |

## 11. Resolved open choices (what we picked)

Each item from §6 that the implementation had to fix to a value, with the
chosen value and where it lives in code. (See REPRODUCTION.md for the run
outcomes and discrepancies.)

| §6 item | choice | location |
|---|---|---|
| 1 synthetic recipe | shared-Q eigenbasis; 95 aligned normal groups (geometric eigen-spread, cond~50) + 5 adversarial (distinct sharp dir, L_big calibrated so κ(A^T A)≈1e5; measured 9.7e4); n_per_group=50 | `gdr/data_synth.py` |
| 2 per-group n_i (synthetic) | 50 | `gdr/data_synth.py make_synth` |
| 3 hyperparameter grids | wide geometric grids per arm (steps 1e-9..1e-2 subgradient, β 0.003..0.02 smoothed, β 0.005/0.02 ball); lowest-worst-loss config selected; ties broken by fewest iterations to reach the tuned loss (paper silent on ties) | `gdr/harness.py GRIDS`/`_tune_and_run` |
| 4 1%-reference | base="init" (gap/gap0) for the FINAL line; base="opt" recorded in results JSON | `gdr/harness.py`/`gdr/metrics.py` |
| 5 warm start (synthetic) | ERM | `gdr/harness.py` (x0=problem.erm()) |
| 6 CVXPY solver | default (CLARABEL fallback to SCS/ECOS) | `gdr/solvers.reference_optimum` |
| 7 trust-region internals | Levenberg damping (H+νM)d=-g, M-ellipsoid boundary projection, Armijo backtracking on f~ | `gdr/solvers._solve_trust_region` |
| 8 fhat regulariser | exposed via reg_on (default off in tuned run); coef ε/(1000·min{rank,m}) with ε=4·β·log m (Algorithm-1 form, `paper/body.tex:181-182`) when on | `gdr/solvers.solve_ball_oracle` |
| 9 Lewis constants | n_iters=ceil(2 ln m), exact leverage solves, p=∞ | `gdr/lewis.block_lewis_weights` |
| 10 IPM internals | centring-based log-barrier; Newton decrement<0.5 ⇒ τ←τ(1+growth/√m); barrier0,growth tuned | `gdr/solvers.solve_ipm` |
| 11 ERM definition | group-averaged (E3) | `gdr/types.GroupProblem.erm` |
| 12 ACS target scale | auto-selected so ERM avg MSE≈108.2 (achieved 104.9 with target_scale=1.0); iteration metric scale-invariant | `gdr/data_acs.py` |
| 13 seeds | seed=0 for ACS subsample, seed=0 for synthetic; recorded in run config | `gdr/data_*.py` |
| 16 ball-oracle radii constant | radius0 grid {50, 500}; does not bind (inner Newton converges) | `gdr/harness.py GRIDS` |
| 17 empirical (β,δ) | decoupled from theory; β∈{0.005,0.02} ball, β∈{0.003..0.01} smoothed, δ=0.01 | `gdr/harness.py GRIDS` |
| 18 argmax ties | lowest index | `gdr/solvers.solve_subgradient` |
