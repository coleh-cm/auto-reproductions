# Reproduction: Explaining and Harnessing Adversarial Examples

Reproduction of **"Explaining and Harnessing Adversarial Examples"** (Ian J.
Goodfellow, Jonathon Shlens, Christian Szegedy; ICLR 2015, arXiv:1412.6572).

The paper's central method is the **Fast Gradient Sign Method (FGSM)**: given a
model with training cost `J(θ, x, y)`, the worst-case max-norm-constrained
perturbation of an input is, to first order,

```
η = ε · sign( ∇ₓ J(θ, x, y) )        # the fast gradient sign method
x̃ = x + η                            # NOT clipped back to the pixel range
```

so `||η||_∞ == ε` exactly. Applied to a trained model this turns clean
inputs into adversarial inputs with high misclassification rate; mixed into
training (`J̃ = α·J(θ,x,y) + (1−α)·J(θ, x+ε·sign(∇ₓJ(θ,x,y)))`, `α = 0.5`) it
acts as a regularizer. We reproduce the MNIST core of the paper: softmax
regression, logistic regression (3-vs-7), maxout networks, adversarial
training, RBF networks, and the rubbish/fooling examples. See `SPEC.md` for
the full specification (algorithms, shapes, equation citations, and the long
list of choices the paper leaves unstated) and `REPRODUCTION.md` for the
running log and target numbers.

## What this repo contains

- `src/fgsm_repro/` — the implementation (models, attacks, objectives,
  train, eval). *(Added by the implementation step.)*
- `run_experiment.py` — the graded-harness runner. `--baseline` (clean maxout
  training) or `--lambda EPS` (FGSM adversarial training, Algorithm B); prints
  exactly one line `FINAL <arm>=<clean test accuracy>`. `--lambda 0` (the method
  at its no-op) prints the same accuracy VALUE as `--baseline` (the degeneracy
  contract, tested by `tests/test_degeneracy.py`).
- `arms.json` — the gate contract: a flat map `{"baseline": "<cmd>",
  "adversarial": "<cmd>"}` from each arm the gate runs to the shell command
  that produces it. The two arms are the paper's headline M4 comparison
  (tex:492-494). The rich per-milestone metadata is in `arms_metadata.json`,
  and the full 14-arm numbers-gate map (arm → `command_per_seed`, results file,
  metric pointers) is `claims.json`.
- `run_all_arms.sh` — runs **every arm** of `claims.json` (14 arms: M1–M9, E1,
  the L1 control, F4 the Figure-4 eps-curve; m5 and m7 each contribute two arms
  from one command) at
  **every seed** in `claims.json['seeds']` (`[0, 1, 2]`), via `make_measured.py`.
  It writes `measured.json` (`{arm: {seed: {metric: value}}}`) and prints exactly
  one `FINAL <arm>=<value>` line per arm to stdout (BLOCKED if the environment
  cannot produce it). m5's paper-full config (1600 units / patience 100 / 5
  seeds) is infeasible on this CPU, so `make_measured.py` runs it at the
  documented sub-scale (`--units 240 --epochs 12`); the m5 headline *magnitude*
  (0.782%) is rated `compute_invariance=low` in claims.json, the HIGH m5 claim
  is the *direction* (adversarial training ≤ baseline), which the sub-scale
  reproduces at every seed. Runtime ~20–25 min on this CPU; real MNIST
  throughout (no synthetic fallback).
- `make_measured.py` — the harness `run_all_arms.sh` delegates to. Runs each
  arm's `command_per_seed` at each seed (concurrent, writing per-seed result
  files via `--out` so runs never clobber), resolves every metric from each
  result via the `claims.json` `<results json>:<json path>` pointer (handles
  dotted keys like `l1_0.0025` and the `per_seed[*]` array wildcard), and
  writes `measured.json`. `--assemble-only` rebuilds it from the per-seed
  files without re-running.
- `smoke.sh` — the same code path at 200 steps (~3s); a path-prover only,
  never evidence about the paper.
- `experiments/` — one script per milestone: `m1_softmax.py`, `m2_logreg.py`,
  `m3_maxout_fgsm.py`, `m4_adversarial.py`, `m5_large_advtrain.py`,
  `m6_robustness_transfer.py`, `m7_noise_controls.py`, `m8_rbf.py`,
  `m9_rubbish.py`, the L1 weight-decay control `m_l1_weight_decay.py`
  (Section 5), and the extended `e1_ensemble.py`. Each writes a parsed
  result JSON to `results/` with the milestone id, all hyperparameters, seed,
  and the grep-able paper target. Defaults are a documented **sub-scale** for
  CPU feasibility; the CLI exposes the full-scale knobs (e.g.
  `--units 1600 --epochs 100 --patience 100 --seeds 0,1,2,3,4` for M5).
- `results/` — committed result JSONs (one per arm, the seed-0 mirror) plus
  `results/_per_seed/` (every arm × every seed, the inputs to `measured.json`).
  Results are committed, not gitignored: a number whose output file is ignored
  is a claim with its evidence deleted.
- `tests/` — degeneracy, shape, invariant, constructed-truth, instrument, and
  data-fingerprint tests (72 nodes). FGSM on a linear model must equal the
  closed-form max-norm adversary; `||η||_∞ == ε`; `x̃ == x` when `ε == 0`; the
  method at its no-op reproduces the baseline bit-for-bit; E6 is the brute-force
  worst case; the MNIST loader is fingerprinted by sha/vocab/shape.
- `instruments.json` / `mutations.json` — the instrument registry (every grader
  with a positive+negative test) and the deliberate-defect suite (each defect
  with a `must_fail` test node, all verified to fail under the defect and pass
  on clean code).
- `requirements.txt` — pinned dependencies (torch CPU, numpy, pytest, and the
  full transitive closure).
- `Dockerfile` — builds the environment from scratch.
- `SPEC.md` / `REPRODUCTION.md` — specification and reproduction log.
- `paper/paper.md` — the paper text (prose only; maths unreliable).
- `paper/source/` — the authoritative arXiv LaTeX source (`iclr2015.tex`).

## Environment

- Python 3.12 or 3.13 (the Dockerfile builds on `python:3.13-slim`; verified in
  two sandboxes: CPython 3.13.5 on 2026-07-30 and CPython 3.12.13 on 2026-08-04 —
  the pinned wheels resolve identically and the full suite passes on both)
- torch 2.7.1 (CPU build), numpy 2.3.2, pytest 8.4.2
- CPU-only; no GPU required

## Quickstart

### With `uv` (recommended; matches the verified environment)

```bash
# from this reproduction folder (.venv/ is gitignored — recreate it in every fresh sandbox,
# otherwise `.venv/bin/python` does not exist)
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt

# verify the environment imports and the FGSM invariant holds
.venv/bin/python -c "import torch,numpy,pytest; print('env OK', torch.__version__, numpy.__version__, pytest.__version__)"

# run the tests
.venv/bin/python -m pytest -q

# run the gate (both arms of the paper's headline M4 comparison; ~1-2 min)
./run_all_arms.sh          # prints: FINAL baseline=<acc>  and  FINAL adversarial=<acc>

# smoke-prove the code path runs (~3s; NOT a paper result)
./smoke.sh

# run a milestone experiment, e.g. the softmax-regression FGSM arm (M1)
.venv/bin/python experiments/m1_softmax.py
```

### With plain pip + a system Python 3.13+

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python experiments/m1_softmax.py
```

### With Docker

```bash
docker build -t fgsm-repro .
docker run --rm fgsm-repro                       # environment smoke test
# once the implementation step wires the experiment runner:
# docker run --rm fgsm-repro python experiments/m1_softmax.py
```

## Target numbers (MNIST core)

| Milestone | Experiment | Paper target |
|-----------|------------|--------------|
| M1 | Softmax regression, FGSM ε=0.25 | adv error 99.9%, mean conf 79.3% |
| M2 | Logistic regression 3-vs-7, FGSM ε=0.25 | clean 1.6%, adv 99% |
| M3 | Maxout 240×2 + dropout, FGSM ε=0.25 | adv error 89.4%, conf 97.6% |
| M4 | Adversarial training of M3 (α=0.5, ε=0.25) | clean 0.94% → 0.84% |
| M5 | Maxout 1600×2, adv-trained, 5 seeds | 1.14% → mean 0.782% |
| M6 | Robustness/transfer of M5 model | 17.9% / 19.6% / 40.9%, conf 81.4% |
| M7 | Noise-training controls | 86.2% / 90.4% |
| M8 | Shallow RBF, FGSM ε=0.25 | adv 55.4%, conf-on-error 1.2% |
| M9 | Rubbish examples N(0, I₇₈₄) | maxout 98.35%, softmax-reg 59.8%, RBF 0% |
| M-L1 | L1 weight-decay control (Section 5) | coeff 0.0025 → >5% train error; smaller → no benefit |
| F4 | Fig. 4 eps-sweep curve (ε ∈ [−15,15], naive maxout) | curve claims fc1–fc3: correct class crossed, wrong classification stable over ε∈[4,15], logits grow extreme (figure shape, not magnitudes) |

The RBF arms (M8/M9) use the paper's unnormalized per-class `exp(q)` form (a
softmax over `q` is bounded below by 1/K and cannot reproduce the paper's
1.2%/60.6%/0%); `β` is negative-definite by construction (the faithful reading
of E8). The M9 sigmoid-top arm is **trained** (per-class BCE), not a frozen
softmax-to-sigmoid swap. The E1 ensemble metric is the error of the ensemble's
mean-prob prediction. See `SPEC.md` §6 for every unstated choice.

Measured numbers land in `results/<milestone>.json` and are compared against
the grep-able paper lines recorded in `SPEC.md` and `REPRODUCTION.md`.
