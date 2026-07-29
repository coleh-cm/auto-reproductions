# Reproduction: Confidence-Weighted Self-Distillation for Learning under Label Noise

Reproduction of **"Confidence-Weighted Self-Distillation for Learning under
Label Noise"** (A. Bergstrom, M. Oyelaran, K. Vasquez).

CWSD replaces the one-hot training target with a convex mixture of the label
and the model's own temperature-softened prediction, where the mixing weight is
gated by the model's confidence on that example. Confident predictions are
trusted and allowed to override the (possibly noisy) label; unconfident ones
fall back to supervision. Setting the mixing coefficient `λ = 0` recovers
standard cross-entropy exactly, which makes the method straightforward to
verify.

Both arms (baseline at `λ = 0` and CWSD at `λ = 1`) are the same program under
a different `--lambda` flag.

## Method in one paragraph

Let `p = softmax(f_θ(x))` and `c = max_k p_k` (Eq. 1). The mixing weight
`w = λ·σ((c − τ)/s)` (Eq. 2) is computed per example. The training target is
`t = (1 − w)·y + w·p̃` where `p̃ = stopgrad(softmax(z/T))` (Eq. 3). The loss is the
cross-entropy between `t` (treated as a constant) and `p`, `L = −(1/B)Σ_iΣ_k
t_ik log p_ik` (Eq. 4), with gradient `dL/dz = (p − t)/B`. Optimiser: vanilla
SGD, lr 0.1, batch 64, 4000 steps. See `SPEC.md` for the full specification
including every choice the paper leaves unstated (notably the gate sharpness
`s`, which has no value anywhere in the paper).

## What this repo contains

- `run_experiment.py` — the single-file experiment runner (numpy + scikit-learn
  only; gradients hand-derived so the stop-gradient semantics are structural).
- `requirements.txt` — pinned dependencies.
- `Dockerfile` — builds the environment from scratch.
- `SPEC.md` — algorithm spec, shapes, equation citations, unstated-items list.
- `REPRODUCTION.md` — running log and target numbers.
- `paper/paper.md` — the paper text, verbatim.

## Target numbers (Table 1)

| Method                     | λ | Paper accuracy |
|----------------------------|---|----------------|
| Cross-entropy (baseline)   | 0 | 0.9370         |
| CWSD (ours)                | 1 | 0.9620          |

Reproduced here (seed 0, defaults): baseline `0.9315`, CWSD `0.9519` at the
default gate sharpness `s=0.05` (CWSD peaks at `0.9574` for `s ∈ {0.08, 0.15,
0.2}`). The **~2.5-point CWSD-over-baseline improvement is reproduced**. The
small absolute offsets from Table 1 are expected: the paper does not state the
RNG stream layout, the weight-initialisation scheme, or the gate sharpness
`s` (see `SPEC.md` §4), so bitwise-exact reproduction of the paper's numbers is
impossible — only statistical reproduction. The structural gate (`λ = 0` ⟹
target equals the one-hot label exactly) holds.

## Quickstart

### With `uv` (recommended; matches the verified environment)

```bash
# from this reproduction folder
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt

# baseline cross-entropy (lambda = 0)
.venv/bin/python run_experiment.py --lambda 0.0
# CWSD (lambda = 1)
.venv/bin/python run_experiment.py --lambda 1.0
```

Each run prints exactly one line on completion: `FINAL accuracy=<float>`.

### With plain pip + a system Python 3.13

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiment.py --lambda 0.0   # baseline
.venv/bin/python run_experiment.py --lambda 1.0   # CWSD
```

### With Docker

```bash
docker build -t cwsd .
docker run --rm cwsd                          # baseline (lambda 0)
docker run --rm cwsd --lambda 1.0             # CWSD
```

## Interface

```
python run_experiment.py --lambda FLOAT   # 0.0 = baseline CE, 1.0 = CWSD (required)
                         [--s 0.05]        # gate sharpness (Eq. 2); paper does not state it
                         [--tau 0.9]        # confidence threshold
                         [--temperature 2.0]
                         [--seed 0] [--steps 4000] [--lr 0.1] [--batch-size 64]
                         [--init he] [--noise-mode uniform-all]
                         [--batch-mode epoch-permutation]
```

Output contract: exactly one line on stdout, `FINAL accuracy=<float>` formatted
`%.4f`. All diagnostics go to stderr.

## Running the tests

```bash
.venv/bin/python -m pytest -q
```

## Environment

- Python 3.13
- numpy 2.5.1, scikit-learn 1.9.0, scipy 1.18.0, joblib 1.5.3,
  threadpoolctl 3.6.0, narwhals 2.24.0
- pytest 9.1.1
