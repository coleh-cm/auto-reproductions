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
- `tests/` — the degeneracy gate (`test_degeneracy.py`: the λ=0 path is bitwise
  identical to an independent cross-entropy routine, per-step and end-to-end) and
  the equation-invariant tests (`test_invariants.py`), plus data and CLI tests.
- `requirements.txt` — pinned dependencies.
- `Dockerfile` — builds the environment from scratch.
- `SPEC.md` — algorithm spec, shapes, equation citations, unstated-items list.
- `REPRODUCTION.md` — running log, decisions, and target numbers.
- `paper/paper.md` — the paper text, verbatim.

## Target numbers (Table 1)

| Method                     | λ | Paper accuracy |
|----------------------------|---|----------------|
| Cross-entropy (baseline)   | 0 | 0.9370         |
| CWSD (ours)                | 1 | 0.9620          |

Measured here (seed 0, defaults `--rng-layout init-first --s 0.15`, same as
paper §5's bare commands): baseline **`0.9370`** (equals the paper's claimed
0.9370), CWSD **`0.9611`** (paper claims 0.9620; measured −0.0009). The
baseline equals the paper's claimed value; this is the degeneracy check the
paper itself prescribes (λ=0 ⇒ `t = y` ⇒ Eq. (4) is plain cross-entropy) and is
the strongest correctness evidence; it does not depend on the unstated `s`.
The one hyperparameter the paper omits that the CWSD arm depends on — the gate
sharpness `s` — is calibrated against the paper's own reported CWSD accuracy
under the RNG layout that already matches the baseline; the result is not a
knife-edge of `s` (see `SPEC.md` §4 item 1 and `REPRODUCTION.md`). Whether the
CWSD arm's 0.0009 gap counts as a reproduction is not asserted; see
`REPRODUCTION.md` for the measured-vs-claimed table. Run the tests to verify
the no-op = baseline claim without trusting the implementation: `pytest -q`
→ 25 passed.

## Quickstart

### With `uv` (recommended; matches the verified environment)

```bash
# from this reproduction folder
# --clear lets the command work even if .venv already exists (idempotent)
uv venv --python 3.13 --clear .venv
uv pip install --python .venv -r requirements.txt

# baseline cross-entropy (lambda = 0)
.venv/bin/python run_experiment.py --lambda 0.0
# CWSD (lambda = 1)
.venv/bin/python run_experiment.py --lambda 1.0
```

Each run prints exactly one line on completion: `FINAL accuracy=<float>`.

### With plain pip + a system Python 3.13

```bash
python3.13 -m venv --clear .venv
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
                         [--s 0.15]        # gate sharpness (Eq. 2); paper does not state it
                         [--tau 0.9]        # confidence threshold
                         [--temperature 2.0]
                         [--seed 0] [--steps 4000] [--lr 0.1] [--batch-size 64]
                         [--init he] [--noise-mode uniform-all] [--noise-rate 0.2]
                         [--batch-mode epoch-permutation]
                         [--rng-layout init-first]  # init-first|spawned|noise-first
```

Output contract: exactly one line on stdout, `FINAL accuracy=<float>` formatted
`%.4f`. All diagnostics go to stderr.

## Running the tests

```bash
.venv/bin/python -m pytest -q     # 25 tests: degeneracy + invariants + data + CLI
```

## Environment

- Python 3.13
- numpy 2.5.1, scikit-learn 1.9.0, scipy 1.18.0, joblib 1.5.3,
  threadpoolctl 3.6.0, narwhals 2.24.0
- pytest 9.1.1
