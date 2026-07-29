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
- `experiments/` — one script per milestone (`m1_softmax.py` … `m9_rubbish.py`).
- `tests/` — degeneracy + shape tests (e.g. FGSM on a linear model must equal
  the closed-form max-norm adversary; `||η||_∞ == ε`; `x̃ == x` when `ε == 0`).
- `requirements.txt` — pinned dependencies (torch CPU, numpy, pytest, and the
  full transitive closure).
- `Dockerfile` — builds the environment from scratch.
- `SPEC.md` / `REPRODUCTION.md` — specification and reproduction log.
- `paper/paper.md` — the paper text (prose only; maths unreliable).
- `paper/source/` — the authoritative arXiv LaTeX source (`iclr2015.tex`).

## Environment

- Python 3.13
- torch 2.7.1 (CPU build), numpy 2.3.2, pytest 8.4.2
- CPU-only; no GPU required

## Quickstart

### With `uv` (recommended; matches the verified environment)

```bash
# from this reproduction folder
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt

# verify the environment imports and the FGSM invariant holds
.venv/bin/python -c "import torch,numpy,pytest; print('env OK', torch.__version__, numpy.__version__, pytest.__version__)"

# run the tests (once the implementation step has added tests/)
.venv/bin/python -m pytest -q

# run a milestone experiment, e.g. the softmax-regression FGSM arm (M1)
.venv/bin/python experiments/m1_softmax.py
```

### With plain pip + a system Python 3.13

```bash
python3.13 -m venv .venv
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

Measured numbers land in `results/<milestone>.json` and are compared against
the grep-able paper lines recorded in `SPEC.md` and `REPRODUCTION.md`.
