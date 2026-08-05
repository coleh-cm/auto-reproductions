# Reproduction: Explaining and Harnessing Adversarial Examples

Reproduction of **Goodfellow, Shlens & Szegedy, "Explaining and Harnessing
Adversarial Examples", ICLR 2015** (arXiv:1412.6572v3).

The paper introduces the **fast gradient sign method (FGSM)** —

```
η = ε · sign(∇ₓ J(θ, x, y)) ,   x̃ = x + η
```

— and **FGSM adversarial training** —

```
J̃(θ, x, y) = α J(θ, x, y) + (1 − α) J(θ, x + ε sign(∇ₓ J(θ, x, y)), y) ,   α = 0.5
```

This repo re-implements both from scratch (no usable author code exists — the
paper's only code link is the dead, Theano/Pylearn2-based CIFAR preprocessing)
and checks them against the paper's reported numbers on MNIST and CIFAR-10.

> **Framework choice (ours, the paper is silent):** PyTorch (CPU). FGSM needs
> the gradient of the cost w.r.t. the *input*; `torch.autograd.grad(loss, x)`
> provides it. See `SPEC.md` §0 for why no upstream code is reused.

## What runs

| Arm | Dataset | What it does | Status |
|---|---|---|---|
| `softmax_reg` | MNIST | linear softmax, FGSM ε=.25, rubbish | runs (3 seeds) |
| `logreg_3v7` | MNIST 3v7 | logistic regression, FGSM, analytic-equivalence (c07) | runs (3 seeds) |
| `maxout_naive` / `maxout_adv` | MNIST | 240-unit maxout MLP, ±FGSM adversarial training | runs (3 seeds) |
| `maxout_large_naive` / `maxout_large_adv` | MNIST | 1600-unit maxout, ±adv training (5 seeds for adv) | runs (sub-scale: 6 epochs, no 60k retrain) |
| `maxout_sigmoid` | MNIST | maxout + independent sigmoid top, rubbish | runs (3 seeds) |
| `noise_rademacher` / `noise_uniform` | MNIST | ±ε / U(−ε,ε) noise controls | runs (3 seeds) |
| `l1_maxout` | MNIST | L¹ weight-decay control (coef .0025, first layer) | runs (3 seeds) |
| `rbf_shallow` | MNIST | 10 RBF units, FGSM, rubbish | runs (3 seeds) |
| `ensemble12` | MNIST | 12-member mean-prob ensemble | runs (3 seeds) |
| `agreement_mnist` | MNIST | cross-model label agreement on FGSM | runs (3 seeds) |
| `transfer_mnist` | MNIST | FGSM transfer between large naive ↔ large adv | runs (3 seeds) |
| `eps_trace` | MNIST | Figure 4 ε-sweep logit curve | runs (3 seeds) |
| `cifar_conv_maxout` | CIFAR-10 | conv maxout, FGSM ε=.1, rubbish, fooling | runs (3 seeds, real CIFAR-10; sub-scale: 25 epochs) |

Not built (see SPEC §9): MP-DBM, GoogLeNet/ImageNet Fig. 1 demo.

## Self-check grader

`selfcheck_claims.py` is this reproduction's OWN grader: it evaluates
`claims.json` (70 claims, 14 high-invariance) against `measured.json` and writes
`selfcheck.json` (with a `produced_by` stamp — never hand-authored). It does
**not** write `claims_result.json`; that filename is owned by the workflow's
numbers gate, and a script here writing it would collide and be refused.
Result on this run: **14/14 HIGH pass, 0 fail, 0 blocked — gate PASS**. The
`low`/`medium` value fails are the tight claims (clean 0.94%, 0.782%, RBF /
softmax-rubbish numbers whose training the paper never states), c12 (the 0.1pp
clean-err reduction below the sub-scale horizon), c62 (frog&truck fooling on
a sub-scale conv net), and c66 (Fig.4 negative-tail thin-manifold on the
deterministic example) — see REPRODUCTION.md. 0 claims are blocked or
unevaluable.

## Quickstart

You need Python 3.13 and [`uv`](https://github.com/astral-sh/uv) (Astral's
Python package manager). On Debian/Ubuntu:

```bash
curl -fsSL https://astral.sh/uv/install.sh | sh
```

### 1. Create the environment (matches the gate exactly)

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt
```

### 2. Sanity-check the imports

```bash
.venv/bin/python -c "import torch, numpy, matplotlib, pytest; \
print('torch', torch.__version__, '| numpy', numpy.__version__, \
'| matplotlib', matplotlib.__version__, '| pytest', pytest.__version__)"
```

Expected (CPU torch):

```
torch 2.7.1+cpu | numpy 2.3.2 | matplotlib 3.11.1 | pytest 8.4.2
```

### 3. Run the tests

```bash
.venv/bin/pytest -q
```

### 4. Run the arms and check the paper's numbers

```bash
.venv/bin/python -m run_all_arms        # trains every arm x seed -> measured.json
.venv/bin/python selfcheck_claims.py   # evaluates claims.json against measured.json -> selfcheck.json
```

Datasets (MNIST, CIFAR-10) download on first run into `./data/` (gitignored).

## Reproduce in Docker (no local tooling needed)

```bash
docker build -t eae-repro .
docker run --rm -it eae-repro pytest -q
docker run --rm -it eae-repro python -m run_all_arms
```

The image is CPU-only; the paper's MNIST / CIFAR-10 experiments are small
enough to run on CPU.

## Notes on reproducibility / our choices

- **No clipping** of `x̃` back into `[0,1]` (the paper never states any; see
  `SPEC.md` §4.9).
- **sign(0) := 0** (`SPEC.md` §4.22).
- Training hyperparameters the paper defers to the (dead) maxout-paper config
  are filled in by us and logged in `SPEC.md` §4 — every such fill is marked
  **our choice, not the paper's**.
- Seeds: `seeds = [0, 1, 2]` for most arms; the 1600-unit adversarially-trained
  maxout uses five seeds `[0..4]`, matching the paper's five runs
  (`paper/source/iclr2015.tex:506-510`). Per seed there are three independent
  RNG streams — weight init, minibatch order, dropout masks.

See `REPRODUCTION.md` for the run log and `SPEC.md` for the full method spec.
