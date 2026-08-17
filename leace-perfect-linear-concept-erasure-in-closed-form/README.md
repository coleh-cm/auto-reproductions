# LEACE: Perfect linear concept erasure in closed form — reproduction

Reproduction of **Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman.**
*LEACE: Perfect linear concept erasure in closed form.* NeurIPS 2023. arXiv:2306.03819.

This directory holds the paper sources (`paper/`), the reproduction spec and claim set
(`SPEC.md`, `claims.json`), and the environment needed to run the method and its
experiments. The method itself is the authors' closed-form eraser from
[`EleutherAI/concept-erasure`](https://github.com/EleutherAI/concept-erasure), pinned to
commit `9f51753`; the experiments (bios, amnesic probing, concept scrubbing) are written
against its public interfaces (see `SPEC.md` §4).

## What is installed

| Dependency | Pin | Role |
|---|---|---|
| `concept-erasure` | git `@9f51753` | LEACE / SAL / Oracle erasers, concept scrubber (the method) |
| `torch` | 2.6.0 | tensor core; CPU-only here |
| `numpy` | 2.1.3 | array interop / experiment code |
| `scikit-learn` | 1.6.1 | logistic-regression probes, INLP baselines |
| `transformers` | 4.46.3 | `bert-base-uncased` (bios, amnesic), `EleutherAI/pythia-160m` (scrubbing) |
| `datasets` | 3.1.0 | `bias_in_bios`, UD English EWT |
| `spacy` + `en_core_web_sm` | 3.8.15 / 3.8.0 | Universal-Dependency POS tags for scrubbing |
| `tqdm` | 4.70.0 | progress bars |
| `pytest` | 8.3.4 | tests |

**Note on the POS tagger.** The paper / upstream default is `en_core_web_trf`
(`paper/main.tex:662`), which is CPU-infeasible in this environment. We use
`en_core_web_sm`, the documented fallback (`SPEC.md` §2.7). This is logged as a
divergence; the POS-tag metric definition is unaffected.

## Quickstart

You need Python 3.13 and [`uv`](https://docs.astral.sh/uv/). All commands are run from
this directory.

```bash
# 1. Create the virtual environment and install every pinned dependency.
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt

# 2. Prove the imports resolve and the method works.
. .venv/bin/activate
python -c "import torch, numpy, sklearn, transformers, datasets, spacy, tqdm, pytest; \
from concept_erasure import LeaceEraser; print('imports OK')"

# 3. Smoke-test the method: LEACE must erase a 3-class concept to machine precision
#    (Theorem 4.1: P @ Sigma_XZ = 0, paper/main.tex:457-467).
python -c "import torch; from concept_erasure import LeaceEraser; \
torch.manual_seed(0); X=torch.randn(2000,16,dtype=torch.float64); \
Z=torch.nn.functional.one_hot(torch.randint(0,3,(2000,)),3).double(); \
e=LeaceEraser.fit(X,Z,affine=True,shrinkage=False,svd_tol=1e-9,constrain_cov_trace=False); \
Y=e(X); Xc=X-X.mean(0); Yc=Y-Y.mean(0); Zc=Z-Z.mean(0); \
cov=(Yc.t()@Zc)/(X.shape[0]-1); \
print('|Cov(Y,Z)|_inf =', cov.abs().max().item()); \
assert cov.abs().max().item() < 1e-12; print('LEACE guardedness OK')"

# 4. Run the test suite (once experiment tests are added).
pytest
```

Step 3 should print `|Cov(Y,Z)|_inf` on the order of `1e-16` — the closed-form eraser
annihilates the empirical cross-covariance exactly, which is the paper's central
guarantee.

## Reproduce in Docker (from scratch)

The `Dockerfile` builds the same environment without needing a local Python/uv install:

```bash
docker build -t leace-repro .
docker run --rm -it -v "$PWD":/work -w /work leace-repro bash
# inside the container the environment is already installed system-wide:
python -c "from concept_erasure import LeaceEraser; print('ok')"
pytest
```

## Layout

```
paper/            # arXiv 2306.03819 LaTeX source (authoritative), rendered HTML, figures
SPEC.md           # method as an explicit algorithm, shapes, cited equations, unstated
                  # items, frozen interfaces, arm/restriction analysis
claims.json       # 20 claims, seeds [0,1,2], metrics, arms, restrictions
requirements.txt  # every pinned dependency (this environment)
Dockerfile        # reproducible from-scratch build of this environment
README.md         # this file
REPRODUCTION.md   # running log of the reproduction status
```

## Paper sources

- `paper/main.tex`, `paper/rebuttal.tex` — LaTeX source (authoritative for equation/number
  quotes; cite as `paper/main.tex:<line>`).
- `paper/paper.html` — arXiv rendered HTML (MathML; reading surface).
- `paper/figures/*.pdf` — quantitative figures (transcript: `paper/figure-transcript.md`).
