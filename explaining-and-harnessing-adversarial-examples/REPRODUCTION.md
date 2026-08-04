# Reproduction: EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

**Paper:** Explaining and Harnessing Adversarial Examples
**Authors:** Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (Google Inc.)
**Venue:** ICLR 2015 (arXiv:1412.6572v3, 20 Mar 2015)
**Date started:** 2026-08-04

## Status

**Setup complete — implementation not yet started (fresh run).**

- [x] Reproduction folder created: `explaining-and-harnessing-adversarial-examples/`
- [x] Branch `repro/explaining-and-harnessing-adversarial-examples` checked out (from `main`, which contains a prior merged run of this paper; this run supersedes it)
- [x] PDF-extracted paper text saved to `paper/paper.md`
- [x] arXiv LaTeX source (e-print 1412.6572) unpacked to `paper/source/`; `.tex`/`.bbl` tracked, figures/styles/tarball gitignored
- [ ] SPEC.md (method spec from the paper)
- [ ] Implementation runs end to end
- [ ] Adversarial review rounds clean
- [ ] Readiness gates
- [ ] Numbers compared and published

## Reference notes (from the LaTeX, which is authoritative)

- Preamble macros: `\eps` = `\epsilon` (the perturbation magnitude, invisible in the PDF text), `\sign` = `\text{sign}`, `\vx,\vw,\veta,\vtheta` = bold vectors.
- FGSM perturbation (`paper/source/iclr2015.tex:309`): **η = ε · sign(∇ₓ J(θ, x, y))**
- Adversarial training objective (line ~487): **J̃ = α J(θ, x, y) + (1−α) J(θ, x + ε sign(∇ₓ J(θ, x, y)))**, with α = 0.5 in all experiments.
- Adversarial logistic regression objective: minimize **E ζ(y(ε‖w‖₁ − wᵀx − b))**, ζ softplus.
- Key reported numbers (verified against the .tex):

| Claim | Value |
|---|---|
| Softmax regression, FGSM ε=.25, MNIST | 99.9% error, avg conf 79.3% |
| Maxout, FGSM ε=.25, MNIST | 89.4% error, avg conf 97.6% |
| Conv maxout, FGSM ε=.1, CIFAR-10 | 87.15% error, avg prob 96.6% |
| Logistic regression 3-vs-7, MNIST | 1.6% clean error; 99% on FGSM ε=.25 |
| Maxout dropout, MNIST clean | 0.94% → 0.84% with adversarial training (ε=.25) |
| Large maxout (1600 units) + adv training, 5 seeds | 0.77% ×4, 0.83% ×1 (avg 0.782%) |
| Same large maxout, FGSM after adv training | 17.9% error (was 89.4%) |
| Transfer: adv examples of original → adv-trained model | 19.6% |
| Transfer: adv examples of adv-trained → original model | 40.9% |
| L1 decay .0025 on first layer | too large, >5% train error |
| RBF shallow, FGSM ε=.25, MNIST | 55.4% error, conf on mistakes 1.2% (clean conf 60.6%) |
| MP-DBM, ε=.25 | 97.5% error |
| Ensemble of 12 maxout nets, ε=.25 | 91.1% (whole-ensemble attack) / 87.9% (single-member attack) |
| Rubbish (Gaussian N(0,I₇₈₄)) maxout MNIST | 98.35% error, conf 92.8% |
| Rubbish softmax regression | 59.8% error, conf 70.8%; RBF 0% |
| Cross-model agreement (maxout mistakes): softmax | 54.6% (84.6% when both err) |
| Cross-model agreement (maxout mistakes): RBF | 16.0% (54.3% when both err) |

## Blockers

None yet.
