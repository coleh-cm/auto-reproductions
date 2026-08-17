"""MidSteer closed-form affine steering core (SPEC.md section 6).

This package implements the paper's training-free affine maps:
  LEACE (Eq.6, paper/content/guardedness.tex:74-83),
  LEACE-Switch (Eq.13, paper/main.tex:363-367),
  MidSteer (Eq.19/23, paper/main.tex:445-448 / :478-481),
  vanilla Householder/projection (Eqs.21/24/25, paper/main.tex:466-469,
  paper/content/suppl.tex:63-73),
with the Welford online covariance estimator (Algorithm 1,
paper/content/suppl.tex:11-52). All covariance/affine algebra is float64.
"""
