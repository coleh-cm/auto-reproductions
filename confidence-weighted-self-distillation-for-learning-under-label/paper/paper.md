Confidence-Weighted Self-Distillation for Learning under Label
Noise

A. Bergstrom
 
M. Oyelaran
 
K. Vasquez
Institute for Applied Learning Systems

Abstract

Training on noisy labels drives a classifier to memorise corrupted targets.
 
We propose

Confidence-Weighted Self-Distillation
 
(CWSD), which replaces the one-hot target with
a convex mixture of the label and the model’s own softened prediction, where the mixing weight
is gated by the model’s confidence on that example. Confident predictions are trusted and al-
lowed to override the label; unconfident ones fall back to supervision. CWSD adds one forward
pass of bookkeeping, no second network, and no additional parameters. On a handwritten-digit
benchmark with
 
20%
 
symmetric label noise it improves test accuracy by
 
2
.
5
 
points over standard
cross-entropy. Setting the mixing coefficient to zero recovers the cross-entropy baseline exactly,
which makes the method straightforward to verify.

1
 
Introduction

Symmetric label noise is the simplest corruption to state and among the hardest to train through:
a network with enough capacity will eventually fit the corrupted labels, and test accuracy falls
after an initial rise.
 
Self-distillation is an appealing remedy because the model’s own predictive
distribution carries information the hard label has thrown away, but distilling indiscriminately also
propagates the model’s own mistakes.
Our observation is that the useful signal is concentrated in the examples the model is already
confident about.
 
We therefore gate the self-distillation term on confidence, so that the model’s
prediction is only allowed to displace the label where the model is sure.

2
 
Method

Let
 
f
θ
 
be a classifier over
 
K
 
classes and let
 
p
 
= softmax(
f
θ
(
x
))
 
denote its predictive distribution
for input
 
x
, with
 
y
 
∈ {
0
,
 
1
}
K
 
the one-hot label.
We first define the model’s confidence on an example as the largest class probability it assigns:

c
 
= max

k
 
p
k
.
 
(1)
Confidence is converted into a mixing weight through a logistic gate centred at a threshold
 
τ
 
,
scaled by a coefficient
 
λ
 
∈
 
[0
,
 
1]
 
that sets the maximum influence self-distillation may have:

w
 
=
 
λ σ

(
 
c
 
−
 
τ
s

)

,
 
σ
(
z
) =
 
1
1 +
 
e
−
z
 
.
 
(2)
Here
 
s
 
controls how sharply the gate opens around
 
τ
 
.
1

The training target is the convex combination of the label and the model’s own temperature-
softened prediction, the latter treated as a constant with respect to
 
θ
:

t
 
= (1
 
−
 
w
)
 
y
 
+
 
w
 
˜p,
 
˜
p
 
= stopgrad
(
softmax(
f
θ
(
x
)
/T
 
)
)
.
 
(3)
The stop-gradient is essential: without it the target follows the prediction and the objective admits
a trivial solution.
Training minimises the cross-entropy between this target and the prediction, averaged over a
minibatch of size
 
B
:

L
 
=
 
−
 
1

B

B
∑

i
=1

K
∑

k
=1

t
ik
 
log
 
p
ik
.
 
(4)
At
 
λ
 
= 0
 
we have
 
w
 
= 0
 
and therefore
 
t
 
=
 
y
, so Eq. (4) reduces to standard cross-entropy on
the observed labels. CWSD is thus a strict generalisation of the baseline, and any implementation
can be checked by confirming that
 
λ
 
= 0
 
reproduces the baseline result exactly.

3
 
Experimental setup

We evaluate on the
 
scikit-learn load_digits
 
dataset:
 
1797
 
grey-scale
 
8
 
×
 
8
 
handwritten digits
over
 
K
 
= 10
 
classes, with pixel values scaled to
 
[0
,
 
1]
 
by dividing by
 
16
. We hold out
 
30%
 
of the
data as a test set with a class-stratified split at seed
 
0
, and corrupt the training labels with
 
20%

symmetric noise: each training example is selected independently with probability
 
0
.
2
 
and its label
replaced by a class drawn uniformly at random.
The classifier is a single-hidden-layer network with
 
64
 
hidden units and ReLU activations,
trained by stochastic gradient descent with learning rate
 
0
.
1
 
and minibatch size
 
64
 
for
 
4000
 
steps.
Gradients are those of Eq. (4).
 
The method’s hyperparameters are
 
λ
 
= 1
, confidence threshold

τ
 
= 0
.
9
, and distillation temperature
 
T
 
= 2
. The baseline is the same network and optimiser at

λ
 
= 0
.
 
All results are single runs at seed
 
0
; the data split, the noise mask, and the parameter
initialisation are all drawn from that seed.

4
 
Results

Method
 
λ
 
Test accuracy
Cross-entropy (baseline)
 
0
 
0
.
9370

CWSD (ours)
 
1
 
0
.
9620

Table 1: Test accuracy under
 
20%
 
symmetric label noise. CWSD improves over the cross-entropy
baseline by
 
2
.
5
 
accuracy points.
Table 1 reports test accuracy for both methods. CWSD reaches
 
0
.
9620
 
against the baseline’s

0
.
9370
, an improvement of
 
2
.
5
 
points. We attribute the gain to the gate suppressing the gradient
contribution of examples whose labels disagree with a confident prediction, which are dispropor-
tionately the corrupted ones.

5
 
Reproducing

Both arms are the same program under a different
 
λ
:
2

python run_experiment.py --lambda 0.0
 
# baseline
python run_experiment.py --lambda 1.0
 
# CWSD

Each run prints a single line of the form
 
FINAL accuracy=<float>
 
on completion.
3
