"""Classifier models for FGSM reproduction.

Implements the four model families of Goodfellow, Shlens, Szegedy (2015),
"Explaining and Harnessing Adversarial Examples":
  - SoftmaxRegression (M1)
  - LogisticRegression (M2, binary y in {-1,+1})
  - MaxoutMLP (M3-M7, E1)
  - RBFNet (M8)

Every classifier implements ``logits(x) -> Tensor[B, K]`` returning
pre-softmax scores, plus the standard ``nn.Module`` ``train()``/``eval()``
mode controls (dropout for MaxoutMLP).
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(Protocol):
    """A classifier exposes pre-softmax ``logits`` of shape [B, K]."""

    def logits(self, x: torch.Tensor) -> torch.Tensor:  # [B, F] -> [B, K]
        ...


# ---------------------------------------------------------------------------
# M1: Softmax regression
# ---------------------------------------------------------------------------
class SoftmaxRegression(nn.Module):
    """Linear softmax classifier (M1)."""

    def __init__(self, in_dim: int = 784, n_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


# ---------------------------------------------------------------------------
# M2: Logistic regression (binary, y in {-1, +1})
# ---------------------------------------------------------------------------
class LogisticRegression(nn.Module):
    """Binary logistic regression (M2).

    score(x) = x @ w + b  -> [B];  logits(x) returns [B, 1] for the
    ``Classifier`` Protocol.  ``w`` and ``b`` are exposed attributes.
    """

    def __init__(self, in_dim: int = 784):
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((), dtype=torch.float32))
        nn.init.zeros_(self.w)
        nn.init.zeros_(self.b)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w + self.b  # [B]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        s = self.score(x)  # [B]
        return s.unsqueeze(-1)  # [B, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


# ---------------------------------------------------------------------------
# M3-M7, E1: Maxout MLP with dropout
# ---------------------------------------------------------------------------
class _MaxoutLayer(nn.Module):
    """A single maxout layer.

    a = x @ W + b  -> [B, U*P]; reshape -> [B, U, P]; h = a.max(-1).values.
    """

    def __init__(self, in_dim: int, units: int, pieces: int):
        super().__init__()
        self.in_dim = in_dim
        self.units = units
        self.pieces = pieces
        # pylearn2-style uniform init, irange 0.005 (external default).
        irange = 0.005
        W = torch.empty(in_dim, units * pieces, dtype=torch.float32)
        nn.init.uniform_(W, -irange, irange)
        b = torch.zeros(units * pieces, dtype=torch.float32)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x @ self.W + self.b  # [B, U*P]
        a = a.view(x.shape[0], self.units, self.pieces)  # [B, U, P]
        return a.max(dim=-1).values  # [B, U]


class MaxoutMLP(nn.Module):
    """Maxout network: 2 hidden maxout layers + linear readout (M3-M7, E1).

    Dropout is applied with a per-module ``torch.Generator(seed)``:
      - input dropout (include-prob ``dropout_input_include``) on the input,
      - hidden dropout (include-prob ``dropout_hidden_include``) on the first
        hidden activation.
    Dropout is applied ONLY in ``train()`` mode; ``eval()`` is identity
    (inverted-dropout scaling: mask / include at train time, no rescale at
    eval).  ``set_seed(seed)`` resets the generator.

    External-recipe alignment (pylearn2 ``mnist_pi.yaml``, fetched 2026-07-29):
      - both maxout layers AND the softmax readout use uniform init
        ``irange .005`` with **zero bias** (the recipe sets ``irange: .005``
        on all three layers, including the ``Softmax`` readout ``y``). The
        readout was previously left at PyTorch's default init
        (uniform ±1/sqrt(fan_in) with random bias), which is a deviation from
        the adopted recipe -- now aligned.
      - the recipe's dropout (``input_include_probs: {h0: .8}``,
        ``input_scales: {h0: 1.}``) applies dropout to the INPUT of h0 only
        (i.e. the raw input x), include-prob 0.8, with NON-inverted scaling
        (scale 1.0: train multiplies by the Bernoulli mask with NO 1/include
        division; eval is identity, so eval-time activations are ~1.25x the
        expected train value). Our implementation uses INVERTED dropout
        (mask/include at train, identity at eval) -- the modern standard,
        mathematically equivalent up to a constant eval-time scale. This is a
        documented minor deviation from the external recipe (the recipe is
        not paper-stated). The recipe has NO dropout on h1's input or the
        readout's input; ``dropout_hidden_include`` defaults to 1.0 (off) to
        match, and the full-scale milestone scripts set input include 0.8 /
        hidden include 1.0.
    """

    def __init__(
        self,
        units: int = 240,
        pieces: int = 5,
        in_dim: int = 784,
        n_classes: int = 10,
        dropout_input_include: float = 1.0,
        dropout_hidden_include: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.units = units
        self.pieces = pieces
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.dropout_input_include = float(dropout_input_include)
        self.dropout_hidden_include = float(dropout_hidden_include)
        self.seed = seed

        self.layer0 = _MaxoutLayer(in_dim, units, pieces)
        self.layer1 = _MaxoutLayer(units, units, pieces)
        # Readout: pylearn2 recipe uses irange .005 + zero bias on the Softmax
        # layer y (aligned; previously PyTorch default ±1/sqrt(fan_in)+rand bias).
        self.readout = nn.Linear(units, n_classes)
        with torch.no_grad():
            nn.init.uniform_(self.readout.weight, -0.005, 0.005)
            nn.init.zeros_(self.readout.bias)

        self.gen = torch.Generator()
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """Reset the per-module dropout generator."""
        self.seed = int(seed)
        self.gen.manual_seed(self.seed)

    def _dropout_mask(self, shape, include: float) -> torch.Tensor:
        """Inverted-dropout mask: Bernoulli(include) / include at train time."""
        if include >= 1.0:
            return None
        mask = (torch.rand(shape, generator=self.gen, dtype=torch.float32) < include).float()
        return mask / include

    def _apply_dropout(self, x: torch.Tensor, include: float) -> torch.Tensor:
        if not self.training or include >= 1.0:
            return x
        mask = self._dropout_mask(x.shape, include)
        if mask is None:
            return x
        return x * mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        h_in = self._apply_dropout(x, self.dropout_input_include)
        h0 = self.layer0(h_in)
        h0 = self._apply_dropout(h0, self.dropout_hidden_include)
        h1 = self.layer1(h0)
        return self.readout(h1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


# ---------------------------------------------------------------------------
# M9: Maxout trunk with independent sigmoid top (rubbish sigmoid-top arm)
# ---------------------------------------------------------------------------
class SigmoidTopMLP(nn.Module):
    """Maxout trunk + independent per-class SIGMOID top (M9, tex:908-909).

    The paper's rubbish appendix contrasts a maxout+softmax net with the same
    net whose top layer is changed to "independent sigmoids" (tex:908-909,
    "Changing the top layer to independent sigmoids dropped the error rate to
    68%"). Each class k has an independent logistic output
    p(y=k|x) = sigmoid(readout_k(h1)); a rubbish sample is an "error" iff ANY
    class's sigmoid probability > 0.5 (tex:906 "assigning a probability greater
    than 0.5 to any class"). ``logits`` returns the raw pre-sigmoid readout
    scores [B, K]; callers apply sigmoid per class. The trunk (two maxout
    layers + input dropout) is identical to ``MaxoutMLP`` so the comparison
    isolates the top layer.
    """

    def __init__(
        self,
        units: int = 240,
        pieces: int = 5,
        in_dim: int = 784,
        n_classes: int = 10,
        dropout_input_include: float = 1.0,
        dropout_hidden_include: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.trunk = MaxoutMLP(
            units=units, pieces=pieces, in_dim=in_dim, n_classes=n_classes,
            dropout_input_include=dropout_input_include,
            dropout_hidden_include=dropout_hidden_include, seed=seed,
        )
        # Reuse the trunk's two maxout layers but REPLACE its softmax readout
        # with an independent-sigmoid readout (same irange .005 / zero bias).
        self.readout = nn.Linear(units, n_classes)
        with torch.no_grad():
            nn.init.uniform_(self.readout.weight, -0.005, 0.005)
            nn.init.zeros_(self.readout.bias)

    def set_seed(self, seed: int) -> None:
        self.trunk.set_seed(seed)

    def _hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Run the trunk up to (but excluding) the readout -> [B, units]."""
        m = self.trunk
        h_in = m._apply_dropout(x, m.dropout_input_include)
        h0 = m.layer0(h_in)
        h0 = m._apply_dropout(h0, m.dropout_hidden_include)
        h1 = m.layer1(h0)
        return h1

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid readout scores [B, K]. Apply sigmoid for per-class probs."""
        return self.readout(self._hidden(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


# ---------------------------------------------------------------------------
# M8: RBF network
# ---------------------------------------------------------------------------
class RBFNet(nn.Module):
    """Shallow RBF network (M8).

    Per class k a quadratic form
        q_k(x) = (x - mu_k)^T beta_k (x - mu_k)     [B]   (NO minus sign, E8)
    with mu [K, F] and beta [K, F, F].  Multiclass normalization (unstated by
    the paper, which prints only the binary form) is taken as softmax over
    the K quadratic forms, so ``logits(x)`` returns q(x) directly [B, K].
    """

    def __init__(self, n_classes: int = 10, in_dim: int = 784):
        super().__init__()
        self.n_classes = n_classes
        self.in_dim = in_dim
        mu = 0.01 * torch.randn(n_classes, in_dim, dtype=torch.float32)
        self.mu = nn.Parameter(mu)
        beta = -0.01 * torch.eye(in_dim, dtype=torch.float32)
        beta = beta.unsqueeze(0).expand(n_classes, in_dim, in_dim).contiguous()
        self.beta = nn.Parameter(beta.clone())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # diff: [B, K, F]
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)
        # temp = diff @ beta_k  : [B, K, F]
        temp = torch.einsum("bkf,kfg->bkg", diff, self.beta)
        # quad_k = diff * temp summed over F : [B, K]
        quad = torch.einsum("bkf,bkf->bk", diff, temp)
        return quad  # softmax over these = p(y=k|x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)
