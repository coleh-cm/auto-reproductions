"""Models for the EAE reproduction.

Every model is an nn.Module exposing:
  .logits(x[B,D])      -> [B,K] f32  (scores; RBF/Ensemble: per-class exp-quadratic)
  .prob(x[B,D])        -> [B,K] f32  (softmax-top: rows sum to 1; RBF: rows need NOT sum to 1)
  .loss(x[B,D], y[B])  -> scalar f32
  .predict(x[B,D])     -> [B] int64
  .confidence(x[B,D])  -> [B] f32  (max prob; logreg_3v7: sigmoid(margin))

Paper equations (paper/source/iclr2015.tex):
  softmax NLL            tex:306
  logistic softplus       tex:399-404
  RBF exp-quadratic      tex:595  (beta negative-semidefinite; printed eq lacks the minus sign)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check(x, name="x"):
    if x is None:
        raise ValueError(f"{name} is None")
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() < 2:
        if x.numel() == 0:
            raise ValueError(f"empty {name}")
        x = x.unsqueeze(0)
    if x.numel() == 0:
        raise ValueError(f"empty {name}")
    return x.float()


# ---------------------------------------------------------------------------
# Softmax regression
# ---------------------------------------------------------------------------
class SoftmaxRegression(nn.Module):
    def __init__(self, D=784, K=10):
        super().__init__()
        self.D, self.K = D, K
        self.linear = nn.Linear(D, K)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def logits(self, x):
        x = _check(x)
        return self.linear(x)

    def prob(self, x):
        return F.softmax(self.logits(x), dim=-1)

    def loss(self, x, y):
        logits = self.logits(x)
        y = y.to(torch.long).view(-1)
        return F.cross_entropy(logits, y)

    def predict(self, x):
        return self.prob(x).argmax(dim=-1).to(torch.long)

    def confidence(self, x):
        return self.prob(x).max(dim=-1).values


# ---------------------------------------------------------------------------
# Binary logistic regression for 3-vs-7
#   P(y=1) = sigma(w.x + b);  y in {-1,+1};  loss = mean zeta(-y*(w.x+b)),
#   zeta(z) = log(1+e^z)  (tex:399-404)
# ---------------------------------------------------------------------------
class LogisticRegression3v7(nn.Module):
    def __init__(self, D=784):
        super().__init__()
        self.D = D
        self.linear = nn.Linear(D, 1)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def margin(self, x):
        x = _check(x)
        return self.linear(x).squeeze(-1)  # [B]

    def logits(self, x):
        # K=1 binary; return [B,1] for API compatibility
        return self.margin(x).unsqueeze(-1)

    def prob(self, x):
        return torch.sigmoid(self.margin(x)).unsqueeze(-1)  # [B,1] P(y=+1)

    def loss(self, x, y):
        m = self.margin(x)
        yf = y.to(torch.float32).view(-1)
        return F.softplus(-yf * m).mean()

    def predict(self, x):
        m = self.margin(x)
        return torch.where(m > 0, torch.tensor(1, dtype=torch.long, device=m.device),
                          torch.tensor(-1, dtype=torch.long, device=m.device))

    def confidence(self, x):
        return torch.sigmoid(self.margin(x))  # [B] confidence on predicted class


# ---------------------------------------------------------------------------
# Maxout MLP (Goodfellow et al. 2013c; tex:498 "240/1600 units per layer")
# Paper silent on pieces/layers/dropout rates (SPEC §4.2); our choices:
#   layers=2, pieces=5, dropout input=0.2 / hidden=0.5 (maxout-paper MNIST config).
# ---------------------------------------------------------------------------
class MaxoutLinear(nn.Module):
    """A maxout layer: max over `pieces` affine slices (no nonlinearity after)."""

    def __init__(self, in_features, out_features, pieces):
        super().__init__()
        self.in_features, self.out_features, self.pieces = in_features, out_features, pieces
        self.weight = nn.Parameter(torch.empty(pieces, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(pieces, out_features))
        nn.init.normal_(self.weight, std=(1.0 / in_features) ** 0.5 * 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [B, in_features]
        z = torch.einsum("bi,pio->bpo", x, self.weight) + self.bias  # [B, pieces, out]
        return z.max(dim=1).values  # [B, out]


class MaxoutMLP(nn.Module):
    def __init__(self, units_per_layer, layers=2, pieces=5, dropout={"input": 0.2, "hidden": 0.5}, D=784, K=10):
        super().__init__()
        self.D, self.K = D, K
        self.units, self.layers, self.pieces = units_per_layer, layers, pieces
        self.dropout = dict(dropout)
        self.blocks = nn.ModuleList()
        in_f = D
        for _ in range(layers):
            self.blocks.append(MaxoutLinear(in_f, units_per_layer, pieces))
            in_f = units_per_layer
        self.head = nn.Linear(in_f, K)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        x = _check(x)
        p_in = self.dropout.get("input", 0.0)
        if p_in > 0 and self.training:
            x = F.dropout(x, p=p_in, training=True)
        for i, block in enumerate(self.blocks):
            x = block(x)
            p_h = self.dropout.get("hidden", 0.0)
            if p_h > 0 and self.training and i < len(self.blocks) - 1:
                x = F.dropout(x, p=p_h, training=True)
        return x

    def logits(self, x):
        h = self.forward_features(x)
        return self.head(h)

    def prob(self, x):
        return F.softmax(self.logits(x), dim=-1)

    def loss(self, x, y):
        return F.cross_entropy(self.logits(x), y.to(torch.long).view(-1))

    def predict(self, x):
        return self.prob(x).argmax(dim=-1).to(torch.long)

    def confidence(self, x):
        return self.prob(x).max(dim=-1).values


def maxout_naive(**kw):
    return MaxoutMLP(240, layers=2, pieces=5, dropout={"input": 0.2, "hidden": 0.5}, **kw)


def maxout_large(**kw):
    return MaxoutMLP(1600, layers=2, pieces=5, dropout={"input": 0.2, "hidden": 0.5}, **kw)


# ---------------------------------------------------------------------------
# Maxout backbone + 10 INDEPENDENT sigmoid outputs (tex:909)
#   rubbish eval uses "any class prob > 0.5" -> rows need NOT sum to 1.
# ---------------------------------------------------------------------------
class MaxoutSigmoid(nn.Module):
    def __init__(self, units_per_layer=240, layers=2, pieces=5,
                 dropout={"input": 0.2, "hidden": 0.5}, D=784, K=10):
        super().__init__()
        self.D, self.K = D, K
        self.backbone = MaxoutMLP(units_per_layer, layers, pieces, dropout, D=D, K=1)
        in_f = units_per_layer
        self.head = nn.Linear(in_f, K)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def logits(self, x):
        # pre-sigmoid activations
        return self.head(self.forward_features(x))

    def prob(self, x):
        return torch.sigmoid(self.logits(x))  # [B,K] independent, rows need NOT sum to 1

    def loss(self, x, y):
        # independent binary BCE per class (one-hot targets)
        logits = self.logits(x)
        y = y.to(torch.long).view(-1)
        target = F.one_hot(y, num_classes=logits.shape[-1]).float()
        return F.binary_cross_entropy_with_logits(logits, target)

    def predict(self, x):
        return self.prob(x).argmax(dim=-1).to(torch.long)

    def confidence(self, x):
        return self.prob(x).max(dim=-1).values


# ---------------------------------------------------------------------------
# Shallow RBF network (tex:595).
#   p_k = exp((x - mu_k)^T beta_k (x - mu_k));  beta_k = -psi_k psi_k^T - nu*I  (NSD)
#   predict = argmax_k p_k;  confidence = max_k p_k;  loss = -log p_{y} (clamped).
# Paper silent on the sign of beta (printed eq lacks the needed minus sign;
# SPEC §4.4) and on training — our choices.
# ---------------------------------------------------------------------------
class RBFNet(nn.Module):
    def __init__(self, K=10, D=784, rank=16, nu=0.01, nu_trainable=False):
        super().__init__()
        self.D, self.K = D, K
        self.rank = rank
        self.mu = nn.Parameter(torch.zeros(K, D))
        nn.init.normal_(self.mu, std=0.1)
        self.psi = nn.Parameter(torch.randn(K, D, rank) * 0.01)
        # nu is a FLOOR on the quadratic decay (fixed by default). Without a
        # positive floor the NLL training collapses: psi,nu -> 0, beta -> 0,
        # quad -> 0, prob -> 1 everywhere (clean_conf ~100%, rubbish ~100%),
        # which contradicts the paper's "naturally immune" RBF (clean conf 60.6%,
        # mistake conf 1.2%, rubbish 0%). The paper never states the training
        # procedure; fixing nu>0 is our choice (SPEC §4.4) to keep the units
        # localized. nu=0.01 gives ~e^{-1} decay at ||x-mu||^2 ~ 100 (digit scale).
        if nu_trainable:
            self.nu = nn.Parameter(torch.tensor(float(nu)))
        else:
            self.register_buffer("nu", torch.tensor(float(nu)))
        self.log_temp = nn.Parameter(torch.zeros(K))

    def _quad(self, x):
        x = _check(x)
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)  # [B, K, D]
        psi = self.psi  # [K, D, rank]
        psi_d = torch.einsum("kdr,bkd->bkr", psi, diff)  # [B, K, rank]
        nu = self.nu if isinstance(self.nu, torch.Tensor) else torch.tensor(self.nu)
        quad = -(psi_d ** 2).sum(-1) - float(nu) * (diff ** 2).sum(-1)  # [B, K] <= 0
        return quad

    def logits(self, x):
        quad = self._quad(x)  # [B, K] <= 0
        return quad - torch.relu(self.log_temp).unsqueeze(0)  # <= 0

    def prob(self, x):
        return torch.exp(self.logits(x))  # [B, K] in (0, 1], rows need NOT sum to 1

    def loss(self, x, y):
        logp = self.logits(x)  # log of prob
        y = y.to(torch.long).view(-1)
        nll = -logp.gather(1, y.unsqueeze(1)).squeeze(1)
        # clamp to avoid -inf -> nan; this is a non-negative loss region
        return nll.clamp(min=-50).mean()

    def predict(self, x):
        return self.prob(x).argmax(dim=-1).to(torch.long)

    def confidence(self, x):
        return self.prob(x).max(dim=-1).values

    def init_means_from(self, x, y, seed=0):
        """Init mu_k from random per-class training examples (helps training)."""
        g = torch.Generator().manual_seed(int(seed))
        for k in range(self.K):
            idx = torch.where(y == k)[0]
            if len(idx) > 0:
                pick = idx[torch.randint(0, len(idx), (1,), generator=g).item()]
                with torch.no_grad():
                    self.mu[k] = x[pick].float()


# ---------------------------------------------------------------------------
# Ensemble of models, mean-probability combine (tex:819-821)
# ---------------------------------------------------------------------------
class Ensemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.K = members[0].K

    def logits(self, x):
        return torch.stack([m.logits(x) for m in self.members], dim=0).mean(0)

    def prob(self, x):
        return torch.stack([m.prob(x) for m in self.members], dim=0).mean(0)

    def loss(self, x, y):
        # NLL of mean prob via logsumexp on mean logits
        logits = self.logits(x)
        return F.cross_entropy(logits, y.to(torch.long).view(-1))

    def predict(self, x):
        return self.prob(x).argmax(dim=-1).to(torch.long)

    def confidence(self, x):
        return self.prob(x).max(dim=-1).values


# ---------------------------------------------------------------------------
# Convolutional maxout net for CIFAR-10 (arch ours; paper silent, tex:340-345).
# Reshape [B,3072] -> [B,3,32,32]; 3 conv-maxout stages + maxpool + maxout MLP head.
# ---------------------------------------------------------------------------
class ConvMaxoutStage(nn.Module):
    """Conv maxout: max over `pieces` conv slices, each in_ch->out_ch."""
    def __init__(self, in_ch, out_ch, pieces, kernel=3, pad=1):
        super().__init__()
        self.pieces = pieces
        self.convs = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel, padding=pad) for _ in range(pieces)])
        for c in self.convs:
            nn.init.kaiming_normal_(c.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(c.bias)

    def forward(self, x):
        outs = torch.stack([c(x) for c in self.convs], dim=1)  # [B, pieces, out_ch, H, W]
        return outs.max(dim=1).values  # [B, out_ch, H, W]


class ConvMaxoutCIFAR(nn.Module):
    # arch (ours, paper silent): 3 conv-maxout stages (32,64,128 ch) + 2x maxpool
    # + maxout MLP head (256,2 pieces) + softmax top. Modest for CPU training.
    def __init__(self, D=3072, K=10):
        super().__init__()
        self.D, self.K = D, K
        self.c1 = ConvMaxoutStage(3, 32, pieces=3)
        self.c2 = ConvMaxoutStage(32, 64, pieces=3)
        self.c3 = ConvMaxoutStage(64, 128, pieces=3)
        self.pool = nn.MaxPool2d(2)
        # after 3 convs (pad=1, stride=1) and 2 maxpools: 32->16->8
        flat = 128 * 8 * 8
        self.fc_max = MaxoutLinear(flat, 256, pieces=2)
        self.head = nn.Linear(256, K)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        x = _check(x)
        B = x.shape[0]
        h = x.view(B, 3, 32, 32)
        h = self.c1(h); h = F.relu(h)  # maxout already nonlinearity, relu keeps positivity
        h = self.pool(h)
        h = self.c2(h); h = F.relu(h)
        h = self.pool(h)
        h = self.c3(h); h = F.relu(h)
        h = h.flatten(1)
        h = self.fc_max(h)
        return h

    def logits(self, x):
        return self.head(self.forward_features(x))

    def prob(self, x):
        return F.softmax(self.logits(x), dim=-1)

    def loss(self, x, y):
        return F.cross_entropy(self.logits(x), y.to(torch.long).view(-1))

    def predict(self, x):
        return self.prob(x).argmax(dim=-1).to(torch.long)

    def confidence(self, x):
        return self.prob(x).max(dim=-1).values
