"""Baseline steering arms (SPEC §5.5): Unsteered, ITI, Angular Steering, Contrastive Decoding.

Each is a controller callable installed on the monitored layers' W_O pre-hook (or, for
CD, a logits processor at the output distribution level). The paper treats correct and
incorrect solution traces as the desired/undesired contrast sets for ITI and AS
(tex:L394); we build each baseline's steering parameters from the same contrastive
activation set MAGS uses.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .manifold import HeadProblemActivations, trajectory_score


# ---------------------------------------------------------------------------
# Unsteered
# ---------------------------------------------------------------------------
class UnsteeredController:
    def __call__(self, layer, x_heads):
        return None


# ---------------------------------------------------------------------------
# ITI (SPEC §4.15): per-head logistic probe on the trace-mean head output;
# top-K heads by held-out probe accuracy; intervention a += alpha * s_h * v_h every step.
# Li et al. 2023 convention: shift along the probe direction by alpha * (head's mean
# activation projected on the probe). We use the difference-in-means direction between
# incorrect and correct trace-means (the contrastive direction), scaled by the probe's
# signed magnitude, applied every decode step (static intervention, tex:L394).
# ---------------------------------------------------------------------------
class ITIController:
    def __init__(self, bank, alpha: float, K: int):
        self.alpha = alpha
        # build per-head probe direction + sign from the contrastive activation set
        # stored in the bank's heads (we reuse mu_c as the correct reference and the
        # direction = -(B^T B)(a-mu_c) is MAGS-specific; for ITI we recompute a plain
        # difference direction at construction time from the activation store).
        # Here we only have the fitted manifold, so ITI's direction is taken as the
        # top-1 right singular vector of D (the dominant error direction), which is the
        # natural probe direction for a difference-in-means contrast set.
        self.directions = {}    # (l,h) -> [d_h] unit vector
        for (l, h), m in bank.heads.items():
            if m.B.shape[0] >= 1:
                self.directions[(l, h)] = m.B[0].astype(np.float32)
        self._selected = set()
        # select top-K by AUROC (use bank's selected_heads if K matches; else take top-K of all)
        ranked = sorted(bank.heads.items(), key=lambda kv: kv[1].auroc, reverse=True)
        for (l, h), _ in ranked[:K]:
            self._selected.add((l, h))

    def __call__(self, layer, x_heads):
        bsz, seq, H, dh = x_heads.shape
        if seq != 1:
            return None
        x = x_heads.detach().to(torch.float32).cpu().numpy()
        modified = False
        out = x.copy()
        for (l, h) in self._selected:
            if l != layer:
                continue
            if (l, h) not in self.directions:
                continue
            v = self.directions[(l, h)]
            a = out[0, 0, h, :]           # [dh]
            # ITI shift: add alpha * (a . v) * v every step (project-then-shift along probe)
            proj = float(a @ v)
            out[0, 0, h, :] = a + self.alpha * proj * v
            modified = True
        if not modified:
            return None
        return torch.from_numpy(out).to(x_heads.device).to(x_heads.dtype).view(bsz, seq, H, dh)


def fit_iti_probes(all_head_acts: dict, train_pids: list, select_pids: list):
    """Optional: fit per-head logistic probes for the held-out-accuracy ranking in §4.15.

    Returns {(l,h): (direction[d_h], accuracy)}. The direction is the LDA-style
    difference of class-trace-means (the contrastive direction). This is the
    selection-time counterpart of ITIController (which uses the dominant SVD direction).
    """
    out = {}
    for (l, h), ha in all_head_acts.items():
        train = _restrict(ha, train_pids)
        sel = _restrict(ha, select_pids)
        if not train.problems():
            continue
        # trace-level mean feature per trace
        Xtr, ytr = [], []
        for pid in train.problems():
            for t in train.correct[pid]:
                Xtr.append(t.mean(axis=0)); ytr.append(0)
            for t in train.incorrect[pid]:
                Xtr.append(t.mean(axis=0)); ytr.append(1)
        Xtr = np.asarray(Xtr); ytr = np.asarray(ytr)
        if len(np.unique(ytr)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        direction = clf.coef_[0].astype(np.float32)
        direction /= (np.linalg.norm(direction) + 1e-12)
        # held-out accuracy
        Xte, yte = [], []
        for pid in sel.problems():
            for t in sel.correct[pid]:
                Xte.append(t.mean(axis=0)); yte.append(0)
            for t in sel.incorrect[pid]:
                Xte.append(t.mean(axis=0)); yte.append(1)
        if not Xte or len(np.unique(yte)) < 2:
            out[(l, h)] = (direction, 0.5); continue
        acc = float(clf.score(np.asarray(Xte), np.asarray(yte)))
        out[(l, h)] = (direction, acc)
    return out


def _restrict(ha: HeadProblemActivations, pids):
    pidset = set(pids)
    sub = HeadProblemActivations()
    for pid in ha.problems():
        if pid in pidset:
            sub.correct[pid] = ha.correct[pid]
            sub.incorrect[pid] = ha.incorrect[pid]
    return sub


# ---------------------------------------------------------------------------
# Angular Steering (SPEC §4.16): Vu & Nguyen target-angle rotation in the
# Span(d_feat, d_PC0) plane, applied at every layer, every step.
# d_feat = difference-in-means direction (correct vs incorrect) at the residual/
# pre-attn output; d_PC0 = first principal component of the pooled activations.
# We approximate the plane from the contrastive activation set we already have:
# d_feat = mean(incorrect) - mean(correct) (trace-level), d_PC0 = top-1 right singular
# vector of D. Rotation applied to the head-output (pre-W_O) at every layer on every
# decode step. Target angle from the ablation (default 30 deg, tex:L654-665).
# ---------------------------------------------------------------------------
class AngularSteeringController:
    def __init__(self, bank, angle_deg: float):
        self.angle = float(np.deg2rad(angle_deg))
        # plane per layer: (d_feat, d_pc0) -> 2x d_h orthonormal basis
        self.planes = {}
        for (l, h), m in bank.heads.items():
            # one plane per layer (use the top-AUROC head's geometry as the layer rep)
            self.planes.setdefault(l, (None, None))
        # use, per layer, the dominant SVD direction (d_PC0) and the global correct
        # centroid's complement as d_feat proxy: d_feat = mu_c shifted by B[0].
        for l in list(self.planes.keys()):
            # pick the best head in this layer from the bank
            cands = [(h, m_) for (ll, h), m_ in bank.heads.items() if ll == l]
            if not cands:
                continue
            cands.sort(key=lambda kv: kv[1].auroc, reverse=True)
            m_best = cands[0][1]
            d_pc0 = m_best.B[0].astype(np.float32) if m_best.B.shape[0] >= 1 else None
            if d_pc0 is None:
                continue
            d_feat = m_best.mu_c.astype(np.float32) - (m_best.mu_c @ d_pc0) * d_pc0
            n = np.linalg.norm(d_feat)
            if n < 1e-8:
                d_feat = np.zeros_like(d_pc0); d_feat[0] = 1.0
            else:
                d_feat = d_feat / n
            self.planes[l] = (d_feat, d_pc0)

    def __call__(self, layer, x_heads):
        bsz, seq, H, dh = x_heads.shape
        if seq != 1:
            return None
        plane = self.planes.get(layer)
        if plane is None or plane[0] is None:
            return None
        d_feat, d_pc0 = plane
        x = x_heads.detach().to(torch.float32).cpu().numpy()
        out = x.copy()
        # rotate every head's output in the (d_feat, d_pc0) plane by self.angle
        cos, sin = np.cos(self.angle), np.sin(self.angle)
        a = out[0, 0]   # [H, dh]
        comp_feat = a @ d_feat      # [H]
        comp_pc0 = a @ d_pc0        # [H]
        new_feat = cos * comp_feat - sin * comp_pc0
        new_pc0 = sin * comp_feat + cos * comp_pc0
        a_new = a + (new_feat - comp_feat)[:, None] * d_feat + (new_pc0 - comp_pc0)[:, None] * d_pc0
        out[0, 0] = a_new
        return torch.from_numpy(out).to(x_heads.device).to(x_heads.dtype).view(bsz, seq, H, dh)


# ---------------------------------------------------------------------------
# Contrastive Decoding (SPEC §4.17): expert vs amateur logits at each decode step.
# Implemented as a HF logits processor (operates at the distribution level, no hooks).
# CD (Li et al. 2023): score = log p_expert - beta * log p_amateur, with adaptive
# plausibility alpha_p masking amateur-improbable tokens. Defaults: alpha_p=0.1, beta=0.5.
# ---------------------------------------------------------------------------
class ContrastiveDecoder:
    """Stateful expert/amateur contrastive decoder.

    Used inside a custom greedy decode loop (mags.generation.cd_generate) because the
    amateur needs its own forward at each step. Not a W_O hook controller.
    """
    def __init__(self, expert_model, amateur_model, tok, alpha_plausibility=0.1, beta=0.5):
        self.expert = expert_model
        self.amateur = amateur_model
        self.tok = tok
        self.alpha_p = alpha_plausibility
        self.beta = beta

    def adapted_logits(self, expert_logits, amateur_logits):
        import torch.nn.functional as F
        logp_e = F.log_softmax(expert_logits, dim=-1)
        logp_a = F.log_softmax(amateur_logits, dim=-1)
        score = logp_e - self.beta * logp_a
        # adaptive plausibility: mask amateur-prob below alpha_p
        p_a = logp_a.exp()
        score = score.masked_fill(p_a < self.alpha_p, float("-inf"))
        return score
