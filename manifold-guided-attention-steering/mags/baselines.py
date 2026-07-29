"""Baseline steering arms (SPEC §5.5): Unsteered, ITI, Angular Steering, Contrastive Decoding.

Each is a controller callable installed on the monitored layers' W_O pre-hook (or, for
CD, a logits processor at the output distribution level). The paper treats correct and
incorrect solution traces as the desired/undesired contrast sets for ITI and AS
(tex:L394); we build each baseline's steering parameters from the same contrastive
activation set MAGS uses.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .manifold import HeadProblemActivations, trajectory_score


def _restrict(ha: HeadProblemActivations, pids):
    pidset = set(pids)
    sub = HeadProblemActivations()
    for pid in ha.problems():
        if pid in pidset:
            sub.correct[pid] = ha.correct[pid]
            sub.incorrect[pid] = ha.incorrect[pid]
    return sub


# ---------------------------------------------------------------------------
# Unsteered
# ---------------------------------------------------------------------------
class UnsteeredController:
    def __call__(self, layer, x_heads):
        return None


# ---------------------------------------------------------------------------
# ITI (SPEC §4.15): per-head logistic probe on the trace-mean head output;
# top-K heads by held-out probe accuracy (K in {24,48,96}); intervention
#   a += alpha * sigma_h * v_h   every decode step (static; Li et al. 2023).
# v_h = probe direction oriented toward the DESIRED (correct) class; sigma_h = the
# probe's held-out accuracy (a fixed per-head scalar). The shift is STATIC (it does
# not depend on the current activation), per the original ITI convention.
# ---------------------------------------------------------------------------
@dataclass
class ITIHead:
    layer: int
    head: int
    direction: np.ndarray       # [d_h] unit vector, oriented toward CORRECT class
    sigma: float                # held-out probe accuracy (fixed scalar)
    accuracy: float = 0.0


@dataclass
class ITIBank:
    model_id: str
    benchmark: str
    K: int
    alpha: float
    layers_monitored: list
    selected_heads: list        # [[l,h], ...] top-K by held-out probe accuracy
    heads: dict                 # (l,h) -> ITIHead  (ALL monitored heads, not just K)
    @classmethod
    def load(cls, path_npz: str, path_manifest: str | None = None) -> "ITIBank":
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        import json
        with open(path_manifest) as f:
            man = json.load(f)
        z = np.load(path_npz)
        heads = {}
        for key_str, info in man["all_heads"].items():
            l, h = eval(key_str)
            heads[(l, h)] = ITIHead(
                layer=l, head=h,
                direction=z[f"dir__l{l}_h{h}"],
                sigma=float(z[f"sigma__l{l}_h{h}"]),
                accuracy=float(info["accuracy"]),
            )
        return cls(model_id=man["model_id"], benchmark=man["benchmark"],
                    K=man["K"], alpha=man["alpha"],
                    layers_monitored=man["layers_monitored"],
                    selected_heads=[list(h) for h in man["selected_heads"]],
                    heads=heads)

    def save(self, path_npz: str, path_manifest: str | None = None):
        arrays = {}
        for (l, h), hd in self.heads.items():
            arrays[f"dir__l{l}_h{h}"] = hd.direction.astype(np.float32)
            arrays[f"sigma__l{l}_h{h}"] = np.float32(hd.sigma)
        np.savez(path_npz, **arrays)
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        import json
        json.dump({
            "model_id": self.model_id, "benchmark": self.benchmark,
            "K": self.K, "alpha": self.alpha,
            "layers_monitored": self.layers_monitored,
            "selected_heads": self.selected_heads,
            "all_heads": {str(k): {"accuracy": v.accuracy} for k, v in self.heads.items()},
        }, open(path_manifest, "w"), indent=2)


def fit_iti_bank(all_head_acts: dict, train_pids: list, select_pids: list, *,
                 model_id: str, benchmark: str, K: int, alpha: float,
                 layers_monitored: list) -> ITIBank:
    """Fit per-head logistic probes (trace-mean features), rank by held-out probe
    accuracy, keep top-K. Stores ALL monitored heads so K in {24,48,96} is reachable
    regardless of the MAGS K. Direction oriented toward the CORRECT class
    (probe predicts y=1 for INCORRECT, so direction = -coef)."""
    heads = {}
    accs = []
    for (l, h), ha in all_head_acts.items():
        train = _restrict(ha, train_pids)
        sel = _restrict(ha, select_pids)
        if not train.problems():
            continue
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
        coef = clf.coef_[0].astype(np.float32)
        n = np.linalg.norm(coef)
        if n < 1e-12:
            continue
        direction = (-coef / n)            # toward CORRECT (y=0), the desired class
        # held-out probe accuracy
        if sel.problems() and len({t for pid in sel.problems() for t in [0, 1]}) >= 2:
            Xte, yte = [], []
            for pid in sel.problems():
                for t in sel.correct[pid]:
                    Xte.append(t.mean(axis=0)); yte.append(0)
                for t in sel.incorrect[pid]:
                    Xte.append(t.mean(axis=0)); yte.append(1)
            if Xte and len(np.unique(yte)) >= 2:
                acc = float(clf.score(np.asarray(Xte), np.asarray(yte)))
            else:
                acc = 0.5
        else:
            acc = 0.5
        heads[(l, h)] = ITIHead(layer=l, head=h, direction=direction, sigma=acc,
                                   accuracy=acc)
        accs.append((acc, l, h))
    accs.sort(reverse=True)
    selected = [[l, h] for _, l, h in accs[:K]]
    return ITIBank(model_id=model_id, benchmark=benchmark, K=K, alpha=alpha,
                   layers_monitored=layers_monitored, selected_heads=selected,
                   heads=heads)


class ITIController:
    """Static ITI intervention (SPEC §4.15, Li et al. 2023): for each selected head,
    every decode step apply  a += alpha * sigma_h * v_h  (v_h toward correct, sigma_h =
    held-out probe accuracy, both fixed)."""
    def __init__(self, iti_bank: ITIBank, alpha: float | None = None):
        self.alpha = iti_bank.alpha if alpha is None else alpha
        self._selected = {}     # (l,h) -> ITIHead
        for h in iti_bank.selected_heads:
            self._selected[tuple(h)] = iti_bank.heads[tuple(h)]

    def __call__(self, layer, x_heads):
        bsz, seq, H, dh = x_heads.shape
        if seq != 1:
            return None
        x = x_heads.detach().to(torch.float32).cpu().numpy()
        out = x.copy()
        modified = False
        for (l, h), hd in self._selected.items():
            if l != layer:
                continue
            # static shift toward the correct class (SPEC §4.15)
            out[0, 0, h, :] = out[0, 0, h, :] + self.alpha * hd.sigma * hd.direction
            modified = True
        if not modified:
            return None
        return torch.from_numpy(out).to(x_heads.device).to(x_heads.dtype).view(bsz, seq, H, dh)


# ---------------------------------------------------------------------------
# Angular Steering (SPEC §4.16): Vu & Nguyen target-angle rotation in the
# Span(d_feat, d_PC0) plane, applied across all monitored layers, every decode step.
#   d_feat = difference-in-means direction (mean_incorrect - mean_correct) over the
#            pooled per-head activations of that layer — the contrastive direction
#            (SPEC §4.16; correct vs incorrect traces as the contrast set, tex:L394).
#   d_PC0 = first principal component of the pooled activations of that layer.
#   Rotation = TARGET-ANGLE form (Vu & Nguyen): rotate each activation so its angle
#            in the (d_feat, d_PC0) plane becomes the target angle (not a fixed offset).
#
# Adaptation note: original AS rotates the residual stream; our hook infrastructure
# is the per-head attention output (pre-W_O). We apply the per-layer rotation to each
# head's attention output using that layer's plane. This is the closest available hook
# point and is recorded in REPRODUCTION.md as a documented adaptation (the paper does
# not specify AS's exact hook location in its reasoning adaptation, tex:L394).
# ---------------------------------------------------------------------------
@dataclass
class ASPlane:
    layer: int
    d_feat: np.ndarray          # [d_h] unit, contrastive direction
    d_pc0: np.ndarray           # [d_h] unit, pooled PC1


@dataclass
class ASBank:
    model_id: str
    benchmark: str
    angle_deg: float
    layers_monitored: list
    planes: dict                # layer -> ASPlane

    @classmethod
    def load(cls, path_npz: str, path_manifest: str | None = None) -> "ASBank":
        import json
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        with open(path_manifest) as f:
            man = json.load(f)
        z = np.load(path_npz)
        planes = {}
        for l in man["layers_monitored"]:
            planes[l] = ASPlane(layer=l, d_feat=z[f"dfeat__l{l}"],
                                 d_pc0=z[f"dpc0__l{l}"])
        return cls(model_id=man["model_id"], benchmark=man["benchmark"],
                   angle_deg=man["angle_deg"], layers_monitored=man["layers_monitored"],
                   planes=planes)

    def save(self, path_npz: str, path_manifest: str | None = None):
        arrays = {}
        for l, p in self.planes.items():
            arrays[f"dfeat__l{l}"] = p.d_feat.astype(np.float32)
            arrays[f"dpc0__l{l}"] = p.d_pc0.astype(np.float32)
        np.savez(path_npz, **arrays)
        import json
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        json.dump({"model_id": self.model_id, "benchmark": self.benchmark,
                   "angle_deg": self.angle_deg,
                   "layers_monitored": self.layers_monitored},
                  open(path_manifest, "w"), indent=2)


def fit_as_bank(all_head_acts: dict, *, model_id: str, benchmark: str,
                angle_deg: float, layers_monitored: list) -> ASBank:
    """Build a per-layer (d_feat, d_PC0) plane from the contrastive activation set.

    d_feat = unit(mean_incorrect - mean_correct) over all pooled per-head activations
    of the layer (across all heads, all traces, all token steps). d_PC0 = top-1 right
    singular vector of the pooled centered activations. Both are contrastive/dataset
    directions, computed at fit time and persisted (the ManifoldBank does not store
    them)."""
    # gather per-layer pooled activations grouped by class
    layer_correct = {l: [] for l in layers_monitored}
    layer_incorrect = {l: [] for l in layers_monitored}
    for (l, h), ha in all_head_acts.items():
        if l not in layer_correct:
            continue
        for pid in ha.problems():
            for t in ha.correct[pid]:
                layer_correct[l].append(t)        # [T,d_h] -> pooled below
            for t in ha.incorrect[pid]:
                layer_incorrect[l].append(t)
    planes = {}
    for l in layers_monitored:
        c = layer_correct[l]; i = layer_incorrect[l]
        if not c or not i:
            continue
        c_all = np.concatenate(c, axis=0)        # [M_c, d_h]
        i_all = np.concatenate(i, axis=0)        # [M_i, d_h]
        d_feat = (i_all.mean(axis=0) - c_all.mean(axis=0))    # difference-in-means
        # d_PC0 from pooled centered activations (correct+incorrect combined)
        pooled = np.concatenate([c_all, i_all], axis=0)
        pooled_c = pooled - pooled.mean(axis=0)
        # top-1 right singular vector
        try:
            _, _, Vh = np.linalg.svd(pooled_c, full_matrices=False)
            d_pc0 = Vh[0]
        except Exception:
            d_pc0 = np.zeros_like(d_feat); d_pc0[0] = 1.0
        nf = np.linalg.norm(d_feat)
        if nf < 1e-12:
            d_feat = np.zeros_like(d_pc0); d_feat[0] = 1.0
        else:
            d_feat = d_feat / nf
        np_ = np.linalg.norm(d_pc0)
        d_pc0 = d_pc0 / np_ if np_ > 1e-12 else np.eye(d_feat.shape[0])[0]
        # orthogonalize d_feat against d_pc0 so the plane basis is orthonormal
        d_feat = d_feat - (d_feat @ d_pc0) * d_pc0
        nf2 = np.linalg.norm(d_feat)
        d_feat = d_feat / nf2 if nf2 > 1e-12 else np.zeros_like(d_feat)
        planes[l] = ASPlane(layer=l, d_feat=d_feat.astype(np.float32),
                             d_pc0=d_pc0.astype(np.float32))
    return ASBank(model_id=model_id, benchmark=benchmark, angle_deg=angle_deg,
                  layers_monitored=layers_monitored, planes=planes)


class AngularSteeringController:
    """Target-angle Angular Steering (SPEC §4.16). Rotates each head's attention
    output in the layer's (d_feat, d_PC0) plane so its angle becomes the target
    angle. Applied at every monitored layer, every decode step."""
    def __init__(self, as_bank: ASBank, angle_deg: float | None = None):
        self.target = float(np.deg2rad(angle_deg if angle_deg is not None
                                       else as_bank.angle_deg))
        self.planes = as_bank.planes    # layer -> ASPlane

    def __call__(self, layer, x_heads):
        bsz, seq, H, dh = x_heads.shape
        if seq != 1:
            return None
        plane = self.planes.get(layer)
        if plane is None:
            return None
        d_feat, d_pc0 = plane.d_feat, plane.d_pc0
        x = x_heads.detach().to(torch.float32).cpu().numpy()
        out = x.copy()
        a = out[0, 0]                      # [H, dh]
        comp_feat = a @ d_feat             # [H]
        comp_pc0 = a @ d_pc0               # [H]
        # current angle of each head in the plane
        cur = np.arctan2(comp_pc0, comp_feat)            # [H]
        delta = self.target - cur                          # rotate to target
        cos, sin = np.cos(delta), np.sin(delta)
        new_feat = cos * comp_feat - sin * comp_pc0
        new_pc0 = sin * comp_feat + cos * comp_pc0
        a_new = a + (new_feat - comp_feat)[:, None] * d_feat \
                  + (new_pc0 - comp_pc0)[:, None] * d_pc0
        out[0, 0] = a_new
        return torch.from_numpy(out).to(x_heads.device).to(x_heads.dtype).view(bsz, seq, H, dh)


# ---------------------------------------------------------------------------
# Contrastive Decoding (SPEC §4.17): expert vs amateur logits at each decode step.
# Implemented as a logits processor (operates at the distribution level, no hooks).
# CD (Li et al. 2023): score = log p_expert - beta * log p_amateur, with adaptive
# plausibility masking on the EXPERT's plausible set. Defaults: alpha_p=0.1, beta=0.5.
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
        # adaptive plausibility (Li et al. 2023 CD): restrict to the EXPERT's plausible
        # set (top-(1-alpha_p) of the expert distribution); discard expert-improbable
        # tokens so the amateur cannot penalize tokens the expert is confident about.
        p_e = logp_e.exp()
        score = score.masked_fill(p_e < self.alpha_p, float("-inf"))
        return score
