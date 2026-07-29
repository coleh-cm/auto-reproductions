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
        # held-out probe accuracy. The functional class-presence check is the
        # inner `len(np.unique(yte)) >= 2` below; the outer guard only needs to
        # confirm the held-out split is non-empty (the prior set-comprehension
        # over the literal [0,1] was a no-op that always read len 2).
        if sel.problems():
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
# Angular Steering (SPEC §4.16): Vu & Nguyen FIXED-OFFSET rotation in the
# Span(d_feat, d_PC0) plane, applied across all monitored layers, every decode step.
#   d_feat = difference-in-means direction (mean_incorrect - mean_correct) over the
#            pooled per-head activations of that layer — the contrastive direction
#            (SPEC §4.16; correct vs incorrect traces as the contrast set, tex:L394).
#   d_PC0 = first principal component of the per-(l,h) candidate difference-in-means
#            directions (Vu & Nguyen §4.5).
#   Rotation = FIXED-OFFSET form (Vu & Nguyen Eq. 1, "Rotation by an Offset Angle"):
#            every activation is rotated by the SAME constant angle theta; theta=0
#            is the IDENTITY. The paper (tex:L394) says "a fixed 2D rotation in the
#            mean-difference span across all layers", and the ablation Table 4
#            (tex:L654-665) confirms the offset form: 0deg ~ unsteered (0.488 vs
#            0.478), with 360deg-periodic accuracy and worst at 90-120deg — the
#            signature of a constant-offset rotation, NOT the target-angle form
#            (Eq. 2) which at 0deg would force every activation onto d_feat (the
#            "incorrect" direction) and crash accuracy. The prior impl used the
#            target-angle form (delta = target - cur); this is the corrected
#            fixed-offset form (delta = theta, a constant).
#
# Adaptation note: original AS rotates the residual stream; our hook infrastructure
# is the per-head attention output (pre-W_O). We apply the per-layer rotation to each
# head's attention output using that layer's plane. This is the closest available hook
# point and is recorded in REPRODUCTION.md as a documented adaptation (the paper does
# not specify AS's exact hook location in its reasoning adaptation, tex:L394).
# ---------------------------------------------------------------------------
@dataclass
class ASPlane:
    layer: int                   # -1 = global (one plane applied at every layer)
    d_feat: np.ndarray           # [d_h] unit, contrastive direction
    d_pc0: np.ndarray            # [d_h] unit, pooled PC1


@dataclass
class ASBank:
    """Angular Steering bank: a SINGLE global (d_feat, d_PC0) plane computed from
    the contrastive activation set and applied as one fixed 2D rotation at EVERY
    layer.

    Paper (tex:L394): Angular Steering "applies a fixed 2D rotation in the
    mean-difference span across all layers"; SPEC §4.16 "rotate all layers";
    SPEC §5.5 "plane from candidate directions at ALL layers, applied at every
    layer". A single fixed rotation applied uniformly is the faithful reading of
    "fixed ... across all layers" (one rotation, all layers), and it does not
    require per-layer activations for non-monitored layers: the plane is pooled
    across the monitored layers/heads (the contrastive set) and reused at every
    layer. ``layers_monitored`` is kept for provenance (which layers fed the
    plane); the controller requests hooks on ALL layers via ``hook_layers='all'``.
    """
    model_id: str
    benchmark: str
    angle_deg: float
    layers_monitored: list        # layers whose activations computed the plane
    plane: ASPlane               # the single global plane (layer=-1)

    @classmethod
    def load(cls, path_npz: str, path_manifest: str | None = None) -> "ASBank":
        import json
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        with open(path_manifest) as f:
            man = json.load(f)
        z = np.load(path_npz)
        plane = ASPlane(layer=-1, d_feat=z["dfeat_global"], d_pc0=z["dpc0_global"])
        return cls(model_id=man["model_id"], benchmark=man["benchmark"],
                   angle_deg=man["angle_deg"], layers_monitored=man["layers_monitored"],
                   plane=plane)

    def save(self, path_npz: str, path_manifest: str | None = None):
        np.savez(path_npz,
                 dfeat_global=self.plane.d_feat.astype(np.float32),
                 dpc0_global=self.plane.d_pc0.astype(np.float32))
        import json
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        json.dump({"model_id": self.model_id, "benchmark": self.benchmark,
                   "angle_deg": self.angle_deg,
                   "layers_monitored": self.layers_monitored},
                  open(path_manifest, "w"), indent=2)


def fit_as_bank(all_head_acts: dict, train_pids: list, *, model_id: str,
                benchmark: str, angle_deg: float, layers_monitored: list) -> ASBank:
    """Build a SINGLE global (d_feat, d_PC0) plane from the contrastive activation
    set, pooled across all monitored layers/heads, RESTRICTED to the train split
    (SPEC §4.8 split discipline; fit_iti_bank and fit_manifold_bank already
    restrict to train_pids/select_pids). The prior impl iterated ALL problems,
    leaking the held-out select/report splits and giving AS ~30% more contrastive
    data than MAGS/ITI.

    d_feat = unit(mean_incorrect - mean_correct) over the pooled TRAIN contrastive
             activations — the contrastive direction (tex:L394; SPEC §4.16).
    d_PC0 = first principal component of the per-(l,h) candidate difference-in-
             means directions (Vu & Nguyen §4.5: "PCA on the candidate directions
             d_feat^i and select the first principal component, d_PC0"). Each
             candidate delta^(l,h) = mean_incorrect^(l,h) - mean_correct^(l,h)
             over the train traces. The prior impl took PC1 of pooled RAW
             activations, which captures dominant token/magnitude variance rather
             than cross-layer feature-direction variance and changes the plane.
    Both are [d_h] directions; head_dim is constant across layers for every
    supported model, so one global plane applies at every layer's per-head output.
    The plane is reused at all layers at inference — the "fixed rotation across
    all layers" the paper describes.
    """
    train_set = set(train_pids)
    c_all = []   # pooled correct activations [M_c, d_h]
    i_all = []   # pooled incorrect activations [M_i, d_h]
    candidates = []   # per-(l,h) difference-in-means directions [n_cand, d_h]
    for (l, h), ha in all_head_acts.items():
        if l not in layers_monitored:
            continue
        c_h, i_h = [], []
        for pid in ha.problems():
            if pid not in train_set:
                continue
            for t in ha.correct[pid]:
                c_h.append(t); c_all.append(t)
            for t in ha.incorrect[pid]:
                i_h.append(t); i_all.append(t)
        if c_h and i_h:
            delta = (np.concatenate(i_h, axis=0).mean(axis=0)
                     - np.concatenate(c_h, axis=0).mean(axis=0))
            candidates.append(delta)
    if not c_all or not i_all or not candidates:
        # degenerate: fall back to an axis-aligned plane so the controller still
        # has a valid orthonormal basis (a real fit never reaches here)
        d_h = next(iter(all_head_acts.values())).correct[
            next(iter(next(iter(all_head_acts.values())).correct))].shape[-1] \
            if all_head_acts else 8
        plane = ASPlane(layer=-1,
                        d_feat=np.eye(d_h)[0].astype(np.float32),
                        d_pc0=np.eye(d_h)[1].astype(np.float32))
        return ASBank(model_id=model_id, benchmark=benchmark, angle_deg=angle_deg,
                      layers_monitored=layers_monitored, plane=plane)
    c_all = np.concatenate(c_all, axis=0)        # [M_c, d_h]
    i_all = np.concatenate(i_all, axis=0)        # [M_i, d_h]
    d_feat = (i_all.mean(axis=0) - c_all.mean(axis=0))    # difference-in-means
    # d_PC0 = PC1 of the per-(l,h) candidate difference-in-means directions
    # (Vu & Nguyen §4.5), NOT PC1 of pooled raw activations.
    C = np.stack(candidates, axis=0)             # [n_candidates, d_h]
    Cc = C - C.mean(axis=0, keepdims=True)
    try:
        _, _, Vh = np.linalg.svd(Cc, full_matrices=False)
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
    plane = ASPlane(layer=-1, d_feat=d_feat.astype(np.float32),
                    d_pc0=d_pc0.astype(np.float32))
    return ASBank(model_id=model_id, benchmark=benchmark, angle_deg=angle_deg,
                  layers_monitored=layers_monitored, plane=plane)


class AngularSteeringController:
    """Fixed-offset Angular Steering (SPEC §4.16; Vu & Nguyen Eq. 1). Rotates each
    head's attention output in the global (d_feat, d_PC0) plane by the SAME constant
    angle theta (a fixed offset). theta=0 is the IDENTITY (matches the ablation
    Table 4 0deg~unsteered signature, tex:L654). The SAME fixed plane is applied at
    EVERY layer (paper tex:L394 "a fixed 2D rotation ... across all layers"; SPEC
    §4.16 "rotate all layers"; SPEC §5.5 "applied at every layer").
    ``hook_layers='all'`` asks the HookRegistry to install W_O pre-hooks on every
    layer so the rotation reaches non-monitored layers too (the prior
    implementation only rotated the monitored subset, which made AS behave like a
    targeted method rather than the uniform-across-all-layers baseline the paper
    compares against)."""
    # request hooks on ALL layers (registry resolves 'all' to range(n_layers))
    hook_layers = "all"

    def __init__(self, as_bank: ASBank, angle_deg: float | None = None):
        self.theta = float(np.deg2rad(angle_deg if angle_deg is not None
                                      else as_bank.angle_deg))
        self.plane = as_bank.plane      # the single global ASPlane

    def __call__(self, layer, x_heads):
        bsz, seq, H, dh = x_heads.shape
        if seq != 1:
            return None
        d_feat, d_pc0 = self.plane.d_feat, self.plane.d_pc0
        x = x_heads.detach().to(torch.float32).cpu().numpy()
        out = x.copy()
        a = out[0, 0]                      # [H, dh]
        comp_feat = a @ d_feat             # [H]
        comp_pc0 = a @ d_pc0               # [H]
        # FIXED-OFFSET rotation by the constant theta (Vu & Nguyen Eq. 1): every
        # activation rotated by the same angle; theta=0 is the identity. The prior
        # target-angle form (delta = target - cur) rotated each activation TO theta
        # and at 0deg forced every head onto d_feat (the "incorrect" direction) —
        # a different intervention that cannot reproduce the ablation.
        cos, sin = np.cos(self.theta), np.sin(self.theta)
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
        # set V_plaus(x_{1:t}) = {x : p_expert(x|x_{1:t}) >= alpha_p * max_x' p_expert(x'|x_{1:t})}
        # — a RELATIVE threshold (alpha_p of the expert's own max probability), NOT an
        # absolute cutoff. An absolute cutoff (p_e < alpha_p) collapses the plausible set
        # to ~1-2 tokens over a large vocab, making the amateur penalty inert and CD
        # degrade to the greedy expert. The relative form keeps a meaningfully large set
        # so the amateur actually reshapes the distribution (Li et al. 2023, Eq. 4-5).
        p_e = logp_e.exp()
        p_max = p_e.max(dim=-1, keepdim=True).values
        plaus_mask = p_e < (self.alpha_p * p_max)
        score = score.masked_fill(plaus_mask, float("-inf"))
        return score
