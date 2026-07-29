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
    layer: int                   # -1 = global (one plane applied at every layer of this dh)
    dh: int                      # head_dim this plane applies to (Gemma-4: 256 or 512)
    d_feat: np.ndarray           # [dh] unit, contrastive direction
    d_pc0: np.ndarray            # [dh] unit, pooled PC1


@dataclass
class ASBank:
    """Angular Steering bank: ONE fixed 2D rotation plane PER distinct head_dim,
    computed from the contrastive activation set and applied as that fixed rotation
    at EVERY layer whose attention output has that head_dim.

    Paper (tex:L394): Angular Steering "applies a fixed 2D rotation in the
    mean-difference span across all layers"; SPEC §4.16 "rotate all layers";
    SPEC §5.5 "plane from candidate directions at ALL layers, applied at every
    layer". The paper rotates the residual stream (d_model, constant), so a single
    plane suffices. Our hook infrastructure is the per-head attention output
    (pre-W_O), and on the REAL google/gemma-4-E4B-it head_dim is NOT constant
    across layers (round-19: sliding_attention 8x256, full_attention 8x512), so a
    single d_h-dim plane cannot apply to both geometries — pooling 256- and
    512-dim activations would raise and `a @ d_feat` would mismatch. We fit one
    plane per distinct head_dim present in the monitored set; the controller
    selects the plane matching each layer's actual head_dim and pass-throughs any
    layer whose head_dim was not seen at fit time. This is the faithful
    "fixed rotation across all layers" reading under the documented head-output
    adaptation (SPEC §4.16); a single-plane model is the degenerate homogeneous
    case (Llama / GPT-OSS / distilgpt2 all have one head_dim). ``layers_monitored``
    is kept for provenance (which layers fed the planes); the controller requests
    hooks on ALL layers via ``hook_layers='all'``.
    """
    model_id: str
    benchmark: str
    angle_deg: float
    layers_monitored: list        # layers whose activations computed the planes
    planes: list                 # list[ASPlane], one per distinct head_dim

    @classmethod
    def load(cls, path_npz: str, path_manifest: str | None = None) -> "ASBank":
        import json
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        with open(path_manifest) as f:
            man = json.load(f)
        z = np.load(path_npz)
        dhs = man.get("plane_dhs", [])
        planes = []
        if dhs:
            for dh in dhs:
                planes.append(ASPlane(layer=-1, dh=int(dh),
                                       d_feat=z[f"dfeat__dh{int(dh)}"],
                                       d_pc0=z[f"dpc0__dh{int(dh)}"]))
        else:
            # back-compat with the prior single-plane format
            planes = [ASPlane(layer=-1, dh=int(z["dpc0_global"].shape[0]),
                              d_feat=z["dfeat_global"], d_pc0=z["dpc0_global"])]
        return cls(model_id=man["model_id"], benchmark=man["benchmark"],
                   angle_deg=man["angle_deg"], layers_monitored=man["layers_monitored"],
                   planes=planes)

    def save(self, path_npz: str, path_manifest: str | None = None):
        arrays = {}
        for p in self.planes:
            arrays[f"dfeat__dh{p.dh}"] = p.d_feat.astype(np.float32)
            arrays[f"dpc0__dh{p.dh}"] = p.d_pc0.astype(np.float32)
        np.savez(path_npz, **arrays)
        import json
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        json.dump({"model_id": self.model_id, "benchmark": self.benchmark,
                   "angle_deg": self.angle_deg,
                   "layers_monitored": self.layers_monitored,
                   "plane_dhs": [p.dh for p in self.planes]},
                  open(path_manifest, "w"), indent=2)


def _fit_one_plane(c_acts, i_acts, candidates, dh):
    """Fit a single (d_feat, d_PC0) orthonormal plane for one head_dim."""
    if not c_acts or not i_acts or not candidates:
        plane = ASPlane(layer=-1, dh=dh,
                        d_feat=np.eye(dh)[0].astype(np.float32),
                        d_pc0=np.eye(dh)[1].astype(np.float32))
        return plane
    c_all = np.concatenate(c_acts, axis=0)        # [M_c, dh]
    i_all = np.concatenate(i_acts, axis=0)        # [M_i, dh]
    d_feat = (i_all.mean(axis=0) - c_all.mean(axis=0))    # difference-in-means
    # d_PC0 = PC1 of the per-(l,h) candidate difference-in-means directions
    # (Vu & Nguyen §4.5), NOT PC1 of pooled raw activations.
    C = np.stack(candidates, axis=0)             # [n_candidates, dh]
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
    return ASPlane(layer=-1, dh=dh, d_feat=d_feat.astype(np.float32),
                   d_pc0=d_pc0.astype(np.float32))


def fit_as_bank(all_head_acts: dict, train_pids: list, *, model_id: str,
                benchmark: str, angle_deg: float, layers_monitored: list) -> ASBank:
    """Build ONE fixed rotation plane PER distinct head_dim from the contrastive
    activation set, pooled across the monitored layers/heads of that head_dim,
    RESTRICTED to the train split (SPEC §4.8 split discipline). A single-plane model
    (Llama/GPT-OSS/distilgpt2: one head_dim) yields one plane; Gemma-4 (two
    head_dims) yields two.

    d_feat = unit(mean_incorrect - mean_correct) over the pooled TRAIN contrastive
             activations of that head_dim — the contrastive direction (tex:L394).
    d_PC0 = first principal component of the per-(l,h) candidate difference-in-
             means directions (Vu & Nguyen §4.5), NOT PC1 of pooled raw activations.
    """
    train_set = set(train_pids)
    # group monitored layers/heads by their head_dim (the activation last-dim).
    by_dh = {}   # dh -> {"c": [...], "i": [...], "cands": [...]}
    for (l, h), ha in all_head_acts.items():
        if l not in layers_monitored:
            continue
        c_h, i_h = [], []
        for pid in ha.problems():
            if pid not in train_set:
                continue
            for t in ha.correct[pid]:
                c_h.append(t)
            for t in ha.incorrect[pid]:
                i_h.append(t)
        if not c_h and not i_h:
            continue
        dh = (c_h[0].shape[-1] if c_h else i_h[0].shape[-1])
        bucket = by_dh.setdefault(dh, {"c": [], "i": [], "cands": []})
        if c_h:
            bucket["c"].extend(c_h)
        if i_h:
            bucket["i"].extend(i_h)
        if c_h and i_h:
            delta = (np.concatenate(i_h, axis=0).mean(axis=0)
                     - np.concatenate(c_h, axis=0).mean(axis=0))
            bucket["cands"].append(delta)
    if not by_dh:
        # degenerate: no train activations at all; emit an axis plane for the
        # registry head_dim so the controller still has a valid basis.
        d_h = 8
        plane = ASPlane(layer=-1, dh=d_h,
                        d_feat=np.eye(d_h)[0].astype(np.float32),
                        d_pc0=np.eye(d_h)[1].astype(np.float32))
        return ASBank(model_id=model_id, benchmark=benchmark, angle_deg=angle_deg,
                      layers_monitored=layers_monitored, planes=[plane])
    planes = [_fit_one_plane(b["c"], b["i"], b["cands"], dh)
              for dh, b in sorted(by_dh.items())]
    return ASBank(model_id=model_id, benchmark=benchmark, angle_deg=angle_deg,
                  layers_monitored=layers_monitored, planes=planes)


class AngularSteeringController:
    """Fixed-offset Angular Steering (SPEC §4.16; Vu & Nguyen Eq. 1). Rotates each
    head's attention output in the (d_feat, d_PC0) plane of its head_dim by the SAME
    constant angle theta (a fixed offset). theta=0 is the IDENTITY (matches the
    ablation Table 4 0deg~unsteered signature, tex:L654). The SAME fixed angle is
    applied at EVERY layer (paper tex:L394 "a fixed 2D rotation ... across all
    layers"; SPEC §4.16 "rotate all layers"; SPEC §5.5 "applied at every layer"),
    selecting the plane matching each layer's head_dim (Gemma-4 has two).
    ``hook_layers='all'`` asks the HookRegistry to install W_O pre-hooks on every
    layer so the rotation reaches non-monitored layers too. Layers whose head_dim
    was not seen at fit time are passed through (no plane to rotate in)."""
    # request hooks on ALL layers (registry resolves 'all' to range(n_layers))
    hook_layers = "all"

    def __init__(self, as_bank: ASBank, angle_deg: float | None = None):
        self.theta = float(np.deg2rad(angle_deg if angle_deg is not None
                                      else as_bank.angle_deg))
        self._planes = {p.dh: p for p in as_bank.planes}

    def __call__(self, layer, x_heads):
        bsz, seq, H, dh = x_heads.shape
        if seq != 1:
            return None
        plane = self._planes.get(dh)
        if plane is None:
            # no fitted plane for this head_dim: pass-through (identity at theta=0
            # semantics for an un-fitted geometry).
            return None
        d_feat, d_pc0 = plane.d_feat, plane.d_pc0
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
