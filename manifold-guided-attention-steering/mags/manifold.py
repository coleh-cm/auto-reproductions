"""Contrastive error manifold construction (SPEC §3.2, Phase A).

Pure-statistics fit: per-class means (Eq.2), contrastive difference (Eq.3), difference
matrix (Eq.4), compact SVD -> error-subspace basis B (Eq.5), global correct centroid
mu_c (Eq.6), threshold percentile over pooled per-token correct-trace scores (Eq.8),
and per-head trajectory AUROC for head selection (tex:L304-305) / drift validation
(tex:L294-298).

No loss, no gradients. The "fit" is descriptive statistics + one SVD per monitored head.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class HeadManifold:
    layer: int
    head: int
    B: np.ndarray            # [k, d_h], orthonormal rows (Eq.5)
    mu_c: np.ndarray         # [d_h], global correct centroid (Eq.6)
    threshold: float         # q-th percentile of pooled per-token correct scores (Eq.8)
    auroc: float = 0.0       # held-out trajectory AUROC (mean aggregation, tex:L305)
    auroc_max: float = 0.0   # held-out trajectory AUROC (max aggregation, tex:L298)

    @property
    def key(self) -> tuple:
        return (self.layer, self.head)

    def proximity(self, a: np.ndarray) -> np.ndarray:
        """Eq.(7): squared norm of projection of centered activation onto error subspace.

        ``a``: [..., d_h] (single or batched head outputs). Returns [...] scalar scores.
        """
        v = a - self.mu_c                      # [..., d_h]
        proj = v @ self.B.T                     # [..., k]
        return np.einsum("...k,...k->...", proj, proj)

    def correct(self, a: np.ndarray, alpha: float) -> np.ndarray:
        """Eq.(9): a - alpha * B^T B (a - mu_c). ``a``: [..., d_h] -> [..., d_h]."""
        v = a - self.mu_c
        proj = v @ self.B.T                     # [..., k]
        return a - alpha * (proj @ self.B)       # broadcast over leading dims


@dataclass
class ManifoldBank:
    """Container for the fitted manifolds of a model x benchmark (SPEC §5.2)."""
    model_id: str
    benchmark: str
    k: int
    q: float
    K: int
    alpha: float
    layers_monitored: list
    selected_heads: list               # [[l,h], ...] top-K by mean-AUROC
    n_problems_fit: int
    n_problems_select: int
    split_seed: int
    heads: dict                        # (l,h) -> HeadManifold  (all monitored, not just selected)
    git_sha: str = ""

    def selected_manifolds(self) -> list:
        return [self.heads[tuple(h)] for h in self.selected_heads]

    # -- persistence --
    def save(self, path_npz: str, path_manifest: str | None = None):
        arrays = {}
        for (l, h), m in self.heads.items():
            arrays[f"B__l{l}_h{h}"] = m.B.astype(np.float32)
            arrays[f"muc__l{l}_h{h}"] = m.mu_c.astype(np.float32)
            arrays[f"thresh__l{l}_h{h}"] = np.float32(m.threshold)
            arrays[f"auroc__l{l}_h{h}"] = np.float32(m.auroc)
        np.savez(path_npz, **arrays)
        manifest = {
            "model_id": self.model_id, "benchmark": self.benchmark,
            "k": self.k, "q": self.q, "K": self.K, "alpha": self.alpha,
            "layers_monitored": self.layers_monitored,
            "selected_heads": self.selected_heads,
            "n_problems_fit": self.n_problems_fit,
            "n_problems_select": self.n_problems_select,
            "split_seed": self.split_seed, "git_sha": self.git_sha,
        }
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        with open(path_manifest, "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, path_npz: str, path_manifest: str | None = None) -> "ManifoldBank":
        if path_manifest is None:
            path_manifest = path_npz + ".manifest.json"
        with open(path_manifest) as f:
            man = json.load(f)
        z = np.load(path_npz)
        heads = {}
        for h in man["selected_heads"]:
            l, hh = h
            key = (l, hh)
            heads[key] = HeadManifold(
                layer=l, head=hh,
                B=z[f"B__l{l}_h{hh}"],
                mu_c=z[f"muc__l{l}_h{hh}"],
                threshold=float(z[f"thresh__l{l}_h{hh}"]),
                auroc=float(z[f"auroc__l{l}_h{hh}"]),
            )
        return cls(
            model_id=man["model_id"], benchmark=man["benchmark"],
            k=man["k"], q=man["q"], K=man["K"], alpha=man["alpha"],
            layers_monitored=man["layers_monitored"],
            selected_heads=[list(h) for h in man["selected_heads"]],
            n_problems_fit=man["n_problems_fit"], n_problems_select=man["n_problems_select"],
            split_seed=man["split_seed"], heads=heads, git_sha=man.get("git_sha", ""),
        )


# ---------------------------------------------------------------------------
# Activation container: per (l,h), per problem -> {correct:[T,d_h], incorrect:[T,d_h]}
# ---------------------------------------------------------------------------
@dataclass
class HeadProblemActivations:
    """All retained-trace activations for one (l,h), keyed by problem id.

    ``correct[pid]`` and ``incorrect[pid]`` are lists of [T,d_h] float32 arrays
    (one per retained trace). Equations 2/6 pool over token steps within each trace.
    """
    correct: dict = field(default_factory=dict)     # pid -> list of [T,d_h]
    incorrect: dict = field(default_factory=dict)    # pid -> list of [T,d_h]

    def problems(self) -> list:
        return list(self.correct.keys())


# ---------------------------------------------------------------------------
# Equations 2 -> 6
# ---------------------------------------------------------------------------
def per_class_means(correct_traces, incorrect_traces):
    """Eq.(2): token-count-weighted per-class means for one problem and one head.

    Each list element is a [T,d_h] array. Returns (mu_c, mu_e) each [d_h].
    Weighting is by token count L_tau (denominator = sum of L_tau), per Eq.(2).
    """
    def mean(traces):
        if not traces:
            return None
        num = sum(t.sum(axis=0) for t in traces)          # [d_h]
        den = sum(t.shape[0] for t in traces)             # sum L_tau
        return num / den
    return mean(correct_traces), mean(incorrect_traces)


def difference_matrix(head_acts: HeadProblemActivations) -> np.ndarray:
    """Eq.(3)+(4): D in R^{N x d_h} (rows = problems). SPEC §7 hazard: rows = problems,
    NOT the transpose. We store D as [N, d_h]; the paper prints D^T in [d_h, N]."""
    rows = []
    for pid in head_acts.problems():
        mu_c, mu_e = per_class_means(head_acts.correct[pid], head_acts.incorrect[pid])
        if mu_c is None or mu_e is None:
            continue   # problem without both classes is discarded (tex:L399)
        rows.append(mu_e - mu_c)                          # Eq.(3)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)      # [N, d_h]


def fit_basis(D: np.ndarray, k: int) -> np.ndarray:
    """Eq.(5): compact SVD D = U S V^T; B = top-k rows of V^T in R^{k x d_h}.

    SPEC §7 hazard: torch.linalg.svd returns Vh == V^T; we take Vh[:k] directly.
    Rows of B are orthonormal (asserted by the caller / tests).
    k is clamped to min(k, N, d_h) so a degenerate smoke model (d_h=1) does not crash.
    """
    N, d_h = D.shape
    k_eff = int(min(k, N, d_h))
    if N == 0 or d_h == 0:
        return np.zeros((0, 0), dtype=np.float32)
    # SVD on D directly (rows = problems). Right singular vectors live in R^{d_h}.
    U, S, Vh = np.linalg.svd(D, full_matrices=False)   # Vh: [r, d_h] = V^T
    B = Vh[:k_eff]                                       # [k_eff, d_h]
    return B.astype(np.float32)


def global_correct_centroid(head_acts: HeadProblemActivations) -> np.ndarray:
    """Eq.(6): token-count-weighted centroid over ALL correct traces of all problems.

    SPEC §7 hazard: denominator pools token counts (sum L_tau), not trace counts.
    """
    num = 0.0
    den = 0
    for pid in head_acts.problems():
        for t in head_acts.correct[pid]:
            num = num + t.sum(axis=0)
            den += t.shape[0]
    if den == 0:
        return None
    return (num / den).astype(np.float32)


def per_token_scores(head_acts: HeadProblemActivations, B, mu_c) -> list:
    """Pooled per-token proximity scores over correct-trace token steps.

    Eq.(8) percentiles act over PER-TOKEN scores pooled across correct traces
    (SPEC §7 hazard: not per-trace maxima/means).
    """
    scores = []
    for pid in head_acts.problems():
        for t in head_acts.correct[pid]:
            v = t - mu_c
            proj = v @ B.T
            scores.extend(np.einsum("tk,tk->t", proj, proj).tolist())
    return scores


def trajectory_score(activations: np.ndarray, B, mu_c, aggregation: str) -> float:
    """Aggregate per-step proximity scores of one trace into a single scalar.

    max: tex:L298 (drift-validation experiment). mean: tex:L305 (head selection).
    SPEC §4.6: we use max for the AUROC diagnostic, mean for production selection.
    """
    if activations.shape[0] == 0:
        return 0.0
    v = activations - mu_c
    proj = v @ B.T
    scores = np.einsum("tk,tk->t", proj, proj)
    return float(scores.max()) if aggregation == "max" else float(scores.mean())


# ---------------------------------------------------------------------------
# Full fit + head selection
# ---------------------------------------------------------------------------
def fit_head(head_acts: HeadProblemActivations, k: int, q: float) -> HeadManifold | None:
    """Fit B, mu_c, threshold for one (l,h). Returns None if degenerate."""
    D = difference_matrix(head_acts)
    if D.shape[0] < 1:
        return None
    B = fit_basis(D, k)
    mu_c = global_correct_centroid(head_acts)
    if mu_c is None:
        return None
    scores = per_token_scores(head_acts, B, mu_c)
    if not scores:
        return None
    threshold = float(np.percentile(np.asarray(scores), q))
    # key unknown until assigned by caller; placeholder 0
    return HeadManifold(layer=-1, head=-1, B=B, mu_c=mu_c, threshold=threshold)


def _safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return 0.5


def head_auroc(head_acts: HeadProblemActivations, B, mu_c, aggregation: str = "mean") -> float:
    """Trajectory-level AUROC for one head on this activation set.

    Aggregation 'mean' -> tex:L305 (head selection), 'max' -> tex:L298 (diagnostic).
    """
    y, s = [], []
    for pid in head_acts.problems():
        for t in head_acts.correct[pid]:
            y.append(0); s.append(trajectory_score(t, B, mu_c, aggregation))
        for t in head_acts.incorrect[pid]:
            y.append(1); s.append(trajectory_score(t, B, mu_c, aggregation))
    return _safe_auc(y, s)


def fit_manifold_bank(
    all_head_acts: dict,              # (l,h) -> HeadProblemActivations  (FULL set)
    train_pids: list,                 # problems used to fit B/mu_c/threshold
    select_pids: list,                # held-out problems for head selection AUROC
    *, model_id: str, benchmark: str, k: int, q: float, K: int, alpha: float,
    layers_monitored: list, split_seed: int, git_sha: str = "",
) -> ManifoldBank:
    """Phase A end-to-end: fit per-head manifolds on train_pids, rank heads by held-out
    (select_pids) mean-AUROC, keep top-K. Returns a ManifoldBank."""
    # restrict activations to train split for fitting
    train_acts = _split_heads(all_head_acts, train_pids)
    select_acts = _split_heads(all_head_acts, select_pids)
    heads = {}
    aurocs = []
    for (l, h), ha in train_acts.items():
        m = fit_head(ha, k, q)
        if m is None:
            continue
        m.layer, m.head = l, h
        # held-out mean-AUROC for selection (tex:L305) and max-AUROC for the diagnostic (tex:L298)
        if select_acts.get((l, h)) is not None and len(select_acts[(l, h)].problems()) > 0:
            m.auroc = head_auroc(select_acts[(l, h)], m.B, m.mu_c, "mean")
            m.auroc_max = head_auroc(select_acts[(l, h)], m.B, m.mu_c, "max")
        else:
            m.auroc = head_auroc(ha, m.B, m.mu_c, "mean")
            m.auroc_max = head_auroc(ha, m.B, m.mu_c, "max")
        heads[(l, h)] = m
        aurocs.append((m.auroc, l, h))
    # top-K by held-out mean-AUROC (SPEC §4.6: mean for production selection)
    aurocs.sort(reverse=True)
    selected = [[l, h] for _, l, h in aurocs[:K]]
    return ManifoldBank(
        model_id=model_id, benchmark=benchmark, k=k, q=q, K=K, alpha=alpha,
        layers_monitored=layers_monitored, selected_heads=selected,
        n_problems_fit=len(train_pids), n_problems_select=len(select_pids),
        split_seed=split_seed, heads=heads, git_sha=git_sha,
    )


def _split_heads(all_head_acts: dict, pids: list) -> dict:
    pidset = set(pids)
    out = {}
    for key, ha in all_head_acts.items():
        sub = HeadProblemActivations()
        for pid in ha.problems():
            if pid in pidset:
                if pid in ha.correct:
                    sub.correct[pid] = ha.correct[pid]
                if pid in ha.incorrect:
                    sub.incorrect[pid] = ha.incorrect[pid]
        if sub.problems():
            out[key] = sub
    return out
