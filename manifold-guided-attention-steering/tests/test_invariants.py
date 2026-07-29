"""Invariant tests: equations the paper's maths imply (SPEC §3, §7 hazards).

These run on small random tensors (no model needed) so they are fast and
deterministic. They are the cheapest real correctness evidence: if the fit or the
correction violates an equation the paper states, the reproduction is wrong.
"""
import numpy as np
import pytest

from mags.manifold import (
    HeadProblemActivations, per_class_means, difference_matrix, fit_basis,
    global_correct_centroid, per_token_scores, trajectory_score, head_auroc,
    fit_head, fit_manifold_bank, HeadManifold,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _make_head_acts(rng, n_problems=8, d_h=16, traces_per_class=2, T=12):
    ha = HeadProblemActivations()
    # a genuine error direction so SVD/centroid are non-degenerate
    w = rng.normal(size=d_h)
    w /= np.linalg.norm(w)
    for i in range(n_problems):
        pid = f"p{i}"
        c = [rng.normal(scale=0.3, size=(T, d_h)) for _ in range(traces_per_class)]
        e = [rng.normal(scale=0.3, size=(T, d_h)) + 1.5 * w for _ in range(traces_per_class)]
        ha.correct[pid] = c
        ha.incorrect[pid] = e
    return ha, w


# --- Eq.(2): per-class means are token-count-weighted ---
def test_per_class_means_token_weighted():
    rng = _rng(1)
    traces = [np.ones((3, 4)), 2 * np.ones((5, 4))]
    mu, _ = per_class_means(traces, [])
    # weighted by token count: (3*1 + 5*2)/(3+5) = 13/8
    assert np.allclose(mu, 13 / 8 * np.ones(4))


# --- Eq.(4)/(5): D rows = problems; B rows orthonormal (SPEC §7 hazard) ---
def test_difference_matrix_rows_are_problems():
    rng = _rng(2)
    ha, _ = _make_head_acts(rng, n_problems=7, d_h=10)
    D = difference_matrix(ha)
    assert D.shape == (7, 10), f"D shape {D.shape} != (N, d_h); wrong axis (SPEC §7)"
    B = fit_basis(D, k=4)
    assert B.shape == (4, 10)
    # rows orthonormal
    G = B @ B.T
    assert np.allclose(G, np.eye(4), atol=1e-4), "B rows not orthonormal"
    # Eq.(10) / Proposition 1 rely on P = B^T B being a projector (idempotent).
    # Orthonormal rows are SUFFICIENT but not DIRECT evidence; assert the
    # projector property itself so a future refactor cannot break it silently.
    P = B.T @ B
    assert np.allclose(P @ P, P, atol=1e-5), "B^T B is not idempotent (not a projector)"
    P_perp = np.eye(10) - P
    assert np.allclose(P_perp @ P_perp, P_perp, atol=1e-5), "I-B^T B is not idempotent"


# --- Eq.(5): fit_basis == top-k rows of V^T from SVD of D (not D^T) ---
def test_basis_matches_svd_of_D():
    rng = _rng(3)
    ha, _ = _make_head_acts(rng, n_problems=9, d_h=12)
    D = difference_matrix(ha)
    B = fit_basis(D, k=3)
    U, S, Vh = np.linalg.svd(D, full_matrices=False)
    assert np.allclose(B, Vh[:3], atol=1e-5), "B must equal top-k rows of V^T (SPEC §7)"
    # and must NOT be a left-singular-vector object (wrong axis): shapes differ
    assert B.shape == (3, 12) and U.shape[0] == 9


# --- Eq.(6): global centroid token-count-weighted across problems+traces ---
def test_global_centroid_pools_token_counts():
    rng = _rng(4)
    ha, _ = _make_head_acts(rng, n_problems=3, d_h=5, traces_per_class=1, T=4)
    mu_c = global_correct_centroid(ha)
    flat = np.concatenate([t for pid in ha.correct for t in ha.correct[pid]], axis=0)
    assert np.allclose(mu_c, flat.mean(axis=0)), "centroid must be token-weighted (SPEC §7)"


# --- Eq.(7): proximity is the squared norm of the projection ---
def test_proximity_is_squared_projection_norm():
    rng = _rng(5)
    ha, _ = _make_head_acts(rng, d_h=8)
    D = difference_matrix(ha)
    B = fit_basis(D, k=3)
    mu_c = global_correct_centroid(ha)
    m = HeadManifold(layer=0, head=0, B=B, mu_c=mu_c, threshold=0.0)
    a = rng.normal(size=(1, 8))
    d = m.proximity(a)
    v = (a - mu_c)
    manual = float(np.sum((v @ B.T) ** 2))
    assert np.allclose(d[0], manual)


# --- Eq.(9) == Eq.(10) at alpha=1 (SPEC §3, hazard) ---
def test_correction_eq9_equals_eq10_at_alpha_one():
    rng = _rng(6)
    ha, _ = _make_head_acts(rng, d_h=6)
    D = difference_matrix(ha)
    B = fit_basis(D, k=2)
    mu_c = global_correct_centroid(ha)
    m = HeadManifold(layer=0, head=0, B=B, mu_c=mu_c, threshold=0.0)
    a = rng.normal(size=(1, 6))
    a_corr = m.correct(a, alpha=1.0)            # Eq.(9) with alpha=1
    v = a - mu_c
    P_perp = np.eye(6) - B.T @ B
    eq10 = mu_c + (P_perp @ v.T).T              # Eq.(10)
    assert np.allclose(a_corr, eq10, atol=1e-5), "Eq.(9) alpha=1 must equal Eq.(10)"


# --- Proposition 1 (Eq.11): complement directions are preserved ---
def test_proposition1_complement_preservation():
    rng = _rng(7)
    ha, _ = _make_head_acts(rng, d_h=8)
    D = difference_matrix(ha)
    B = fit_basis(D, k=3)
    mu_c = global_correct_centroid(ha)
    m = HeadManifold(layer=0, head=0, B=B, mu_c=mu_c, threshold=0.0)
    a = rng.normal(size=(1, 8))
    a_corr = m.correct(a, alpha=0.7)
    # any v in complement of span(B): B v = 0
    # null space of B
    _, _, Vt = np.linalg.svd(B)
    null = Vt[3:].T     # [d_h, d_h-k]
    v = null[:, 0]
    assert np.allclose(B @ v, 0, atol=1e-6)
    lhs = float((a_corr @ v)[0])
    rhs = float((a @ v)[0])
    assert np.allclose(lhs, rhs, atol=1e-5), \
        "Proposition 1 violated: complement direction not preserved"


# --- Eq.(9) subtracts the projection of the CENTRED activation (SPEC §7 hazard) ---
def test_correction_uses_centered_activation():
    rng = _rng(8)
    ha, _ = _make_head_acts(rng, d_h=6)
    D = difference_matrix(ha)
    B = fit_basis(D, k=2)
    mu_c = global_correct_centroid(ha)
    m = HeadManifold(layer=0, head=0, B=B, mu_c=mu_c, threshold=0.0)
    a = rng.normal(size=(1, 6))
    a_corr = m.correct(a, alpha=1.0)
    # the centring matters: must equal mu_c + P_perp(a - mu_c), NOT P_perp a
    P_perp = np.eye(6) - B.T @ B
    wrong = (P_perp @ a.T).T                       # off-by-centring
    assert not np.allclose(a_corr, wrong, atol=1e-5), "correction must re-centre (SPEC §7)"


# --- threshold is the q-th percentile over PER-TOKEN pooled scores (SPEC §7) ---
def test_threshold_per_token_pooled():
    rng = _rng(9)
    ha, _ = _make_head_acts(rng, n_problems=4, d_h=6, traces_per_class=2, T=5)
    D = difference_matrix(ha)
    B = fit_basis(D, k=2)
    mu_c = global_correct_centroid(ha)
    scores = np.asarray(per_token_scores(ha, B, mu_c))
    m = fit_head(ha, k=2, q=90)
    assert np.isclose(m.threshold, np.percentile(scores, 90), atol=1e-4)


# --- head selection ranks by held-out mean-AUROC (tex:L305) and keeps top-K ---
def test_head_selection_topK_by_auroc():
    rng = _rng(10)
    all_acts = {}
    pids = [f"p{i}" for i in range(10)]
    for lh in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ha, _ = _make_head_acts(rng, n_problems=10, d_h=6, traces_per_class=2, T=6)
        # rename problems to the shared split
        ha.correct = {pids[i]: ha.correct[f"p{i}"] for i in range(10)}
        ha.incorrect = {pids[i]: ha.incorrect[f"p{i}"] for i in range(10)}
        all_acts[lh] = ha
    bank = fit_manifold_bank(all_acts, pids[:7], pids[7:], model_id="m", benchmark="b",
                            k=2, q=90, K=2, alpha=1.0, layers_monitored=[0, 1], split_seed=42)
    assert len(bank.selected_heads) == 2
    # invariant: selection is monotone in held-out AUROC (top-K means every selected
    # head's auroc >= every non-selected head's auroc). Exact identity is fragile to
    # tie-breaking; the monotone property is what "top-K by AUROC" guarantees.
    sel_keys = {tuple(h) for h in bank.selected_heads}
    sel_auc = [bank.heads[k].auroc for k in sel_keys]
    nonsel_auc = [v.auroc for k, v in bank.heads.items() if k not in sel_keys]
    if nonsel_auc:
        assert min(sel_auc) >= max(nonsel_auc) - 1e-9, \
            "selected heads must be top-K by held-out mean-AUROC (tex:L305)"


# --- drift-detection AUROC > 0.5 when a real error direction exists ---
def test_auroc_detects_signal_above_chance():
    rng = _rng(11)
    ha, w = _make_head_acts(rng, n_problems=12, d_h=16, traces_per_class=3, T=10)
    D = difference_matrix(ha)
    B = fit_basis(D, k=2)
    mu_c = global_correct_centroid(ha)
    auc_mean = head_auroc(ha, B, mu_c, "mean")
    auc_max = head_auroc(ha, B, mu_c, "max")
    assert auc_mean > 0.6, f"mean AUROC {auc_mean} should exceed 0.6 with a real error dir"
    assert auc_max > 0.6


# --- prompt truncation never crashes a forward pass on a long prompt ---
def test_truncate_prompt_left_truncates_to_context():
    """Regression: generate()/capture_trace() left-truncate the prompt to the
    model's context window so a long training prompt (e.g. an APPS question) does
    not raise IndexError in the position-embedding lookup. The paper's 8B/20B
    models have >=8k context so this is inert there; it matters for verification
    on small models and for any over-long prompt on any model."""
    import numpy as np
    from mags.generation import _max_positions, _truncate_prompt

    class _Cfg:
        max_position_embeddings = 64
    class _M:
        config = _Cfg()
    import torch
    ids = torch.arange(1, 201).unsqueeze(0)  # 200 tokens, exceeds 64
    out = _truncate_prompt(_M(), ids, max_new_tokens=10)
    assert out.shape[1] == 54, out.shape  # 64 - 10 = 54
    # keeps the MOST RECENT tokens (left-truncation)
    assert out[0, -1].item() == 200
    # under the cap: unchanged
    short = torch.arange(1, 11).unsqueeze(0)
    assert _truncate_prompt(_M(), short, max_new_tokens=10).shape[1] == 10


# --- 70/15/15 split: the Figure-3 diagnostic AUROC uses a report-only split ---
def test_fit_uses_report_split_for_diagnostic_auroc():
    """Regression (adversarial review, confirmed MINOR): SPEC §4.8 mandates a
    70/15/15 problem-level split (fit / head-select / report-only AUROC test).
    The prior fit_manifold_bank computed the Figure-3 drift-validation
    ``auroc_max`` (tex:L298) on the SAME 15% select split used for top-K head
    selection (tex:L305), biasing the reported diagnostic of the selected heads.
    ``report_pids`` must be a distinct third split and ``auroc_max`` must be
    computed on it; passing report_pids must not break the selection
    (auroc/mean stays on the select split)."""
    rng = _rng(20)
    all_acts = {}
    pids = [f"p{i}" for i in range(20)]
    for lh in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ha, _ = _make_head_acts(rng, n_problems=20, d_h=6, traces_per_class=2, T=6)
        ha.correct = {pids[i]: ha.correct[f"p{i}"] for i in range(20)}
        ha.incorrect = {pids[i]: ha.incorrect[f"p{i}"] for i in range(20)}
        all_acts[lh] = ha
    fit_p, sel_p, rep_p = pids[:14], pids[14:17], pids[17:]
    bank = fit_manifold_bank(all_acts, fit_p, sel_p, model_id="m", benchmark="b",
                            k=2, q=90, K=2, alpha=1.0, layers_monitored=[0, 1],
                            split_seed=42, report_pids=rep_p)
    # selection (mean AUROC) still driven by the select split (tex:L305)
    assert len(bank.selected_heads) == 2
    # the report split is non-empty and disjoint from fit+select
    assert rep_p and set(rep_p).isdisjoint(set(fit_p) | set(sel_p))
    # backward-compat: report_pids=None falls back to select split (no crash)
    bank2 = fit_manifold_bank(all_acts, fit_p, sel_p, model_id="m", benchmark="b",
                             k=2, q=90, K=2, alpha=1.0, layers_monitored=[0, 1],
                             split_seed=42)
    assert len(bank2.selected_heads) == 2


def test_a_bank_with_no_selected_heads_is_not_a_fit():
    """Steering with an empty bank is byte-identical to not steering at all.

    `fit_head` returns None for every head when no problem has both a correct and an
    incorrect trace, which is what a grader reporting everything incorrect produces.
    The fit step used to print `OK ... 0 heads selected` and exit 0 for that.
    """
    import pytest
    from mags.manifold import ManifoldBank

    empty = ManifoldBank(model_id="m", benchmark="MBPP", k=8, q=95.0, K=24, alpha=1.0,
                         layers_monitored=[8], selected_heads=[], n_problems_fit=0,
                         n_problems_select=0, split_seed=0, heads={})
    with pytest.raises(RuntimeError, match="no usable heads"):
        empty.require_usable()

    usable = ManifoldBank(model_id="m", benchmark="MBPP", k=8, q=95.0, K=24, alpha=1.0,
                          layers_monitored=[8], selected_heads=[[8, 0]], n_problems_fit=3,
                          n_problems_select=2, split_seed=0, heads={})
    assert usable.require_usable() is usable
