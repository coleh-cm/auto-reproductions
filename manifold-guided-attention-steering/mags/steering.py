"""Steering controllers (SPEC §5.4, §5.5).

A controller is a callable ``(layer, x_heads) -> Optional[x_heads]`` installed as the
HookRegistry callback. ``x_heads`` is a head-contiguous view ``[bsz, seq, H, d_h]``.

MAGS (Eq.7-9) is implemented in fp32. The other arms live in mags/baselines.py.
"""
from __future__ import annotations
import numpy as np
import torch


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(torch.float32).cpu().numpy()


class MAGSController:
    """Algorithm 1: per decode step, per monitored head, score+conditional-correct.

    Installed on monitored layers' ``W_O`` pre-hook. Pass-through on prefill
    (seq > 1) per SPEC §4.9: prefill positions are not steered, only decode steps are.
    Corrections apply in fp32 (SPEC §5: fitted params fp32).
    """

    def __init__(self, bank, alpha: float | None = None, score_only: bool = False,
                 log_path: str | None = None, problem_id: str | None = None):
        # bank: ManifoldBank (only selected heads are active)
        self.alpha = bank.alpha if alpha is None else alpha
        self.score_only = score_only     # True for unsteered-with-scoring / diagnostics
        self.log_path = log_path
        self.problem_id = problem_id
        self._heads = {tuple(h): bank.heads[tuple(h)] for h in bank.selected_heads}
        self._log = []
        self._decode_step = 0          # per-decode-step counter (one increment per NEW step)
        self._last_layer_seen = None  # detect step boundary across monitored layers

    def begin_problem(self, problem_id: str):
        """Reset per-problem state so the decode-step index and log are per-problem."""
        self.problem_id = problem_id
        self._decode_step = 0
        self._last_layer_seen = None

    def __call__(self, layer: int, x_heads: torch.Tensor):
        bsz, seq, H, dh = x_heads.shape
        if seq != 1:
            # prefill (or any multi-position forward): pass-through (SPEC §4.9)
            return None
        # decode step: only the single generated position. Hooks fire once per
        # monitored layer per decode step; increment the decode-step counter once per
        # NEW step (detected when the layer index wraps back to the first monitored
        # layer), so `t` is the decode-step index, not the cumulative hook-call index.
        layers_order = sorted({l for l, _ in self._heads.keys()})
        if not layers_order:
            return None
        if self._last_layer_seen is None or layer <= self._last_layer_seen:
            self._decode_step += 1
        self._last_layer_seen = layer
        t = self._decode_step
        x = _to_np(x_heads)               # [bsz,1,H,dh] -> [bsz,1,H,dh]
        modified = np.array(x, copy=True)
        for (l, h), m in self._heads.items():
            if l != layer:
                continue
            a = x[0, 0, h, :]              # [dh] (batch size 1)
            d = float(m.proximity(a.reshape(1, -1))[0])     # Eq.(7)
            fired = d > m.threshold                          # Eq.(8)
            if self.log_path is not None:
                self._log.append({"problem": self.problem_id, "t": t,
                                  "head": [l, h], "d": d, "fired": bool(fired)})
            if fired and not self.score_only:
                modified[0, 0, h, :] = m.correct(a.reshape(1, -1), self.alpha)[0]   # Eq.(9)
        # if nothing changed, return None to skip an unnecessary tensor copy
        if self.score_only:
            return None
        if not np.any(modified != x):
            return None
        out = torch.from_numpy(modified).to(x_heads.device).to(x_heads.dtype)
        return out.view(bsz, seq, H, dh)

    def flush_log(self):
        if self.log_path and self._log:
            import json, os
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a") as f:
                for rec in self._log:
                    f.write(json.dumps(rec) + "\n")
        self._log = []


class NoOpController:
    """Unsteered baseline: never modifies. Used as the reference arm and the
    degeneracy no-op (MAGS at alpha=0 / untriggerable threshold reduces to this)."""
    def __init__(self, score_only: bool = True):
        self.score_only = score_only

    def __call__(self, layer: int, x_heads: torch.Tensor):
        return None
