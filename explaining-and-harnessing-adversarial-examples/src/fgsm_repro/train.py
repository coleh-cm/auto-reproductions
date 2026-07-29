"""Training loop for the FGSM reproduction.

Implements the SGD-with-momentum trainer shared by every milestone. The only
optimizer details the paper states are alpha=0.5, eps=0.25 for adversarial
training (tex:488, tex:428-429); *everything* about the optimizer itself (lr,
momentum schedule, weight-norm constraint, batch size, epoch count) is taken
from the EXTERNAL pylearn2 maxout recipe
(`lisa-lab/pylearn2` `scripts/papers/maxout/mnist_pi.yaml`, fetched 2026-07-29),
NOT from the paper. See SPEC.md section 6 "External". Each external choice is
flagged in a comment below.

SCOPE (SPEC.md section 6 "External"): the external maxout recipe — lr
exponential adjust (*1.000004 each step, floor 1e-6), momentum ramp
(0.5 -> 0.7 linearly by epoch 250), and max_col_norm 1.9365 column-norm
clamping — is adopted "for M3/M4 arms", i.e. it applies to MaxoutMLP ONLY.
SoftmaxRegression (M1), LogisticRegression (M2) and RBFNet (M8) train with
plain fixed-lr / fixed-momentum SGD and NO weight-norm constraint. Applying
the maxout recipe to M1/M2 is an unstated deviation, so this trainer gates
every external piece on ``isinstance(model, MaxoutMLP)``.

Frozen interface (SPEC.md section 4):
    @dataclass TrainConfig: batch_size, lr, momentum, max_epochs, seed,
        adv_train=False, eps=0.25, alpha=0.5, early_stop="clean", patience=100,
        max_steps=None, init_seed=None
    @dataclass TrainResult: best_state_dict, history, epochs_run, best_metric,
        steps_run
    def train(model, cfg, data) -> TrainResult
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol, TYPE_CHECKING

import torch

from fgsm_repro.objectives import adversarial_train_cost, cross_entropy_cost
from fgsm_repro.eval import eval_clean, eval_fgsm
from fgsm_repro.models import MaxoutMLP

if TYPE_CHECKING:
    from fgsm_repro.data import MNISTData


@dataclass
class TrainConfig:
    """Hyperparameters for one training run.

    ``max_steps`` and ``init_seed`` are implementation additions (not in the
    paper); they let the degeneracy self-check bound compute and let callers
    separate data-shuffle RNG from weight-init RNG.
    """

    batch_size: int
    lr: float
    momentum: float
    max_epochs: int
    seed: int
    adv_train: bool = False
    eps: float = 0.25
    alpha: float = 0.5
    early_stop: Literal["clean", "adversarial"] = "clean"
    patience: int = 100
    max_steps: Optional[int] = None
    init_seed: Optional[int] = None


@dataclass
class TrainResult:
    best_state_dict: dict
    history: list  # list of {'train_loss','clean_valid_error','adv_valid_error'}
    epochs_run: int
    best_metric: float
    steps_run: int


class _Trainable(Protocol):
    """Minimal structural type train() relies on."""

    def logits(self, x: torch.Tensor) -> torch.Tensor: ...
    def parameters(self) -> list: ...
    def state_dict(self) -> dict: ...
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...


# External maxout recipe constants (pylearn2 mnist_pi.yaml). Not paper-stated.
# Scoped to MaxoutMLP only (SPEC.md section 6 "External").
_MAX_COL_NORM = 1.9365     # column L2-norm clamp on maxout weight matrices.
_LR_SCALE = 1.000004       # lr *= 1.000004 each update (pylearn2); not paper-stated.
_LR_FLOOR = 1e-6          # clamp lr >= 1e-6 (pylearn2); not paper-stated.
_MOM_RAMP_EPOCHS = 250     # momentum ramps over 250 epochs (pylearn2); not paper-stated.
_MOM_RAMP_WIDTH = 0.2      # pylearn2 ramps .5 -> .7; width 0.2; not paper-stated.


def _set_init_seed(model: Any, seed: int) -> None:
    """Re-seed the model's per-module RNG once (determinism of dropout).

    ``MaxoutMLP.set_seed`` resets its dropout ``torch.Generator``; models with
    no RNG (softmax / logistic / RBF) simply lack the method and are skipped.
    Weight init itself already happened in the model constructor; this only
    fixes the dropout stream so identical ``cfg.seed`` reproduces identical
    dropout masks.
    """
    fn = getattr(model, "set_seed", None)
    if callable(fn):
        fn(int(seed))


def _apply_max_col_norm(model: "MaxoutMLP", max_col_norm: float) -> None:
    """External maxout recipe: clamp the L2-norm of every *column* (i.e. every
    OUTPUT unit's incoming-weight vector) of each maxout weight matrix to
    ``max_col_norm`` AFTER each SGD step (pylearn2 mnist_pi.yaml,
    max_col_norm 1.9365). Not paper-stated.

    This operates DIRECTLY on the MaxoutMLP weight matrices. The 1-D bias
    parameters are left untouched. We clamp the per-output-unit norm for EACH
    weight matrix explicitly, because the two layer kinds store their weights
    on DIFFERENT axes:
      - ``_MaxoutLayer.W`` has shape ``[in, out]``  -> each column W[:, j] is
        one output unit's incoming weights -> norm over ``dim=0``.
      - the readout ``nn.Linear(units, n_classes).weight`` has shape
        ``[out=n_classes, in=units]`` (PyTorch convention) -> each output
        class's incoming weights are a ROW -> norm over ``dim=1``.
    pylearn2 ``max_col_norm`` constrains the per-OUTPUT-unit incoming-weight
    L2 norm, so we must reduce over the INPUT axis of each matrix. Using a
    blanket ``p.norm(dim=0)`` for all 2-D weights (the prior bug) clamped the
    wrong axis on the readout (per-INPUT-unit norm over 10 classes instead of
    per-OUTPUT-class norm over 240 units) and silently no-op'd on the readout's
    real constraint; the self-check below asserts the correct axis too.

    Rescaling: multiply each output unit's column/row by
    ``min(1, max_col_norm / norm)`` so under-normed units are unchanged and
    only over-normed units are shrunk (in place).
    """
    with torch.no_grad():
        # _MaxoutLayer.W : [in, out] -> per-output norm over the input axis (dim 0)
        for layer in (model.layer0, model.layer1):
            W = layer.W  # [in, out]
            norms = W.norm(dim=0, keepdim=True)  # [1, out]
            factor = (max_col_norm / norms.clamp_min(1e-12)).clamp(max=1.0)
            W.mul_(factor)
        # readout nn.Linear.weight : [out=n_classes, in=units] -> per-output norm
        # over the input axis (dim 1).
        Rw = model.readout.weight  # [n_classes, units]
        norms = Rw.norm(dim=1, keepdim=True)  # [n_classes, 1]
        factor = (max_col_norm / norms.clamp_min(1e-12)).clamp(max=1.0)
        Rw.mul_(factor)


def train(model: Any, cfg: TrainConfig, data: "MNISTData") -> TrainResult:
    """Train ``model`` with SGD + momentum, returning the best checkpoint.

    Determinism contract: two calls with identical ``cfg`` (and identical
    ``data``) produce bit-identical ``best_state_dict`` and ``history``.
    """
    # --- RNG: weight init seeded once from init_seed (or seed); shuffle RNG
    # seeded from seed. Keeping the two separate lets a caller fix init while
    # varying the data order. ---
    init_seed = cfg.init_seed if cfg.init_seed is not None else cfg.seed
    _set_init_seed(model, init_seed)

    # External maxout recipe is scoped to MaxoutMLP ONLY (SPEC.md section 6
    # "External": adopted for M3/M4 arms). Non-maxout models (M1 softmax, M2
    # logistic, M8 RBF) train with plain fixed-lr / fixed-momentum SGD and no
    # weight-norm constraint.
    is_maxout = isinstance(model, MaxoutMLP)

    gen = torch.Generator()
    gen.manual_seed(int(cfg.seed))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(cfg.lr), momentum=float(cfg.momentum)
    )

    x_train, y_train = data.x_train, data.y_train
    x_valid, y_valid = data.x_valid, data.y_valid
    n_train = int(x_train.size(0))

    cur_lr = float(cfg.lr)
    best_metric = float("inf")
    best_state_dict: Optional[dict] = None
    patience_counter = 0
    steps = 0
    history: list = []
    epochs_run = 0

    reached_max_steps = False
    for epoch in range(int(cfg.max_epochs)):
        epochs_run = epoch + 1

        model.train()
        # External maxout recipe (MaxoutMLP only): momentum ramps linearly from
        # cfg.momentum to cfg.momentum + 0.2 over 250 epochs. With the maxout
        # default cfg.momentum = 0.5 this reproduces the pylearn2 .5 -> .7
        # schedule. Not paper-stated. Non-maxout models keep cfg.momentum fixed.
        if is_maxout:
            mom = float(cfg.momentum) + _MOM_RAMP_WIDTH * min(
                epoch / _MOM_RAMP_EPOCHS, 1.0
            )
        else:
            mom = float(cfg.momentum)
        for group in optimizer.param_groups:
            group["momentum"] = mom

        perm = torch.randperm(n_train, generator=gen)
        loss_sum = 0.0
        loss_count = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            xb = x_train[idx]
            yb = y_train[idx]

            if cfg.adv_train:
                loss = adversarial_train_cost(
                    model, xb, yb, float(cfg.eps), float(cfg.alpha)
                )
            else:
                loss = cross_entropy_cost(model, xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # External maxout recipe (MaxoutMLP only): lr *= 1.000004 each step,
            # clamped >= 1e-6 (pylearn2 mnist_pi.yaml). Not paper-stated.
            # Non-maxout models keep cfg.lr fixed for the whole run.
            if is_maxout:
                cur_lr = max(_LR_FLOOR, cur_lr * _LR_SCALE)
                for group in optimizer.param_groups:
                    group["lr"] = cur_lr
            optimizer.step()

            # External maxout recipe (MaxoutMLP only): max_col_norm column-norm
            # clamp AFTER the step. Enforced directly on the weight matrices so
            # it cannot silently no-op. Not paper-stated.
            if is_maxout:
                _apply_max_col_norm(model, _MAX_COL_NORM)

            steps += 1
            loss_sum += float(loss.item()) * int(xb.size(0))
            loss_count += int(xb.size(0))

            if cfg.max_steps is not None and steps >= cfg.max_steps:
                reached_max_steps = True
                break

        train_loss = loss_sum / max(loss_count, 1)

        # --- per-epoch validation (model in eval mode => dropout off) ---
        model.eval()
        clean_acc = float(eval_clean(model, x_valid, y_valid))
        clean_valid_error = 1.0 - clean_acc
        adv_valid_error = float(
            eval_fgsm(model, x_valid, y_valid, float(cfg.eps)).error_rate
        )

        history.append(
            {
                "train_loss": train_loss,
                "clean_valid_error": clean_valid_error,
                "adv_valid_error": adv_valid_error,
            }
        )

        if cfg.early_stop == "adversarial":
            metric = adv_valid_error
        else:
            metric = clean_valid_error

        if metric < best_metric:
            best_metric = metric
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

        if reached_max_steps:
            break

    if best_state_dict is None:
        # No epoch ever improved on +inf (e.g. max_steps == 0): keep the final
        # state so the checkpoint is never empty.
        best_state_dict = copy.deepcopy(model.state_dict())

    return TrainResult(
        best_state_dict=best_state_dict,
        history=history,
        epochs_run=epochs_run,
        best_metric=best_metric,
        steps_run=steps,
    )


# --------------------------------------------------------------------------- #
# Self-check (run as ``python -m fgsm_repro.train``)
# --------------------------------------------------------------------------- #
def _build_synthetic_data(
    n_train: int = 200, n_valid: int = 80, n_test: int = 80, seed: int = 0
) -> "MNISTData":
    """Tiny 10-class synthetic MNISTData for the degeneracy self-check."""
    from fgsm_repro.data import MNISTData

    g = torch.Generator().manual_seed(seed)
    n_classes = 10
    in_dim = 784

    def make(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, in_dim, generator=g, dtype=torch.float32)
        # class-discriminative signal so the model has something to learn:
        # bias each class's mean by a fixed per-class offset.
        y = torch.randint(0, n_classes, (n,), generator=g, dtype=torch.int64)
        x[torch.arange(n), y] += 0.5  # brighten the pixel indexed by the label
        return x, y

    x_train, y_train = make(n_train)
    x_valid, y_valid = make(n_valid)
    x_test, y_test = make(n_test)
    return MNISTData(
        x_train=x_train, y_train=y_train,
        x_valid=x_valid, y_valid=y_valid,
        x_test=x_test, y_test=y_test,
    )


def _state_equal(a: dict, b: dict) -> bool:
    """Bit-identical comparison of two state_dicts (keys + tensor bytes)."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        ta, tb = a[k], b[k]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            if ta != tb:
                return False
            continue
        if ta.dtype != tb.dtype or ta.shape != tb.shape:
            return False
        if not torch.equal(ta, tb):
            return False
    return True


def _self_check() -> str:
    """Degeneracy self-check: with eps=0, adversarial training (E7) must reduce
    EXACTLY to clean training, so two runs of the SAME seed — one clean, one
    adversarial with eps=0 — must produce bit-identical best_state_dict.

    Uses a tiny MaxoutMLP (units=8, pieces=2, dropout include 1.0/1.0 so
    dropout is identity) and synthetic MNISTData built inline. Runs max_steps=5
    so the check is fast.
    """
    lines: list[str] = []

    data = _build_synthetic_data()
    common = dict(
        batch_size=32, lr=0.1, momentum=0.5, max_epochs=10, seed=0,
        max_steps=5, early_stop="clean", patience=100,
    )

    # MaxoutMLP.__init__ draws its weights from the *global* torch RNG (the
    # ``seed`` ctor arg only fixes the dropout generator). So to give both
    # runs bit-identical starting weights we re-seed the global RNG before
    # each construction. train() itself uses a private torch.Generator for
    # shuffling and consumes no global RNG, so the two runs stay aligned.
    # Clean run.
    torch.manual_seed(0)
    model_clean = MaxoutMLP(units=8, pieces=2, in_dim=784, n_classes=10,
                            dropout_input_include=1.0, dropout_hidden_include=1.0,
                            seed=0)
    cfg_clean = TrainConfig(adv_train=False, eps=0.25, alpha=0.5, **common)
    res_clean = train(model_clean, cfg_clean, data)

    # Adversarial run with eps=0 -> x_tilde == x exactly, so E7 == clean cost.
    torch.manual_seed(0)
    model_adv = MaxoutMLP(units=8, pieces=2, in_dim=784, n_classes=10,
                          dropout_input_include=1.0, dropout_hidden_include=1.0,
                          seed=0)
    cfg_adv = TrainConfig(adv_train=True, eps=0.0, alpha=0.5, **common)
    res_adv = train(model_adv, cfg_adv, data)

    lines.append(
        f"clean: steps={res_clean.steps_run} epochs={res_clean.epochs_run} "
        f"best_metric={res_clean.best_metric:.6f}"
    )
    lines.append(
        f"adv(eps=0): steps={res_adv.steps_run} epochs={res_adv.epochs_run} "
        f"best_metric={res_adv.best_metric:.6f}"
    )

    assert res_clean.steps_run == res_adv.steps_run == 5, (
        f"steps mismatch: {res_clean.steps_run} vs {res_adv.steps_run}"
    )

    ok = _state_equal(res_clean.best_state_dict, res_adv.best_state_dict)
    lines.append(f"best_state_dict bit-identical: {ok}")
    assert ok, (
        "Degeneracy FAILED: adv_train(eps=0) != clean_train under same seed; "
        "E7 must reduce to the clean cost when eps=0."
    )

    # Sanity: confirm max_col_norm is actually enforced on every output unit's
    # incoming-weight vector. _MaxoutLayer.W is [in,out] -> norm over dim 0;
    # the readout nn.Linear.weight is [out,in] -> norm over dim 1 (PyTorch
    # convention). A blanket p.norm(dim=0) (the prior bug) would assert the
    # wrong axis on the readout and pass tautologically.
    max_col = 0.0
    for layer in (model_clean.layer0, model_clean.layer1):
        max_col = max(max_col, layer.W.norm(dim=0).max().item())
    max_col = max(max_col, model_clean.readout.weight.norm(dim=1).max().item())
    lines.append(f"max per-output column-norm after training (<=1.9365 expected): {max_col:.6f}")
    assert max_col <= 1.9365 + 1e-5, (
        f"max_col_norm not enforced: column norm {max_col} > 1.9365"
    )

    # Sanity: a non-maxout model (SoftmaxRegression) trains and the recipe is
    # NOT applied (lr stays fixed; no norm clamp). Just confirm it runs and
    # produces a non-empty checkpoint.
    from fgsm_repro.models import SoftmaxRegression

    model_sr = SoftmaxRegression(in_dim=784, n_classes=10)
    cfg_sr = TrainConfig(batch_size=32, lr=0.1, momentum=0.5, max_epochs=2,
                         seed=0, max_steps=3, early_stop="clean", patience=100)
    res_sr = train(model_sr, cfg_sr, data)
    lines.append(
        f"softmax: steps={res_sr.steps_run} epochs={res_sr.epochs_run} "
        f"best_metric={res_sr.best_metric:.6f} keys={len(res_sr.best_state_dict)}"
    )
    assert len(res_sr.best_state_dict) > 0 and res_sr.steps_run == 3

    lines.append("PASS")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_self_check())
