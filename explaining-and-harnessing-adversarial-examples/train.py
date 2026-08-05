"""Training loop for the EAE reproduction.

Supports (SPEC §1.2, §5, §6):
  - plain SGD+momentum (our choice; paper silent)
  - adversarial training: J_tilde = alpha*J(x,y) + (1-alpha)*J(x_adv,y), alpha=0.5,
    x_adv built from CURRENT theta with the perturbation detached (sign under
    no_grad) so gradients flow into theta but not through sign (tex:486-488).
  - noise controls: per-pixel +/-eps (Rademacher) or U(-eps,eps), resampled per
    minibatch (tex:555-557).
  - L1 weight decay on the FIRST layer only (tex:429-430).
  - early stopping on val_err or adv_val_err with patience (tex:501-505).
  - retrain on the full 60k after choosing the epoch count (tex:505-506).

Fails LOUDLY on empty data / degenerate runs. history["final_model"] holds the
trained nn.Module.
"""
import random

import numpy as np
import torch
import torch.nn as nn

import eval as ev


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _data_tensor(d, key):
    v = d[key]
    if not torch.is_tensor(v):
        v = torch.as_tensor(v)
    return v.float() if v.dtype == torch.float64 or v.dtype == torch.float32 else v.to(torch.long)


def _add_noise(x, cfg, gen):
    noise = cfg.get("noise")
    if noise is None:
        return x
    eps = float(noise.get("eps", 0.25))
    if eps == 0.0:
        return x  # degeneracy: eps=0 noise is a true no-op (no RNG consumed)
    typ = noise.get("type", "uniform")
    if typ == "rademacher":
        signs = torch.randint(0, 2, x.shape, generator=gen).float() * 2 - 1
        n = eps * signs
    elif typ == "uniform":
        n = (torch.rand(x.shape, generator=gen) * 2 - 1) * eps
    else:
        raise ValueError(f"unknown noise type {typ}")
    return x + n


def _l1_first_layer_penalty(model, coef):
    if coef is None or coef <= 0:
        return 0.0
    # first weight-bearing parameter: linear/maxout first block weight or conv first stage
    for m in model.modules():
        if isinstance(m, (nn.Linear)) or m.__class__.__name__ in ("MaxoutLinear",):
            if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                return coef * m.weight.abs().sum()
        if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad \
                and m.weight.dim() >= 2 and m.__class__.__name__ == "Conv2d":
            return coef * m.weight.abs().sum()
    return 0.0


def _build_x_adv(model, x, y, eps):
    """Compute the FGSM perturbation of the CURRENT theta, returning x_adv
    detached so the training loss J(theta, x_adv, y) backprops into theta only
    (not through the sign of the input gradient). The input-gradient step must
    run with grad ENABLED (torch.autograd.grad needs a graph), then the result
    is detached. (tex:486-488; sign is non-differentiable, so grads flow
    through x_adv as a constant w.r.t. theta.)

    The FGSM direction is computed with the model in EVAL mode (deterministic
    network, dropout OFF). The paper's FGSM (tex:309, Fig.1) is defined on the
    cost of the deterministic network; computing it under an active dropout
    mask zeroes the input gradient on masked pixels and yields a materially
    different/weaker perturbation (review finding: the train-mode FGSM sign is
    exactly 0 on ~20% of pixels). The adversarial-half LOSS is then evaluated
    in the caller's (train) mode, with dropout, as part of training — only the
    perturbation direction is made deterministic.
    """
    was_training = model.training
    model.eval()
    try:
        x_req = x.detach().requires_grad_(True)
        loss0 = model.loss(x_req, y)
        g = torch.autograd.grad(loss0, x_req, create_graph=False)[0]
        x_req.requires_grad_(False)
        eta = eps * torch.sign(g)
        return (x + eta).detach()
    finally:
        model.train(was_training)


def _eval_train_err(model, x, y, batch=2000):
    model.eval()
    idx = torch.randperm(x.shape[0])[:min(batch, x.shape[0])]
    with torch.no_grad():
        pred = model.predict(x[idx])
    return float((pred != y[idx].view(-1)).float().mean().item()) * 100.0


def train(model, data, cfg):
    """Train `model` on `data` per `cfg`. Returns history dict."""
    if data is None or "x_train" not in data:
        raise ValueError("train: missing data['x_train']")
    _set_seed(cfg.get("seed", 0))
    device = torch.device("cpu")

    x_train = _data_tensor(data, "x_train")
    y_train = _data_tensor(data, "y_train")
    x_val = _data_tensor(data, "x_val")
    y_val = _data_tensor(data, "y_val")
    if x_train.numel() == 0:
        raise ValueError("train: empty x_train")

    lr = float(cfg.get("lr", 0.05))
    momentum = float(cfg.get("momentum", 0.9))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    batch_size = int(cfg.get("batch_size", 128))
    max_epochs = int(cfg.get("max_epochs", 60))
    adversarial = cfg.get("adversarial")
    l1_coef = cfg.get("l1_first_layer")
    early_stop = cfg.get("early_stop")
    retrain_full = bool(cfg.get("retrain_full_60k", False))

    adv_eps = float((adversarial or {}).get("eps", 0.25)) if adversarial else 0.25
    adv_alpha = float((adversarial or {}).get("alpha", 0.5)) if adversarial else 0.5
    # degeneracy: adversarial eps=0 is a true no-op — x_adv == x, so skip the
    # redundant adversarial half entirely (keeps the code path bit-identical to
    # plain training, which the degeneracy test asserts).
    adv_active = adversarial is not None and adv_eps > 0.0

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Capture the INITIAL weights (tex:505-506: the retrain-on-60k is FROM
    # SCRATCH — it must restart from the init weights with a FRESH optimizer,
    # not continue from the Phase-1 state with carried momentum). Stored so
    # Phase 2 can reload them; also exposed in history for the guard test.
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    history = {"epochs": [], "train_err": [], "val_err": [], "adv_val_err": []}
    best_metric = float("inf")
    best_state = None
    best_epoch = 0
    patience = int((early_stop or {}).get("patience", 100)) if early_stop else None
    monitor = (early_stop or {}).get("monitor", "val_err") if early_stop else "val_err"
    since_best = 0

    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 1)

    # Phase 1: train up to max_epochs with optional early stopping
    best_state, best_metric, best_epoch = None, float("inf"), 0
    since_best = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(x_train.shape[0], generator=gen)
        for i in range(0, x_train.shape[0], batch_size):
            batch_idx = perm[i:i + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            xb = _add_noise(xb, cfg, gen)
            loss = model.loss(xb, yb)
            if adv_active:
                x_adv = _build_x_adv(model, xb, yb, adv_eps)
                loss_adv = model.loss(x_adv, yb)
                loss = adv_alpha * loss + (1 - adv_alpha) * loss_adv
            l1 = _l1_first_layer_penalty(model, l1_coef)
            if not torch.is_tensor(l1):
                l1 = torch.tensor(0.0)
            total = loss + l1
            opt.zero_grad()
            total.backward()
            opt.step()
        tr_err = _eval_train_err(model, x_train, y_train)
        with torch.no_grad():
            val_pred = model.predict(x_val)
        val_err = float((val_pred != y_val.view(-1)).float().mean().item()) * 100.0
        adv_val = ev.adv_eval(model, x_val, y_val, eps=adv_eps)["adv_err"]
        history["epochs"].append(len(history["epochs"]) + 1)
        history["train_err"].append(tr_err)
        history["val_err"].append(val_err)
        history["adv_val_err"].append(adv_val)
        metric = val_err if monitor == "val_err" else adv_val
        if metric < best_metric - 1e-6:
            best_metric = metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            since_best = 0
        else:
            since_best += 1
        print(f"  epoch {ep} train_err={tr_err:.3f} val_err={val_err:.3f} adv_val_err={adv_val:.3f}", flush=True)
        if early_stop is not None and since_best >= patience:
            print(f"  early stop at epoch {ep} (patience {patience} on {monitor})", flush=True)
            break

    # degeneracy guard: nothing learned
    if best_state is None:
        raise ValueError("train: no epoch improved the monitor metric (degenerate run)")

    # Phase 2: optional retrain on full 60k for best_epoch epochs (tex:505-506).
    # This is a FROM-SCRATCH retrain: reload the INITIAL weights and use a FRESH
    # optimizer (no carried momentum) + a FRESH RNG stream, then train for the
    # early-stopped epoch count on the full 60k. The earlier implementation
    # continued Phase 2 from the Phase-1 state with the same optimizer, which
    # contradicts the paper's from-scratch protocol; guarded by
    # tests/test_degeneracy.py::test_retrain_full_60k_is_from_scratch.
    if retrain_full and "x_train_full" in data:
        x_full = _data_tensor(data, "x_train_full")
        y_full = _data_tensor(data, "y_train_full")
        print(f"  retraining on full 60k for {best_epoch} epochs (from scratch)", flush=True)
        model.load_state_dict(init_state)
        # expose the Phase-2 starting point for the guard test (== init_state)
        history["phase2_start_state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history["init_state"] = init_state
        opt2 = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        gen2 = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 7)
        model.train()
        for ep in range(1, best_epoch + 1):
            perm = torch.randperm(x_full.shape[0], generator=gen2)
            for i in range(0, x_full.shape[0], batch_size):
                batch_idx = perm[i:i + batch_size]
                xb = x_full[batch_idx]
                yb = y_full[batch_idx]
                xb = _add_noise(xb, cfg, gen2)
                loss = model.loss(xb, yb)
                if adv_active:
                    x_adv = _build_x_adv(model, xb, yb, adv_eps)
                    loss_adv = model.loss(x_adv, yb)
                    loss = adv_alpha * loss + (1 - adv_alpha) * loss_adv
                l1 = _l1_first_layer_penalty(model, l1_coef)
                if not torch.is_tensor(l1):
                    l1 = torch.tensor(0.0)
                total = loss + l1
                opt2.zero_grad()
                total.backward()
                opt2.step()
            # record retrain epoch metrics (overwrite history tail to reflect final model)
            tr_err = _eval_train_err(model, x_full, y_full)
            with torch.no_grad():
                val_pred = model.predict(x_val)
            val_err = float((val_pred != y_val.view(-1)).float().mean().item()) * 100.0
            adv_val = ev.adv_eval(model, x_val, y_val, eps=adv_eps)["adv_err"]
            print(f"  retrain epoch {ep} train_err={tr_err:.3f} val_err={val_err:.3f} adv_val_err={adv_val:.3f}", flush=True)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    history["final_model"] = model
    history["best_epoch"] = best_epoch
    return history
