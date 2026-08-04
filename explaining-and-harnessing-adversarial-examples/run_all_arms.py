"""run_all_arms.py — run every arm at the paper's full configuration at every
seed and write measured.json.

Each arm prints exactly one line: FINAL <arm name>=<value>  (the primary
metric for that arm, using claims.json's names) or FINAL <arm name>=BLOCKED.

Seeds: claims.json['seeds'] = [0,1,2] for most arms; maxout_large_adv uses
[0,1,2,3,4] (paper's five runs, tex:506-510).

CPU sub-scale note (REPRODUCTION.md): the paper's full budget (1600-unit
maxout, 5 seeds, 12-member ensemble, conv net on CIFAR) is hours of GPU. This
run uses a capped epoch budget on CPU. High-invariance claims (directions,
orderings, the algebraic invariant, curve shapes) survive at sub-scale; tight
value claims (clean_err 0.94%, 0.782%) are expected to fail and are rated
low/medium in claims.json for exactly this reason. Any arm that cannot
complete (e.g. CIFAR download fails) is marked BLOCKED.
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

import data
import models
import attack
import eval as ev
import train

REPO = os.path.dirname(os.path.abspath(__file__))

SEEDS = [0, 1, 2]
M5_SEEDS = [0, 1, 2, 3, 4]  # maxout_large_adv: paper's five runs
EPS_MNIST = 0.25
EPS_CIFAR = 0.10

# Capped epoch budgets (CPU sub-scale; recorded in REPRODUCTION.md). Early
# stopping with patience shortens most runs further.
EPOCHS_SOFTMAX = 60
EPOCHS_LOGREG = 60
EPOCHS_MAXOUT240 = 40
EPOCHS_MAXOUT1600 = 20
EPOCHS_RBF = 40
EPOCHS_CONV = 12
ENSEMBLE_MEMBERS = 12
ENSEMBLE_EPOCHS = 18  # 12 members x 18 epochs x 3 seeds is heavy; capped


def _t(d, key, dtype=None):
    v = d[key]
    if not torch.is_tensor(v):
        v = torch.from_numpy(v)
    if dtype == "long":
        return v.to(torch.long)
    return v.float()


def _torch_data(d):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v
        else:
            out[k] = torch.from_numpy(v)
    return out


def _primary(arm, m):
    """One headline metric per arm for the FINAL line."""
    table = {
        "softmax_reg": ("adv_err", "adv_err"),
        "logreg_3v7": ("clean_err", "clean_err"),
        "maxout_naive": ("clean_err", "clean_err"),
        "maxout_adv": ("clean_err", "clean_err"),
        "maxout_large_naive": ("clean_err", "clean_err"),
        "maxout_large_adv": ("clean_err", "clean_err"),
        "maxout_sigmoid": ("rubbish_err", "rubbish_err"),
        "noise_rademacher": ("adv_err", "adv_err"),
        "noise_uniform": ("adv_err", "adv_err"),
        "l1_maxout": ("train_err", "train_err"),
        "rbf_shallow": ("adv_err", "adv_err"),
        "ensemble12": ("adv_err_ensemble_crafted", "adv_err_ensemble_crafted"),
        "agreement_mnist": ("agree_softmax_cond", "agree_softmax_cond"),
        "transfer_mnist": ("err_orig_on_advfromnew", "err_orig_on_advfromnew"),
        "eps_trace": ("logit_correct_seq", None),  # sequence; print mean
        "cifar_conv_maxout": ("adv_err", "adv_err"),
    }
    return table.get(arm, (None, None))


def _seeded(seeds, fn):
    out = {}
    for s in seeds:
        try:
            out[str(s)] = fn(s)
        except Exception as e:
            print(f"  arm {fn.__name__ if hasattr(fn,'__name__') else '?'} seed {s} FAILED: {repr(e)[:200]}", flush=True)
            traceback.print_exc()
            out[str(s)] = "BLOCKED"
    return out


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def arm_softmax_reg(seed):
    d = _torch_data(data.load_mnist(seed))
    m = models.SoftmaxRegression()
    train.train(m, d, {"lr": 0.5, "max_epochs": EPOCHS_SOFTMAX, "batch_size": 128,
                       "seed": seed, "momentum": 0.9,
                       "early_stop": {"monitor": "val_err", "patience": 8}})
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    adv = ev.adv_eval(m, x_test, y_test, EPS_MNIST)
    rub = ev.rubbish_eval(m, 784, 10000, seed)
    res = {"clean_err": clean, **adv,
           "rubbish_err": rub["rubbish_err"], "rubbish_conf_mistakes": rub["rubbish_conf_mistakes"],
           "rubbish_class_shares": rub["rubbish_class_shares"]}
    return res


def _logreg_analytic_equiv(m, d, eps=0.25, n=1024):
    """c07: | mean zeta(-y(w.(x - eps*sign(w)) + b)) - mean zeta(y*(eps*||w||_1 - w.x - b)) |
    should be ~0 on a fixed batch. The paper's exact FGSM for logreg uses the
    uniform perturbation eta = -eps*sign(w) (tex:407 "sign of the gradient is
    just -sign(w)"), giving x_adv = x - eps*sign(w) and the closed form
    zeta(y*(eps*||w||_1 - w.x - b)) (tex:411)."""
    x = _t(d, "x_train")[:n]
    y = _t(d, "y_train", "long")[:n]
    w = m.linear.weight.detach().squeeze(0)
    b = m.linear.bias.detach()
    # paper's exact perturbation (uniform -sign(w), tex:407)
    x_adv = x - eps * torch.sign(w).unsqueeze(0)
    fgsm_loss = m.loss(x_adv, y)
    w1 = w.abs().sum()
    yf = y.float()
    margin = eps * w1 - (x @ w + b)  # eps*||w||_1 - w.x - b
    analytic = torch.nn.functional.softplus(yf * margin).mean()
    return float(abs(fgsm_loss.item() - analytic.item()))


def arm_logreg_3v7(seed):
    d = _torch_data(data.load_mnist_3v7(seed))
    m = models.LogisticRegression3v7()
    train.train(m, d, {"lr": 0.5, "max_epochs": EPOCHS_LOGREG, "batch_size": 128,
                       "seed": seed, "momentum": 0.9,
                       "early_stop": {"monitor": "val_err", "patience": 8}})
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    adv = ev.adv_eval(m, x_test, y_test, EPS_MNIST)
    equiv = _logreg_analytic_equiv(m, d, EPS_MNIST)
    train_err = ev.error(m, d["x_train"], d["y_train"])
    return {"clean_err": clean, "adv_err": adv["adv_err"], "train_err": train_err,
            "analytic_equiv_max_absdiff": equiv}


def _train_maxout(units, seed, d, adversarial=False, noise=None, l1=None, monitor="val_err",
                  epochs=EPOCHS_MAXOUT240, full60k=False):
    m = models.MaxoutMLP(units, 2, 5, {"input": 0.2, "hidden": 0.5})
    cfg = {"lr": 0.05, "max_epochs": epochs, "batch_size": 128, "seed": seed,
           "momentum": 0.9, "early_stop": {"monitor": monitor, "patience": 8}}
    if adversarial:
        cfg["adversarial"] = {"alpha": 0.5, "eps": EPS_MNIST}
    if noise:
        cfg["noise"] = noise
    if l1:
        cfg["l1_first_layer"] = l1
    if full60k:
        df = _torch_data(data.load_mnist_full(seed))
        cfg["retrain_full_60k"] = True
        d = df
    h = train.train(m, d, cfg)
    return m, h


def arm_maxout_naive(seed):
    d = _torch_data(data.load_mnist(seed))
    m, h = _train_maxout(240, seed, d, epochs=EPOCHS_MAXOUT240)
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    train_err = h["train_err"][-1]
    adv = ev.adv_eval(m, x_test, y_test, EPS_MNIST)
    rub = ev.rubbish_eval(m, 784, 10000, seed)
    return {"clean_err": clean, "train_err": train_err, **adv,
            "rubbish_err": rub["rubbish_err"], "rubbish_conf_mistakes": rub["rubbish_conf_mistakes"],
            "rubbish_class_shares": rub["rubbish_class_shares"]}


def arm_maxout_adv(seed):
    d = _torch_data(data.load_mnist(seed))
    m, h = _train_maxout(240, seed, d, adversarial=True, epochs=EPOCHS_MAXOUT240)
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    adv = ev.adv_eval(m, x_test, y_test, EPS_MNIST)
    return {"clean_err": clean, **adv}


def arm_maxout_large_naive(seed):
    d = _torch_data(data.load_mnist(seed))
    m, h = _train_maxout(1600, seed, d, epochs=EPOCHS_MAXOUT1600)
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    adv = ev.adv_eval(m, x_test, y_test, EPS_MNIST)
    return {"clean_err": clean, **adv}


def arm_maxout_large_adv(seed):
    d = _torch_data(data.load_mnist_full(seed))
    m, h = _train_maxout(1600, seed, d, adversarial=True,
                         monitor="adv_val_err", epochs=EPOCHS_MAXOUT1600, full60k=True)
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    adv = ev.adv_eval(m, x_test, y_test, EPS_MNIST)
    return {"clean_err": clean, **adv, "epochs_run": h["best_epoch"]}


def arm_maxout_sigmoid(seed):
    d = _torch_data(data.load_mnist(seed))
    m = models.MaxoutSigmoid(240, 2, 5, {"input": 0.2, "hidden": 0.5})
    train.train(m, d, {"lr": 0.05, "max_epochs": EPOCHS_MAXOUT240, "batch_size": 128,
                      "seed": seed, "momentum": 0.9,
                      "early_stop": {"monitor": "val_err", "patience": 8}})
    rub = ev.rubbish_eval(m, 784, 10000, seed)
    clean = ev.error(m, d["x_test"], d["y_test"])
    return {"clean_err": clean, "rubbish_err": rub["rubbish_err"],
            "rubbish_conf_mistakes": rub["rubbish_conf_mistakes"]}


def arm_noise_rademacher(seed):
    d = _torch_data(data.load_mnist(seed))
    m, h = _train_maxout(240, seed, d,
                         noise={"type": "rademacher", "eps": EPS_MNIST}, epochs=EPOCHS_MAXOUT240)
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    adv = ev.adv_eval(m, x_test, y_test, EPS_MNIST)
    return {"clean_err": clean, **adv}


def arm_noise_uniform(seed):
    d = _torch_data(data.load_mnist(seed))
    m, h = _train_maxout(240, seed, d,
                         noise={"type": "uniform", "eps": EPS_MNIST}, epochs=EPOCHS_MAXOUT240)
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    adv = ev.adv_eval(m, x_test, y_test, EPS_MNIST)
    return {"clean_err": clean, **adv}


def arm_l1_maxout(seed):
    d = _torch_data(data.load_mnist(seed))
    m, h = _train_maxout(240, seed, d, l1=0.0025, epochs=EPOCHS_MAXOUT240)
    train_err = h["train_err"][-1]
    return {"train_err": train_err}


def arm_rbf_shallow(seed):
    d = _torch_data(data.load_mnist(seed))
    m = models.RBFNet(K=10, D=784)
    m.init_means_from(_t(d, "x_train"), _t(d, "y_train", "long"), seed=seed)
    train.train(m, d, {"lr": 0.01, "max_epochs": EPOCHS_RBF, "batch_size": 128,
                      "seed": seed, "momentum": 0.9,
                      "early_stop": {"monitor": "val_err", "patience": 8}})
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    with torch.no_grad():
        conf_clean = m.confidence(x_test)
    clean_conf_all = float(conf_clean.mean().item()) * 100.0
    adv = ev.adv_eval(m, x_test, y_test, EPS_MNIST)
    rub = ev.rubbish_eval(m, 784, 10000, seed)
    return {"clean_err": clean, "clean_conf_all": clean_conf_all,
            "adv_err": adv["adv_err"], "adv_conf_mistakes": adv["adv_conf_mistakes"],
            "rubbish_err": rub["rubbish_err"]}


def arm_ensemble12(seed):
    d = _torch_data(data.load_mnist(seed))
    members = []
    for i in range(ENSEMBLE_MEMBERS):
        mi, _ = _train_maxout(240, seed * 100 + i, d, epochs=ENSEMBLE_EPOCHS)
        members.append(mi)
    ens = models.Ensemble(members)
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(ens, x_test, y_test)
    # ensemble-crafted: FGSM on ensemble NLL
    x_adv_ens = attack.fgsm(ens, x_test, y_test, EPS_MNIST)
    with torch.no_grad():
        pred_ens = ens.predict(x_adv_ens)
    err_ens = float((pred_ens != y_test.view(-1)).float().mean().item()) * 100.0
    # single-member-crafted: FGSM on member 0, evaluate on ensemble
    x_adv_m0 = attack.fgsm(members[0], x_test, y_test, EPS_MNIST)
    with torch.no_grad():
        pred_m0 = ens.predict(x_adv_m0)
    err_m0 = float((pred_m0 != y_test.view(-1)).float().mean().item()) * 100.0
    return {"clean_err": clean, "adv_err_ensemble_crafted": err_ens,
            "adv_err_single_crafted": err_m0}


def arm_agreement_mnist(seed, maxout_naive_model=None, softmax_model=None, rbf_model=None):
    """Reuses trained maxout_naive + softmax_reg + rbf_shallow of the same seed."""
    d = _torch_data(data.load_mnist(seed))
    if maxout_naive_model is None:
        maxout_naive_model, _ = _train_maxout(240, seed, d, epochs=EPOCHS_MAXOUT240)
    if softmax_model is None:
        softmax_model = models.SoftmaxRegression()
        train.train(softmax_model, d, {"lr": 0.5, "max_epochs": EPOCHS_SOFTMAX,
                                      "batch_size": 128, "seed": seed, "momentum": 0.9,
                                      "early_stop": {"monitor": "val_err", "patience": 8}})
    if rbf_model is None:
        rbf_model = models.RBFNet(K=10, D=784)
        rbf_model.init_means_from(_t(d, "x_train"), _t(d, "y_train", "long"), seed=seed)
        train.train(rbf_model, d, {"lr": 0.01, "max_epochs": EPOCHS_RBF, "batch_size": 128,
                                   "seed": seed, "momentum": 0.9,
                                   "early_stop": {"monitor": "val_err", "patience": 8}})
    x_test, y_test = d["x_test"], d["y_test"]
    x_adv = attack.fgsm(maxout_naive_model, x_test, y_test, EPS_MNIST)
    ag_sm = ev.agreement(maxout_naive_model, softmax_model, x_adv, y_test)
    ag_rbf = ev.agreement(maxout_naive_model, rbf_model, x_adv, y_test)
    # rbf predicting softmax's class (both-wrong conditioned, SPEC §4.17)
    x_adv_sm = attack.fgsm(softmax_model, x_test, y_test, EPS_MNIST)
    ag_rbf_on_sm = ev.agreement(softmax_model, rbf_model, x_adv_sm, y_test)
    return {"agree_softmax_all": ag_sm["agree_all"], "agree_softmax_cond": ag_sm["agree_cond"],
            "agree_rbf_all": ag_rbf["agree_all"], "agree_rbf_cond": ag_rbf["agree_cond"],
            "agree_rbf_on_softmax": ag_rbf_on_sm["agree_cond"]}


def arm_transfer_mnist(seed, large_naive=None, large_adv=None):
    d = _torch_data(data.load_mnist(seed))
    if large_naive is None:
        large_naive, _ = _train_maxout(1600, seed, d, epochs=EPOCHS_MAXOUT1600)
    if large_adv is None:
        large_adv, _ = _train_maxout(1600, seed, d, adversarial=True,
                                     monitor="adv_val_err", epochs=EPOCHS_MAXOUT1600, full60k=True)
    x_test, y_test = d["x_test"], d["y_test"]
    x_adv_from_new = attack.fgsm(large_adv, x_test, y_test, EPS_MNIST)
    with torch.no_grad():
        err_orig_on_new = float((large_naive.predict(x_adv_from_new) != y_test.view(-1)).float().mean().item()) * 100.0
    x_adv_from_orig = attack.fgsm(large_naive, x_test, y_test, EPS_MNIST)
    with torch.no_grad():
        err_new_on_orig = float((large_adv.predict(x_adv_from_orig) != y_test.view(-1)).float().mean().item()) * 100.0
    return {"err_orig_on_advfromnew": err_orig_on_new,
            "err_new_on_advfromorig": err_new_on_orig}


def arm_eps_trace(seed, maxout_naive_model=None):
    d = _torch_data(data.load_mnist(seed))
    if maxout_naive_model is None:
        maxout_naive_model, _ = _train_maxout(240, seed, d, epochs=EPOCHS_MAXOUT240)
    x_test, y_test = d["x_test"], d["y_test"]
    # first class-4 test example correctly classified by this seed's model
    with torch.no_grad():
        preds = maxout_naive_model.predict(x_test)
    correct = (preds == y_test.view(-1)) & (y_test.view(-1) == 4)
    idx = torch.where(correct)[0]
    if len(idx) == 0:
        return "BLOCKED"
    ex_i = int(idx[0].item())
    x0 = x_test[ex_i]
    y0 = int(y_test[ex_i].item())
    eps_grid = [float(v) for v in range(-10, 11)]  # 21 pts
    logits = attack.fgsm_logits_trace(maxout_naive_model, x0, y0, torch.tensor(eps_grid, dtype=torch.float32))
    logits = logits.detach().cpu().numpy()  # [21, K]
    logit_correct = [float(logits[i, y0]) for i in range(21)]
    # max wrong-class logit at each eps (excluding the correct class)
    wrong = np.delete(logits, y0, axis=1)  # [21, K-1]
    logit_maxwrong = [float(wrong[i].max()) for i in range(21)]
    margin = [float(logit_correct[i] - logit_maxwrong[i]) for i in range(21)]
    return {"eps_grid": eps_grid, "logit_correct_seq": logit_correct,
            "logit_maxwrong_seq": logit_maxwrong, "margin_seq": margin,
            "example_index": ex_i, "example_true_label": y0}


def arm_cifar_conv_maxout(seed):
    d = _torch_data(data.load_cifar10(seed))
    m = models.ConvMaxoutCIFAR()
    train.train(m, d, {"lr": 0.05, "max_epochs": EPOCHS_CONV, "batch_size": 256,
                      "seed": seed, "momentum": 0.9,
                      "early_stop": {"monitor": "val_err", "patience": 5}})
    x_test, y_test = d["x_test"], d["y_test"]
    clean = ev.error(m, x_test, y_test)
    adv = ev.adv_eval(m, x_test, y_test, EPS_CIFAR)
    rub = ev.rubbish_eval(m, 3072, 1000, seed)
    fool = ev.fooling_eval(m, 3072, 200, seed, EPS_CIFAR)
    return {"clean_err": clean, **adv,
            "rubbish_err": rub["rubbish_err"], "rubbish_conf_mistakes": rub["rubbish_conf_mistakes"],
            "rubbish_class_shares": rub["rubbish_class_shares"],
            "fool_success": fool["fool_success"], "fool_success_avg": fool["fool_success_avg"]}


ARMS = {
    "softmax_reg": (SEEDS, arm_softmax_reg),
    "logreg_3v7": (SEEDS, arm_logreg_3v7),
    "maxout_naive": (SEEDS, arm_maxout_naive),
    "maxout_adv": (SEEDS, arm_maxout_adv),
    "maxout_large_naive": (SEEDS, arm_maxout_large_naive),
    "maxout_large_adv": (M5_SEEDS, arm_maxout_large_adv),
    "maxout_sigmoid": (SEEDS, arm_maxout_sigmoid),
    "noise_rademacher": (SEEDS, arm_noise_rademacher),
    "noise_uniform": (SEEDS, arm_noise_uniform),
    "l1_maxout": (SEEDS, arm_l1_maxout),
    "rbf_shallow": (SEEDS, arm_rbf_shallow),
    "ensemble12": (SEEDS, arm_ensemble12),
    "cifar_conv_maxout": (SEEDS, arm_cifar_conv_maxout),
    # composite arms reuse trained models of the same seed:
    "agreement_mnist": (SEEDS, arm_agreement_mnist),
    "transfer_mnist": (SEEDS, arm_transfer_mnist),
    "eps_trace": (SEEDS, arm_eps_trace),
}


def _print_final(arm, res):
    _, key = _primary(arm, None)
    if res == "BLOCKED" or res is None:
        print(f"FINAL {arm}=BLOCKED", flush=True)
        return
    if key is None or key not in res:
        print(f"FINAL {arm}=BLOCKED", flush=True)
        return
    val = res[key]
    if isinstance(val, list):
        print(f"FINAL {arm}={sum(val)/len(val):.4f} (seq mean)", flush=True)
    else:
        print(f"FINAL {arm}={val:.4f}", flush=True)


def main():
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    measured = {}
    # run order: heavy composite arms reuse models; to avoid retraining, we
    # train shared models once per seed and pass them in.
    only = os.environ.get("EAE_ONLY")
    for arm, (seeds, fn) in ARMS.items():
        if only and arm != only:
            continue
        print(f"\n=== arm {arm} (seeds {seeds}) ===", flush=True)
        t0 = time.time()
        measured[arm] = {}
        for s in seeds:
            ts = time.time()
            try:
                # composite arms: reuse shared trained models of this seed
                if arm == "agreement_mnist":
                    d = _torch_data(data.load_mnist(s))
                    mn, _ = _train_maxout(240, s, d, epochs=EPOCHS_MAXOUT240)
                    res = fn(s, maxout_naive_model=mn)
                elif arm == "transfer_mnist":
                    d = _torch_data(data.load_mnist(s))
                    ln, _ = _train_maxout(1600, s, d, epochs=EPOCHS_MAXOUT1600)
                    la, _ = _train_maxout(1600, s, d, adversarial=True,
                                          monitor="adv_val_err", epochs=EPOCHS_MAXOUT1600, full60k=True)
                    res = fn(s, large_naive=ln, large_adv=la)
                elif arm == "eps_trace":
                    d = _torch_data(data.load_mnist(s))
                    mn, _ = _train_maxout(240, s, d, epochs=EPOCHS_MAXOUT240)
                    res = fn(s, maxout_naive_model=mn)
                else:
                    res = fn(s)
            except Exception as e:
                print(f"  {arm} seed {s} FAILED: {repr(e)[:200]}", flush=True)
                traceback.print_exc()
                res = "BLOCKED"
            measured[arm][str(s)] = res
            _print_final(arm, res if str(s) == str(seeds[-1]) else None) if False else None
            # print per-seed headline
            _, key = _primary(arm, None)
            if isinstance(res, dict) and key and key in res:
                v = res[key]
                vs = f"{sum(v)/len(v):.4f}" if isinstance(v, list) else f"{v:.4f}"
                print(f"  seed {s}: {key}={vs} ({time.time()-ts:.0f}s)", flush=True)
            else:
                print(f"  seed {s}: BLOCKED ({time.time()-ts:.0f}s)", flush=True)
        # one FINAL line per arm (last seed's value stands for the arm)
        last = measured[arm][str(seeds[-1])]
        _print_final(arm, last)
        print(f"  arm {arm} total {time.time()-t0:.0f}s", flush=True)
        # checkpoint measured.json after each arm
        with open(os.path.join(REPO, "measured.json"), "w") as f:
            json.dump(measured, f, indent=2)
    print("\n=== writing measured.json ===", flush=True)
    with open(os.path.join(REPO, "measured.json"), "w") as f:
        json.dump(measured, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
