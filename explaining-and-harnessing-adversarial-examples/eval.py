"""Evaluation metrics for the EAE reproduction.

All returns are PERCENTS in [0,100] (multiply fractions by 100), matching the
paper's printed numbers (SPEC §metric_convention).

Paper references (paper/source/iclr2015.tex):
  error / confidence   tex:333-341
  rubbish              tex:905-911  (error iff any class prob > 0.5)
  agreement            tex:681-688
  fooling              tex:953-955   (success iff p(y=i|x_tilde) > 0.5)
"""
import torch

import attack
import data


def _to_tensor(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    return x


def error(model, x, y):
    x = _to_tensor(x); y = _to_tensor(y).to(torch.long)
    if x.numel() == 0:
        raise ValueError("error: empty input")
    pred = model.predict(x)
    return float((pred != y.view(-1)).float().mean().item()) * 100.0


def confidence_stats(model, x, y):
    x = _to_tensor(x); y = _to_tensor(y).to(torch.long)
    if x.numel() == 0:
        raise ValueError("confidence_stats: empty input")
    with torch.no_grad():
        conf = model.confidence(x)  # [B]
    conf_all = float(conf.mean().item()) * 100.0
    pred = model.predict(x)
    wrong = pred != y.view(-1)
    if wrong.sum().item() == 0:
        conf_mistakes = 0.0  # undefined; SPEC §4.10 — report 0 when no mistakes
    else:
        conf_mistakes = float(conf[wrong].mean().item()) * 100.0
    return {"conf_all": conf_all, "conf_mistakes": conf_mistakes}


def adv_eval(model, x, y, eps=0.25):
    """FGSM adversarial error + confidence on x with eps."""
    x = _to_tensor(x); y = _to_tensor(y).to(torch.long)
    if x.numel() == 0:
        raise ValueError("adv_eval: empty input")
    model.eval()
    x_adv = attack.fgsm(model, x, y, eps)
    with torch.no_grad():
        pred = model.predict(x_adv)
        conf = model.confidence(x_adv)
    wrong = pred != y.view(-1)
    adv_err = float(wrong.float().mean().item()) * 100.0
    adv_conf_all = float(conf.mean().item()) * 100.0
    if wrong.sum().item() == 0:
        adv_conf_mistakes = 0.0
    else:
        adv_conf_mistakes = float(conf[wrong].mean().item()) * 100.0
    return {"adv_err": adv_err, "adv_conf_all": adv_conf_all, "adv_conf_mistakes": adv_conf_mistakes}


def rubbish_eval(model, dim, n, seed=0):
    """Gaussian rubbish eval (tex:905-911).

    error := any class prob > 0.5. For softmax-top: max_k > 0.5. For sigmoid-top
    / RBF (rows need NOT sum to 1): ANY p_k > 0.5.
    """
    if n <= 0 or dim <= 0:
        raise ValueError("rubbish_eval: n>0 and dim>0 required")
    x = torch.from_numpy(data.rubbish(dim, n, seed))
    model.eval()
    with torch.no_grad():
        prob = model.prob(x)  # [n, K]
        conf = prob.max(dim=-1).values  # max_k p_k
        pred = prob.argmax(dim=-1)
        # "any class prob > 0.5" = max_k p_k > 0.5 (since max is the largest)
        mistakes = conf > 0.5
    rubbish_err = float(mistakes.float().mean().item()) * 100.0
    if mistakes.sum().item() == 0:
        rubbish_conf_mistakes = 0.0
    else:
        rubbish_conf_mistakes = float(conf[mistakes].mean().item()) * 100.0
    total_mistakes = int(mistakes.sum().item())
    class_shares = {}
    for k in range(prob.shape[-1]):
        if total_mistakes > 0:
            cnt = int(((pred == k) & mistakes).sum().item())
            class_shares[str(k)] = 100.0 * cnt / total_mistakes
        else:
            class_shares[str(k)] = 0.0
    return {"rubbish_err": rubbish_err, "rubbish_conf_mistakes": rubbish_conf_mistakes,
            "rubbish_class_shares": class_shares}


def agreement(modelA, modelB, x_adv, y):
    """Label agreement on examples modelA misclassifies (tex:681-688).

    agree_all  : over examples where A is wrong, fraction where B predicts A's class.
    agree_cond : over examples where BOTH are wrong, fraction where B predicts A's class.
    """
    x_adv = _to_tensor(x_adv); y = _to_tensor(y).to(torch.long)
    if x_adv.numel() == 0:
        raise ValueError("agreement: empty input")
    modelA.eval(); modelB.eval()
    with torch.no_grad():
        predA = modelA.predict(x_adv)
        predB = modelB.predict(x_adv)
    a_wrong = predA != y.view(-1)
    if a_wrong.sum().item() == 0:
        return {"agree_all": 0.0, "agree_cond": 0.0}
    agree_all = float((predB[a_wrong] == predA[a_wrong]).float().mean().item()) * 100.0
    both_wrong = a_wrong & (predB != y.view(-1))
    if both_wrong.sum().item() == 0:
        agree_cond = 0.0
    else:
        agree_cond = float((predB[both_wrong] == predA[both_wrong]).float().mean().item()) * 100.0
    return {"agree_all": agree_all, "agree_cond": agree_cond}


def fooling_eval(model, dim, n_per_class, seed=0, eps=0.1, classes=range(10)):
    """Targeted fooling (Fig.5 caption, tex:953-955).

    For each target class i, draw n_per_class fresh N(0,I_dim) samples, take one
    targeted sign step toward i with eps, success iff p(y=i|x1) > 0.5.
    """
    if n_per_class <= 0 or dim <= 0:
        raise ValueError("fooling_eval: n_per_class>0 and dim>0 required")
    model.eval()
    fool_success = {}
    for i in classes:
        x0 = torch.from_numpy(data.rubbish(dim, n_per_class, seed + i))
        x1 = attack.targeted_fool_step(model, x0, i, eps)
        with torch.no_grad():
            p_target = model.prob(x1)[:, i]
        rate = float((p_target > 0.5).float().mean().item()) * 100.0
        fool_success[str(i)] = rate
    fool_success_avg = sum(fool_success.values()) / len(fool_success)
    return {"fool_success": fool_success, "fool_success_avg": fool_success_avg}
