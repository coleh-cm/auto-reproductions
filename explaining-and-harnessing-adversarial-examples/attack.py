"""FGSM attack and variants for the EAE reproduction.

Paper equations (paper/source/iclr2015.tex):
  FGSM              tex:309      eta = eps * sign(grad_x J(theta, x, y)); x_tilde = x + eta
  targeted fooling  tex:953-955  x_tilde = x + eps * sign(grad_x p(y=i|x))  (Fig.5 caption)
  eps trace (Fig.4) tex:762-770  direction computed ONCE at eps=0, held fixed while eps sweeps

SPEC §4.9: NO clipping of x_tilde. SPEC §4.22: sign(0) := 0 (torch.sign already does this).
"""
import torch


def _prep(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    if x.numel() == 0:
        raise ValueError("empty input to attack")
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x.float()


def fgsm(model, x, y, eps):
    """eta = eps*sign(grad_x J(theta,x,y)); x_adv = x + eta; NO clipping.

    Uses torch.autograd.grad on model.loss so it works for ANY model that
    differentiates in x (softmax, logistic, maxout, RBF, conv, ensemble).
    """
    x = _prep(x).clone().detach()
    if eps <= 0:
        return x
    x_in = x.clone().detach().requires_grad_(True)
    loss = model.loss(x_in, y)
    g = torch.autograd.grad(loss, x_in, create_graph=False)[0]
    x_in.requires_grad_(False)
    eta = eps * torch.sign(g)
    return (x + eta).detach()


def fgsm_logits_trace(model, x, y, eps_grid):
    """Record model.logits(x + eps*sign(g)) over eps_grid, with the FGSM
    direction g computed ONCE at eps=0 (the unperturbed example) and held
    fixed. This makes the logit lines exactly (piecewise) linear in eps
    (SPEC §4.15, tex:762-770).

    x: single example, [D] or [1,D]. y: scalar or [1]. eps_grid: [21].
    Returns logits [len(eps_grid), K].
    """
    x = _prep(x)
    y_t = torch.as_tensor([y]) if not torch.is_tensor(y) else (y.view(1) if y.dim() == 0 else y)
    x_in = x.clone().detach().requires_grad_(True)
    loss = model.loss(x_in, y_t)
    g = torch.autograd.grad(loss, x_in, create_graph=False)[0]
    x_in.requires_grad_(False)
    direction = torch.sign(g)  # [1, D] (or [B,D]); fixed
    eps_grid = torch.as_tensor(eps_grid, dtype=torch.float32).view(-1, 1, 1)  # [E,1,1]
    x_exp = x.unsqueeze(0)  # [1,B,D]
    x_eps = x_exp + eps_grid * direction.unsqueeze(0)  # [E,B,D]
    E, B, D = x_eps.shape
    flat = x_eps.view(E * B, D)
    logits = model.logits(flat)  # [E*B, K]
    K = logits.shape[-1]
    return logits.view(E, B, K).squeeze(1)  # [E, K] (B=1)


def targeted_fool_step(model, x0, target, eps):
    """x1 = x0 + eps*sign(grad_x p(y=target|x)); NO clipping (Fig.5 caption,
    tex:953-955)."""
    x0 = _prep(x0).clone().detach()
    x_in = x0.clone().detach().requires_grad_(True)
    prob = model.prob(x_in)  # [B, K]
    target = int(target)
    if target < 0 or target >= prob.shape[-1]:
        raise ValueError(f"bad target class {target}")
    obj = prob[:, target].sum()
    g = torch.autograd.grad(obj, x_in, create_graph=False)[0]
    x_in.requires_grad_(False)
    return (x0 + eps * torch.sign(g)).detach()
