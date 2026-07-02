import torch
import torch.optim as optim
from typing import List


def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(optim.Optimizer):
    def __init__(
        self,
        muon_params: List[torch.nn.Parameter],
        adam_params: List[torch.nn.Parameter],
        lr: float = 0.02,
        adam_lr: float = 3e-4,
        momentum: float = 0.9,
        betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, adam_lr=adam_lr, momentum=momentum, betas=betas,
            weight_decay=weight_decay, ns_steps=ns_steps, eps=eps,
        )

        groups = []
        if muon_params:
            groups.append(dict(params=muon_params, use_muon=True, **defaults))
        if adam_params:
            groups.append(dict(params=adam_params, use_muon=False, **defaults))

        super().__init__(groups, defaults)

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            else:
                for p in group["params"]:
                    self.state[p]["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adam_step(group)
        return loss

    def _muon_step(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g.float())
            update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
            if p.ndim == 2:
                update = update * max(1.0, p.size(0) / p.size(1)) ** 0.5
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)
            p.data.add_(update.to(p.dtype), alpha=-lr)

    def _adam_step(self, group):
        lr = group["adam_lr"]
        betas = group["betas"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]
        beta1, beta2 = betas

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(g.float(), alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).add_(g.float().pow(2), alpha=1 - beta2)
            bias_c1 = 1 - beta1 ** (state.get("step", 1))
            bias_c2 = 1 - beta2 ** (state.get("step", 1))
            state["step"] = state.get("step", 0) + 1
            update = exp_avg / (exp_avg_sq.sqrt() + eps)
            update = update * (bias_c1 / (bias_c2 ** 0.5))
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)
            p.data.add_(update.to(p.dtype), alpha=-lr)