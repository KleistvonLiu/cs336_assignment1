# optim/adamw_from_scratch.py
import math
from typing import Iterable, Tuple
import torch
from torch.optim.optimizer import Optimizer


class AdamWFromScratch(Optimizer):
    r"""
    Pure PyTorch implementation of AdamW (Loshchilov & Hutter, 2019).
    Decoupled weight decay: p <- p * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        params: iterable of parameters
        lr (float): learning rate (α)
        betas (Tuple[float, float]): (β1, β2)
        eps (float): ϵ for numerical stability
        weight_decay (float): λ (decoupled L2)
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0) or not (0.0 <= b2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWFromScratch does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    # State initialization
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps) # 这里是吧alpha_t直接带入到公式，同时分子分母都除以自己的bias
                step_size = lr / bias_correction1

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size) # self.addcdiv_(tensor1, tensor2, value=v) 等价于 self = self + tensor1/tensor2 * value

        return loss
