# optim/adamw_from_scratch.py
import math
from typing import Iterable, Tuple
import torch
from torch.optim.optimizer import Optimizer


class AdamWFromScratch(Optimizer):
    r"""
    Pure PyTorch implementation of AdamW (Loshchilov & Hutter, 2019).
    Follows the reference AdamW pseudocode:
      1) update first/second moments
      2) apply the bias-corrected Adam step
      3) apply decoupled weight decay

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

                # Adam moments
                exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (grad * grad)
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                # Bias-corrected Adam step: alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                denom = torch.sqrt(exp_avg_sq) + eps

                # Parameter update: p <- p - step_size * exp_avg / denom
                update = exp_avg / denom
                p -= step_size * update

                # Decoupled weight decay is applied after the Adam step, matching the reference order.
                if wd != 0.0:
                    p -= lr * wd * p

        return loss
