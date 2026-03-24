# optim/grad_clip.py
from __future__ import annotations
from typing import Iterable, Optional
import torch

def clip_gradients(params: Iterable[torch.nn.Parameter],
                   max_norm: float,
                   eps: float = 1e-6) -> float:
    """
    In-place global L2 gradient clipping.

    Args:
        params: iterable of nn.Parameter (with .grad possibly None)
        max_norm: maximum allowed global L2 norm
        eps: small constant to avoid division by zero (default 1e-6)

    Returns:
        total_norm_before_clipping (float)
    """
    # Collect grads (skip None)
    grads = [p.grad for p in params if (p.grad is not None)]
    if not grads:
        return 0.0

    # Compute global L2 norm in float32 for stability (like PyTorch)
    # Keep the reduction on device to avoid needless transfers.
    device = grads[0].device
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    for g in grads:
        total_sq = total_sq + g.detach().to(torch.float32).pow(2).sum()
    total_norm = total_sq.sqrt()  # tensor scalar on device

    # Compute scaling coef
    if max_norm <= 0:
        clip_coef = 0.0
    else:
        clip_coef = float(max_norm) / (float(total_norm.item()) + float(eps))

    # Scale in-place only if clipping is needed
    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)

    return float(total_norm.item())