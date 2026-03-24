# linear.py
from __future__ import annotations
from typing import Optional
import torch
from torch import nn


class Linear(nn.Module):
    """
    Bias-free linear layer: y = x @ W^T

    - Parameter stored as W with shape (out_features, in_features)
    - No bias
    - Weight init: N(0, 2/(din + dout)) truncated to [-3σ, 3σ]
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        factory_kwargs = {"device": device, "dtype": dtype}
        # Store as W (not W^T): (out_features, in_features)
        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, **factory_kwargs))

        # --- Initialization per assignment ---
        # sigma^2 = 2 / (din + dout)  -> sigma = sqrt(2/(din + dout))
        sigma = (2.0 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        return x @ self.W.transpose(-1, -2)
