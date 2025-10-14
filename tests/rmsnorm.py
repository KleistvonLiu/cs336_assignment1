# rmsnorm.py
from __future__ import annotations
from typing import Optional
import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    - Param: weight (gamma), shape = (d_model,), init to 1
    - Forward: upcast to float32 for stability, then downcast back to input dtype
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model) -> (..., d_model)
        """
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)

        # mean of squares over the last dimension
        ms = x32.pow(2).mean(dim=-1, keepdim=True)          # (..., 1)
        inv_rms = torch.rsqrt(ms + self.eps)                # (..., 1)

        y32 = x32 * inv_rms * self.weight.to(dtype=x32.dtype)  # broadcast gamma
        return y32.to(orig_dtype)
