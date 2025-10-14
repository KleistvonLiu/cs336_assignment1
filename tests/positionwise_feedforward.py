# positionwise_feedforward.py
from __future__ import annotations
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from .linear import Linear  # 复用你之前实现的 Linear


def _round_to_multiple(x: int, m: int) -> int:
    """四舍五入到 m 的倍数。"""
    return ((x + m // 2) // m) * m


class SwiGLU(nn.Module):
    """
    Position-wise FFN with SwiGLU:
        out = W2( silu(W1(x)) * W3(x) )
    - W1: (d_model -> d_ff)
    - W3: (d_model -> d_ff)
    - W2: (d_ff -> d_model)
    - 无 bias
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,   # 允许外部显式指定，便于与给定权重完全匹配
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        if d_ff is None:
            approx = int(round((8.0 / 3.0) * self.d_model))
            d_ff = _round_to_multiple(approx, 64)
        self.d_ff = int(d_ff)

        # 命名与常见 state_dict 对应（w1/w2/w3）
        # 习惯：w1=gate_proj, w3=up_proj, w2=down_proj（与 LLaMA 系一致）
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)  # gate
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)  # up
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model) -> (..., d_model)
        """
        gate = self.w1(x)         # (..., d_ff)
        up   = self.w3(x)         # (..., d_ff)
        h = F.silu(gate) * up     # SwiGLU
        return self.w2(h)         # (..., d_model)
