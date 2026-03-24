# rotary_positional_embedding.py
from __future__ import annotations
from typing import Optional
import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Apply RoPE to the last dimension of shape (..., seq_len, d_k).
    Precomputes cos/sin for positions [0, max_seq_len).

    Args:
        theta: RoPE base (e.g., 1e4)
        d_k:   last-dim size of query/key (must be even)
        max_seq_len: maximum positions to cache
    """
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        half = self.d_k // 2
        # inv_freq_i = theta^(-2i/d_k) == theta^(-i/(d_k/2))
        i = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = self.theta ** (-i / half)  # (half,)
        # inv_freq = self.theta ** (-(2*i+1) / d_k) ## 为啥会报错啊？

        # Positions 0..max_seq_len-1
        pos = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)  # (L,)
        # freqs[p, i] = pos[p] * inv_freq[i]
        freqs = torch.einsum("p,i->pi", pos, inv_freq)  # (L, half)

        cos = torch.cos(freqs)  # (L, half)
        sin = torch.sin(freqs)  # (L, half)

        # 缓存为 buffer；默认 float32，前向时会转换到输入 dtype/device
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x:               (..., seq_len, d_k)
        token_positions: (..., seq_len) 或 (seq_len,)
        return:          (..., seq_len, d_k)
        """
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Last dim of x must be d_k={self.d_k}, got {x.shape[-1]}")

        # 基本形状，x 形状 (..., L, d_k)，token_positions 形状可以是 (..., L) 或仅 (L,)。
        B = x.shape[:-2]  # 任意批维
        L = x.shape[-2]  # seq_len

        # 规范化 positions：long + 设备对齐
        pos = token_positions.to(dtype=torch.long, device=x.device)

        # 要求最后一维等于 seq_len
        if pos.shape[-1] != L:
            raise ValueError(f"token_positions last dim must be seq_len={L}, got {pos.shape[-1]}")

        # 将 1D 或较少前缀维的 pos 广播到 B + (L,)
        need_dims = len(B) + 1
        while pos.dim() < need_dims:
            pos = pos.unsqueeze(0)  # 在最前面补单维
        # 现在可以安全 expand 到批维
        pos = pos.expand(*B, L)  # 形状变为 (..., L)

        # 越界检查
        if pos.min().item() < 0 or pos.max().item() >= self.max_seq_len:
            raise IndexError("token_positions out of cached range [0, max_seq_len).")

        # 选择对应 cos/sin：结果形状 (..., L, d_k//2)
        cos = self.cos_cached[pos].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[pos].to(device=x.device, dtype=x.dtype)

        # 偶/奇位分量
        x_even = x[..., 0::2]  # (..., L, d_k//2)
        x_odd = x[..., 1::2]  # (..., L, d_k//2)

        # 旋转
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        # 交错回拼
        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out