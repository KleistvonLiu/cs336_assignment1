# multihead_self_attention.py
from __future__ import annotations
from typing import Optional
import math
import torch
from torch import nn, Tensor

from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .attention import scaled_dot_product_attention  # 你已实现的 SDPA（内部会做稳定 softmax）


class MultiheadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE.

    Args:
        d_model:    Transformer hidden size
        num_heads:  number of attention heads (d_model % num_heads == 0)
        max_seq_len: maximum sequence length for RoPE cache
        theta:      RoPE base (e.g., 1e4)
        device/dtype: optional
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 1e4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by num_heads({num_heads})")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads

        # Projections (no bias), dk=dv=head_dim
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.o_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        # RoPE buffers for Q/K
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device
        )

    def _reshape_to_heads(self, x: Tensor) -> Tensor:
        """
        (..., L, d_model) -> (..., H, L, head_dim)
        不打乱前缀批维
        """
        *prefix, L, _ = x.shape
        H, Dh = self.num_heads, self.head_dim
        x = x.view(*prefix, L, H, Dh).movedim(-2, -3)  # -> (..., H, L, Dh)
        return x

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        (..., H, L, head_dim) -> (..., L, d_model)
        """
        *prefix, H, L, Dh = x.shape
        x = x.movedim(-3, -2)  # (..., L, H, Dh)
        return x.reshape(*prefix, L, H * Dh)

    def forward(
        self,
        x: Tensor,                               # (..., L, d_model)
        token_positions: Optional[Tensor] = None # (..., L) 或 (L,)
    ) -> Tensor:
        # 线性投影
        q = self.q_proj(x)  # (..., L, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 拆分多头
        q = self._reshape_to_heads(q)  # (..., H, L, Dh)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)

        # 对 Q/K 应用 RoPE（rope 支持任意批维；positions 可为 (L,) 自动广播）
        L = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(L, device=x.device, dtype=torch.long)  # (L,)
        q = self.rope(q, token_positions)  # (..., H, L, Dh)
        k = self.rope(k, token_positions)

        # 因果 mask：允许看自身及左侧 → 下三角 True
        causal_mask = torch.ones(L, L, dtype=torch.bool, device=x.device).tril_(0)  # (L, L)

        # 缩放点积注意力（可处理任意批维；这里批维包含了 H）
        attn_out = scaled_dot_product_attention(
            q, k, v, mask=causal_mask
        )  # (..., H, L, Dh)

        # 合并多头 + 输出投影
        y = self._merge_heads(attn_out)       # (..., L, d_model)
        y = self.o_proj(y)                    # (..., L, d_model)
        return y
