# transformer_block.py
from __future__ import annotations
from typing import Optional
import torch
from torch import nn, Tensor

from .rmsnorm import RMSNorm
from .positionwise_feedforward import SwiGLU
from .multi_head_self_attention import MultiheadSelfAttention


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block (RMSNorm -> Self-Attn -> residual -> RMSNorm -> FFN -> residual).

    Components:
      - input_layernorm                : RMSNorm(d_model)
      - attn (causal, with RoPE)      : MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
      - post_attention_layernorm      : RMSNorm(d_model)
      - ffn (SwiGLU)                  : SwiGLU(d_model, d_ff)

    Args:
        d_model:    hidden size
        num_heads:  number of attention heads
        d_ff:       inner size of feed-forward (SwiGLU)
        max_seq_len: RoPE cache length
        theta:      RoPE base
        device/dtype: optional
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 1e4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

        self.input_layernorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,                                # (..., L, d_model)
        token_positions: Optional[Tensor] = None  # (..., L) 或 (L,)
    ) -> Tensor:
        # Pre-norm + Self-Attention
        h = self.input_layernorm(x)
        attn_out = self.attn(h, token_positions=token_positions)
        x = x + attn_out

        # Pre-norm + FFN
        h2 = self.post_attention_layernorm(x)
        ffn_out = self.ffn(h2)
        x = x + ffn_out
        return x
