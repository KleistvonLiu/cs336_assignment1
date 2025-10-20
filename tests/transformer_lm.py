# transformer_lm.py
from __future__ import annotations
from typing import Optional
import torch
from torch import nn, Tensor

from .embedding import Embedding
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 1e4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.num_layers = int(num_layers)
        self.d_model = int(d_model)

        self.tok_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # 可选的独立 lm head（默认不用，适配器里若提供则注册）
        self.lm_head_weight: Optional[nn.Parameter] = None  # 不注册为 Parameter，等适配器显式注册

    def forward(
        self,
        token_ids: Tensor,                  # (B, L) 或 (..., L)
        token_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns logits: (..., L, vocab_size)
        """
        x = self.tok_embeddings(token_ids)                       # (..., L, d_model)
        L = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(L, device=x.device, dtype=torch.long)

        for blk in self.blocks:
            x = blk(x, token_positions=token_positions)

        x = self.final_norm(x)

        # --- 关键改动：如果提供了独立 lm_head.weight 就用它；否则用 embedding 权重共享 ---
        W_out = (self.lm_head_weight if isinstance(self.lm_head_weight, nn.Parameter)
                 else self.tok_embeddings.weight)
        logits = x @ W_out.t()                                   # (..., L, vocab_size)
        return logits
