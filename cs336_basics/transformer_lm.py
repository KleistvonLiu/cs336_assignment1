# transformer_lm.py
from __future__ import annotations
from typing import Optional
import torch
from torch import nn, Tensor

from .embedding import Embedding
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock, TransformerBlockNoNorm, TransformerBlockPostNorm


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
        use_rope: bool = True,
        activation: str = "swiglu",
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
                use_rope = use_rope,
                activation=activation,
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

class TransformerLMNoNorm(nn.Module):
    """
    全模型移除 RMSNorm：block 内两处 + 末尾 final_norm。
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 1e4,
        use_rope: bool = True,
        activation: str = "swiglu",
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
            TransformerBlockNoNorm(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                use_rope=use_rope,
                activation=activation,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.Identity()  # 替换 RMSNorm

        self.lm_head_weight: Optional[nn.Parameter] = None

    def forward(self, token_ids: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        x = self.tok_embeddings(token_ids)  # (..., L, d_model)
        L = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(L, device=x.device, dtype=torch.long)
        for blk in self.blocks:
            x = blk(x, token_positions=token_positions)
        x = self.final_norm(x)              # no-op

        W_out = (self.lm_head_weight if isinstance(self.lm_head_weight, nn.Parameter)
                 else self.tok_embeddings.weight)
        logits = x @ W_out.t()              # (..., L, V)
        return logits

class TransformerLMPostNorm(nn.Module):
    """
    使用 Post-norm Block 的语言模型。
    说明：经典 Post-LN 通常不再额外加“final norm”；为了与现有 Pre-norm 版本对齐，
    这里提供一个可切换的 final_norm（默认 Identity；如需可改为 RMSNorm）。
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 1e4,
        use_rope: bool = True,
        activation: str = "swiglu",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_final_norm: bool = False,  # 若想与 Pre-norm 一样在末尾再做一次 norm，可设 True
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.num_layers = int(num_layers)
        self.d_model = int(d_model)

        self.tok_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlockPostNorm(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                use_rope=use_rope,
                activation=activation,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        # Post-norm 通常不需要额外 final norm；如需实验可切换为 RMSNorm(d_model,...)
        self.final_norm = (RMSNorm(d_model, device=device, dtype=dtype)
                           if use_final_norm else nn.Identity())

        # 与原实现一致：可选独立 lm head；默认与嵌入权重 tied
        self.lm_head_weight: Optional[nn.Parameter] = None

    def forward(
        self,
        token_ids: Tensor,                  # (B, L) 或 (..., L)
        token_positions: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.tok_embeddings(token_ids)  # (..., L, d_model)
        L = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(L, device=x.device, dtype=torch.long)

        for blk in self.blocks:
            x = blk(x, token_positions=token_positions)

        x = self.final_norm(x)

        W_out = (self.lm_head_weight if isinstance(self.lm_head_weight, nn.Parameter)
                 else self.tok_embeddings.weight)
        logits = x @ W_out.t()              # (..., L, vocab_size)
        return logits