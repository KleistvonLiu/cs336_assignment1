# embedding.py
from __future__ import annotations
from typing import Optional
import torch
from torch import nn


class Embedding(nn.Module):
    """
    A simple embedding lookup layer (no padding_idx / no sparse features).
    Stores weights as (num_embeddings, embedding_dim), so output's last dim is d_model.

    Init: N(0, 1) truncated to [-3, 3].
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        factory_kwargs = {"device": device, "dtype": dtype}
        # Parameter name "weight" 与 PyTorch 习惯保持一致
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, **factory_kwargs)
        )

        # ��(0, 1) 截断到 [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (...,) 整型张量
        return:    (..., embedding_dim)
        把所有的token id直接使用索引找到對應的 embedding，這種索引的方式實際上等價於使用矩陣乘法，還是節省了空間。
        """
        # 保证索引用 long
        idx = token_ids.to(dtype=torch.long, device=self.weight.device)
        # 直接张量索引（不使用 F.embedding）
        return self.weight[idx]
