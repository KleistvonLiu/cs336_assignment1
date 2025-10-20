# attention.py
from __future__ import annotations
from typing import Optional
import math
import torch
from torch import Tensor

from tests.softmax_stable import softmax_stable


def scaled_dot_product_attention(
    Q: Tensor,                       # (..., Lq, d_k)
    K: Tensor,                       # (..., Lk, d_k)
    V: Tensor,                       # (..., Lk, d_v)
    mask: Optional[Tensor] = None,   # (..., Lq, Lk) 或 (Lq, Lk), bool; True=keep, False=block
) -> Tensor:
    """
    Scaled Dot-Product Attention (稳定 softmax，支持广播 mask)
    Returns: (..., Lq, d_v)
    """
    # 上浮到 float32 提高稳定性
    q = Q.to(torch.float32) # Lk = Lv，因为它们是同源的，Lq可以不同，因为可以是其他sequence
    k = K.to(torch.float32)
    v = V.to(torch.float32)
    # print(f"q.shape = {q.shape}, k.shape = {k.shape}, v.shape = {v.shape}")

    d_k = q.shape[-1]
    Lq = q.shape[-2]
    Lk = k.shape[-2]
    assert k.shape[-1] == d_k, "Q and K must have the same d_k"
    assert v.shape[-2] == Lk, "V length must match K length"

    # 注意力打分: (..., Lq, Lk)
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    # print(f"scores.shape = {scores.shape}")

    if mask is not None:
        # 将 mask 转成可广播到 scores 的形状，True=允许，False=屏蔽
        m = mask.to(dtype=torch.bool, device=scores.device)
        # 补齐前缀维度数以便广播（把缺的前面维度补 1）
        while m.dim() < scores.dim():
            m = m.unsqueeze(0)
        # 先将禁用位置设为一个很大的负数（避免 -inf 带来的 NaN）
        very_neg = torch.tensor(-1e9, dtype=scores.dtype, device=scores.device)
        masked_scores = torch.where(m, scores, very_neg) # 如果true，用score，如果false，用very_neg

        # 调用数值稳定 softmax
        attn = softmax_stable(masked_scores, dim=-1)

        # 进一步确保禁用位置概率=0，并对允许位置重新归一（防全屏蔽行 NaN）
        # attn = attn * m.to(attn.dtype)
        # row_sum = attn.sum(dim=-1, keepdim=True)           # (..., Lq, 1)
        # attn = torch.where(row_sum > 0, attn / row_sum, torch.zeros_like(attn))
    else:
        attn = softmax_stable(scores, dim=-1)

    # 输出: (..., Lq, d_v)
    out = torch.matmul(attn, v)
    return out.to(Q.dtype)