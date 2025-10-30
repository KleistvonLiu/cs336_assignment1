# decoding.py
from __future__ import annotations
from typing import Iterable, Optional, Sequence, List
import torch
import torch.nn.functional as F

@torch.no_grad()
def decode(
    model,                              # TransformerLM
    prompt_ids: Sequence[int] | torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_p: Optional[float] = None,      # 0<p<=1，None 表示不启用 nucleus
    device: str | torch.device = "cpu",
) -> List[int]:
    """
    自回归解码：支持 temperature 与 nucleus (top-p) 采样。
    返回：包含 prompt 与生成内容的完整 token 序列（list[int]）。
    """
    model.eval()
    device = torch.device(device)
    if isinstance(prompt_ids, torch.Tensor):
        x = prompt_ids.to(device=device, dtype=torch.long).unsqueeze(0)  # (1,T0) or (B=1,T0)
    else:
        x = torch.tensor(list(prompt_ids), dtype=torch.long, device=device).unsqueeze(0)

    # 读取上下文窗口（若类里无该属性，可改为常量或参数传入）
    ctx_len = getattr(model, "context_length", x.size(1))

    for _ in range(int(max_new_tokens)):
        # 只保留最近 ctx_len 个 token 作为条件
        x_cond = x[:, -ctx_len:]
        logits = model(x_cond)                # (1,T,V)
        logits_last = logits[:, -1, :]        # (1,V)

        # temperature 缩放（<=0 视为贪心）
        if temperature is None or temperature <= 0:
            next_id = torch.argmax(logits_last, dim=-1, keepdim=True)  # (1,1)
        else:
            logits_last = logits_last / float(temperature)

            # 先做 softmax 得到概率
            probs = F.softmax(logits_last, dim=-1)  # (1,V)

            # nucleus / top-p
            if top_p is not None:
                p = float(top_p)
                if not (0.0 < p <= 1.0):
                    raise ValueError(f"top_p must be in (0,1], got {top_p}")
                # 按概率降序排序
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)   # (1,V), (1,V)
                cumsum = torch.cumsum(sorted_probs, dim=-1)                              # (1,V)
                # 保留使累计概率 >= p 的最小集合：keep 首次超出阈值之后的元素置 0
                # 方式：keep = cumsum <= p；至少保留第一个
                keep = cumsum <= p
                keep[..., 0] = True
                # 反投影到原索引
                mask = torch.zeros_like(probs, dtype=torch.bool)                         # (1,V)
                mask.scatter_(-1, sorted_idx, keep)
                probs = torch.where(mask, probs, torch.zeros_like(probs))
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            # 从分布采样
            next_id = torch.multinomial(probs, num_samples=1)  # (1,1)

        x = torch.cat([x, next_id], dim=1)

        # 早停：遇到 eos
        if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
            break

    return x.squeeze(0).tolist()
