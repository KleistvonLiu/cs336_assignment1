# decoding.py
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


def _prepare_prompt_ids(
    prompt_ids: Sequence[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Normalize prompt ids to shape (1, T) on the requested device."""
    if isinstance(prompt_ids, torch.Tensor):
        input_ids = prompt_ids.to(device=device, dtype=torch.long)
    else:
        input_ids = torch.tensor(list(prompt_ids), device=device, dtype=torch.long)

    if input_ids.dim() == 1:
        return input_ids.unsqueeze(0)
    if input_ids.dim() == 2 and input_ids.size(0) == 1:
        return input_ids

    raise ValueError("prompt_ids must be 1D or have shape (1, T)")


def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest high-probability set whose cumulative mass reaches top_p."""
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0,1], got {top_p}")

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens once the cumulative mass is already above the threshold,
    # but keep the first token that crosses the threshold.
    remove_sorted = cumulative_probs > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False

    filtered_sorted_probs = sorted_probs.masked_fill(remove_sorted, 0.0)
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(-1, sorted_indices, filtered_sorted_probs)

    return filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _select_next_token(
    logits_last: torch.Tensor,
    temperature: float,
    top_p: Optional[float],
) -> torch.Tensor:
    """Choose the next token from the final-step logits."""
    if temperature is None or temperature <= 0:
        return torch.argmax(logits_last, dim=-1, keepdim=True)

    scaled_logits = logits_last / float(temperature)
    probs = F.softmax(scaled_logits, dim=-1)

    if top_p is not None:
        probs = _apply_top_p(probs, float(top_p))

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def decode(
    model,  # TransformerLM
    prompt_ids: Sequence[int] | torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    device: str | torch.device = "cpu",
) -> List[int]:
    """
    自回归解码：支持 greedy、temperature 采样和 nucleus (top-p) 采样。
    返回包含 prompt 与新生成 token 的完整序列。
    """
    model.eval()

    device = torch.device(device)
    input_ids = _prepare_prompt_ids(prompt_ids, device=device)
    context_length = int(getattr(model, "context_length", input_ids.size(1)))

    for _ in range(int(max_new_tokens)):
        model_input = input_ids[:, -context_length:]
        logits = model(model_input)
        logits_last = logits[:, -1, :]

        next_token = _select_next_token(
            logits_last=logits_last,
            temperature=temperature,
            top_p=top_p,
        )
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break

    return input_ids.squeeze(0).tolist()
