# data_loading.py  或  tests/adapters.py 中相同作用域
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import torch


def sample_lm_batch_np(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    纯 NumPy 的批采样接口：返回 (inputs_np, labels_np)，形状均为 (B, T)。

    Args:
        dataset: 1D numpy array of integer token IDs
        batch_size: B
        context_length: T
        rng: 可选的 numpy 随机数发生器（便于外部控制随机性）；默认为 np.random

    Returns:
        inputs_np, labels_np: 两个 np.int64 数组，形状 (B, T)
    """
    x = np.asarray(dataset).reshape(-1)
    B = int(batch_size)
    T = int(context_length)

    if x.size < T + 1:
        raise ValueError(
            f"dataset must have at least context_length+1 tokens "
            f"(got {x.size}, context_length={T})"
        )

    n = x.size
    rng = rng if rng is not None else np.random

    # 选择起点 i ∈ [0, n - T - 1]。np.randint 的 high 不包含，因此用 high=n-T
    starts = rng.randint(0, n - T, size=B, dtype=np.int64)  # (B,)

    # 构造 (B, T) 的切片索引
    arange_T = np.arange(T, dtype=np.int64)                 # (T,)
    idx = starts[:, None] + arange_T                        # (B, T)
    idx_next = idx + 1                                      # (B, T)

    inputs_np = x[idx]                                      # (B, T)
    labels_np = x[idx_next]                                 # (B, T)
    # 用 int64（后面转 torch.long）
    inputs_np = inputs_np.astype(np.int64, copy=False)
    labels_np = labels_np.astype(np.int64, copy=False)
    return inputs_np, labels_np



