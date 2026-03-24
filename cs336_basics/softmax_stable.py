import torch
from torch import Tensor

def softmax_stable(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along dimension `dim`."""
    if dim < 0:
        dim = x.dim() + dim
    x32 = x.to(torch.float32)
    m = x32.amax(dim=dim, keepdim=True)     # 按 dim 的最大值
    shifted = x32 - m                        # 减最大值避免溢出
    exp = shifted.exp()
    out32 = exp / exp.sum(dim=dim, keepdim=True)
    return out32.to(x.dtype)