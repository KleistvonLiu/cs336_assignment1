import torch
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy_stable(
    logits: Float[Tensor, "... vocab"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """
    数值稳定版交叉熵：CE = logsumexp(logits - max) - (logits - max)[y]
    支持任意批维（最后一维是类别维）。
    返回：标量平均损失。
    """
    x = logits.to(torch.float32)
    y = targets.to(dtype=torch.long, device=x.device)

    if y.shape != x.shape[:-1]:
        raise ValueError(f"targets shape {tuple(y.shape)} must match logits shape {tuple(x.shape[:-1])}")

    # 减最大值避免溢出
    m = x.max(dim=-1, keepdim=True).values
    x_shift = x - m

    # logsumexp over classes
    lse = torch.logsumexp(x_shift, dim=-1)

    # 选取正确类别的 logit
    """
    按标签把每个样本对应类别的logit取出来
    设 x_shift 形状是 (..., vocab)：最后一维是类别（词表）维，前面是任意“批维”（batch / 时间步等）。
    设 y 形状是 (...)：和 x_shift 去掉最后一维后的形状一致，每个位置是一条样本的正确类别索引（int/long）。
    y.unsqueeze(-1),y最后增加一个维度，为1
    gather(dim=-1, index) 会在最后一维（类别维）上，按 index 指定的列号去取值。
    里面的数就是每个样本 x_shift[..., y]
    """
    x_y = x_shift.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    # print(x_shift)
    # print(y)
    # print(x_shift.gather(-1, y.unsqueeze(-1)))

    # 平均损失
    loss = (lse - x_y).mean()
    return loss
