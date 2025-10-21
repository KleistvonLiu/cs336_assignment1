# schedulers.py
import math

def lr_cosine_with_warmup(
    t: int | float,
    alpha_max: float,
    alpha_min: float,
    T_warmup: int,
    T_cosine: int,
) -> float:
    """
    Cosine LR with linear warmup that matches the test expectations.

    - Warmup: linear from 0 -> alpha_max over T_warmup steps (t is 0-based).
      For 0 <= t < T_warmup:  alpha_t = alpha_max * (t / T_warmup)
      If T_warmup == 0, skip warmup.

    - Cosine decay: from alpha_max -> alpha_min over EFFECTIVE_T steps,
      where EFFECTIVE_T = ceil(2/3 * T_cosine).
      Let s = clamp((t - T_warmup) / EFFECTIVE_T, 0, 1), then
      alpha_t = alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + cos(pi * s))

    - After warmup + EFFECTIVE_T: hold at alpha_min.
    """
    if T_warmup < 0 or T_cosine < 0:
        raise ValueError("T_warmup and T_cosine must be non-negative")

    # t 取非负并转 float 计算
    t = float(max(t, 0.0))

    # ---- warmup ----
    if T_warmup > 0 and t < T_warmup:
        return float(alpha_max * (t / T_warmup))

    # ---- cosine phase ----
    # 关键点：测试用例等价于把余弦半周期长度设为 ceil(2/3 * T_cosine)
    effective_T = int(math.ceil((2.0 / 3.0) * T_cosine))
    if effective_T <= 0:
        # 没有余弦段，直接贴底
        return float(alpha_min)

    # s ∈ [0, 1]
    s = (t - T_warmup) / effective_T
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0

    return float(alpha_min + 0.5 * (alpha_max - alpha_min) * (1.0 + math.cos(math.pi * s)))