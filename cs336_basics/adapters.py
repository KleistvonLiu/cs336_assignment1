from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Dict

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from tests.attention import scaled_dot_product_attention
from tests.multi_head_self_attention import MultiheadSelfAttention
from tests.positionwise_feedforward import SwiGLU
from tests.rope import RotaryPositionalEmbedding
from tests.softmax_stable import softmax_stable
from tests.tokenizer import Tokenizer
from tests.train_bpe import train_bpe
from tests.linear import Linear
from tests.embedding import Embedding
from tests.rmsnorm import RMSNorm
from tests.transformer_block import TransformerBlock
from tests.transformer_lm import TransformerLM
from .checkpointing import save_checkpoint, load_checkpoint
from .cross_entrophy import cross_entropy_stable
from .adamw import AdamWFromScratch
from .get_batch import sample_lm_batch_np
from .gradient_clipping import clip_gradients
from .lr_scheduler import lr_cosine_with_warmup


def run_linear(
        d_in: int,
        d_out: int,
        weights: Float[Tensor, " d_out d_in"],
        in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.
    Loads the provided weights into your bias-free Linear and runs a forward pass.
    """
    # 1) 设备/精度对齐
    if not isinstance(weights, torch.Tensor):
        W = torch.as_tensor(weights, device=in_features.device, dtype=in_features.dtype)
    else:
        W = weights.to(device=in_features.device, dtype=in_features.dtype)

    # 2) 基本校验
    assert W.shape == (d_out, d_in), f"weights shape {W.shape} != ({d_out}, {d_in})"
    assert in_features.shape[-1] == d_in, f"in_features last dim {in_features.shape[-1]} != {d_in}"

    # 3) 构造自定义 Linear，并加载权重（参数名必须是 'W'）
    layer = Linear(in_features=d_in, out_features=d_out,
                   device=in_features.device, dtype=in_features.dtype)
    layer.load_state_dict({"W": W}, strict=True)

    # 4) 前向计算
    layer.eval()  # 開啓評估模式，影響dropout和normalization等
    with torch.no_grad():
        return layer(in_features)


def run_embedding(
        vocab_size: int,
        d_model: int,
        weights: Float[Tensor, " vocab_size d_model"] | "numpy.ndarray",
        token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.
    """
    # 1) 设备/精度对齐
    if not isinstance(weights, torch.Tensor):
        W = torch.as_tensor(weights)
    else:
        W = weights
    W = W.to(device=token_ids.device)

    # 2) 基本形状校验
    assert W.shape == (vocab_size, d_model), f"weights shape {W.shape} != ({vocab_size}, {d_model})"

    # 3) 构造自定义 Embedding 并加载权重（参数名必须是 "weight"）
    emb = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device=token_ids.device,
        dtype=W.dtype,
    )
    emb.load_state_dict({"weight": W}, strict=True)

    # 4) 前向计算
    emb.eval()
    with torch.no_grad():
        return emb(token_ids)


def run_swiglu(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return the output of your implementation with these weights."""
    dev, dt = in_features.device, in_features.dtype

    # to() 确保 device/dtype 对齐
    W1 = torch.as_tensor(w1_weight, device=dev, dtype=dt)
    W2 = torch.as_tensor(w2_weight, device=dev, dtype=dt)
    W3 = torch.as_tensor(w3_weight, device=dev, dtype=dt)

    # 形状校验
    assert W1.shape == (d_ff, d_model), f"W1 shape {W1.shape} != {(d_ff, d_model)}"
    assert W3.shape == (d_ff, d_model), f"W3 shape {W3.shape} != {(d_ff, d_model)}"
    assert W2.shape == (d_model, d_ff), f"W2 shape {W2.shape} != {(d_model, d_ff)}"

    # 构造模块并加载权重（注意：我们的 Linear 参数名是 "W"）
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=dev, dtype=dt)
    swiglu.load_state_dict({"w1.W": W1, "w2.W": W2, "w3.W": W3}, strict=True)

    swiglu.eval()
    with torch.no_grad():
        return swiglu(in_features)


def run_scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Adapter: call your SDPA implementation. Supports broadcastable mask.
    """
    if mask is not None and not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=Q.device)
    elif mask is not None:
        mask = mask.to(device=Q.device, dtype=torch.bool)

    return scaled_dot_product_attention(Q, K, V, mask=mask)


def run_multihead_self_attention(
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    x = in_features
    dev, dt = x.device, x.dtype
    L = x.shape[-2]

    # 权重放到与输入一致的 device/dtype
    Wq = torch.as_tensor(q_proj_weight, device=dev, dtype=dt)
    Wk = torch.as_tensor(k_proj_weight, device=dev, dtype=dt)
    Wv = torch.as_tensor(v_proj_weight, device=dev, dtype=dt)
    Wo = torch.as_tensor(o_proj_weight, device=dev, dtype=dt)

    # 构造模块（max_seq_len 设为当前序列长度；RoPE 将被中和）
    mha = MultiheadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=L,
        theta=1e4,  # 任意值都可以，因为我们会把 positions 置零来中和 RoPE
        device=dev,
        dtype=dt,
    )

    # 加载权重（你的 Linear 参数名为 "W"）
    state = {
        "q_proj.W": Wq,
        "k_proj.W": Wk,
        "v_proj.W": Wv,
        "o_proj.W": Wo,
    }
    mha.load_state_dict(state, strict=True)

    # 传入全 0 的 position，令 RoPE 变为恒等（cos=1, sin=0）
    zero_positions = torch.zeros(L, device=dev, dtype=torch.long)  # 形状 (L,)

    mha.eval()
    with torch.no_grad():
        out = mha(x, token_positions=zero_positions)

    return out


def run_multihead_self_attention_with_rope(
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    x = in_features
    dev, dt = x.device, x.dtype
    L = x.shape[-2]

    # 权重对齐到输入的 device / dtype
    Wq = torch.as_tensor(q_proj_weight, device=dev, dtype=dt)
    Wk = torch.as_tensor(k_proj_weight, device=dev, dtype=dt)
    Wv = torch.as_tensor(v_proj_weight, device=dev, dtype=dt)
    Wo = torch.as_tensor(o_proj_weight, device=dev, dtype=dt)

    # 构建带 RoPE 的多头自注意力；RoPE 维度 = head_dim = d_model // num_heads（在类中使用）
    mha = MultiheadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        device=dev,
        dtype=dt,
    )

    # 按参数名加载（你的 Linear 参数名为 "W"）
    state = {
        "q_proj.W": Wq,
        "k_proj.W": Wk,
        "v_proj.W": Wv,
        "o_proj.W": Wo,
    }
    mha.load_state_dict(state, strict=True)

    # 准备 token 位置；未提供则用 [0..L-1]
    if token_positions is None:
        pos = torch.arange(L, device=dev, dtype=torch.long)  # (L,)
    else:
        pos = token_positions.to(device=dev, dtype=torch.long)

    mha.eval()
    with torch.no_grad():
        out = mha(x, token_positions=pos)  # (..., L, d_model)
    return out


def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.
    """
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    rope.eval()
    with torch.no_grad():
        return rope(in_query_or_key, token_positions.to(device=in_query_or_key.device, dtype=torch.long))


def run_transformer_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
        in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    x = in_features
    dev, dt = x.device, x.dtype
    B, L, Din = x.shape
    assert Din == d_model, f"in_features last dim {Din} != d_model {d_model}"
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # 1) 构建模块
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        device=dev,
        dtype=dt,
    )

    # 2) 准备权重（对齐 device/dtype；必要时做转置以匹配 Linear.W 的 (out, in) 约定）
    def to_dev_dtype(t: Tensor) -> Tensor:
        return t.to(device=dev, dtype=dt)

    # Attn 投影（Linear.W 形状是 (out, in)=(d_model, d_model)）
    Wq = to_dev_dtype(weights["attn.q_proj.weight"])
    Wk = to_dev_dtype(weights["attn.k_proj.weight"])
    Wv = to_dev_dtype(weights["attn.v_proj.weight"])
    Wo = to_dev_dtype(weights["attn.output_proj.weight"])

    # LN 权重（RMSNorm.weight 为 (d_model,)）
    ln1_w = to_dev_dtype(weights["ln1.weight"])
    ln2_w = to_dev_dtype(weights["ln2.weight"])

    # FFN（三个 Linear：w1: d_model->d_ff；w2: d_ff->d_model；w3: d_model->d_ff）
    # 我们 Linear.W 为 (out, in)。若提供的形状是反的，就转置。
    W1 = to_dev_dtype(weights["ffn.w1.weight"])
    W2 = to_dev_dtype(weights["ffn.w2.weight"])
    W3 = to_dev_dtype(weights["ffn.w3.weight"])

    def ensure_shape(mat: Tensor, desired: tuple[int, int]) -> Tensor:
        if mat.shape == desired:
            return mat
        elif mat.t().shape == desired:
            return mat.t()
        else:
            raise ValueError(f"FFN weight shape {tuple(mat.shape)} incompatible with {desired}")

    W1 = ensure_shape(W1, (d_ff, d_model))  # w1: (out=d_ff, in=d_model)
    W2 = ensure_shape(W2, (d_model, d_ff))  # w2: (out=d_model, in=d_ff)
    W3 = ensure_shape(W3, (d_ff, d_model))  # w3: (out=d_ff, in=d_model)

    # 3) 组装 state_dict（注意我们模块内各层参数名）
    state = {
        # LayerNorm (RMSNorm)
        "input_layernorm.weight": ln1_w,
        "post_attention_layernorm.weight": ln2_w,
        # Attention projections
        "attn.q_proj.W": Wq,
        "attn.k_proj.W": Wk,
        "attn.v_proj.W": Wv,
        "attn.o_proj.W": Wo,
        # FFN (SwiGLU)
        "ffn.w1.W": W1,
        "ffn.w2.W": W2,
        "ffn.w3.W": W3,
    }
    block.load_state_dict(state, strict=True)

    # 4) 准备 RoPE 的 token positions（未提供就用 0..L-1）
    pos = torch.arange(L, device=dev, dtype=torch.long)

    # 5) 前向
    block.eval()
    with torch.no_grad():
        out = block(x, token_positions=pos)  # (B, L, d_model)

    return out


def run_transformer_lm(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    dev, dt = in_indices.device, torch.float32

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta,
        device=dev,
        dtype=dt,
    )

    def td(x: Tensor) -> Tensor:
        return x.to(device=dev, dtype=dt)

    def fit_linear_weight(mat: Tensor, desired: tuple[int, int]) -> Tensor:
        if mat.shape == desired:
            return td(mat)
        mt = mat.t()
        if mt.shape == desired:
            return td(mt)
        raise ValueError(f"Weight shape {tuple(mat.shape)} incompatible with desired {desired}")

    # ---- 映射并加载除 lm_head 以外的所有参数 ----
    state: Dict[str, Tensor] = {}

    # token embedding
    if "token_embeddings.weight" not in weights:
        raise KeyError("Missing 'token_embeddings.weight' in weights.")
    state["tok_embeddings.weight"] = td(weights["token_embeddings.weight"])

    # 每层参数
    for i in range(num_layers):
        p = f"layers.{i}"
        state[f"blocks.{i}.input_layernorm.weight"] = td(weights[f"{p}.ln1.weight"])
        state[f"blocks.{i}.post_attention_layernorm.weight"] = td(weights[f"{p}.ln2.weight"])
        state[f"blocks.{i}.attn.q_proj.W"] = td(weights[f"{p}.attn.q_proj.weight"])
        state[f"blocks.{i}.attn.k_proj.W"] = td(weights[f"{p}.attn.k_proj.weight"])
        state[f"blocks.{i}.attn.v_proj.W"] = td(weights[f"{p}.attn.v_proj.weight"])
        state[f"blocks.{i}.attn.o_proj.W"] = td(weights[f"{p}.attn.output_proj.weight"])
        state[f"blocks.{i}.ffn.w1.W"] = fit_linear_weight(weights[f"{p}.ffn.w1.weight"], (d_ff, d_model))
        state[f"blocks.{i}.ffn.w2.W"] = fit_linear_weight(weights[f"{p}.ffn.w2.weight"], (d_model, d_ff))
        state[f"blocks.{i}.ffn.w3.W"] = fit_linear_weight(weights[f"{p}.ffn.w3.weight"], (d_ff, d_model))

    # final norm（测试里常叫 ln_final.weight）
    if "ln_final.weight" in weights:
        state["final_norm.weight"] = td(weights["ln_final.weight"])
    elif "final_norm.weight" in weights:
        state["final_norm.weight"] = td(weights["final_norm.weight"])
    elif "ln_f.weight" in weights:
        state["final_norm.weight"] = td(weights["ln_f.weight"])

    # 先加载其余参数
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = set(missing) - {"final_norm.weight"}
    if missing:
        raise RuntimeError(f"Missing keys when loading TransformerLM: {sorted(missing)}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading TransformerLM: {sorted(unexpected)}")

    # ---- 如果提供了独立 lm_head.weight，则注册为模型参数（否则走共享）----
    if "lm_head.weight" in weights:
        # 显式注册为可训练参数；这样 forward 会走独立 head，不与 embedding 共享
        model.lm_head_weight = torch.nn.Parameter(td(weights["lm_head.weight"]))

    # ---- 前向 ----
    model.eval()
    with torch.no_grad():
        L = in_indices.shape[-1]
        pos = torch.arange(L, device=dev, dtype=torch.long)
        logits = model(in_indices, token_positions=pos)  # (B, L, vocab)

    return logits.to(torch.float32)


def run_rmsnorm(
        d_model: int,
        eps: float,
        weights: Float[Tensor, " d_model"] | "numpy.ndarray",
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.
    """
    # 设备/精度对齐
    if not isinstance(weights, torch.Tensor):
        gamma = torch.as_tensor(weights, device=in_features.device, dtype=in_features.dtype)
    else:
        gamma = weights.to(device=in_features.device, dtype=in_features.dtype)

    assert gamma.shape == (d_model,), f"weights shape {gamma.shape} != ({d_model},)"

    # 构造并加载权重（参数名必须是 "weight"）
    layer = RMSNorm(d_model=d_model, eps=eps, device=in_features.device, dtype=gamma.dtype)
    layer.load_state_dict({"weight": gamma}, strict=True)

    layer.eval()
    with torch.no_grad():
        return layer(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    inputs_np, labels_np = sample_lm_batch_np(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        rng=None,  # 如需确定性，可在外部传入 np.random.default_rng(seed)
    )

    inputs = torch.as_tensor(inputs_np, dtype=torch.long, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    return inputs, labels

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax_stable(in_features, dim)


def run_cross_entropy(
        inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy_stable(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    clip_gradients(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamWFromScratch


def run_get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return lr_cosine_with_warmup(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    return save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens=special_tokens)


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return train_bpe(input_path, vocab_size, special_tokens=special_tokens, **kwargs)
