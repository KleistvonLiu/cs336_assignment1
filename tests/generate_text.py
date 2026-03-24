#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate text by calling YOUR decoding.decode() and YOUR Tokenizer.

Usage (推荐用“模块名”导入，保证相对导入 .train_bpe 能工作)：
  uv run ./tests/generate_text.py \
    --ckpt /home/kleist/Documents/Courses/assignment1-basics/runs_lr_single/exp_20251031_160954/ckpt_iter_10000.pt \
    --vocab-size 10000 --context-length 256 \
    --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 --theta 10000 \
    --tokenizer-module tests.tokenizer \
    --vocab-json /home/kleist/Documents/Courses/assignment1-basics/tokenizer_out_tiny_valid/vocab.json --merges-txt /home/kleist/Documents/Courses/assignment1-basics/tokenizer_out_tiny_valid/merges.txt \
    --special "<|endoftext|>" \
    --prompt "Once upon a time" \
    --max-new-tokens 256 --temperature 0.9 --top-p 0.95 \
    --device cuda:0 \
    --save ./sample.txt

或（没有文本 tokenizer，就直接给 prompt 的 token id）：
  python generate_with_user_tokenizer.py \
    --ckpt ./runs_lr_single/ckpt_iter_6000.pt \
    --vocab-size 10000 --context-length 256 \
    --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 --theta 10000 \
    --prompt-ids "12 34 56" \
    --max-new-tokens 256 --temperature 1.0 --top-p 0.9 \
    --eos-id 0 \
    --device cuda:0
"""
from __future__ import annotations

import time
import argparse
import importlib
from typing import List, Optional

import torch

# 你的课程组件
from tests.transformer_lm import TransformerLM
from tests.checkpointing import load_checkpoint

# 你的解码函数（decoding.py）
try:
    from decode import decode as user_decode
except Exception as e:
    raise SystemExit(f"请确保 decoding.py 在 PYTHONPATH 可导入；当前导入失败：{e}")

def _import_tokenizer_class(module_name: str):
    """
    以模块名方式导入你的 Tokenizer（例如 tests.tokenizer）。
    这样相对导入 .train_bpe 能正常解析。
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise SystemExit(f"无法导入模块 {module_name!r}：{e}")
    if not hasattr(mod, "Tokenizer"):
        raise SystemExit(f"模块 {module_name!r} 中没有名为 Tokenizer 的类")
    return mod.Tokenizer

def _infer_eos_id(tokenizer, eos_text: Optional[str]) -> Optional[int]:
    if eos_text is None:
        return None
    try:
        ids = tokenizer.encode(eos_text)
        if isinstance(ids, list) and len(ids) == 1:
            return int(ids[0])
    except Exception:
        pass
    return None

def main():
    ap = argparse.ArgumentParser(description="Generate text using your Tokenizer + decoding.decode().")
    # 模型结构（需与训练一致）
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--context-length", type=int, required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--d-model", type=int, required=True)
    ap.add_argument("--num-heads", type=int, required=True)
    ap.add_argument("--d-ff", type=int, required=True)
    ap.add_argument("--theta", type=float, default=1e4)

    # 设备与 ckpt
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", type=str, required=True)

    # 采样参数
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=None)

    # Tokenizer（模块名方式）
    ap.add_argument("--tokenizer-module", type=str, default=None,
                    help="模块名（例如 tests.tokenizer），其中应定义 Tokenizer 类")
    ap.add_argument("--vocab-json", type=str, default=None)
    ap.add_argument("--merges-txt", type=str, default=None)
    ap.add_argument("--special", type=str, action="append", default=[],
                    help="可重复：追加 special token，例如 --special \"<|endoftext|>\"")

    # prompt（文本 或 直接给 ids）
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt-ids", type=str, default=None)

    # EOS
    ap.add_argument("--eos-text", type=str, default="<|endoftext|>")
    ap.add_argument("--eos-id", type=int, default=None)

    # 输出
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    device = args.device

    # 构建模型并加载权重
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=torch.device(device),
        dtype=None,
    ).to(device).eval()

    try:
        _ = load_checkpoint(args.ckpt, model, optimizer=None)
    except TypeError:
        class _DummyOpt(torch.optim.Optimizer):
            def __init__(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, s): pass
        _ = load_checkpoint(args.ckpt, model, _DummyOpt())

    # 构建 tokenizer（若要用文本 prompt）
    tokenizer = None
    if args.prompt_ids is None:
        if not args.tokenizer_module:
            raise SystemExit("需要 --tokenizer-module、--vocab-json、--merges-txt 来编码文本 prompt；或改用 --prompt-ids。")
        if not (args.vocab_json and args.merges_txt):
            raise SystemExit("请提供 --vocab-json 与 --merges-txt。")
        Tokenizer = _import_tokenizer_class(args.tokenizer_module)
        tokenizer = Tokenizer.from_files(
            vocab_filepath=args.vocab_json,
            merges_filepath=args.merges_txt,
            special_tokens=list(args.special or []),
        )

    # 准备 prompt ids
    if args.prompt_ids is not None:
        prompt_ids: List[int] = [int(x) for x in args.prompt_ids.strip().split()]
        decode_text = None  # 无法解码为文本
    else:
        text = args.prompt or ""
        prompt_ids = tokenizer.encode(text)
        decode_text = tokenizer.decode

    # 解析 top-p
    top_p = None if args.top_p is None else float(args.top_p)
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise SystemExit(f"--top-p must be in (0,1], got {args.top_p}")

    # 确定 EOS
    eos_id = args.eos_id
    if eos_id is None and tokenizer is not None:
        eos_id = _infer_eos_id(tokenizer, args.eos_text)

    # 生成（调用你的 decoding.decode）
    t0 = time.time()
    out_ids = user_decode(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
        temperature=float(args.temperature),
        top_p=top_p,
        device=device,
    )
    dt = time.time() - t0
    new_ids = out_ids[len(prompt_ids):]

    print("\n=== Sampling config ===")
    print(f"device={device}  temperature={args.temperature}  top_p={top_p}  max_new_tokens={args.max_new_tokens}")
    print(f"prompt_len={len(prompt_ids)}  new_tokens={len(new_ids)}  total_len={len(out_ids)}  time={dt:.3f}s")
    if eos_id is not None:
        print(f"eos_id={eos_id}")

    print("\n=== Generated token IDs (first 256 new or until EOS) ===")
    print(" ".join(str(t) for t in new_ids[:256]))

    if decode_text is not None:
        try:
            print("\n=== Decoded text ===")
            txt = decode_text(out_ids)
            print(txt)
            if args.save:
                with open(args.save, "w", encoding="utf-8") as f:
                    f.write(txt)
                print(f"[saved] {args.save}")
        except Exception as e:
            print(f"[WARN] tokenizer.decode() 失败：{e}")
    else:
        print("\n(no tokenizer provided for decode; only token IDs shown)")

if __name__ == "__main__":
    main()
