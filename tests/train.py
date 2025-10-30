#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：
python train.py \
  --train ./train.npy --val ./val.npy \
  --vocab-size 50257 \
  --context-length 256 \
  --num-layers 6 --d-model 384 --num-heads 6 --d-ff 1536 \
  --batch-size 32 --iters 5000 \
  --lr 3e-4 --min-lr 3e-5 --warmup-iters 200 --cosine-iters 2000 \
  --outdir ./runs_lr_sweep \
  --device cuda:0 \
  --log-to-csv \
  --diverge-patience 4

学习率扫描：
uv run tests/train.py \
  --train ./ts_train_ids.npy --val ./ts_dev_ids.npy \
  --vocab-size 10000 --context-length 256 \
  --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 --theta 10000\
  --batch-size 64 --iters 6000 \
  --lr 3e-4 --min-lr 3e-5 --warmup-iters 200 --cosine-iters 5800 \
  --outdir ./runs_lr_sweep \
  --device cuda:0 \
  --log-to-csv \
  --sweep-lrs "4e-3,5e-3,7e-3,1e-2"
"""
from __future__ import annotations

import os
import time
import json
import argparse
import pathlib
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.adamw import AdamWFromScratch
from tests.checkpointing import load_checkpoint, save_checkpoint
from tests.get_batch import sample_lm_batch_np
from tests.gradient_clipping import clip_gradients
from tests.lr_scheduler import lr_cosine_with_warmup
from tests.transformer_lm import TransformerLM

# ========== 轻量实验记录器（CSV + JSONL，可叠加 W&B） ==========
class CSVJSONLogger:
    def __init__(self, run_dir: str, enable_csv: bool = True, enable_jsonl: bool = True, flush_every: int = 1, save_config: dict | None = None):
        self.run_dir = run_dir
        pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.enable_csv = enable_csv
        self.enable_jsonl = enable_jsonl
        self.flush_every = max(1, int(flush_every))
        self._cnt = 0
        self._csv_fp = None
        self._jsonl_fp = None
        if save_config is not None:
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(save_config, f, indent=2, ensure_ascii=False)
        if self.enable_csv:
            import csv
            newfile = not os.path.exists(os.path.join(run_dir, "metrics.csv"))
            self._csv_fp = open(os.path.join(run_dir, "metrics.csv"), "a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=["step", "wall_time", "tag", "value"])
            if newfile:
                self._csv_writer.writeheader()
        if self.enable_jsonl:
            self._jsonl_fp = open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8")
        self.t0 = time.time()

    def log(self, step: int, wall_time: float | None = None, **metrics: float):
        if wall_time is None:
            wall_time = time.time()
        if self._csv_fp is not None:
            for k, v in metrics.items():
                self._csv_writer.writerow({"step": step, "wall_time": wall_time, "tag": k, "value": float(v)})
        if self._jsonl_fp is not None:
            rec = {"step": step, "wall_time": wall_time, **{k: float(v) for k, v in metrics.items()}}
            self._jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._cnt += 1
        if self._cnt % self.flush_every == 0:
            self.flush()

    def flush(self):
        if self._csv_fp is not None:
            self._csv_fp.flush()
        if self._jsonl_fp is not None:
            self._jsonl_fp.flush()

    def close(self):
        self.flush()
        if self._csv_fp is not None:
            self._csv_fp.close()
        if self._jsonl_fp is not None:
            self._jsonl_fp.close()

# --------- 数据：省内存 memmap 打开 & 备用批采样 ---------
def _open_memmap(path: str, dtype: str = "int64"):
    p = pathlib.Path(path)
    if p.suffix == ".npy":
        return np.load(p, mmap_mode="r")
    return np.memmap(p, mode="r", dtype=dtype)

@torch.no_grad()
def _estimate_val_loss(model: nn.Module, val_ds, B: int, T: int, device: str, steps: int = 50) -> float:
    model.eval()
    total = 0.0
    for _ in range(steps):
        x_np, y_np = sample_lm_batch_np(val_ds, B, T)
        x = torch.as_tensor(x_np, dtype=torch.long, device=device)
        y = torch.as_tensor(y_np, dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total += loss.item()
    model.train()
    return total / steps

# --------- 单次训练（可被 sweep 循环调用） ---------
def run_one(args, lr_override: float | None = None, run_name_suffix: str = "") -> float:
    # 打开 memmap
    train_ds = _open_memmap(args.train, dtype=args.dtype)
    val_ds = _open_memmap(args.val, dtype=args.dtype)
    device = args.device

    # 目录 & 配置保存
    base_outdir = pathlib.Path(args.outdir)
    lr_used = float(args.lr if lr_override is None else lr_override)
    run_dir = base_outdir / (args.run_name if args.run_name else time.strftime("exp_%Y%m%d_%H%M%S"))
    if run_name_suffix:
        run_dir = pathlib.Path(str(run_dir) + f"_{run_name_suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 模型 & 优化器
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
    ).to(device)

    optimizer = AdamWFromScratch(
        model.parameters(),
        lr=lr_used,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 记录器
    save_cfg = vars(args).copy()
    save_cfg["lr"] = lr_used
    logger = CSVJSONLogger(str(run_dir), enable_csv=args.log_to_csv, enable_jsonl=args.log_to_csv, save_config=save_cfg)

    # 可选恢复
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        try:
            start_iter = load_checkpoint(args.resume, model, optimizer)
            print(f"[INFO] Resumed from {args.resume} @ iter={start_iter}")
        except Exception as e:
            print(f"[WARN] Resume failed: {e}")

    # 可选 W&B
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project,
                   name=(args.wandb_runname or run_dir.name),
                   config=save_cfg)

    last = time.time()
    model.train()

    # 连续验证上升计数（发散/早停）
    val_bad_streak = 0
    best_val = float("inf")

    final_val = float("inf")

    for it in range(start_iter, args.iters):
        # 学习率调度
        lr_t = lr_cosine_with_warmup(
            t=it,
            alpha_max=lr_used,
            alpha_min=args.min_lr,
            T_warmup=args.warmup_iters,
            T_cosine=args.cosine_iters,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t

        # 采样 batch
        x_np, y_np = sample_lm_batch_np(train_ds, args.batch_size, args.context_length)
        x = torch.as_tensor(x_np, dtype=torch.long, device=device)
        y = torch.as_tensor(y_np, dtype=torch.long, device=device)

        # 前向/反向
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪
        if args.clip_grad_norm and args.clip_grad_norm > 0:
            clip_gradients(model.parameters(), args.clip_grad_norm)

        # 梯度范数监控（可选）
        with torch.no_grad():
            gnorm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gnorm_sq += float(p.grad.detach().float().pow(2).sum().item())
            gnorm = float(gnorm_sq ** 0.5)

        # 发散检测1：loss 非有限
        if not torch.isfinite(loss):
            print("[diverged] loss is NaN/Inf; abort this run.")
            logger.log(step=it+1, wall_time=time.time(), **{"status/diverged": 1})
            if args.wandb:
                import wandb
                wandb.log({"status/diverged": 1, "iter": it+1})
            break

        optimizer.step()

        # 日志（step & wallclock）
        if (it + 1) % args.log_interval == 0 or it == start_iter:
            wall = time.time()
            dt = wall - last
            last = wall
            print(f"[it {it+1}/{args.iters}] lr={lr_t:.4e} loss={loss.item():.4f} gnorm={gnorm:.2f} ({dt*1000:.1f} ms/iter)")
            logger.log(step=it+1, wall_time=wall, **{
                "train/loss": float(loss.item()),
                "lr": float(lr_t),
                "time/iter_ms": dt * 1000.0,
                "grad/gnorm": gnorm,
            })
            if args.wandb:
                import wandb
                wandb.log({"train/loss": loss.item(), "lr": lr_t, "grad/gnorm": gnorm, "iter": it+1})

        # 验证
        if (it + 1) % args.eval_interval == 0:
            val_loss = _estimate_val_loss(model, val_ds, args.batch_size, args.context_length, device, steps=args.eval_steps)
            final_val = val_loss
            print(f"[eval] it {it+1}: val_loss={val_loss:.4f}")
            logger.log(step=it+1, wall_time=time.time(), **{"val/loss": float(val_loss)})

            # 连涨早停 / 发散检测2
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                val_bad_streak = 0
            else:
                val_bad_streak += 1
                if args.diverge_patience > 0 and val_bad_streak >= args.diverge_patience:
                    print(f"[early-stop/diverged] val loss increased {val_bad_streak} times in a row; stop.")
                    logger.log(step=it+1, wall_time=time.time(), **{"status/early_stop": 1})
                    break

            if args.wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "iter": it+1})

        # 保存 checkpoint
        if (it + 1) % args.save_interval == 0 or (it + 1) == args.iters:
            ckpt_path = pathlib.Path(run_dir) / f"ckpt_iter_{it+1}.pt"
            try:
                save_checkpoint(model, optimizer, it+1, str(ckpt_path))
                print(f"[ckpt] saved to {ckpt_path}")
                logger.log(step=it+1, wall_time=time.time(), **{"ckpt/saved": 1})
            except Exception as e:
                print(f"[WARN] Failed to save checkpoint: {e}")

    logger.close()
    if args.wandb:
        import wandb
        wandb.finish()
    return final_val

# --------- 主入口：支持 sweep 或单次训练 ---------
def main():
    ap = argparse.ArgumentParser(description="Train TransformerLM with AdamWFromScratch, cosine LR warmup, memmap data, checkpointing.")
    # 数据与设备
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--val", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="int64")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--theta", type=float, default=1e4)
    # 优化
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-iters", type=int, default=200)
    ap.add_argument("--cosine-iters", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    # 训练控制
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    # Checkpoint & 路径
    ap.add_argument("--outdir", type=str, default="./checkpoints")
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--resume", type=str, default="")
    # 日志
    ap.add_argument("--log-to-csv", action="store_true", help="写 runs/<run>/metrics.csv 与 metrics.jsonl")
    ap.add_argument("--diverge-patience", type=int, default=0, help="验证损失连续上升多少次后早停 (0=关闭)")
    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="cs336-basics-train")
    ap.add_argument("--wandb-runname", type=str, default="")
    # 学习率扫描
    ap.add_argument("--sweep-lrs", type=str, default="", help='逗号分隔的学习率列表，如 "3e-4,5e-4,7e-4,1e-3"')
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.sweep_lrs:
        # 多组 LR 循环
        lr_list: List[float] = [float(s) for s in args.sweep_lrs.split(",") if s.strip()]
        results = []
        for lr in lr_list:
            print(f"\n===== SWEEP LR = {lr:.3e} =====\n")
            val = run_one(args, lr_override=lr, run_name_suffix=f"lr_{str(lr).replace('.', 'p')}")
            results.append((lr, val))
        # 打印汇总
        print("\n=== Sweep Summary (final val loss) ===")
        for lr, val in results:
            print(f"lr={lr:.3e} -> val_loss={val:.4f}")
        # 找最好的一组
        best = min(results, key=lambda x: x[1] if np.isfinite(x[1]) else float("inf"))
        print(f"\nBest: lr={best[0]:.3e}, val_loss={best[1]:.4f}")
    else:
        # 单次训练
        run_one(args, lr_override=None, run_name_suffix="")

if __name__ == "__main__":
    main()
