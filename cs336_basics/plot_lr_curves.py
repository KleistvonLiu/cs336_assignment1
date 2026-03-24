#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot learning curves for LR sweep (or any runs directory produced by train.py).

Usage:
  python plot_lr_curves.py --root ./runs_lr_sweep
  python plot_lr_curves.py --root ./checkpoints --include-train
  python plot_lr_curves.py --root ./runs_lr_sweep --tag "val/loss" --dpi 150

It expects each run directory to contain:
  - config.json   (includes "lr" if available)
  - metrics.csv   (columns: step, wall_time, tag, value)

Outputs in --root:
  - summary.csv
  - val_loss_vs_step.png
  - val_loss_vs_time.png
  - (optional) train_loss_vs_step.png  if --include-train is set
"""

from __future__ import annotations
import os
import csv
import json
import math
import argparse
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


@dataclass
class RunMetrics:
    name: str
    lr: Optional[float]
    batch_size: Optional[int]
    steps_val: List[float]
    vals_val: List[float]
    times_val: List[float]  # absolute wallclock (s)
    steps_train: List[float]
    vals_train: List[float]
    diverged: bool
    final_val: float  # inf if no val


def _list_run_dirs(root: str) -> list[str]:
    """
    Recursively find all run directories that contain a metrics.csv.
    Supports either passing a single run dir or a root dir with nested runs.
    """
    root = os.path.abspath(root)
    if os.path.isfile(os.path.join(root, "metrics.csv")):
        return [root]
    run_dirs = {os.path.dirname(p) for p in glob(os.path.join(root, "**", "metrics.csv"), recursive=True)}
    return sorted(run_dirs)


def _load_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_metrics_csv(run_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(run_dir, "metrics.csv")
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _to_float(x: str, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _get_batch_size_from_config(cfg: dict) -> Optional[int]:
    # 常见/兼容的键名
    candidates = [
        "batch_size", "batch-size", "batch", "bs",
        "global_batch_size", "global-batch-size",
        "micro_batch_size", "micro-batch-size",
    ]
    for k in candidates:
        if k in cfg:
            try:
                return int(cfg[k])
            except Exception:
                pass
    # 兜底：找任意包含 "batch" 的键
    for k, v in cfg.items():
        if "batch" in str(k).lower():
            try:
                return int(v)
            except Exception:
                continue
    return None


def _parse_run(run_dir: str, tag_val: str = "val/loss", tag_train: str = "train/loss") -> RunMetrics:
    cfg = _load_config(run_dir)
    rows = _load_metrics_csv(run_dir)

    steps_val, vals_val, times_val = [], [], []
    steps_train, vals_train = [], []
    diverged = False

    for r in rows:
        tag = r.get("tag", "")
        step = _to_float(r.get("step", "nan"))
        wall = _to_float(r.get("wall_time", "nan"))
        val = _to_float(r.get("value", "nan"))

        if tag == tag_val:
            steps_val.append(step)
            vals_val.append(val)
            times_val.append(wall)
        elif tag == tag_train:
            steps_train.append(step)
            vals_train.append(val)
        elif tag == "status/diverged" and val > 0:
            diverged = True

    # sort by step/time
    val_sorted = sorted(zip(steps_val, times_val, vals_val), key=lambda x: x[0])
    if val_sorted:
        steps_val, times_val, vals_val = map(list, zip(*val_sorted))
    train_sorted = sorted(zip(steps_train, vals_train), key=lambda x: x[0])
    if train_sorted:
        steps_train, vals_train = map(list, zip(*train_sorted))

    lr = cfg.get("lr", None)
    if lr is not None:
        try:
            lr = float(lr)
        except Exception:
            lr = None

    bsz = _get_batch_size_from_config(cfg)

    final_val = vals_val[-1] if vals_val else float("inf")

    return RunMetrics(
        name=os.path.basename(run_dir.rstrip(os.sep)),
        lr=lr,
        batch_size=bsz,
        steps_val=steps_val,
        vals_val=vals_val,
        times_val=times_val,
        steps_train=steps_train,
        vals_train=vals_train,
        diverged=diverged,
        final_val=final_val,
    )


def _label_for(run: RunMetrics) -> str:
    parts = []
    if run.batch_size is not None:
        parts.append(f"bs={run.batch_size}")
    if run.lr is not None and not math.isnan(run.lr):
        parts.append(f"lr={run.lr:.1e}")
    return ", ".join(parts) if parts else run.name


def _write_summary(root: str, runs: List[RunMetrics], summary_name: str = "summary.csv") -> None:
    out = os.path.join(root, summary_name)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_name", "batch_size", "lr", "final_val_loss", "diverged"])
        for r in runs:
            lr_str = f"{r.lr:.6g}" if r.lr is not None and not math.isnan(r.lr) else ""
            bsz_str = f"{r.batch_size}" if r.batch_size is not None else ""
            fin = f"{r.final_val:.6f}" if math.isfinite(r.final_val) else ""
            w.writerow([r.name, bsz_str, lr_str, fin, int(r.diverged)])
    print(f"[saved] {out}")


def plot_lr_curves(
    root: str,
    include_train: bool = False,
    tag_val: str = "val/loss",
    tag_train: str = "train/loss",
    dpi: int = 150,
    width: int = 8,
    height: int = 5,
    no_legend: bool = False,
):
    run_dirs = _list_run_dirs(root)
    if not run_dirs:
        print(f"[WARN] No run directories with metrics.csv found under: {root}")
        return

    runs = [_parse_run(d, tag_val=tag_val, tag_train=tag_train) for d in run_dirs]

    # ---- Print summary to console
    print("\n=== Runs Summary ===")
    for r in runs:
        label = _label_for(r)
        print(f"{label:>16} | final val: {r.final_val:.4f} | diverged: {r.diverged}")

    # ---- Save summary.csv
    _write_summary(root, runs)

    # ---- Plot: val loss vs step
    plt.figure(figsize=(width, height), dpi=dpi)
    plotted = 0
    for r in runs:
        if not r.steps_val:
            continue
        plt.plot(r.steps_val, r.vals_val, label=_label_for(r))
        plotted += 1
    plt.xlabel("step")
    plt.ylabel(tag_val)
    plt.title(f"{tag_val} vs step")
    plt.grid(True, alpha=0.3)
    if not no_legend and plotted > 0:
        plt.legend()
    out1 = os.path.join(root, "val_loss_vs_step.png")
    plt.tight_layout()
    plt.savefig(out1)
    print(f"[saved] {out1}")

    # ---- Plot: val loss vs wallclock (seconds since first val point of each run)
    plt.figure(figsize=(width, height), dpi=dpi)
    plotted = 0
    for r in runs:
        if not r.times_val:
            continue
        t0 = r.times_val[0]
        xs = [t - t0 for t in r.times_val]
        plt.plot(xs, r.vals_val, label=_label_for(r))
        plotted += 1
    plt.xlabel("wallclock (s)")
    plt.ylabel(tag_val)
    plt.title(f"{tag_val} vs wallclock time")
    plt.grid(True, alpha=0.3)
    if not no_legend and plotted > 0:
        plt.legend()
    out2 = os.path.join(root, "val_loss_vs_time.png")
    plt.tight_layout()
    plt.savefig(out2)
    print(f"[saved] {out2}")

    # ---- (Optional) Plot: train loss vs step
    if include_train:
        plt.figure(figsize=(width, height), dpi=dpi)
        plotted = 0
        for r in runs:
            if not r.steps_train:
                continue
            plt.plot(r.steps_train, r.vals_train, label=_label_for(r))
            plotted += 1
        plt.xlabel("step")
        plt.ylabel(tag_train)
        plt.title(f"{tag_train} vs step")
        plt.grid(True, alpha=0.3)
        if not no_legend and plotted > 0:
            plt.legend()
        out3 = os.path.join(root, "train_loss_vs_step.png")
        plt.tight_layout()
        plt.savefig(out3)
        print(f"[saved] {out3}")

    # ---- Best run hint
    best = min(runs, key=lambda r: r.final_val if math.isfinite(r.final_val) else float("inf"))
    if math.isfinite(best.final_val):
        print(f"\nBest run: {_label_for(best)} | val_loss={best.final_val:.4f}")
        if best.final_val <= 1.45:
            print("✅ Target met: val loss ≤ 1.45 on TinyStories.")
        else:
            print("ℹ️  Consider extending iters / refining lr grid / adjusting wd/min_lr.")
    else:
        print("\nNo run has a finite final validation loss. Check logs for divergence or missing evals.")


def main():
    ap = argparse.ArgumentParser(description="Plot learning curves from train.py logs (metrics.csv).")
    ap.add_argument("--root", type=str, required=True, help="Root runs directory or a single run directory containing metrics.csv")
    ap.add_argument("--include-train", action="store_true", help="Also plot train/loss vs step")
    ap.add_argument("--tag", type=str, default="val/loss", help="Validation tag name (default: val/loss)")
    ap.add_argument("--train-tag", type=str, default="train/loss", help="Train tag name (default: train/loss)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--height", type=int, default=5)
    ap.add_argument("--no-legend", action="store_true")
    args = ap.parse_args()

    plot_lr_curves(
        root=args.root,
        include_train=args.include_train,
        tag_val=args.tag,
        tag_train=args.train_tag,
        dpi=args.dpi,
        width=args.width,
        height=args.height,
        no_legend=args.no_legend,
    )


if __name__ == "__main__":
    main()
