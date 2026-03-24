#!/usr/bin/env bash
# 批量跑不同 batch size，并自动线性缩放学习率 + 打印汇总
# 用法：
#   bash sweep_bsz.sh
# 可改动的参数见“CONFIG”段落

set -o pipefail

########################################
# ============ CONFIG 开始 ============
########################################

# 你的基础命令（除 batch-size/lr/min-lr/outdir 之外）
COMMON_ARGS="\
  --train ./ts_train_ids.npy --val ./ts_dev_ids.npy \
  --vocab-size 10000 --context-length 256 \
  --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 --theta 10000 \
  --iters 6000 --warmup-iters 200 --cosine-iters 5800 \
  --device cuda:0 \
  --log-to-csv \
  --diverge-patience 4"

# 扫描的 batch sizes（按需修改/增减；遇到 OOM 就注释掉更大值）
BATCH_SIZES=(1 8 32 64 128 256)

# 线性缩放的参考点：当 bsz=64 时，lr=1e-2，min_lr=1e-3
BASE_BSZ=64
BASE_LR=1e-2

# min_lr 相对 lr 的比例（你当前是 0.1）
MIN_LR_RATIO=0.1

# 输出根目录：每个 bsz 会建一个子目录
ROOT_OUTDIR="./runs_bsz_sweep_$(date +%Y%m%d_%H%M%S)"

# 可选：是否开启 W&B（默认关闭）
USE_WANDB=0
WANDB_FLAGS=""
if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_FLAGS="--wandb --wandb-project cs336-basics-train"
fi

########################################
# ============ CONFIG 结束 ============
########################################

mkdir -p "$ROOT_OUTDIR"

# 小工具：线性缩放学习率（用 awk 做浮点）
scale_lr() {
  local base_lr="$1"
  local new_bsz="$2"
  local base_bsz="$3"
  awk -v lr="$base_lr" -v b="$new_bsz" -v bref="$base_bsz" 'BEGIN{printf "%.8g", lr*b/bref}'
}

echo "[INFO] Root outdir: $ROOT_OUTDIR"
echo "[INFO] Batch sizes: ${BATCH_SIZES[*]}"
echo

for BS in "${BATCH_SIZES[@]}"; do
  LR=$(scale_lr "$BASE_LR" "$BS" "$BASE_BSZ")
  MINLR=$(awk -v lr="$LR" -v r="$MIN_LR_RATIO" 'BEGIN{printf "%.8g", lr*r}')
  OUTDIR_BSZ="$ROOT_OUTDIR/bsz_${BS}"

  echo "===== RUN: bsz=${BS}, lr=${LR}, min_lr=${MINLR} ====="
  echo "Outdir: $OUTDIR_BSZ"
  echo

  # 运行（失败时不中断整个 sweep）
  uv run tests/train.py \
    $COMMON_ARGS \
    --batch-size "$BS" \
    --lr "$LR" --min-lr "$MINLR" \
    --outdir "$OUTDIR_BSZ" \
    $WANDB_FLAGS \
    || echo "[WARN] Run failed for bsz=$BS (continuing...)"

  echo
done

# ===== 汇总结果：扫描 ROOT_OUTDIR 下所有 metrics.csv，取最后一次 val/loss =====
python - <<'PY'
import os, csv, json, math
from glob import glob

root = os.environ.get("ROOT_OUTDIR", "")
if not root:
    # 从 bash 传入
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."

def find_runs(root_dir):
    runs = []
    for p in glob(os.path.join(root_dir, "**/metrics.csv"), recursive=True):
        runs.append(os.path.dirname(p))
    return sorted(set(runs))

runs = find_runs(root)
rows_out = []
print("\n=== Sweep Summary (final val/loss) ===")
for rd in runs:
    cfg_path = os.path.join(rd, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass
    bsz = cfg.get("batch_size", None) or cfg.get("batch-size", None)
    lr  = cfg.get("lr", None)

    metrics_path = os.path.join(rd, "metrics.csv")
    last_val = None
    diverged = 0
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                tag = r.get("tag", "")
                if tag == "val/loss":
                    try:
                        last_val = float(r.get("value", "nan"))
                    except Exception:
                        pass
                if tag == "status/diverged":
                    try:
                        if float(r.get("value", "0")) > 0:
                            diverged = 1
                    except Exception:
                        pass
    except FileNotFoundError:
        continue

    name = os.path.basename(rd.rstrip(os.sep))
    print(f"{name:>24} | bsz={bsz} | lr={lr} | final val: {last_val if last_val is not None else float('inf'):.4f} | diverged: {bool(diverged)}")
    rows_out.append((name, bsz, lr, last_val, diverged))

# 另存 summary.csv 在根目录
out_csv = os.path.join(root, "summary.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run_name", "batch_size", "lr", "final_val_loss", "diverged"])
    for row in rows_out:
        w.writerow(row)
print(f"\n[saved] {out_csv}")

PY "$ROOT_OUTDIR"
