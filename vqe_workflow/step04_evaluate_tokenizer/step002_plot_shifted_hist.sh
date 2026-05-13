#!/usr/bin/env bash
# ==============================================================================
# 脚本功能：一键生成碱基信号分布直方图
# 核心逻辑：读取指定 jsonl.gz，计算每个碱基(经过 trim)的信号均值并绘图
# ==============================================================================

set -euo pipefail

# --- 1. 环境配置 ---
PYTHON_BIN="python3"  # 如果环境里是 python3，请修改此处
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="scripts/step002_plot_shifted_signal_hist.py"

# --- 2. 输入输出配置 (直接修改这里) ---
# 输入文件路径
INPUT_PATH="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.reformed.shiftl4.jsonl.gz"
INPUT_PATH="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.reformed.shiftl3.jsonl.gz"
INPUT_PATH="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.reformed.shiftl5.jsonl.gz"
INPUT_PATH="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.reformed.shiftr4.jsonl.gz"
INPUT_PATH="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.reformed.shiftr3.jsonl.gz"
INPUT_PATH="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.reformed.shiftr5.jsonl.gz"
INPUT_PATH="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.modified.reformed.shiftr4.jsonl.gz"

# 输出目录
OUTPUT_DIR="step002_plot_shifted_hist"

# --- 3. 绘图超参数 ---
BINS=120     # 直方图柱数
ALPHA=0.5    # 透明度 (0.0 到 1.0)
TRIM=2       # 每个碱基 Span 左右各剔除的点数 (建议 1-2，用于消除转换噪声)

# ==============================================================================
# 执行逻辑 (无需修改)
# ==============================================================================

# 检查 Python 脚本是否存在
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "[ERROR] 未找到 Python 脚本: $PYTHON_SCRIPT" >&2
    exit 1
fi

# 检查输入文件
if [[ ! -f "$INPUT_PATH" ]]; then
    echo "[ERROR] 输入文件不存在: $INPUT_PATH" >&2
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 自动推导 PNG 文件名，方便你查看
STEM=$(basename "$INPUT_PATH")
STEM="${STEM%%.jsonl*}"
EXPECTED_PNG="$OUTPUT_DIR/${STEM}.hist.png"

echo "------------------------------------------------"
echo "[*] 开始绘图任务..."
echo "[*] 输入文件: $INPUT_PATH"
echo "[*] 输出目录: $OUTPUT_DIR"
echo "[*] 预期结果: $EXPECTED_PNG"
echo "[*] 裁剪参数: trim=$TRIM"
echo "------------------------------------------------"

# 调用 Python
"$PYTHON_BIN" "$PYTHON_SCRIPT" \
    --input "$INPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --bins "$BINS" \
    --trim "$TRIM" \
    --xmin 0 \
    --xmax 150
if [[ $? -eq 0 ]]; then
    echo "------------------------------------------------"
    echo "[+] 绘图成功！"
    echo "[+] 请查看文件: $EXPECTED_PNG"
    echo "------------------------------------------------"
else
    echo "[!] 绘图过程中发生错误。"
    exit 1
fi
