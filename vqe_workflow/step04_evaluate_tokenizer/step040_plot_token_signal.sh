#!/bin/bash

# =================================================================
# 脚本名称: step230_plot_token_based_signal_nobase.sh
# 脚本用途: 封装 Token 信号全叠加绘图（不区分碱基）
# 使用方法: ./run.sh [TOKEN] [LAYER] [STRIDE] [INPUT_GZ]
# =================================================================

# --- 1. 参数配置区 ---
TOKEN=${1:-348130}       # 目标 Token ID
LAYER=${2:-1}         # 0: tokens字段, >0: 拼接模式
STRIDE=${3:-4}        # 步长因子 (纳米孔模型通常为 4 或 5)
INPUT_FILE=${4:-"/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"}

# 其他硬编码参数
CB_SIZE=625           # 计算 uni_id 时的码本大小
OUT_DIR="step040_plot_token_signal"
MAX_LINES=10000         # 叠加线条数，不看碱基建议多设一点（如 200）
ALPHA=0.1             # 透明度，线条多时调低一点效果更好

# --- 2. 环境准备 ---
mkdir -p "$OUT_DIR"
export MPLBACKEND=Agg  # 生产环境强制无界面绘图

# 生成文件名: token_406_L0_nobase_20260410.png
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_NAME="${OUT_DIR}/token_${TOKEN}_L${LAYER}_nobase_${TIMESTAMP}.png"

# --- 3. 执行绘图 ---
echo "-------------------------------------------------------"
echo "🚀 正在绘制 Token 全叠加波形图..."
echo "📍 Token ID  : $TOKEN"
echo "📑 层级模式  : $LAYER"
echo "📏 步长因子  : $STRIDE"
echo "📂 输入文件  : $INPUT_FILE"
echo "🖼️ 保存路径  : $OUT_NAME"
echo "-------------------------------------------------------"

python3 step040_plot_token_signal.py \
    --input "$INPUT_FILE" \
    --target_token "$TOKEN" \
    --layer "$LAYER" \
    --codebook_size "$CB_SIZE" \
    --stride_factor "$STRIDE" \
    --output "$OUT_NAME" \
    --max_plot_lines "$MAX_LINES" \
    --alpha "$ALPHA"

# --- 4. 结果检查 ---
if [ $? -eq 0 ]; then
    echo "-------------------------------------------------------"
    echo "✅ 绘图成功！"
    echo "🔗 图片位置: $OUT_NAME"
    echo "💡 提示: 红色线条代表所有信号的平均值 (Mean Signal)"
    echo "-------------------------------------------------------"
else
    echo "❌ 绘图失败，请检查参数或文件路径。"
    exit 1
fi
