#!/bin/bash

# =================================================================
# 脚本名称: step230_plot_token_based_signal.sh
# 脚本用途: 封装 Nanopore 信号 Token 叠加绘图脚本 (支持多层 ID 计算)
# 使用方法: ./step230_plot_token_based_signal.sh [TOKEN] [LAYER] [STRIDE] [INPUT_GZ]
# =================================================================

# --- 1. 参数配置区 ---
# 调整了参数顺序，方便常用修改
TOKEN=${1:-254336}       # 目标 Token ID (可以是单层 ID 或拼接后的 uni_id)
LAYER=${2:-0}         # 模式：0 使用 tokens 字段, >0 使用 tokens_layered 拼接前 N 层
STRIDE=${3:-4}        # 步长因子
INPUT_FILE=${4:-"/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"}

# 其他内部变量
CB_SIZE=512           # Codebook 大小 (计算多层 uni_id 时使用)
OUT_DIR="step230_plot_token_based_signal"
MAX_LINES=150
ALPHA=0.15

# --- 2. 目录准备 ---
mkdir -p "$OUT_DIR"

# 生成带参数的文件名，增加 layer 标识以便区分
# 格式: token_406_L0_s4_20260410.png
OUT_NAME="${OUT_DIR}/token_${TOKEN}_L${LAYER}_s${STRIDE}_$(date +%Y%m%d_%H%M%S).png"

# --- 3. 环境准备 ---
# 强制使用 Agg 后端，防止在无显示器的服务器上报错
export MPLBACKEND=Agg

# --- 4. 执行绘图 ---
echo "-------------------------------------------------------"
echo "正在启动绘图程序..."
echo "目标 Token  : $TOKEN"
echo "层级模式    : $LAYER (0: tokens 字段, >0: 拼接模式)"
echo "步长因子    : $STRIDE"
echo "输入文件    : $INPUT_FILE"
echo "保存路径    : $OUT_NAME"
echo "-------------------------------------------------------"

python3 step230_plot_token_based_signal.py \
    --input "$INPUT_FILE" \
    --target_token "$TOKEN" \
    --target_base "G" \
    --layer "$LAYER" \
    --codebook_size "$CB_SIZE" \
    --stride_factor "$STRIDE" \
    --output "$OUT_NAME" \
    --max_plot_lines "$MAX_LINES" \
    --alpha "$ALPHA"

# --- 5. 结果检查 ---
if [ $? -eq 0 ]; then
    echo "✅ 成功! 图片已保存至: $OUT_NAME"
else
    echo "❌ 失败! 请检查 Python 脚本报错信息或输入路径。"
    exit 1
fi
