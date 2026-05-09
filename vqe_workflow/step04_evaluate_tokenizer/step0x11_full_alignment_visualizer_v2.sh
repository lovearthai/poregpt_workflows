#!/bin/bash

# --- 1. 路径配置 ---
# PYTHON_EXE="/mnt/modeldisk/conda_home/anaconda3/envs/bonito_py310/bin/python"
# PROJECT_DIR="/home/jiaoshuai/step04_evaluate_tokenizer"

# 🔥 tokenizer 控制
TOKENIZER_NAME="vqe342s036000l1"

# 脚本与数据定义
PY_SCRIPT="scripts/step0x11_full_alignment_visualizer_v2.py"
INPUT_GZ="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.${TOKENIZER_NAME}.aligned.jsonl.gz"

# 输出图片路径
# 所有输出统一放在一个目录
OUT_DIR="step0x11_full_alignment_visualizer_v2"
mkdir -p "$OUT_DIR"

# 输出文件（带 tokenizer，避免覆盖）
OUTPUT_IMAGE="${OUT_DIR}/step0x11_full_alignment_plot_${TOKENIZER_NAME}.png"

# 支持外部覆盖输出路径
if [ -n "$2" ]; then
    OUTPUT_IMAGE="$2"
fi

# 定义显示范围 (根据需要修改)
RANGE_START=1000
RANGE_END=1200

# --- 2. 运行绘图 ---
echo "🎨 正在生成单条数据的对齐可视化图..."
# echo "📍 解释器: $PYTHON_EXE"

python3 "$PY_SCRIPT" \
    --input-file "$INPUT_GZ" \
    --output-file "$OUTPUT_IMAGE" \
    --range $RANGE_START $RANGE_END

# --- 3. 结果验证 ---
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✨ 任务完成！"
    echo "🖼️ 图片已保存至: $OUTPUT_IMAGE"
else
    echo "❌ 绘图失败，请检查 Python 报错信息。"
fi