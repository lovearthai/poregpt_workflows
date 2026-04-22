#!/bin/bash

# --- 1. 路径配置 ---
PYTHON_EXE="/mnt/modeldisk/conda_home/anaconda3/envs/bonito_py310/bin/python"
PROJECT_DIR="/home/jiaoshuai/step04_evaluate_tokenizer"

# 脚本与数据定义
PY_SCRIPT="$PROJECT_DIR/step0x11_full_alignment_visualizer_v2.py"
INPUT_GZ="$PROJECT_DIR/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"
OUTPUT_IMG="$PROJECT_DIR/step0x11_single_alignment_plot.png"

# 定义显示范围 (根据需要修改)
RANGE_START=1000
RANGE_END=1200

# --- 2. 运行绘图 ---
echo "🎨 正在生成单条数据的对齐可视化图..."
echo "📍 解释器: $PYTHON_EXE"

$PYTHON_EXE "$PY_SCRIPT" \
    --input-file "$INPUT_GZ" \
    --output-file "$OUTPUT_IMG" \
    --range $RANGE_START $RANGE_END

# --- 3. 结果验证 ---
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✨ 任务完成！"
    echo "🖼️ 图片已保存至: $OUTPUT_IMG"
else
    echo "❌ 绘图失败，请检查 Python 报错信息。"
fi