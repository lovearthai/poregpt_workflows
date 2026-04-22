#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================
INPUT_FILE="/home/jiaoshuai/step04_evaluate_tokenizer/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"
PYTHON_SCRIPT="step0x5_analyze_fsq_high_layer.py"

# --- 实验参数 ---
TOP_K=10          # 你想标注的高频点数量
MAX_READS=10000  # 读取的数据量
OUTPUT_IMAGE="fsq_high_layer_top${TOP_K}_analysis.png"

# ==============================================================================
# 🚀 执行
# ==============================================================================

echo "--------------------------------------------------"
echo "📊 正在寻找特征编码 (Top $TOP_K Peaks)..."
echo "--------------------------------------------------"

python3 "$PYTHON_SCRIPT" \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_IMAGE" \
    -n "$MAX_READS" \
    -k "$TOP_K"

if [ $? -eq 0 ]; then
    echo "✨ 绘图成功: $OUTPUT_IMAGE"
    # 如果你想直接在终端看到数值结果，可以加一行 grep 过滤 python 的打印
fi