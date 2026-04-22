#!/bin/bash

# --- 1. 配置绝对路径 (基于你的 ECS 环境) ---
PROJECT_DIR="/home/jiaoshuai/step04_evaluate_tokenizer"

# 文件路径定义
PY_SCRIPT="$PROJECT_DIR/step0x14_count_token_base_pairs_layer.py"
INPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"

OUTPUT_CSV="$PROJECT_DIR/step0x14_count_token_base_pairs_layer.csv"

echo "🚀 开始执行统计任务..."
echo "📍 解释器: $PYTHON_EXE"

# --- 2. 运行 Python 程序 ---
python3 "$PY_SCRIPT" -i "$INPUT_FILE" -o "$OUTPUT_CSV"

# --- 3. 结果验证 ---
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✨ 任务完成！"
    echo "📊 结果已保存至: $OUTPUT_CSV"
    echo "📝 数据预览 (前 5 行):"
    column -s, -t < <(head -n 5 "$OUTPUT_CSV")
else
    echo "❌ 统计过程中出现错误。"
fi
