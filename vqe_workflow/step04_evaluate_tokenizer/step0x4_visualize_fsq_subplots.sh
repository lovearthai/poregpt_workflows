#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================
INPUT_FILE="/mnt/nas_syy/default/poregpt/poregpt/poregpt/workflows/vqe_workflow/step04_evaluate_tokenizer/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"

# 输出图片名称
OUTPUT_IMAGE="fsq_base_subplots_2x3.png"

# 读取的 Read 数量 (子图模式建议多读一点数据，例如 3000-5000)
NUM_READS=3000

PYTHON_SCRIPT="step0x4_visualize_fsq_subplots.py"

# ==============================================================================
# 🚀 执行逻辑
# ==============================================================================

echo "=================================================="
echo "🎨 开始绘制 2x3 碱基分布子图"
echo "=================================================="

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: 找不到 Python 脚本 '$PYTHON_SCRIPT'"
    exit 1
fi

# 执行绘图
python3 "$PYTHON_SCRIPT" \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_IMAGE" \
    -n "$NUM_READS"

if [ $? -eq 0 ]; then
    echo "✨ 绘图成功！请检查: $OUTPUT_IMAGE"
else
    echo "❌ 绘图失败。"
    exit 1
fi