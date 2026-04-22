#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================
INPUT_FILE="/mnt/nas_syy/default/poregpt/poregpt/poregpt/workflows/vqe_workflow/step04_evaluate_tokenizer/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"

# 输出图片名称
OUTPUT_IMAGE="fsq_base_distribution.png"

# 读取的 Read 数量 (建议先用少量数据测试，如 1000)
NUM_READS=2000

PYTHON_SCRIPT="step0x3_visualize_fsq_bases.py"

# ==============================================================================
# 🚀 执行逻辑
# ==============================================================================

echo "=================================================="
echo "🎨 开始绘制 FSQ 潜空间分布图"
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
    echo "✨ 绘图成功！请查看文件: $OUTPUT_IMAGE"
else
    echo "❌ 绘图失败。"
    exit 1
fi