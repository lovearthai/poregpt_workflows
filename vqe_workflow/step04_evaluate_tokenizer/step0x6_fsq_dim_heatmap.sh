#!/bin/bash

# 配置路径
INPUT_FILE="/home/jiaoshuai/step04_evaluate_tokenizer/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"
OUTPUT_IMAGE="fsq_dimension_feature_map.png"
PYTHON_SCRIPT="step0x6_fsq_dim_heatmap.py"

# 数据读取量建议
# 对于统计热图，10000条记录左右能得到非常稳定的分布
NUM_READS=10000

echo "=================================================="
echo "🌡️ 正在执行 FSQ 维度解码与热图分析"
echo "=================================================="

python3 "$PYTHON_SCRIPT" \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_IMAGE" \
    -n "$NUM_READS"

if [ $? -eq 0 ]; then
    echo "✨ 分析成功！"
    echo "图片位置: $OUTPUT_IMAGE"
    echo "请观察热图中的高亮块（热点），那是该碱基在 FSQ 空间中的特征‘指纹’。"
fi