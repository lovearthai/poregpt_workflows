#!/bin/bash

# 输入文件路径
INPUT_FILE="/home/jiaoshuai/step04_evaluate_tokenizer/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"
OUTPUT_IMAGE="raw_signal_overlay_full_n10_r100_300.png"

# 根据你上传的统计图：
# 平均长度约1750，最大约2440。我们将绘图范围设为 0 到 2440。
python3 step0x7_signal_overlay.py \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_IMAGE" \
    -n 10 \
    -r 100 300

if [ $? -eq 0 ]; then
    echo "✨ 绘图成功！现在 X 轴对应的是原始信号点索引 (0-2440)。"
fi