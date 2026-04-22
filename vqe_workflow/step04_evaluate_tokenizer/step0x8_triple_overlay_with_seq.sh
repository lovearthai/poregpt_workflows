#!/bin/bash

# 定义输入文件（根据你的统计结果，该文件包含约 96 行数据）
INPUT_FILE="/home/jiaoshuai/step04_evaluate_tokenizer/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"
OUTPUT_IMAGE="raw_signal_single_subplots.png"

# 执行脚本
# 根据你的统计，单行最大长度约 2440
python3 step0x8_triple_overlay_with_seq.py \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_IMAGE" \
    -r 0 1000