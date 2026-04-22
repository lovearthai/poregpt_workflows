#!/bin/bash

INPUT="/home/jiaoshuai/step04_evaluate_tokenizer/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"

# 尝试绘制一个更窄的区间（如 0-1000），你会看到 Token 的对齐更加精准
python3 step0x9_final_signal_debug.py \
    -i "$INPUT" \
    -o "aligned_signal_debug_0_2440.png" \
    -r 0 1000

#python3 step06_aligned_debug_visualizer.py \
    -i "$INPUT" \
    -o "aligned_signal_debug_zoom.png" \
    -r 500 1200