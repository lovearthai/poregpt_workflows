#!/bin/bash

INPUT="/home/jiaoshuai/step04_evaluate_tokenizer/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"

# 建议先看一个较短的区间，这样碱基和辅助线的对应关系会非常清晰
python3 step0x10_base_aligned_visualizer.py \
    -i "$INPUT" \
    -o "base_signal_mapping_zoom.png" \
    -r 800 1300

# 同时也生成一个全量程的概览图
python3 step0x10_base_aligned_visualizer.py \
    -i "$INPUT" \
    -o "base_signal_mapping_full.png" \
    -r 0 2440