#!/bin/bash

# 配置
INPUT="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"
PATTERN="GGCGG"
MIN_REPEAT=5

# 执行
python3 step202_find_repeated_pattern.py \
    -i "$INPUT" \
    --pattern "$PATTERN" \
    --min-repeat "$MIN_REPEAT"
