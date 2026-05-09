#!/bin/bash

# 配置
# 🔥 tokenizer 控制
TOKENIZER_NAME="vqe342s036000l1"

INPUT="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.shiftr4.mongoq30.${TOKENIZER_NAME}.aligned.jsonl.gz"
PATTERN="GGCGG"
MIN_REPEAT=5

# 执行
python3 scripts/step202_find_repeated_pattern.py \
    -i "$INPUT" \
    --pattern "$PATTERN" \
    --min-repeat "$MIN_REPEAT"
