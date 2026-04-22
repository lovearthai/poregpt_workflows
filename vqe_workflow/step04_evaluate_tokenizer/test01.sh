#!/bin/bash

INPUT_FILE="/home/jiaoshuai/step04_evaluate_tokenizer/signal_none_mongoq30_tokenized_pair_based.jsonl.gz"

# 统计全量数据
python3 test01.py -i "$INPUT_FILE"