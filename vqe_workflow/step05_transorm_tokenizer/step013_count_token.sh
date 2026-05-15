#!/bin/bash

# --- 配置参数 ---
# 数据存放的根目录
INPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256_repeated"

# 统计结果输出路径
OUTPUT_CSV="token_statistics_$(date +%Y%m%d_%H%M%S).csv"

# 并行进程数 (建议设为 CPU 核心数)
WORKERS=$(nproc)

echo "Starting token counting task..."
echo "Input Directory: $INPUT_DIR"
echo "Output File: $OUTPUT_CSV"
echo "Workers: $WORKERS"

# --- 执行 Python 脚本 ---
python3 count_tokens.py \
    --input_dir "$INPUT_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --workers "$WORKERS"

if [ $? -eq 0 ]; then
    echo "Task finished successfully."
else
    echo "Task failed. Please check the logs."
    exit 1
fi
