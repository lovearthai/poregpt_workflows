#!/bin/bash

# ==============================================================================
# @Module:   0x16 Data Integration - Ratio Alignment
# @Desc:     将碱基频率 (Ratio) 合并入潜空间向量表，为后续多维聚类做准备
# ==============================================================================

# --- 配置区 ---
readonly COUNT_FILE="step0x14_count_token_base_pairs_layer.csv"
readonly VEC_FILE="step0x15_transform_fsqtoken_to_vector.csv"
readonly OUTPUT_FILE="step0x17_token_vector_with_ratio.csv"
readonly WORKER="step0x17_merge_fsq_vector_and_ratio.py"

echo "[$(date +'%H:%M:%S')] INFO: Starting data merge pipeline..."

# 检查输入
if [[ ! -f "$COUNT_FILE" || ! -f "$VEC_FILE" ]]; then
    echo "ERROR: Input CSV files not found."
    exit 1
fi

# 执行合并
python3 "$WORKER" \
    --count_csv "$COUNT_FILE" \
    --vec_csv "$VEC_FILE" \
    --output "$OUTPUT_FILE"

if [[ $? -eq 0 ]]; then
    echo "-------------------------------------------------------"
    echo "✅ SUCCESS: Integrated dataset is ready."
    echo "📊 Sample of merged header:"
    head -n 1 "$OUTPUT_FILE"
    echo "-------------------------------------------------------"
else
    echo "❌ FATAL: Merge process failed."
    exit 1
fi
