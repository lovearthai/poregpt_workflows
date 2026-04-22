#!/bin/bash

# =================================================================
# 脚本名称: step0x14_count_token_distribution.sh
# 脚本用途: 统计 jsonl.gz 中 tokens 字段的全局分布（频次与比例）
# 使用方法: ./step0x14_count_token_distribution.sh [INPUT_GZ] [OUT_NAME]
# =================================================================

# --- 1. 配置区 ---
# 默认输入文件路径
DEFAULT_INPUT="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"
INPUT_FILE=${1:-$DEFAULT_INPUT}

# 获取不带路径的文件名作为前缀，方便区分不同的数据集
BASE_NAME=$(basename "$INPUT_FILE" .jsonl.gz)

# 获取当前日期
DATE_STR=$(date +%Y%m%d_%H%M%S)

# 输出文件名 (如果未提供第二个参数，则自动生成)
OUT_CSV="step030_count_token.csv"
OUTPUT_FILE=${2:-$OUT_CSV}

# --- 2. 执行统计 ---
echo "-------------------------------------------------------"
echo "🚀 启动 Token 分布统计..."
echo "📂 输入文件: $INPUT_FILE"
echo "📊 输出文件: $OUTPUT_FILE"
echo "-------------------------------------------------------"

# 运行 Python 脚本
python3 step030_count_token.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE"

# --- 3. 结果汇总 ---
if [ $? -eq 0 ]; then
    echo "-------------------------------------------------------"
    echo "✅ 统计成功完成！"
    # 打印前 5 个最频繁出现的 Token 预览
    echo "🔝 出现频次最高的 Top 5 Tokens (预览):"
    head -n 6 "$OUTPUT_FILE" | column -t -s ','
    echo "-------------------------------------------------------"
else
    echo "❌ 统计过程中发生错误，请检查输入文件格式。"
    exit 1
fi
