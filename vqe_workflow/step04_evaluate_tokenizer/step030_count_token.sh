#!/bin/bash

# =================================================================
# 脚本名称: step0x14_count_token_distribution.sh
# 脚本用途: 统计 jsonl.gz 中 tokens 字段的全局分布（频次与比例）
# 使用方法: ./step0x14_count_token_distribution.sh [INPUT_GZ] [OUT_NAME]
# =================================================================

# --- 1. 配置区 ---
# 默认输入文件路径

# tokenizer的名字,一旦固定，为了保持后续查找一致,不要改动了
TOKENIZER_NAME="vqe342s036000l1"

DEFAULT_INPUT="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.shiftr4.mongoq30.${TOKENIZER_NAME}.aligned.jsonl.gz"
INPUT_FILE=${1:-$DEFAULT_INPUT}

# 获取不带路径的文件名作为前缀，方便区分不同的数据集
BASE_NAME=$(basename "$INPUT_FILE" .jsonl.gz)

# 获取当前日期
DATE_STR=$(date +%Y%m%d_%H%M%S)

# 【新增】定义输出目录
OUT_DIR="step030_count_token"

# 输出文件名 (如果未提供第二个参数，则自动生成)
# OUT_CSV="step030_count_token_layer${LAYER}.csv"
# OUTPUT_FILE=${2:-$OUT_CSV}

# 如果layer==0,就是统计tokens字段
# 如果layer==1,就是统计tokens_layered字段的第0层
LAYER=1

# 拼接输出文件路径
# 如果用户传了第二个参数就用用户的，否则生成默认路径
if [ -n "$2" ]; then
    OUTPUT_FILE="$2"
else
    # 这里将 OUT_DIR 拼接到文件名之前
    OUTPUT_FILE="${OUT_DIR}/step030_count_token_layer${LAYER}.csv"
fi

# --- 2. 准备工作 ---
# 【新增】创建输出目录，-p 参数保证目录存在时不报错，不存在时自动创建多级目录
mkdir -p "$OUT_DIR"

# --- 3. 执行统计 ---
echo "-------------------------------------------------------"
echo "🚀 启动 Token 分布统计..."
echo "📂 输入文件: $INPUT_FILE"
echo "📁 输出目录: $OUT_DIR"
echo "📊 输出文件: $OUTPUT_FILE"
echo "-------------------------------------------------------"

# 运行 Python 脚本
python3 scripts/step030_count_token.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --layer $LAYER

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
