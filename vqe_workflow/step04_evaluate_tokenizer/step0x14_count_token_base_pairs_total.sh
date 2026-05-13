#!/bin/bash

# --- 1. 配置绝对路径 (基于你的 ECS 环境) ---

# tokenizer的名字,一旦固定，为了保持后续查找一致,不要改动了
TOKENIZER_NAME="vqe342s036000l1"

# 文件路径定义
PY_SCRIPT="scripts/step0x14_count_token_base_pairs_total.py"
INPUT_FILE="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.shiftr4.mongoq30.${TOKENIZER_NAME}.aligned.jsonl.gz"

# 【新增】定义输出目录
OUT_DIR="step0x14_count_token_base_pairs_total"

# 拼接输出文件路径
# 如果用户传了第二个参数就用用户的，否则生成默认路径
if [ -n "$2" ]; then
    OUTPUT_FILE="$2"
else
    # 这里将 OUT_DIR 拼接到文件名之前
    OUTPUT_FILE="${OUT_DIR}/step0x14_count_token_base_pairs_total.csv"
fi

# 【新增】创建输出目录，-p 参数保证目录存在时不报错，不存在时自动创建多级目录
mkdir -p "$OUT_DIR"


echo "🚀 开始执行统计任务..."
echo "📍 解释器: $PYTHON_EXE"

# --- 2. 运行 Python 程序 ---
python3 "$PY_SCRIPT" -i "$INPUT_FILE" -o "$OUTPUT_FILE"

# --- 3. 结果验证 ---
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✨ 任务完成！"
    echo "📊 结果已保存至: $OUTPUT_FILE"
    echo "📝 数据预览 (前 5 行):"
    column -s, -t < <(head -n 5 "$OUTPUT_FILE")
else
    echo "❌ 统计过程中出现错误。"
fi
