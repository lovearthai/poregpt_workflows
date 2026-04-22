#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================

# 输入文件 (上一步生成的包含 tokens_layered 的文件)
INPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.jsonl.gz"

# 输出文件
OUTPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"

# 映射因子 (1 token = 4 signals)
FACTOR=4

# Python 脚本名称
PYTHON_SCRIPT="step0x2_map_tokens_to_bases.py"

# ==============================================================================
# 🚀 执行逻辑
# ==============================================================================

echo "=================================================="
echo "🧬 开始执行 Token-Base 序列映射"
echo "=================================================="
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_FILE"
echo "比例: 1 token : $FACTOR signals"
echo "=================================================="

# 检查脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: 找不到 Python 脚本 '$PYTHON_SCRIPT'"
    exit 1
fi

# 执行转换
python3 "$PYTHON_SCRIPT" \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    -f "$FACTOR"

if [ $? -eq 0 ]; then
    echo -e "\n✅ 映射任务成功结束！"
    echo "📊 新字段 'tokens_based' 已添加。"
else
    echo -e "\n❌ 执行失败，请检查报错日志。"
    exit 1
fi
