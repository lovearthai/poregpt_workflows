#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域 (在此处修改参数)
# ==============================================================================

FILE_PATTERN="LB07"
# 信号截断阈值 (对应 Python 脚本的 --clip-value)
CLIP_VALUE=3.0

# 信号处理策略 (对应 Python 脚本的 --process-strategy)
PROCESS_STRATEGY="mongo"


INPUT_FILE="/mnt/zzbnew/poregpt/dnadata/movetable/signal_${FILE_PATTERN}.modified.reformed.shiftr4.jsonl.gz"

# 预定义基础路径（不包含 r/l 标识的部分）
OUTPUT_FILE="/mnt/zzbnew/poregpt/dnadata/movetable/signal_${FILE_PATTERN}.modified.reformed.shiftr4.mongoq30.jsonl.gz"

# Python 脚本名称 (确保该文件在当前目录存在)
PYTHON_SCRIPT="scripts/step010_filter_movetable_jsonlgz.py"

# ==============================================================================
# 🚀 执行逻辑 (下方代码通常无需修改)
# ==============================================================================

echo "=================================================="
echo "🚀 开始执行 Nanopore 信号过滤"
echo "=================================================="
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_FILE"
echo "阈值: $CLIP_VALUE"
echo "策略: $PROCESS_STRATEGY"
echo "=================================================="

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: 找不到 Python 脚本 '$PYTHON_SCRIPT'"
    exit 1
fi

# 执行 Python 脚本
python3 "$PYTHON_SCRIPT" \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    --clip-value "$CLIP_VALUE" \
    --process-strategy "$PROCESS_STRATEGY"

echo ""
echo "✅ 处理完成！"
