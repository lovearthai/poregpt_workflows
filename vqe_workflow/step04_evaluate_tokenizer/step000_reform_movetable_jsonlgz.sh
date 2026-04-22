#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域 (在此处修改文件路径和参数)
# ==============================================================================

# 输入文件路径
INPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.original.jsonl.gz"

# 输出文件路径
OUTPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.reformed.jsonl.gz"

# 保留的小数位数
PRECISION=1

# Python 脚本名称
PYTHON_SCRIPT="step000_reform_movetable_jsonlgz.py"

# ==============================================================================
# 🚀 执行逻辑
# ==============================================================================

echo "=================================================="
echo "🚀 开始执行信号精度处理"
echo "=================================================="
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_FILE"
echo "精度: 保留 $PRECISION 位小数"
echo "=================================================="

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: 找不到 Python 脚本 '$PYTHON_SCRIPT'"
    exit 1
fi

# 执行 Python 脚本，并传入参数
python3 "$PYTHON_SCRIPT" \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    -p "$PRECISION"

echo ""
echo "✅ 任务执行完毕！"
