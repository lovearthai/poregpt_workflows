#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================

# tokenizer的名字,一旦固定，为了保持后续查找一致,不要改动了
TOKENIZER_NAME="vqe342s036000l1"

# 输入文件 (包含 tokens_layered 和 tokens_based 字段的 jsonl.gz 文件)
INPUT_FILE="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.shiftr4.mongoq30.${TOKENIZER_NAME}.aligned.jsonl.gz"

# 【新增】定义输出目录
OUT_DIR="step0x4_fsq_base_distribution_subplots"

# 拼接输出文件路径
# 如果用户传了第二个参数就用用户的，否则生成默认路径
if [ -n "$2" ]; then
    OUTPUT_FILE="$2"
else
    # 这里将 OUT_DIR 拼接到文件名之前
    OUTPUT_FILE="${OUT_DIR}/step0x4_fsq_base_distribution_subplots.png"
fi

# 【新增】创建输出目录，-p 参数保证目录存在时不报错，不存在时自动创建多级目录
mkdir -p "$OUT_DIR"

# 处理的最大 Read 数量 (子图模式建议多读一点数据，例如 3000-5000)
NUM_READS=3000

# Python 脚本名称
PYTHON_SCRIPT="scripts/step0x4_fsq_base_distribution_scatter_subplot_movetable_jsonlgz.py"

# ==============================================================================
# 🚀 执行逻辑
# ==============================================================================

echo "=================================================="
echo "🎨 FSQ 潜空间碱基分布子图可视化"
echo "=================================================="
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_FILE"
echo "分析: 最多处理 $NUM_READS 条 Read"
echo "=================================================="

# 检查脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: 找不到 Python 脚本 '$PYTHON_SCRIPT'"
    exit 1
fi

# 执行可视化
python3 "$PYTHON_SCRIPT" \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    -n "$NUM_READS"

if [ $? -eq 0 ]; then
    echo -e "\n✅ 可视化任务成功结束！"
    echo "📊 已生成 2x3 子图: $OUTPUT_FILE"
else
    echo -e "\n❌ 执行失败，请检查报错日志。"
    exit 1
fi