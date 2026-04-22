#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================

# 输入文件 (包含 tokens_layered 和 tokens_based 字段的 jsonl.gz 文件)
INPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"

# 输出图片路径
OUTPUT_IMAGE="step0x3_fsq_base_distribution.png"

# 处理的最大 Read 数量 (建议先用少量数据测试，如 1000)
NUM_READS=3000

# Python 脚本名称
PYTHON_SCRIPT="step0x3_fsq_base_distribution_scatter_plot_movetable_jsonlgz.py"

# ==============================================================================
# 🚀 执行逻辑
# ==============================================================================

echo "=================================================="
echo "🎨 FSQ 潜空间碱基分布可视化"
echo "=================================================="
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_IMAGE"
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
    -o "$OUTPUT_IMAGE" \
    -n "$NUM_READS"

if [ $? -eq 0 ]; then
    echo -e "\n✅ 可视化任务成功结束！"
    echo "📊 高层/低层坐标的碱基分布已保存至: $OUTPUT_IMAGE"
else
    echo -e "\n❌ 执行失败，请检查报错日志。"
    exit 1
fi