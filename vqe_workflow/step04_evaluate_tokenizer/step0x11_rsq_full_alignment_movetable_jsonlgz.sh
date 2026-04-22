#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================

# 输入文件：包含 signal, pattern, base_sample_spans_rel, tokens_layered 字段的 jsonl.gz 文件
INPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"

# Python 脚本名称
PYTHON_SCRIPT="step0x11_rsq_full_alignment_movetable_jsonlgz.py"

# 输出图片路径
OUTPUT_IMAGE="step0x11_rsq_full_alignment_plot.png"

# 可视化参数
RANGE_START=1000   # 显示区间的起始位置
RANGE_END=1200     # 显示区间的结束位置

# ==============================================================================
# 🚀 脚本功能说明
# ==============================================================================
# 该脚本用于生成 RSQ 编码的长序列对齐可视化图：
# - 3个子图数据来源于从文件开头顺序读取的3行数据
# - 显示指定区间内的原始信号曲线
# - 标注对应区间的碱基序列（来自 pattern 字段）
# - 显示 token 编码位置和值，格式为 [ID(3位):4维坐标索引串]
# - 便于观察信号与碱基/Token/RSQ坐标 的对齐关系
# ==============================================================================

# ==============================================================================
# 🚀 执行逻辑
# ==============================================================================

echo "=================================================="
echo "📊 RSQ 长序列对齐可视化"
echo "=================================================="
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_IMAGE"
echo "分析: 显示区间 $RANGE_START-$RANGE_END"
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
    -r "$RANGE_START" "$RANGE_END"

if [ $? -eq 0 ]; then
    echo -e "\n✅ 可视化任务成功结束！"
    echo "📊 已生成对齐图: $OUTPUT_IMAGE"
    echo "💡 提示：Token格式 [ID:坐标串]，内嵌参数 num_quantizers=1。"
else
    echo -e "\n❌ 可视化失败，请检查报错日志。"
    exit 1
fi