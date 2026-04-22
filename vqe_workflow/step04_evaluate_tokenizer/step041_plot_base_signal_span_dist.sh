#!/bin/bash

# --- 环境配置 ---
# 建议根据你的系统环境修改 conda 激活命令
# source activate bonito_py310

# --- 路径配置 ---
INPUT_JSONL="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"
OUTPUT_DIR="step041_plot_base_signal_span_dist"
OUTPUT_PNG="${OUTPUT_DIR}/base_signal_span_distribution.png"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# --- 脚本调用 ---
echo "----------------------------------------------------------------"
echo "🚀 启动碱基信号长度分布统计..."
echo "📂 输入文件: ${INPUT_JSONL}"
echo "📊 输出文件: ${OUTPUT_PNG}"
echo "----------------------------------------------------------------"

# 执行 Python 脚本
# 假设脚本接受 -i/--input 和 -o/--output 参数
python step041_plot_base_signal_span_dist.py \
    --input "${INPUT_JSONL}" \
    --output "${OUTPUT_PNG}"

# 检查执行状态
if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------------"
    echo "✅ 统计任务成功完成！"
    echo "🖼️ 请查看结果: ${OUTPUT_PNG}"
else
    echo "----------------------------------------------------------------"
    echo "❌ 脚本执行出错，请检查 step041_plot_base_signal_span_dist.py 的日志。"
fi
