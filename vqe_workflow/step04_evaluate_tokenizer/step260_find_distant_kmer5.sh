#!/bin/bash

# --- 1. 配置参数 ---
# 输入文件 (如果运行时没传参数，默认使用 sigma3 文件)
INPUT_CSV=${1:-"step252_cal_bases_pattern_feature/step252_strategy_all_block5_sigma3.csv"}

# 输出目录 (用于存放可能的 log 或结果，避免路径错误)
OUTPUT_DIR="step260_find_distant_kmer5"

# 脚本名称
PY_SCRIPT="step260_find_distant_kmer5.py"

# --- 2. 环境检查 ---
if [ ! -f "$INPUT_CSV" ]; then
    echo "❌ 错误: 找不到输入文件 $INPUT_CSV"
    exit 1
fi

if [ ! -f "$PY_SCRIPT" ]; then
    echo "❌ 错误: 找不到 Python 脚本 $PY_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- 3. 执行分析 ---
echo "------------------------------------------------"
echo "🚀 开始寻找特征空间中距离最远的碱基模式"
echo "时间:      $(date)"
echo "输入文件:  $INPUT_CSV"
echo "Top N:     10"
echo "------------------------------------------------"

# 运行 Python 脚本
# 我们将结果同时输出到屏幕和 log 文件中
python3 "$PY_SCRIPT" \
    --input "$INPUT_CSV" \
    --top_n 5 | tee "$OUTPUT_DIR/analysis_result_$(date +%Y%m%d_%H%M%S).log"

echo "------------------------------------------------"
echo "✅ 分析完成！结果已保存至 $OUTPUT_DIR"
