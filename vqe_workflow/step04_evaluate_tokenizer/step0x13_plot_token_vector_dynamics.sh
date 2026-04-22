#!/bin/bash

# ==========================================
# 脚本名称: 隐空间动力学可视化运行脚本 (分段增强版)
# 功能: 配置特定行、Token 范围及滑动窗口，调用 Python 绘图工具
# ==========================================

# --- 1. 路径配置 ---
INPUT_GZ=${1:-"/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.recon1.jsonl.gz"}
OUTPUT_DIR="step0x13_plot_token_vector_dynamics"
mkdir -p $OUTPUT_DIR

# --- 2. 数据定位参数 (新增) ---

# 指定分析 JSONL 文件中的第几行 (0 代表第一行)
LINE_ID=1851

# Token 的起始和结束索引 (不再直接输入信号点范围，而是按 Token 算)
# 这里的范围会被平分成 3 个 Subplot 绘制
TOKEN_START=350
TOKEN_END=650

# 每个 Token 对应的信号步长 (stride)，用于计算坐标对齐
# 你的 VQ 模型通常是 4
SIGNAL_STRIDE=4

# --- 3. 动力学分析参数 ---

# 窗口大小 (Window Size): 1 表示计算相邻 Token 间的移动
WIN_SIZE=1
WIN_STRIDE=1

# --- 4. 生成输出文件名 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TAG="line${LINE_ID}_tk${TOKEN_START}-${TOKEN_END}_sz${WIN_SIZE}"
OUTPUT_PNG="${OUTPUT_DIR}/dynamics_segmented_${TAG}_${TIMESTAMP}.png"

# --- 5. 执行分析与绘图 ---

echo "------------------------------------------------"
echo "🚀 开始执行隐空间动力学联合分析 (分段模式)"
echo "时间:        $(date)"
echo "输入文件:    $INPUT_GZ"
echo "目标行号:    $LINE_ID"
echo "Token 范围:  $TOKEN_START 到 $TOKEN_END (Stride: $SIGNAL_STRIDE)"
echo "窗口配置:    Size=$WIN_SIZE, Stride=$WIN_STRIDE"
echo "------------------------------------------------"

# 统计执行耗时
start_time=$(date +%s)

# 注意：这里调用的参数名已更新，以匹配最新的 Python 脚本
python3 step0x13_plot_token_vector_dynamics.py \
    --input-file "$INPUT_GZ" \
    --output-file "$OUTPUT_PNG" \
    --line-id $LINE_ID \
    --token-start $TOKEN_START \
    --token-end $TOKEN_END \
    --signal-stride $SIGNAL_STRIDE \
    --window-size $WIN_SIZE \
    --window-stride $WIN_STRIDE

# --- 6. 检查结果 ---

if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "------------------------------------------------"
    echo "✅ 分析完成！耗时: ${duration}秒"
    echo "结果保存至: $OUTPUT_PNG"
else
    echo "------------------------------------------------"
    echo "❌ 错误：Python 绘图脚本执行失败，请确认 Python 脚本已更新为分段版本。"
    exit 1
fi
