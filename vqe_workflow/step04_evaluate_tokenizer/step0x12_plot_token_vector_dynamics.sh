#!/bin/bash

# ==========================================
# 脚本名称: 隐空间动力学可视化运行脚本
# 功能: 配置滑动窗口参数并调用 Python 绘图工具
# ==========================================

# --- 1. 路径配置 ---
# 默认输入文件 (根据你的项目结构调整)
INPUT_GZ=${1:-"/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"}
OUTPUT_DIR="step0x12_plot_token_vector_dynamics"
mkdir -p $OUTPUT_DIR

# --- 2. 核心分析参数 (专业术语对齐) ---

# 窗口大小 (Window Size): 每次计算包含的 Token 数量
# 较小的值 (2-5) 对局部突变更敏感；较大的值 (10+) 趋势更平稳
WIN_SIZE=1

# 窗口步长 (Window Stride): 窗口滑动的跨度
# 设为 1 可以获得最高分辨率的平滑曲线
WIN_STRIDE=1

# 绘图范围 (Sample Range): 原始电信号的采样点起止索引
PLOT_START=800
PLOT_END=1200

# --- 3. 生成输出文件名 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TAG="sz${WIN_SIZE}_st${WIN_STRIDE}"
OUTPUT_PNG="${OUTPUT_DIR}/dynamics_alignment_${TAG}_${TIMESTAMP}.png"

# --- 4. 执行分析与绘图 ---

echo "------------------------------------------------"
echo "🚀 开始执行隐空间动力学联合分析"
echo "时间:       $(date)"
echo "输入文件:   $INPUT_GZ"
echo "窗口配置:   Size=$WIN_SIZE, Stride=$WIN_STRIDE"
echo "显示范围:   $PLOT_START 到 $PLOT_END"
echo "------------------------------------------------"

# 统计执行耗时
start_time=$(date +%s)

python3 step0x12_plot_token_vector_dynamics.py \
    --input-file "$INPUT_GZ" \
    --output-file "$OUTPUT_PNG" \
    --range $PLOT_START $PLOT_END \
    --window-size $WIN_SIZE \
    --window-stride $WIN_STRIDE

# --- 5. 检查结果 ---

if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "------------------------------------------------"
    echo "✅ 分析完成！耗时: ${duration}秒"
    echo "结果保存至: $OUTPUT_PNG"
else
    echo "------------------------------------------------"
    echo "❌ 错误：Python 绘图脚本执行失败，请检查输入文件格式。"
    exit 1
fi
