#!/bin/bash

# ================================================================= #
# PoreGPT Token 差异分布分析启动脚本 (并行版)
# ================================================================= #

# 1. 环境与路径配置
PYTHON_EXEC="python"
SCRIPT_PATH="scripts/step01_transform_deltatoken.py"

# 2. 数据输入输出配置
INPUT_DIR="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_150m_DNA595G_RSQ542_C625_CNN12_V340S147000/basecall"

# 定义输出目录和文件名
OUTPUT_DIR="step01_transform_deltatoken"
OUTPUT_FILENAME="step01_transform_deltatoken_$(date +%Y%m%d).csv"
OUTPUT_CSV="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

# 3. FSQ 与 并行参数配置
FSQ_LEVELS="5 5 5 5"
NUM_QUANTIZERS=1
PROCESSES=32  # <--- 新增：建议设置为 CPU 核心数的一半或 80%，根据内存情况调整

# ================================================================= #
# 执行逻辑
# ================================================================= #

echo "[$(date)] 正在准备环境..."

# 创建输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "创建输出目录: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "[$(date)] 开始并行分析任务..."
echo "输入目录: $INPUT_DIR"
echo "FSQ Levels: $FSQ_LEVELS"
echo "并发进程: $PROCESSES"
echo "输出结果: $OUTPUT_CSV"

# 运行 Python 脚本
$PYTHON_EXEC "$SCRIPT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --levels $FSQ_LEVELS \
    --num_quantizers $NUM_QUANTIZERS \
    --processes $PROCESSES

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "[$(date)] 任务成功完成！统计结果已保存至 $OUTPUT_CSV"

    # 快速预览结果
    echo "------------------------------------------------------"
    echo "结果预览 (Head -n 5):"
    head -n 5 "$OUTPUT_CSV"
    echo "------------------------------------------------------"
else
    echo "[$(date)] 错误: 分析任务失败，请检查 Python 报错信息。"
    exit 1
fi
