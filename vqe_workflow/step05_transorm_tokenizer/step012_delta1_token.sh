#!/bin/bash

# ================= 配置区 =================
# 输入目录 (支持递归查找子目录下的 jsonl.gz)
INPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256"

# 输出目录 (修改了后缀名以区分旧的 repeated 压缩)
OUTPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256_delta"

# 原始量化级别 (用空格分隔，对应代码中的 --levels)
LEVELS="5 5 5 5"

# 并行工作进程数 (建议设置为 CPU 核心数，0 表示自动检测)
NUM_WORKERS=32

# 日志输出文件
LOG_FILE="logs/step012_delta1_token.log"

# Python 脚本路径 (请确保该路径与你保存新 Python 代码的文件名一致)
PYTHON_SCRIPT="scripts/step012_delta1_token.py"
# ==========================================

# 自动创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: 找不到 Python 脚本 $PYTHON_SCRIPT"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 错误: 找不到输入目录 $INPUT_DIR"
    exit 1
fi

echo "=========================================="
echo "🚀 开始执行 Token 差分编码 (Delta) 任务"
echo "输入目录 : $INPUT_DIR"
echo "输出目录 : $OUTPUT_DIR"
echo "Levels   : $LEVELS"
echo "进程数   : $NUM_WORKERS"
echo "日志文件 : $LOG_FILE"
echo "=========================================="

# 执行 Python 脚本 (转入后台运行)
# 注意：$LEVELS 没有加引号，Shell 会自动将其解包为 4 个参数传给 --levels
python3 "$PYTHON_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --levels $LEVELS \
    --num_workers "$NUM_WORKERS" \
    > "$LOG_FILE" 2>&1 &

# 获取后台任务的 PID
PID=$!

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 任务已成功提交至后台运行！"
    echo "进程 PID : $PID"
    echo "运行日志 : $LOG_FILE"
    echo "提示     : 你可以使用 'tail -f $LOG_FILE' 实时查看流式处理进度和人工检查样例。"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ 任务提交后台失败，请检查脚本配置。"
    echo "=========================================="
fi
