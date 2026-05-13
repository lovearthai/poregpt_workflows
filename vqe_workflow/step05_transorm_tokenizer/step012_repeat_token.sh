#!/bin/bash

# ================= 配置区 =================
# 输入目录 (支持递归查找子目录下的 jsonl.gz)
INPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example"

# 输出目录
OUTPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256_repeated/example"

# 量化级别 (用空格分隔，对应代码中的 --levels)
LEVELS="5 5 5 5"

# 并行工作进程数 (建议设置为 CPU 核心数，0 表示自动检测)
NUM_WORKERS=1

# Python 脚本路径
PYTHON_SCRIPT="scripts/step012_repeat_token.py"
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
echo "🚀 开始执行 Token 展开任务"
echo "输入目录 : $INPUT_DIR"
echo "输出目录 : $OUTPUT_DIR"
echo "Levels   : $LEVELS"
echo "进程数   : $NUM_WORKERS"
echo "=========================================="

# 执行 Python 脚本
# 注意：$LEVELS 变量没有加引号，这样 Shell 会把 "5 5 5 5" 解析为 4 个独立的参数传给 --levels
python3 "$PYTHON_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --levels $LEVELS \
    --num_workers "$NUM_WORKERS"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 任务全部执行成功！"
    echo "结果保存在: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ 任务执行出错，请检查上方日志。"
    echo "=========================================="
fi
