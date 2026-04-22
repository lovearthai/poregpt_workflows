#!/bin/bash

# --- 配置区 ---
# FSQ 的能级配置 (levels)
LEVELS="5 5 5 5"
# 残差层数 (num_q)
NUM_Q=1
# 输出文件名
OUTPUT_FILE="step0x15_transform_fsqtoken_to_vector.csv"
# Python 脚本名 (确保与你保存的 Python 文件名一致)
PY_SCRIPT="step0x15_transform_fsqtoken_to_vector.py"

echo "========================================================="
echo "🚀 开始执行 ResidualFSQ 全量 Token 映射任务"
echo "📅 时间: $(date)"
echo "⚙️ 配置: Levels=[$LEVELS], Num_Q=$NUM_Q"
echo "========================================================="

# 1. 环境检查
if [ ! -f "$PY_SCRIPT" ]; then
    echo "❌ 错误: 找不到 Python 脚本 $PY_SCRIPT"
    exit 1
fi

# 2. 确保 PYTHONPATH 包含当前目录，以便导入 poregpt 模块
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. 执行 Python 任务
# 使用 time 命令记录耗时
time python3 "$PY_SCRIPT" \
    --levels $LEVELS \
    --num_q $NUM_Q \
    --output "$OUTPUT_FILE"

# 4. 检查执行结果
if [ $? -eq 0 ]; then
    echo "========================================================="
    echo "✅ 任务成功完成！"
    echo "💾 结果保存在: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
    echo "========================================================="
else
    echo "❌ 任务执行过程中出现错误，请检查日志。"
    exit 1
fi
