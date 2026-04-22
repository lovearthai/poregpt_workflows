#!/bin/bash

# 1. 脚本参数配置
# 请确保 Python 环境中已安装 pandas, numpy, tqdm 和你的 poregpt 包
INPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"
OUTPUT_FILE="step250_cal_base_patten_feature_mp13.csv"
MIN_REPEAT=13
LEVELS="5 5 5 5"

# 2. 执行前检查
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: 输入文件不存在 -> $INPUT_FILE"
    exit 1
fi

# 3. 执行 Python 脚本
echo "------------------------------------------------"
echo "开始处理纳米孔数据 (当前系统 Python)"
echo "时间:   $(date)"
echo "输入:   $INPUT_FILE"
echo "输出:   $OUTPUT_FILE"
echo "参数:   min_repeat=$MIN_REPEAT, levels=$LEVELS"
echo "------------------------------------------------"

# 使用 python3 确保调用正确，也可以直接改回 python
python3 step250_cal_base_patten_feature.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --min_repeat $MIN_REPEAT \
    --levels $LEVELS

# 4. 检查执行结果
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "处理成功完成！"
    echo "结果已保存至: $OUTPUT_FILE"
else
    echo "------------------------------------------------"
    echo "处理过程中出现错误，请检查日志。"
fi
