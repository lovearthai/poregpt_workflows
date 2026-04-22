#!/bin/bash

# ==========================================
# 脚本名称: 碱基特征提取运行脚本 (Updated for Strategy Selection)
# 功能: 支持 all, boundary, dynamic 三种策略
# ==========================================

# --- 1. 路径与文件名配置 ---

# 输入文件路径
INPUT_GZ="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"

# 输出目录
OUTPUT_DIR="step252_cal_bases_pattern_feature"

# --- 2. 核心模式配置 ---

# 策略选择: all | boundary | dynamic
SELECT_STRATEGY="boundary"
SELECT_STRATEGY="all"
SELECT_STRATEGY="dynamic"

# [策略: boundary] 专用配置
# 对应 Python 中的 --boundary_num
# 1 代表每个 block 首尾各取 1 个，2 代表各取 2 个...
BOUNDARY_NUM=1

# [策略: dynamic] 专用配置
# 对应 Python 中的 --dynamic_top_n
DYNAMIC_TOP_N=9

# [通用] 滑动窗口配置
BLOCK_COUNT=5          # 连续块的数量 (例如 5 代表 A...AT...TC...CG...G)
MIN_REPEAT=2           # 每个碱基块的最小重复次数

# [通用] 高斯加权配置
# 权重分布参考示例 (若 sigma = 2.0): 距离中心 0 步权重 1.0, 2 步 0.6, 4 步 0.135
WEIGHT_SIGMA=3

# --- 3. 码本相关参数 ---
LEVELS="5 5 5 5"
NUM_QUANTIZERS=1

# --- 4. 动态生成输出文件名 ---
mkdir -p $OUTPUT_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 根据策略生成不同的文件名后缀
if [ "$SELECT_STRATEGY" == "boundary" ]; then
    MODE_DESC="边界提取 (Num=$BOUNDARY_NUM)"
    OUTPUT_CSV="$OUTPUT_DIR/step252_strategy_${SELECT_STRATEGY}_bnum${BOUNDARY_NUM}_block${BLOCK_COUNT}_sigma${WEIGHT_SIGMA}.csv"
elif [ "$SELECT_STRATEGY" == "dynamic" ]; then
    MODE_DESC="动态活跃度 (TopN=$DYNAMIC_TOP_N)"
    OUTPUT_CSV="$OUTPUT_DIR/step252_strategy_${SELECT_STRATEGY}_topn${DYNAMIC_TOP_N}_block${BLOCK_COUNT}_sigma${WEIGHT_SIGMA}.csv"
else
    MODE_DESC="全量提取 (All)"
    OUTPUT_CSV="$OUTPUT_DIR/step252_strategy_${SELECT_STRATEGY}_block${BLOCK_COUNT}_sigma${WEIGHT_SIGMA}.csv"
fi

# --- 5. 执行提取脚本 ---

echo "------------------------------------------------"
echo "🚀 开始提取特征"
echo "时间:       $(date)"
echo "输入文件:   $INPUT_GZ"
echo "输出文件:   $OUTPUT_CSV"
echo "运行策略:   $MODE_DESC"
echo "码本层级:   $LEVELS"
echo "------------------------------------------------"

python3 step252_cal_bases_pattern_feature.py \
    --input "$INPUT_GZ" \
    --output "$OUTPUT_CSV" \
    --levels $LEVELS \
    --num_quantizers $NUM_QUANTIZERS \
    --block_count $BLOCK_COUNT \
    --min_repeat $MIN_REPEAT \
    --select_token_strategy $SELECT_STRATEGY \
    --boundary_num $BOUNDARY_NUM \
    --dynamic_top_n $DYNAMIC_TOP_N \
    --weight_sigma $WEIGHT_SIGMA

# --- 6. 检查状态 ---

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✅ 特征提取成功！"
    echo "结果保存在: $OUTPUT_CSV"
else
    echo "------------------------------------------------"
    echo "❌ 错误：特征提取失败，请检查 Python 脚本。 "
    exit 1
fi
