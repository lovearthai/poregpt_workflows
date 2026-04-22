#!/bin/bash

# =================================================================
# 脚本名称: step280_cal_bases_pattern_feature_by_olmo.sh
# 描述: 自动化运行 OLMo 特征提取脚本 (恢复 Sigma 命名)
# =================================================================

# --- 1. 路径与文件名配置 ---

# 输入文件路径
INPUT_GZ="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"
# 模型路径
MODEL_PATH="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_150m_DNA595G_RSQ542_C625_CNN12_V340S147000/base"

# 输出目录
OUTPUT_DIR="step280_cal_bases_pattern_feature_by_olmo"

# --- 2. 核心模式配置 ---

# 词表偏移量
VOCAB_CODE_SHIFT=128

# 策略选择: all | boundary | dynamic
SELECT_STRATEGY="boundary"
SELECT_STRATEGY="all"
SELECT_STRATEGY="dynamic"

# [策略: boundary] 专用配置
BOUNDARY_NUM=1

# [策略: dynamic] 专用配置
DYNAMIC_TOP_N=10

# [通用] 滑动窗口配置
BLOCK_COUNT=5           # 连续块的数量
MIN_REPEAT=2            # 每个碱基块的最小重复次数

# [通用] 高斯加权配置
WEIGHT_SIGMA=3          # 权重分布 Sigma


# --- 4. 动态生成输出文件名 ---
mkdir -p $OUTPUT_DIR

# 恢复原始命名逻辑，确保包含 sigma
if [ "$SELECT_STRATEGY" == "boundary" ]; then
    MODE_DESC="边界提取 (Num=$BOUNDARY_NUM)"
    OUTPUT_CSV="$OUTPUT_DIR/step280_strategy_${SELECT_STRATEGY}_bnum${BOUNDARY_NUM}_block${BLOCK_COUNT}_sigma${WEIGHT_SIGMA}.csv"
elif [ "$SELECT_STRATEGY" == "dynamic" ]; then
    MODE_DESC="动态活跃度 (TopN=$DYNAMIC_TOP_N)"
    OUTPUT_CSV="$OUTPUT_DIR/step280_strategy_${SELECT_STRATEGY}_topn${DYNAMIC_TOP_N}_block${BLOCK_COUNT}_sigma${WEIGHT_SIGMA}.csv"
else
    MODE_DESC="全量提取 (All)"
    OUTPUT_CSV="$OUTPUT_DIR/step280_strategy_${SELECT_STRATEGY}_block${BLOCK_COUNT}_sigma${WEIGHT_SIGMA}.csv"
fi

# --- 5. 检查路径是否存在 ---
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在 -> $MODEL_PATH"
    exit 1
fi

if [ ! -f "$INPUT_GZ" ]; then
    echo "❌ 错误: 输入文件不存在 -> $INPUT_GZ"
    exit 1
fi

# --- 6. 执行 Python 脚本 ---

echo "------------------------------------------------"
echo "🚀 开始 OLMo 特征提取任务"
echo "时间:        $(date)"
echo "模型路径:    $MODEL_PATH"
echo "输入文件:    $INPUT_GZ"
echo "输出文件:    $OUTPUT_CSV"
echo "运行策略:    $MODE_DESC (Sigma: $WEIGHT_SIGMA)"
echo "------------------------------------------------"

python step280_cal_bases_pattern_feature_by_olmo.py \
    --input "$INPUT_GZ" \
    --output "$OUTPUT_CSV" \
    --model_path "$MODEL_PATH" \
    --vocab_code_shift $VOCAB_CODE_SHIFT \
    --block_count $BLOCK_COUNT \
    --min_repeat $MIN_REPEAT \
    --select_token_strategy "$SELECT_STRATEGY" \
    --boundary_num $BOUNDARY_NUM \
    --dynamic_top_n $DYNAMIC_TOP_N \
    --weight_sigma $WEIGHT_SIGMA

# --- 7. 检查执行结果 ---

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✅ 任务完成成功！"
    echo "结果保存在: $OUTPUT_CSV"
else
    echo "------------------------------------------------"
    echo "❌ 错误：任务运行出错，请检查日志。"
    exit 1
fi
