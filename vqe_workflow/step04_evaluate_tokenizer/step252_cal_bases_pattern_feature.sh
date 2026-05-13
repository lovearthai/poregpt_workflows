#!/bin/bash

# ==========================================
# 脚本名称: 碱基特征提取运行脚本 (Updated for Multi-Strategy K-mer)
# 功能: 一次性提取并输出 all, boundary, dynamic 三种复合特征
# ==========================================

# --- 1. 路径与文件名配置 ---

# 🔥 tokenizer 控制
TOKENIZER_NAME="vqe340s147000l1"
# 量化器数量定义
NUM_QUANTIZERS=1

# --- 2. 逻辑判断 (确定码本层级 LEVELS) ---
if [ "$TOKENIZER_NAME" == "vqe342s036000l1" ]; then
    LEVELS="11 11 11 11"
    echo "命中新版本 ($TOKENIZER_NAME)，LEVELS 设为: $LEVELS"
elif [ "$TOKENIZER_NAME" == "vqe340s147000l1" ]; then
    LEVELS="5 5 5 5"
    echo "命中新版本 ($TOKENIZER_NAME)，LEVELS 设为: $LEVELS"
else
    LEVELS=""
    echo "未知版本 ($TOKENIZER_NAME)，默认 LEVELS 设为空。"
fi

# --- 3. 输出执行参数 ---
echo "最终执行参数 LEVELS='$LEVELS'"


# 输入文件路径

# 输出目录
OUTPUT_DIR="step252_cal_bases_pattern_feature"
mkdir -p $OUTPUT_DIR

# --- 4. 核心模式与参数配置 ---

# [滑动窗口配置]
KMER_K=5               # 滑动窗口 K-mer 长度 (代替原来的 BLOCK_COUNT)

# [策略: boundary] 专用配置
# 对应 Python 中的 --boundary_num (首尾碱基各取几个 Token)
BOUNDARY_NUM=1

# [策略: dynamic] 专用配置
# 对应 Python 中的 --dynamic_top_n (每个 K-mer 提取活跃度最高的前 N 个 Token)
DYNAMIC_TOP_N=9

# [通用] 高斯加权配置
# 权重分布的标准差 sigma
WEIGHT_SIGMA=3

TOKEN_STRIDE=5

FILE_PATTERN="LB07"

INPUT_GZ="/mnt/zzbnew/poregpt/dnadata/movetable/signal_${FILE_PATTERN}.modified.reformed.shiftr4.mongoq30.${TOKENIZER_NAME}.aligned.jsonl.gz"

# --- 5. 动态生成输出文件名 ---
# 由于一次运行同时包含三种策略结果，统一输出到一个 CSV 文件中
OUTPUT_CSV="$OUTPUT_DIR/step252_${FILE_PATTERN}.modified.reformed.shiftr4.mongoq30.${TOKENIZER_NAME}_kmer${KMER_K}_bnum${BOUNDARY_NUM}_topn${DYNAMIC_TOP_N}_sigma${WEIGHT_SIGMA}.csv"


# --- 6. 执行提取脚本 ---

echo "------------------------------------------------"
echo "🚀 开始提取特征 (三合一多策略并行)"
echo "时间:       $(date)"
echo "输入文件:   $INPUT_GZ"
echo "输出文件:   $OUTPUT_CSV"
echo "K-mer 长度: $KMER_K"
echo "边界数:     $BOUNDARY_NUM"
echo "动态 TopN:  $DYNAMIC_TOP_N"
echo "高斯 Sigma: $WEIGHT_SIGMA"
echo "码本层级:   $LEVELS"
echo "------------------------------------------------"

python3 scripts/step252_cal_bases_pattern_feature.py \
    --input "$INPUT_GZ" \
    --output "$OUTPUT_CSV" \
    --levels $LEVELS \
    --token_stride $TOKEN_STRIDE \
    --num_quantizers $NUM_QUANTIZERS \
    --kmer_k $KMER_K \
    --boundary_num $BOUNDARY_NUM \
    --dynamic_top_n $DYNAMIC_TOP_N \
    --weight_sigma $WEIGHT_SIGMA

# --- 7. 检查运行状态 ---

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✅ 特征提取成功！"
    echo "结果（包含 feature_all, feature_dynamic, feature_boundary）保存在: $OUTPUT_CSV"
else
    echo "------------------------------------------------"
    echo "❌ 错误：特征提取失败，请检查 Python 脚本。 "
    exit 1
fi
