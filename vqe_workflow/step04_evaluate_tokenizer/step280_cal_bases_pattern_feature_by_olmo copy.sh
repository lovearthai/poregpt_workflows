#!/bin/bash

# =================================================================
# 脚本名称: step280_cal_bases_pattern_feature_by_olmo.sh
# 描述: 自动化运行 OLMo 特征提取脚本 (恢复 Sigma 命名)
# =================================================================

# --- 1. 路径与文件名配置 ---
# 🔥 tokenizer 控制
TOKENIZER_NAME="vqe342s036000l1"

# 输入文件路径
INPUT_GZ="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.${TOKENIZER_NAME}.aligned.jsonl.gz"
# 模型路径
# MODEL_PATH="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_150m_DNA595G_RSQ542_C625_CNN12_V340S147000/base"
MODEL_PATH="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_300m_DNA595G_RSQ1142_C16k_CNN12_V342S036000L1/base"

# 输出目录
OUTPUT_DIR="step280_cal_bases_pattern_feature_by_olmo"

# --- 2. 核心模式配置 ---

# 词表偏移量
VOCAB_CODE_SHIFT=128

# [K-mer 滑动窗口配置] (改为 K-mer 方式)
KMER_K=5                # K-mer 长度
TOKEN_STRIDE=5          # Token stride 下采样步长

# [策略: boundary] 专用配置
BOUNDARY_NUM=1

# [策略: dynamic] 专用配置
DYNAMIC_TOP_N=9

# [通用] 高斯加权配置
WEIGHT_SIGMA=3          # 权重分布 Sigma


# --- 4. 动态生成输出文件名 ---
mkdir -p $OUTPUT_DIR

# 改为基于 K-mer 的命名方式（并行三种策略）
OUTPUT_CSV="$OUTPUT_DIR/step280_kmer${KMER_K}_stride${TOKEN_STRIDE}_bnum${BOUNDARY_NUM}_topn${DYNAMIC_TOP_N}_sigma${WEIGHT_SIGMA}.csv"

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
echo "🚀 开始 OLMo 特征提取任务 (K-mer 方式并行三策略)"
echo "时间:        $(date)"
echo "模型路径:    $MODEL_PATH"
echo "输入文件:    $INPUT_GZ"
echo "输出文件:    $OUTPUT_CSV"
echo "K-mer 长度:  $KMER_K"
echo "Token stride: $TOKEN_STRIDE"
echo "边界数:      $BOUNDARY_NUM"
echo "动态 TopN:   $DYNAMIC_TOP_N"
echo "高斯 Sigma:  $WEIGHT_SIGMA"
echo "------------------------------------------------"

python scripts/step280_cal_bases_pattern_feature_by_olmo_v1.py \
    --input "$INPUT_GZ" \
    --output "$OUTPUT_CSV" \
    --model_path "$MODEL_PATH" \
    --vocab_code_shift $VOCAB_CODE_SHIFT \
    --kmer_k $KMER_K \
    --token_stride $TOKEN_STRIDE \
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
