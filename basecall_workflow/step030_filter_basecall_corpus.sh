#!/bin/bash

MODEL_NAME="HF_150m_DNA595G_RSQ542_C625_CNN12_V340S147000"

# --- 参数设置 ---
# 你的统计结果 CSV
STATS_CSV="step010_count_kmer_dist/basecall_kmer9_dist.csv"
# 原始 jsonl.gz 路径
INPUT_GLOB="/mnt/zzbnew/rnamodel/model/signalDNAmodel/${MODEL_NAME}/basecall/*.jsonl.gz"
K_VAL=9
# 每个 kmer 期望收集的数量。建议：对于 9-mer，5000-10000 是个不错的平衡点
TARGET=9000 
LEAST_REPEAT_N=1000
RATIO=0.3
THREADS=$(nproc) # 使用所有核心
THREADS=32
OUT_DIR="/mnt/zzbnew/rnamodel/model/signalDNAmodel/${MODEL_NAME}/basecall_kmer${K_VAL}_t${TARGET}_r${RATIO}_least${LEAST_REPEAT_N}"
# 1. 创建目录
mkdir -p "$OUT_DIR"

echo "------------------------------------------------"
echo "Running Importance Sampling based on Pre-stats"
echo "Target: $TARGET per K-mer"
echo "------------------------------------------------"

# 2. 运行 Python 筛选
# 我们直接利用现有的统计数据，结合流式实时计数进行“削峰填谷”
time python3 scripts/step030_filter_basecall_corpus.py \
    --input "$INPUT_GLOB" \
    --csv "$STATS_CSV" \
    --k $K_VAL \
    --target $TARGET \
    --output "$OUT_DIR" \
    --ratio $RATIO \
    --least_repeat_n $LEAST_REPEAT_N \
    --threads $THREADS

if [ $? -eq 0 ]; then
    echo "Success. Balanced dir: $OUT_DIR"
    ls -lh "$OUT_DIR"
else
    echo "Error: Sampling failed."
    exit 1
fi
