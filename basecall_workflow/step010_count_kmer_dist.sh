#!/bin/bash

# --- 参数配置 ---
KMER_LEN=13
THREADS=$(nproc)
THREADS=32
# 示例2
INPUT_DIR="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_150m_DNA595G_RSQ542_C625_CNN12_V340S147000/basecall_kmer9_t9000_r0.3"
OUTPUT_FILE="step010_count_kmer_dist/basecall_kmer9_t9000_r0.3_kmer${KMER_LEN}_dist.csv"
# 示例1
INPUT_DIR="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_150m_DNA595G_RSQ542_C625_CNN12_V340S147000/basecall"
OUTPUT_FILE="step010_count_kmer_dist/basecall_kmer${KMER_LEN}_dist.csv"


# --- 关键步骤：创建输出目录 ---
# 使用 dirname 获取文件所在的文件夹路径
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")

echo "Checking output directory: $OUTPUT_DIR"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# --- 执行统计命令 ---
echo "Starting K-mer statistics (K=$KMER_LEN) with $THREADS threads..."

python3 scripts/step010_count_kmer_dist.py \
    --input_dir "$INPUT_DIR" \
    --kmer_len "$KMER_LEN" \
    --output_csv "$OUTPUT_FILE" \
    --workers "$THREADS"

if [ $? -eq 0 ]; then
    echo "Success! Results saved to: $OUTPUT_FILE"
else
    echo "Error: K-mer counting failed."
    exit 1
fi
