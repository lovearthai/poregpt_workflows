#!/bin/bash

# --- 参数配置 ---
K_VAL=11
INPUT_CSV="step010_count_kmer_dist/kmer${K_VAL}_dist.csv"
INPUT_CSV="step010_count_kmer_dist/basecall_kmer9_t9000_r0.3_kmer9_dist.csv"
PARENT_DIR="step020_plot_kmer_dist"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINAL_OUT_DIR="${PARENT_DIR}/run_${TIMESTAMP}"

echo "------------------------------------------------"
echo "Step: Plotting K-mer Frequency"
echo "------------------------------------------------"

# 1. 在脚本中创建固定父目录
echo "Creating parent directory: $PARENT_DIR"
mkdir -p "$PARENT_DIR"

# 3. 运行 Python，传入带时间戳的完整路径
# Python 代码内部的 os.makedirs 会创建 run_${TIMESTAMP} 子目录
python3 scripts/step020_plot_kmer_dist.py \
    --input "$INPUT_CSV" \
    --k $K_VAL \
    --output_dir "$FINAL_OUT_DIR" \
    #--plot_spectrum \

# 4. 检查结果
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "SUCCESS!"
    echo "Report and Plot saved in: $FINAL_OUT_DIR"
    echo "------------------------------------------------"
else
    echo "ERROR: Analysis failed."
    exit 1
fi
