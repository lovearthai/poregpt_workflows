#!/bin/bash

# 1. 路径与文件名配置
INPUT_CSV="step250_cal_base_patten_feature_mp13.csv"
OUTPUT_DIR="step251_plot_base_pattern"
METHOD="umap"  # 可选: pca, tsne, umap
N_CLUSTERS=4
MAX_POINTS=100000

# 2. 创建输出目录
mkdir -p $OUTPUT_DIR

# 3. 定义输出文件名（包含时分秒以防覆盖）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PNG="${OUTPUT_DIR}/vis_${METHOD}_k${N_CLUSTERS}_${TIMESTAMP}.png"

# 4. 执行绘图脚本
echo "------------------------------------------------"
echo "开始生成特征分布图..."
echo "时间:     $(date)"
echo "输入:     $INPUT_CSV"
echo "输出:     $OUTPUT_PNG"
echo "聚类数:   $N_CLUSTERS"
echo "------------------------------------------------"

python3 step251_plot_base_pattern.py \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_PNG" \
    --n_clusters "$N_CLUSTERS" \
    --max_points "$MAX_POINTS"

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "绘图成功！"
else
    echo "------------------------------------------------"
    echo "绘图失败。"
    exit 1
fi
