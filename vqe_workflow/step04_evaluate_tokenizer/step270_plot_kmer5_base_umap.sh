#!/bin/bash

# ==========================================
# 脚本名称: 4D特征空间降维聚类分析脚本 (t-SNE/PCA)
# 功能: 匹配 CGXGA 等通配符模式，提取变量 X 并绘制 2D 聚类图
# ==========================================

# --- 1. 路径与文件名配置 ---

# 输入 CSV 文件路径
INPUT_CSV="step252_cal_bases_pattern_feature/step252_boundary0_block7_repeat2_sigma3.csv"
INPUT_CSV="step252_cal_bases_pattern_feature/step252_strategy_dynamic_topn5_block5_sigma3.csv"
INPUT_CSV="step252_cal_bases_pattern_feature/step252_boundary0_block5_repeat2_sigma3.csv"
INPUT_CSV="step252_cal_bases_pattern_feature/step252_boundary0_block7_repeat2_sigma3.csv"
INPUT_CSV="step252_cal_bases_pattern_feature/step252_strategy_boundary_bnum1_block7_sigma3.csv"
INPUT_CSV="step252_cal_bases_pattern_feature/step252_strategy_all_block5_sigma3.csv"
INPUT_CSV="step252_cal_bases_pattern_feature/step252_strategy_boundary_bnum1_block5_sigma3.csv"

if [ ! -z "$1" ]; then
    INPUT_CSV=$1
fi

# 输出目录
OUTPUT_DIR="step270_plot_kmer5_base_umap"

# --- 2. 核心绘图参数配置 ---

# 匹配模式：X 代表任意碱基字符 (如 CGXGA 会匹配 CGAGA, CGTGA 等)
CATO_PATTERN="ACXCA"
CATO_PATTERN="TGXCA"
CATO_PATTERN="GAXCA"
CATO_PATTERN="GCAXTCA"
CATO_PATTERN="TCGXGAC"
CATO_PATTERN="CGTXAGA"
CATO_PATTERN="GTXAG"
CATO_PATTERN="GCXCT"
CATO_PATTERN="CGCXCTA"
CATO_PATTERN="TCXGT"
CATO_PATTERN="TGXCA"
CATO_PATTERN="ACXTA"
CATO_PATTERN="TCXCG"
CATO_PATTERN="GTCXCGA"
CATO_PATTERN="ATXGA"
CATO_PATTERN="AGXAC"

# 降维方法：可选 tsne 或 pca
# tsne: 擅长展示局部簇（适合看 K-mer 区分度）
# pca: 线性投影（适合看全局分布）
METHOD="tsne"

# 输出图片标签（用于区分不同的实验参数）
TAG="wildcard_clustering"

# --- 3. 创建输出目录 ---
mkdir -p "$OUTPUT_DIR"

# --- 4. 生成带时间戳的输出文件名 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILE_TAG=$(basename "$INPUT_CSV" .csv)

# 动态生成文件名，包含模式和降维方法
OUTPUT_PNG="${OUTPUT_DIR}/Clustering_${FILE_TAG}_${CATO_PATTERN}_${METHOD}_${TIMESTAMP}.png"

# --- 5. 执行绘图脚本 ---

echo "------------------------------------------------"
echo "🚀 开始执行分类降维聚类分析"
echo "时间:       $(date)"
echo "输入文件:   $INPUT_CSV"
echo "匹配模式:   $CATO_PATTERN"
echo "降维方法:   $METHOD"
echo "------------------------------------------------"

# 统计运行耗时
start_time=$(date +%s)

# 调用 Python 脚本
# 注意：请确保 plot_category_clustering.py 文件在当前目录或 Python 路径中
python3 step270_plot_kmer5_base_umap.py \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_PNG" \
    --cato_pattern "$CATO_PATTERN" \
    --method "$METHOD"

# --- 6. 检查执行状态 ---

if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "------------------------------------------------"
    echo "✅ 绘图成功！降维分析耗时: ${duration}秒"
    echo "结果保存至: $OUTPUT_PNG"
else
    echo "------------------------------------------------"
    echo "❌ 错误：Python 绘图脚本执行失败。"
    echo "请检查是否安装了必要的库 (pandas, matplotlib, scikit-learn)。"
    exit 1
fi
