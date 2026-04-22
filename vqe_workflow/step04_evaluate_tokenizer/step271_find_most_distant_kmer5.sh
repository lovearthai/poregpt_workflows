#!/bin/bash

# ==========================================
# 脚本名称: 碱基区分度掩码环境搜索脚本 (Fast Version)
# 功能: 遍历所有 B1B2_X_B3B4 组合，计算碱基 X 在 4D 空间的聚类区分度
# ==========================================

# --- 1. 路径与文件名配置 ---

# 默认输入文件
INPUT_CSV="step252_cal_bases_pattern_feature/step252_strategy_boundary_bnum1_block5_sigma3.csv"
if [ ! -z "$1" ]; then
    INPUT_CSV=$1
fi

# 输出目录 (用于保存可能的日志或结果备份)
OUTPUT_DIR="step271_find_most_distant_kmer5"
mkdir -p "$OUTPUT_DIR"

# --- 2. 核心参数配置 ---

# 是否开启对称模式 (如只看 AAXAA)
# 开启则测试 16 种组合，关闭则测试 256 种组合
SYMMETRIC_MODE=false

# 每个 K-mer 最少需要的样本数 (样本太少会导致均值不稳定，产生虚假区分度)
MIN_SAMPLES=3

# 显示前 N 个最具有区分度的模式
TOP_N=20

# --- 3. 执行搜索脚本 ---

echo "------------------------------------------------"
echo "🚀 开始搜索最具区分度的碱基掩码环境"
echo "时间:       $(date)"
echo "输入文件:   $INPUT_CSV"
echo "对称搜索:   $SYMMETRIC_MODE"
echo "样本阈值:   $MIN_SAMPLES"
echo "------------------------------------------------"

# 统计运行耗时
start_time=$(date +%s)

# 构建基础命令
CMD="python3 step271_find_most_distant_kmer5.py -i $INPUT_CSV --min_samples $MIN_SAMPLES --top_n $TOP_N"

# 根据配置添加对称参数
if [ "$SYMMETRIC_MODE" = true ] ; then
    CMD="$CMD --symmetric"
fi

# 执行并实时打印进度条
eval $CMD

# --- 4. 检查执行状态 ---

if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "------------------------------------------------"
    echo "✅ 搜索完成！总计耗时: ${duration}秒"
    echo "提示: 请关注上方打印的 Total Dist 排名，数值越高区分度越强。"
else
    echo "------------------------------------------------"
    echo "❌ 错误：Python 脚本执行失败。"
    echo "请检查是否安装了 tqdm: pip install tqdm"
    exit 1
fi
