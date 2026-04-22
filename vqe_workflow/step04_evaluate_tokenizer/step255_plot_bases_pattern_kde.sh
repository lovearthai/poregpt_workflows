#!/bin/bash

# ==========================================
# 脚本名称: 碱基特征可视化脚本 (支持固定类别)
# 功能: 调用 Python 脚本绘制维度两两组合的散点图
# ==========================================

# --- 1. 路径与文件名配置 ---

# 输入文件 (支持命令行参数传入)
INPUT_CSV=${1:-"step252_cal_bases_pattern_feature/step252_boundary1.csv"}
INPUT_CSV=${1:-"step252_cal_bases_pattern_feature/step252_block5_repeat2.csv"}
INPUT_CSV=${1:-"step252_cal_bases_pattern_feature/step252_boundary1_block5_repeat2.csv"}
INPUT_CSV=${1:-"step252_cal_bases_pattern_feature/step252_boundary0_block5_repeat2.csv"}
INPUT_CSV=${1:-"step252_cal_bases_pattern_feature/step252_strategy_dynamic_topn5_block7_sigma3.csv"}

# 输出目录
OUTPUT_DIR="step255_plot_bases_pattern_kde"

# --- 2. 绘图参数配置 ---

# 最大采样点数
MAX_POINTS=100000

# 最大绘制类别总数
MAX_CATEGORIES=2

# --- 核心修改：必须包含的碱基模式 (用空格分隔) ---
# 脚本会优先保证这些模式出现在图中，然后再随机抽取其他模式补齐到 MAX_CATEGORIES
REQUIRED_PATTERNS="GAGAG AGAGA CTCTC TCTCT "
REQUIRED_PATTERNS="GAGAG AGAGA CTCTC"
REQUIRED_PATTERNS="AGAGA CTCTC"
REQUIRED_PATTERNS="CTCTC AGAGA"
REQUIRED_PATTERNS="CGTCA GATAG GCTAT"

# --- 3. 创建输出目录 ---
mkdir -p $OUTPUT_DIR

# --- 4. 生成带时间戳的输出文件名 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILE_TAG=$(basename "$INPUT_CSV" .csv)

# 文件名标识包含类别数限制
if [ "$MAX_CATEGORIES" != "" ] && [ "$MAX_CATEGORIES" -gt 0 ]; then
    OUTPUT_PNG="${OUTPUT_DIR}/vis_${FILE_TAG}_cat${MAX_CATEGORIES}_${TIMESTAMP}.png"
else
    OUTPUT_PNG="${OUTPUT_DIR}/vis_${FILE_TAG}_all_${TIMESTAMP}.png"
fi

# --- 5. 执行绘图脚本 ---

echo "------------------------------------------------"
echo "🚀 开始执行可视化分析 (维度网格图)"
echo "时间:       $(date)"
echo "输入文件:   $INPUT_CSV"
echo "输出图片:   $OUTPUT_PNG"
echo "最大点数:   $MAX_POINTS"

if [ "$MAX_CATEGORIES" != "" ] && [ "$MAX_CATEGORIES" -gt 0 ]; then
    echo "类别限制: 总计 $MAX_CATEGORIES 个类别"
    if [ "$REQUIRED_PATTERNS" != "" ]; then
        echo "强制包含: $REQUIRED_PATTERNS"
    fi
else
    echo "类别限制: 绘制所有类别"
fi
echo "------------------------------------------------"

# 统计运行耗时
start_time=$(date +%s)

# 注意：双引号引用变量以处理空格
python3 step255_plot_bases_pattern_kde.py \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_PNG" \
    --max_points $MAX_POINTS \
    --max_categories "$MAX_CATEGORIES" \
    --required_base_patterns "$REQUIRED_PATTERNS"

# --- 6. 检查执行状态 ---

if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "------------------------------------------------"
    echo "✅ 绘图成功！耗时: ${duration}秒"
    echo "结果已保存至: $OUTPUT_PNG"
else
    echo "------------------------------------------------"
    echo "❌ 错误：绘图脚本执行失败。"
    exit 1
fi
