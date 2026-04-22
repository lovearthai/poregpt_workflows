#!/bin/bash

# ==========================================
# 脚本名称: 碱基特征可视化脚本 (v2)
# 功能: 调用 Python 脚本绘制 UMAP 聚类图 (支持指定 Pattern)
# ==========================================

# --- 1. 路径与文件名配置 ---

# 输入文件 (支持命令行第一个参数传入)
INPUT_CSV=${1:-"step252_cal_bases_pattern_feature/step252_strategy_boundary_bnum1_block5_sigma3.csv"}

# 输出目录
OUTPUT_DIR="step253_plot_bases_pattern"

# --- 2. 绘图参数配置 ---

# 聚类数量
N_CLUSTERS=100

# 最大采样点数
MAX_POINTS=100000

# 最大绘制类别数 (包含必须显示的类别后，随机补齐到此数量)
MAX_CATEGORIES=2

# --- [新增] 必须显示的类别列表 ---
# 多个类别请用空格分隔。如果不需要，留空即可：REQUIRED_BASE_PATTERNS=""
REQUIRED_BASE_PATTERNS="CGTCA GATAG"
REQUIRED_BASE_PATTERNS="CGTCA GATAG GCTAT TCTAC TAGAG GCTGT GATAT GTCGT"
REQUIRED_BASE_PATTERNS="CGTCA GATAG GCTAT TAGAG"
REQUIRED_BASE_PATTERNS="CGTCA GATAG GCTAT"

# --- 3. 创建输出目录 ---
mkdir -p "$OUTPUT_DIR"

# --- 4. 生成带时间戳的输出文件名 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILE_TAG=$(basename "$INPUT_CSV" .csv)

if [ $MAX_CATEGORIES -gt 0 ]; then
    OUTPUT_PNG="${OUTPUT_DIR}/vis_${FILE_TAG}_cat${MAX_CATEGORIES}_${TIMESTAMP}.png"
else
    OUTPUT_PNG="${OUTPUT_DIR}/vis_${FILE_TAG}_all_${TIMESTAMP}.png"
fi

# --- 5. 执行绘图脚本 ---

echo "------------------------------------------------"
echo "🚀 开始执行可视化分析 (UMAP + Clustering Context)"
echo "时间:       $(date)"
echo "输入文件:   $INPUT_CSV"
echo "输出图片:   $OUTPUT_PNG"
echo "最大点数:   $MAX_POINTS"

if [ $MAX_CATEGORIES -gt 0 ]; then
    echo "类别限制: 总计 $MAX_CATEGORIES 个类别"
    if [ -n "$REQUIRED_BASE_PATTERNS" ]; then
        echo "强制包含: $REQUIRED_BASE_PATTERNS"
    fi
else
    echo "类别限制: 绘制所有类别"
fi

echo "------------------------------------------------"

# 统计运行耗时
start_time=$(date +%s)

# 注意：$REQUIRED_BASE_PATTERNS 必须用双引号包裹传递，防止空格导致参数解析断开
python3 step253_plot_bases_pattern.py \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_PNG" \
    --max_points $MAX_POINTS \
    --max_categories $MAX_CATEGORIES \
    --required_base_patterns "$REQUIRED_BASE_PATTERNS"

# --- 6. 检查执行状态 ---

if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "------------------------------------------------"
    echo "✅ 绘图成功！耗时: ${duration}秒"
    echo "结果保存至: $OUTPUT_PNG"
else
    echo "------------------------------------------------"
    echo "❌ 错误：绘图脚本执行失败。"
    exit 1
fi
