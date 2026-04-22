#!/bin/bash

# ==========================================
# 脚本名称: 4D特征空间全视角3D投影脚本
# 功能: 绘制 4列 (维度组合) x N行 (观察视角) 的 3D 散点矩阵
# ==========================================

# --- 1. 路径与文件名配置 ---

# 默认输入文件 (按顺序取最后定义的有效值，或通过命令行参数传入 $1)
# 建议通过命令传参: ./step257_plot_bases_pattern_kde3d.sh input.csv
INPUT_CSV="step252_cal_bases_pattern_feature/step252_strategy_all_block5_sigma3.csv"
if [ ! -z "$1" ]; then
    INPUT_CSV=$1
fi

# 输出目录
OUTPUT_DIR="step258_plot_bases_pattern_kde3d_pro"

# --- 2. 核心绘图参数配置 ---

# [新增/重点] 观察的角度行数 (NUM_ROWS)
# 作用说明：
# 1. 脚本会生成 N 行子图。
# 2. 每一行会自动旋转观察角度 (Elevation 和 Azimuth)。
# 3. 行数越多，越能全方位检查特征簇在 3D 空间中是否有重叠或被遮挡。
# 建议值: 3-5 行 (过大会导致图片极长且渲染缓慢)
NUM_ROWS=4

# 散点大小：3D 图中点太多建议调小 (5-15)，点少可以调大 (20-40)
POINT_SIZE=20

# 最大采样点数：3D 渲染非常消耗内存和计算，建议控制在 5000 以内
MAX_POINTS=3000

# 最大绘制类别总数 (包含强制显示的模式)
MAX_CATEGORIES=3

# --- 核心修改：必须包含的碱基模式 ---
# 配合 step260 脚本找到的距离最远的 K-mer 使用
REQUIRED_PATTERNS="CGTCA GATAG GCTAT"

# --- 3. 创建输出目录 ---
mkdir -p "$OUTPUT_DIR"

# --- 4. 生成带时间戳的输出文件名 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILE_TAG=$(basename "$INPUT_CSV" .csv)

# 动态生成文件名
OUTPUT_PNG="${OUTPUT_DIR}/3D_MultiView_${FILE_TAG}_row${NUM_ROWS}_cat${MAX_CATEGORIES}_${TIMESTAMP}.png"

# --- 5. 执行绘图脚本 ---

echo "------------------------------------------------"
echo "🚀 开始执行多视角 3D 可视化分析"
echo "时间:       $(date)"
echo "输入文件:   $INPUT_CSV"
echo "布局架构:   4列 (维度组合) x $NUM_ROWS行 (旋转视角)"
echo "最大点数:   $MAX_POINTS"
echo "最大类别:   $MAX_CATEGORIES"
echo "强制包含:   $REQUIRED_PATTERNS"
echo "------------------------------------------------"

# 统计运行耗时
start_time=$(date +%s)

# 注意：传递参数给新的 python 脚本，包含 --num_rows 和 --point_size
python3 step258_plot_bases_pattern_kde3d_pro.py \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_PNG" \
    --max_points "$MAX_POINTS" \
    --max_categories "$MAX_CATEGORIES" \
    --required_base_patterns "$REQUIRED_PATTERNS" \
    --num_rows "$NUM_ROWS" \
    --point_size "$POINT_SIZE"

# --- 6. 检查执行状态 ---

if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "------------------------------------------------"
    echo "✅ 绘图成功！总计渲染 $((NUM_ROWS * 4)) 副子图，耗时: ${duration}秒"
    echo "结果已保存至: $OUTPUT_PNG"
else
    echo "------------------------------------------------"
    echo "❌ 错误：绘图脚本执行失败。"
    exit 1
fi
