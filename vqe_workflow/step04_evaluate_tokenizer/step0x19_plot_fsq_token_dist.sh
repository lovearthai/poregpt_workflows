#!/bin/bash

# =================================================================
# 脚本用途：生成纳米孔电信号 Token 的碱基频率分布图 (增强对比度版)
# 核心逻辑：将 FSQ 潜空间的坐标与碱基计数合并，通过圆点大小展示频率
# =================================================================

# --- 1. 路径与文件名配置 ---

# 输入：碱基统计文件 (由 step0x14 生成，包含 token_id, base, count)
BASE_CSV="step0x14_count_token_base_pairs_layer.csv"  # 注意使用正确的文件名，确保与前面步骤一致

# 输入：Token 坐标与编码文件 (由 step0x15 生成，包含 x, y 坐标)
LOC_CSV="step0x15_fsqcode_and_loc.csv" # 注意使用正确的文件名，确保与前面步骤一致

# 输出：最终生成的分布散点图
OUTPUT="step0x19_plot_fsq_token_dist.png"

# --- 2. 绘图参数配置 (关键) ---

# 筛选比例 (Top Ratio)：
# 仅对 count 排序前 10% 的记录进行着色。
# 如果画面太乱，请调小（如 0.05）；如果想看更多 token，请调大（如 0.2）。
TOP_PERCENT=0.2

# 放大系数 (Power)：解决“圆点看起来一样大”的核心参数。
# 原理：点面积 ∝ count^POWER_VAL。
# 1.0 为线性增长；1.5~2.5 会显著拉开高频（如 Poly-A）与低频 Token 的视觉差距。
POWER_VAL=1.5

# --- 3. 执行环境准备 ---

# 确保 Python 能找到当前目录下的自定义模块 (如 poregpt)
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "-------------------------------------------------------"
echo "🚀 开始绘制碱基分布图..."
echo "📊 采样比例: 前 $(echo "$TOP_PERCENT * 100" | bc)%"
echo "🔥 放大系数: $POWER_VAL (数值越大，高频点对比越强烈)"
echo "-------------------------------------------------------"

# --- 4. 运行 Python 绘图脚本 ---

# 调用 step0x19 脚本，传入配置参数
python3 step0x19_plot_fsq_token_dist.py \
    --base_csv "$BASE_CSV" \
    --loc_csv "$LOC_CSV" \
    --output "$OUTPUT" \
    --top "$TOP_PERCENT" \
    --power "$POWER_VAL"

# --- 5. 结果检查 ---

if [ $? -eq 0 ]; then
    echo "-------------------------------------------------------"
    echo "✅ 绘图成功: $OUTPUT"
    ls -lh "$OUTPUT"
    echo "-------------------------------------------------------"
else
    echo "❌ 绘图失败，请确认：1. step0x19 脚本支持 --power 参数；2. 环境中已安装 matplotlib/seaborn"
    exit 1
fi
