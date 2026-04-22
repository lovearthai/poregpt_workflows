#!/bin/bash

# =================================================================
# 脚本用途：2行3列矩阵可视化 (A, T, G, C, N + 叠加图)
# 使用说明：此脚本将调用 Python 生成包含 6 个子图的大图
# =================================================================

# --- 配置区 ---
BASE_CSV="step0x14_count_token_base_pairs_layer.csv"
LOC_CSV="step0x15_fsqcode_and_loc.csv"
OUTPUT="step0x18_plot_fsq_token_subplot.png"

# 采样与放大参数
TOP_PERCENT=0.2
POWER_VAL=1.5  # 增加到 2.2 以便在子图中更清晰地分辨大小

# --- 环境准备 ---
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "-------------------------------------------------------"
echo "📊 启动 2x3 矩阵分布绘图任务..."
echo "⚙️  配置: Top $TOP_PERCENT | Power $POWER_VAL"
echo "-------------------------------------------------------"

# --- 执行 ---
python3 step0x18_plot_fsq_token_dist.py \
    --base_csv "$BASE_CSV" \
    --loc_csv "$LOC_CSV" \
    --output "$OUTPUT" \
    --top "$TOP_PERCENT" \
    --power "$POWER_VAL"

if [ $? -eq 0 ]; then
    echo "-------------------------------------------------------"
    echo "✅ 绘图成功！请查看: $OUTPUT"
    ls -lh "$OUTPUT"
    echo "-------------------------------------------------------"
else
    echo "❌ 绘图失败，请检查脚本逻辑。"
    exit 1
fi
