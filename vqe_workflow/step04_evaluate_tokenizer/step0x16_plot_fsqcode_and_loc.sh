#!/bin/bash

# --- 配置 ---
CSV_INPUT="step0x15_fsqcode_and_loc.csv"
PNG_OUTPUT="step0x16_fsqcode_and_loc.png"
PLOT_SCRIPT="step0x16_plot_fsqcode_and_loc.py"
# 标记的点数 (建议 50-200 之间，取决于你的 UMAP 簇的紧密程度)
LABEL_SAMPLES=120

echo "========================================================="
echo "🎨 启动 ResidualFSQ 可视化标记任务"
echo "📅 时间: $(date)"
echo "---------------------------------------------------------"

# 1. 检查文件
if [ ! -f "$CSV_INPUT" ]; then
    echo "❌ 错误: 找不到 CSV 输入文件: $CSV_INPUT"
    echo "请先运行生成 CSV 的脚本。"
    exit 1
fi

if [ ! -f "$PLOT_SCRIPT" ]; then
    echo "❌ 错误: 找不到绘图脚本: $PLOT_SCRIPT"
    exit 1
fi

# 2. 执行绘图
echo "🚀 正在运行绘图脚本，采样 $LABEL_SAMPLES 个点进行标记..."
python3 "$PLOT_SCRIPT" \
    --input "$CSV_INPUT" \
    --output "$PNG_OUTPUT" \
    --num_samples $LABEL_SAMPLES

# 3. 结果检查
if [ $? -eq 0 ]; then
    echo "---------------------------------------------------------"
    echo "✅ 绘图成功！图片已保存至: $PNG_OUTPUT"
    ls -lh "$PNG_OUTPUT"
    echo "========================================================="
else
    echo "❌ 绘图失败，请检查 Python 错误日志。"
    exit 1
fi
