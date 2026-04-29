#!/bin/bash

# ================================================================= #
# PoreGPT 相关性分析绘图启动脚本 (支持限制维度)
# ================================================================= #

# 1. 环境配置
PYTHON_EXEC="python"
PLOT_SCRIPT="scripts/step021_plot_deltatoken_distribution.py"
INPUT_DATA_DIR="step01_transform_deltatoken"
OUTPUT_DIR="step021_plot_deltatoken_distribution"

# 2. 自动定位最新的 CSV 文件
# 这里的路径修改为 step01 的输出目录进行搜索
LATEST_CSV=$(ls -t ${INPUT_DATA_DIR}/step01_transform_deltatoken_*.csv 2>/dev/null | head -n 1)

# 如果没找到最新的，则使用你指定的默认路径
CSV_PATH=${LATEST_CSV:-"step01_transform_deltatoken/step01_transform_deltatoken_20260425.csv"}

# 3. 绘图参数配置
MAX_DIMS=16      # <--- 修改这里：绘制前几维。如果想画全部，请设为空或注释掉
COLS=4          # 每行显示 5 个子图

# 定义图片输出路径 (文件名包含维度信息方便区分)
DIM_SUFFIX=${MAX_DIMS:+"_top${MAX_DIMS}"}
OUTPUT_PNG="${OUTPUT_DIR}/base_vs_diff_correlation${DIM_SUFFIX}_$(date +%Y%m%d).png"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ================================================================= #
# 执行逻辑
# ================================================================= #

echo "[$(date)] 开始执行相关性分析绘图..."
echo "输入数据: $CSV_PATH"
echo "输出图片: $OUTPUT_PNG"
echo "绘制维度: ${MAX_DIMS:-全部}"

# 检查 Python 脚本是否存在
if [ ! -f "$PLOT_SCRIPT" ]; then
    echo "错误: 找不到绘图脚本 $PLOT_SCRIPT"
    exit 1
fi

# 检查 CSV 文件是否存在
if [ ! -f "$CSV_PATH" ]; then
    echo "错误: 找不到 CSV 数据文件: $CSV_PATH"
    exit 1
fi

# 组装运行命令
CMD_ARGS="--csv_path $CSV_PATH --output_png $OUTPUT_PNG --cols $COLS --max_plot_lines 10000 "

# 如果指定了维度限制，则添加参数
if [ ! -z "$MAX_DIMS" ]; then
    CMD_ARGS="$CMD_ARGS --max_dims $MAX_DIMS"
fi

# 运行 Python 绘图
$PYTHON_EXEC "$PLOT_SCRIPT" $CMD_ARGS

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "------------------------------------------------------"
    echo "[$(date)] 绘图任务完成！"
    echo "分析结果保存路径: $OUTPUT_PNG"
    echo "------------------------------------------------------"
else
    echo "[$(date)] 错误: 绘图脚本运行失败，请检查报错信息。"
    exit 1
fi
