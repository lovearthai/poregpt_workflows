#!/bin/bash

# 默认绘制数量 (0 表示全部)
DEFAULT_TOP_N=300
# ================= 配置区域 =================
DEFAULT_INPUT="step012_count_repeated_token_20260515_215237.csv"
DEFAULT_OUTPUT="step012_count_repeated_token_p${DEFAULT_TOP_N}_20260515_215237.png"
# 默认输入文件
DEFAULT_INPUT="step012_count_delta1_token_20260515_204732.csv"
DEFAULT_OUTPUT="step012_count_delta1_token_p${DEFAULT_TOP_N}_20260515_204732.png"


# Python 脚本名称
PY_SCRIPT="scripts/step013_ploted_token.py"
# ============================================

# 检查 Python 脚本是否存在
if [ ! -f "$PY_SCRIPT" ]; then
    echo "❌ 错误: 找不到 $PY_SCRIPT，请确保它在当前目录下。"
    exit 1
fi

# 检查必要的 Python 库
echo "🔍 检查环境..."
python3 -c "import pandas, matplotlib, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 缺少必要库，正在尝试安装 (pandas, matplotlib, seaborn)..."
    pip install pandas matplotlib seaborn
fi

# 使用传入参数或默认值
INPUT_CSV=${1:-$DEFAULT_INPUT}
TOP_N=${2:-$DEFAULT_TOP_N}
OUTPUT_PNG=${3:-$DEFAULT_OUTPUT}
TOKENS_PER_ROW=100       # 每行 50 个，最终会生成 3 行 subplot
echo "🚀 开始绘制图表..."
echo "--------------------------------"
echo "输入文件: $INPUT_CSV"
echo "绘制数量: $TOP_N (0 代表全部)"
echo "输出文件: $OUTPUT_PNG"
echo "--------------------------------"

# 运行 Python 脚本
python3 "$PY_SCRIPT" --input "$INPUT_CSV" --top_n "$TOP_N" --output "$OUTPUT_PNG" --plot_tokens_per_row "$TOKENS_PER_ROW"

if [ $? -eq 0 ]; then
    echo "✨ 大功告成！"
else
    echo "💥 绘图失败，请检查 CSV 格式。"
    exit 1
fi
