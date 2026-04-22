#!/bin/bash

# --- 基础配置 ---
# 输入数据路径
INPUT_JSONL="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.jsonl.gz"
# 模型路径
MODEL_PATH="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_150m_DNA595G_RSQ542_C625_CNN12_V340S147000/base"

#!/bin/bash

# --- 基础环境配置 ---
# 如果需要激活特定环境，请取消下行注释
# source activate bonito_py310

# 图片和结果输出目录
OUT_DIR="step291_infer_base_pattern_from_olmo"

# --- 运行参数控制 ---
# 你可以通过 ./run_infer_analysis.sh 10 256 这种方式传参
# 默认值：行号=0, Prompt长度=128
LINE_ID=${1:-1851}
PROMPT_LEN=${2:-395}

VOCAB_CODE_SHIFT=128
TOP_K=100
# 创建输出目录
mkdir -p "$OUT_DIR"

echo "================================================================"
echo "📊 PoreGPT 推理与物理空间距离分析"
echo "================================================================"
echo "📅 执行时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📂 数据文件: $INPUT_DATA"
echo "🤖 模型路径: $MODEL_PATH"
echo "📍 目标行号: $LINE_ID"
echo "📏 上下文长度: $PROMPT_LEN"
echo "🖼️ 输出目录: $OUT_DIR"
echo "----------------------------------------------------------------"

# 执行 Python 脚本
python3 step291_infer_base_pattern_from_olmo.py \
    --input "$INPUT_JSONL" \
    --model_path "$MODEL_PATH" \
    --line_id "$LINE_ID" \
    --prompt_len "$PROMPT_LEN" \
    --vocab_code_shift $VOCAB_CODE_SHIFT \
    --output_dir "$OUT_DIR" \
    --top_k "$TOP_K"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------------"
    echo "✅ 分析完成！请在 $OUT_DIR 查看生成的散点图。"
else
    echo "----------------------------------------------------------------"
    echo "❌ 脚本执行失败，请检查 Python 报错信息。"
fi
