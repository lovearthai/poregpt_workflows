#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域 (在此处修改参数)
# ==============================================================================

# 输入文件路径
INPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.jsonl.gz"

# 输出文件路径
OUTPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.jsonl.gz"

# 模型检查点路径 (.pth 文件)
MODEL_CKPT="/mnt/si003067jezr/default/poregpt/workflows/workflows/vqe_workflow/step02_train_vqe_model/pass340_w64_5x4x2_cnn12_dna595g_lr4e5_mongoq30_m12_scratch_f01k_lc12000/models/porepgt_vqe_tokenizer.step147000.pth"

# 运行设备 (cuda 或 cpu)
DEVICE="cuda"

# token化的层
LAYER=0




# ==============================================================================
# 🚀 执行逻辑 (下方代码通常无需修改)
# ==============================================================================

echo "=================================================="
echo "🚀 开始执行 Tokenization 任务"
echo "=================================================="
echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_FILE"
echo "模型: $MODEL_CKPT"
echo "设备: $DEVICE"
echo "层数: $LAYER"
echo "策略: $PROCESS_STRATEGY"
echo "=================================================="


# 执行 Python 脚本
python3 step020_tokenize_movetable_jsonlgz.py \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    --model-ckpt "$MODEL_CKPT" \
    --tokenize-layer "$LAYER" \
    --device "$DEVICE" \

echo ""
echo "✅ 任务执行完毕！"
