#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域 (在此处修改参数)
# ==============================================================================

# tokenizer的名字,一旦固定，为了保持后续查找一致,不要改动了
# MODEL_CKPT的名字也要改
TOKENIZER_NAME="vqe342s036000l1"

# 输入文件路径
INPUT_FILE="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.shiftr4.mongoq30.jsonl.gz"

# 输出文件路径
OUTPUT_FILE="/mnt/zzbnew/poregpt/dnadata/movetable/signal_LB06.shiftr4.mongoq30.${TOKENIZER_NAME}.jsonl.gz"

# 模型检查点路径, 因为我们使用accelerate框架，所以检查点路径是个目录
MODEL_CKPT="/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_300m_DNA595G_RSQ1142_C16k_CNN12_V342S036000L1/encoder"

# 运行设备 (cuda 或 cpu)
DEVICE="cuda"

# token化的层,这个地方必须填写0
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
python3 scripts/step020_tokenize_movetable_jsonlgz.py \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    --model-ckpt "$MODEL_CKPT" \
    --tokenize-layer "$LAYER" \
    --device "$DEVICE" \

echo ""
echo "✅ 任务执行完毕！"
