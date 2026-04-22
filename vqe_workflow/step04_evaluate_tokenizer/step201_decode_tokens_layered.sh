#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================

# 输入文件路径
INPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.recon.jsonl.gz"
# 解码层级参数：0=全部(K层), 1=第一层, 2=前两层
LAYER=1

# 直接定义最终输出文件的全路径 (不再动态拼接文件夹)
OUTPUT_FILE="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.vqe340s147000.aligned.recon1.jsonl.gz"

# 模型检查点路径
MODEL_CKPT="/mnt/si003067jezr/default/poregpt/workflows/workflows/vqe_workflow/step02_train_vqe_model/pass340_w64_5x4x2_cnn12_dna595g_lr4e5_mongoq30_m12_scratch_f01k_lc12000/models/porepgt_vqe_tokenizer.step147000.pth"


# 计算设备
DEVICE="cuda:0"

# ==============================================================================

# 1. 确保目标文件所在的父目录存在
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "------------------------------------------------"
echo "🚀 任务启动: VQE 信号重构"
echo "📍 输入: $INPUT_FILE"
echo "📍 层级: $LAYER"
echo "📍 输出: $OUTPUT_FILE"
echo "------------------------------------------------"

# 执行 Python 脚本
python3 step201_decode_tokens_layered.py \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    --model-ckpt "$MODEL_CKPT" \
    --layer "$LAYER" \
    --device "$DEVICE"

echo "------------------------------------------------"
echo "✅ 处理完成！"
