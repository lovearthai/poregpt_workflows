#!/bin/bash

# ==============================================================================
# ⚙️ 配置区域
# ==============================================================================

# 输入的 JSONL.GZ 文件
INPUT_JSONL="/mnt/zzbnew/dnadata/movetable/signal_LB06.mongoq30.jsonl.gz"

# 模型检查点路径
MODEL_CKPT="/mnt/si003067jezr/default/poregpt/workflows/workflows/vqe_workflow/step02_train_vqe_model/pass340_w64_5x4x2_cnn12_dna595g_lr4e5_mongoq30_m12_scratch_f01k_lc12000/models/porepgt_vqe_tokenizer.step147000.pth"

# 绘图输出根目录
EVAL_ROOT_DIR="step050_compare_signal_and_recon"

DEVICE="cuda"
BATCH_SIZE=16  # 推理 batch
MAX_PLOTS=30   # 想要查看的对比图数量

# ==============================================================================
# 🚀 执行逻辑
# ==============================================================================

STEP_TAG=$(echo $MODEL_CKPT | grep -oP 'step\d+' || echo "vqe_eval")
OUTPUT_DIR="${EVAL_ROOT_DIR}/eval_${STEP_TAG}"

mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "📊 VQE 重建评估 (从 JSONL 读取)"
echo "=================================================="
echo "📂 输入文件 : $INPUT_JSONL"
echo "🧠 模型权重 : $MODEL_CKPT"
echo "🖼️ 保存目录 : $OUTPUT_DIR"
echo "🔢 样本总数 : $MAX_PLOTS"
echo "=================================================="

python3 step050_compare_signal_and_recon.py \
    -i "$INPUT_JSONL" \
    -o "$OUTPUT_DIR" \
    --model-ckpt "$MODEL_CKPT" \
    --batch-size "$BATCH_SIZE" \
    --max-plots "$MAX_PLOTS" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✅ 评估完成！图片已存至: $OUTPUT_DIR"
else
    echo "❌ 执行失败。"
    exit 1
fi
