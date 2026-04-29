#!/bin/bash

# ==============================================================================
# Nanopore Signal Tokenizer 训练启动脚本
# 功能：配置环境路径，启动 Python 训练任务，并记录详细日志
# ==============================================================================

# 1. 路径与参数配置
# 请根据实际服务器挂载点修改以下路径
INPUT_DIR="/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example_min"
OUTPUT_FILE="bwav_unicode_bpe_1024.json"
VOCAB_SIZE=1024
LOG_FILE="tokenizer_train_$(date +%Y%m%d_%H%M%S).log"

# 2. 自动创建工作目录（如果需要）
mkdir -p ./step030_bpe_tokenizer

# 3. 打印任务元数据
echo "----------------------------------------------------------------"
echo "开始执行 Nanopore 信号 Tokenizer 训练任务"
echo "执行时间: $(date)"
echo "输入目录: $INPUT_DIR"
echo "词表大小: $VOCAB_SIZE"
echo "日志文件: ./logs/$LOG_FILE"
echo "----------------------------------------------------------------"

# 4. 执行训练任务
# 使用 time 命令记录总耗时
# 2>&1 | tee 将输出同时显示在屏幕并写入日志文件
{
  time python3 scripts/step030_bpe_tokenizer.py \
    --input_dir "$INPUT_DIR" \
    --output_file "$OUTPUT_FILE" \
    --vocab_size $VOCAB_SIZE \
    --unicode_start 57344  # 对应 0xE000 的十进制
} 2>&1 | tee "./step030_bpe_tokenizer/$LOG_FILE"

# 5. 任务状态检查
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "----------------------------------------------------------------"
    echo "✅ 任务成功完成！"
    echo "模型保存至: $OUTPUT_FILE"
    echo "----------------------------------------------------------------"
else
    echo "----------------------------------------------------------------"
    echo "❌ 任务失败，请检查 ./logs/$LOG_FILE 中的错误详情。"
    echo "----------------------------------------------------------------"
    exit 1
fi
