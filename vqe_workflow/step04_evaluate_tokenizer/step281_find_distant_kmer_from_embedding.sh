#!/bin/bash

# =================================================================
# 脚本名称: run_distant_kmer_analysis.sh
# 描述: 自动解析高维 Embedding CSV 并寻找区分度最高的 K-mer 模式
# =================================================================

# 1. 路径配置
INPUT_CSV="step280_cal_bases_pattern_feature_by_olmo/step280_strategy_all_block5_sigma3.csv"
PYTHON_SCRIPT="step281_find_distant_kmer_from_embedding.py"

# 2. 分析参数配置
MIN_SAMPLES=5      # 每个 K-mer 至少出现的次数，增加此值可提高统计稳定性
TOP_N=300           # 输出前多少个最具区分度的模式
SYMMETRIC="--symmetric" # 如果你想比较左右侧翼完全一致的模式 (如 AAXAA)，保留此项
                        # 如果想比较所有组合 (如 AAXTT)，请将其设为空串 ""
SYMMETRIC=""
# 3. 环境准备
mkdir -p "$LOG_DIR"

if [ ! -f "$INPUT_CSV" ]; then
    echo "❌ 错误: 找不到输入文件 -> $INPUT_CSV"
    echo "请确认 step280 的提取任务已完成。"
    exit 1
fi

# 4. 执行分析
echo "-------------------------------------------------------"
echo "开始 K-mer 区分度分析..."
echo "输入文件: $INPUT_CSV"
echo "最小样本数: $MIN_SAMPLES"
echo "模式类型: $( [ -n "$SYMMETRIC" ] && echo "对称侧翼 (Symmetric)" || echo "全组合侧翼" )"
echo "-------------------------------------------------------"

# 记录开始时间
start_time=$(date +%s)

python "$PYTHON_SCRIPT" \
    -i "$INPUT_CSV" \
    --min_samples $MIN_SAMPLES \
    --top_n $TOP_N \
    $SYMMETRIC 

# 5. 执行结果检查
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "-------------------------------------------------------"
    echo "✅ 分析完成！耗时: ${duration} 秒"
else
    echo "-------------------------------------------------------"
    echo "❌ 运行出错，请查看日志。"
    exit 1
fi
