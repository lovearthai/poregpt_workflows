import json
import gzip
import argparse
import torch
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_hf_model(model_path):
    """
    加载自回归语言模型及对应的分词器，并自动搬运至 GPU
    """
    print(f"正在加载模型及分词器 (CausalLM 模式): {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

def get_full_row_embeddings(model, tokenizer, high_layer_ids, vocab_code_shift):
    """
    将原始 Token ID 加上指定的词表偏移量后送入模型，提取最后一层完整的隐藏状态隐藏向量
    """
    input_ids = torch.tensor([tid + vocab_code_shift for tid in high_layer_ids]).long().unsqueeze(0)

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][0].cpu().numpy()

    return embeddings

def select_boundary_tokens_embeddings(
    kmer_spans,
    window_embeddings,
    boundary_num,
    stride,
    pattern_start,
    pattern_end
):
    """
    针对 K-mer 窗口的边界特征提取策略：
    严格按照 spans 的原始物理时间顺序，顺序截取每个碱基在当前窗口内的首尾 boundary_num 个位置
    """
    if not kmer_spans:
        return [], np.array([])

    combined_indices = []

    for span in kmer_spans:
        span_start = max(span[0] // stride, pattern_start)
        span_end = min(span[1] // stride, pattern_end)
        local_start = span_start - pattern_start
        local_end = span_end - pattern_start

        if local_start >= local_end:
            continue

        head_indices = list(range(local_start, min(local_start + boundary_num, local_end)))
        tail_indices = list(range(max(local_start, local_end - boundary_num), local_end))

        combined_indices.extend(head_indices)
        combined_indices.extend(tail_indices)

    selected_vecs = []
    valid_indices = []

    for idx in combined_indices:
        if 0 <= idx < len(window_embeddings):
            selected_vecs.append(window_embeddings[idx])
            valid_indices.append(idx)

    return valid_indices, np.array(selected_vecs)

def select_dynamic_tokens_embeddings(full_token_embeddings, top_n):
    """
    针对 K-mer 窗口的动态活跃特征提取策略：
    计算相邻 Token 嵌入向量的一阶差分欧氏距离，筛选波动最剧烈的 Top N 个物理位置
    """
    if len(full_token_embeddings) <= 1:
        return list(range(len(full_token_embeddings))), full_token_embeddings

    activities = [0.0]
    for i in range(1, len(full_token_embeddings)):
        dist = np.linalg.norm(full_token_embeddings[i] - full_token_embeddings[i - 1])
        activities.append(dist)

    activities = np.array(activities)
    k = min(top_n, len(full_token_embeddings))

    top_indices = np.argsort(activities)[-k:]
    top_indices = sorted(top_indices)
    selected_embeddings = full_token_embeddings[top_indices]

    return top_indices, selected_embeddings

def compute_weighted_feature(vecs, sigma):
    """
    根据给定的高斯标准差 sigma，对输入的向量集合进行中心对称的高斯权重加权求和
    """
    if len(vecs) == 0:
        return np.array([])

    L = len(vecs)
    mid_idx = (L - 1) / 2.0
    offsets = np.arange(L) - mid_idx
    raw_weights = np.exp(-(offsets**2) / (2 * sigma**2))
    weights = raw_weights / np.sum(raw_weights)
    mean_vec = np.sum(vecs * weights[:, np.newaxis], axis=0)
    return mean_vec

def process_nanopore_data(args):
    """
    核心流水线：滑动 K-mer 窗口，全量提取多策略聚合特征与对应的无歧义原始电信号
    """
    model, tokenizer = load_hf_model(args.model_path)
    global_id = 0

    print(f"开始处理数据集: {args.input} (Vocab Shift: {args.vocab_code_shift})")

    csv_file = open(args.output, 'w', newline='')
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            'id', 'kmer_seq', 
            'tokens_all', 'tokens_dyn', 'tokens_bnd', 
            'feature_all', 'feature_dyn', 'feature_bnd',
            'signal'
        ]
    )
    writer.writeheader()

    with gzip.open(args.input, 'rt') as f:
        for line_idx, line in enumerate(tqdm(f, desc="Processing Rows")):
            try:
                record = json.loads(line)

                # 数据基础核心字段抽取
                pattern_ref = record.get('pattern_ref', '')
                high_layer_ids = [t[0] for t in record['tokens_layered']]
                spans = record.get('base_sample_spans_rel', [])
                raw_signal_list = record.get('signal', [])

                if len(pattern_ref) != len(spans) or len(pattern_ref) == 0:
                    continue

                # 提取一整行长序列对应的最后一层 Embeddings 后进入窗口循环
                full_row_embeddings = get_full_row_embeddings(
                    model, tokenizer, high_layer_ids, args.vocab_code_shift
                )

                if len(full_row_embeddings) != len(high_layer_ids):
                    print(f"CRITICAL: Alignment error at line {line_idx}!")
                    continue

                N = len(pattern_ref)
                K = args.kmer_k
                stride = args.token_stride

                # K-mer 滑动窗口主循环
                for i in range(N - K + 1):
                    kmer_seq = pattern_ref[i : i + K]
                    kmer_spans = spans[i : i + K]

                    # 物理时间轴坐标到下采样 Token 轴索引的 Stride 映射
                    pattern_start = kmer_spans[0][0] // stride
                    pattern_end = kmer_spans[-1][1] // stride

                    if pattern_start >= len(high_layer_ids) or pattern_end > len(high_layer_ids):
                        continue

                    full_token_ids = high_layer_ids[pattern_start : pattern_end]
                    if not full_token_ids:
                        continue

                    window_full_vecs = full_row_embeddings[pattern_start : pattern_end]

                    # --- 策略 1: All 全量提取 ---
                    vecs_all = window_full_vecs
                    mean_all = compute_weighted_feature(vecs_all, args.weight_sigma)
                    feature_all = "_".join([f"{v:.6f}" for v in mean_all]) if len(mean_all) > 0 else ""

                    # --- 策略 2: Boundary 边界提取 ---
                    bnd_indices_local, vecs_boundary = select_boundary_tokens_embeddings(
                        kmer_spans, window_full_vecs, args.boundary_num, stride, pattern_start, pattern_end
                    )
                    bnd_indices_global = [pattern_start + idx for idx in bnd_indices_local]
                    tokens_boundary = [high_layer_ids[idx] for idx in bnd_indices_global]

                    mean_boundary = compute_weighted_feature(vecs_boundary, args.weight_sigma)
                    feature_boundary = "_".join([f"{v:.6f}" for v in mean_boundary]) if len(mean_boundary) > 0 else ""

                    # --- 策略 3: Dynamic 动态活跃度提取 ---
                    dyn_indices_local, vecs_dynamic = select_dynamic_tokens_embeddings(window_full_vecs, args.dynamic_top_n)
                    dyn_indices_global = [pattern_start + idx for idx in dyn_indices_local]
                    tokens_dynamic = [high_layer_ids[idx] for idx in dyn_indices_global]

                    mean_dynamic = compute_weighted_feature(vecs_dynamic, args.weight_sigma)
                    feature_dynamic = "_".join([f"{v:.6f}" for v in mean_dynamic]) if len(mean_dynamic) > 0 else ""

                    # --- 信号处理功能：将 K-mer spans 映射的一维原始电信号打包转化为无 CSV 歧义的格式 ---
                    signal_str = ""
                    if raw_signal_list:
                        segments_list = []
                        for span in kmer_spans:
                            sig_start = span[0]
                            sig_end = span[1]
                            segment = raw_signal_list[sig_start:sig_end]  # 严格对齐当前碱基信号片段
                            seg_str = "_".join(map(str, segment))          # 片段内部元素以下划线连接
                            segments_list.append(seg_str)
                        signal_str = "|".join(segments_list)              # 碱基片段间以竖线隔离

                    # 构建统一行字典并持久化输出
                    row = {
                        'id': global_id,
                        'kmer_seq': kmer_seq,
                        'tokens_all': "-".join(map(str, full_token_ids)),
                        'tokens_dyn': "-".join(map(str, tokens_dynamic)),
                        'tokens_bnd': "-".join(map(str, tokens_boundary)),
                        'feature_all': feature_all,
                        'feature_dyn': feature_dynamic,
                        'feature_bnd': feature_boundary,
                        'signal': signal_str
                    }
                    writer.writerow(row)
                    global_id += 1

            except Exception:
                continue

    csv_file.close()
    print(f"\n处理成功完成。输出文件：{args.output}")
    print(f"成功提取保存的 K-mer 总条目数：{global_id}")

def main():
    parser = argparse.ArgumentParser(description="PoreGPT Feature Extractor (K-mer based)")
    parser.add_argument('--input', type=str, required=True, help='输入的 .json.gz 数据集路径')
    parser.add_argument('--output', type=str, default='signal_features.csv', help='特征输出目标 .csv 路径')
    parser.add_argument('--model_path', type=str, required=True, help='HuggingFace 模型权重/配置的本地或云端路径')

    # 业务超参数定义
    parser.add_argument('--kmer_k', type=int, default=5, help='滑动窗口 K-mer 的窗口跨度 (K值)')
    parser.add_argument('--token_stride', type=int, default=4, help='分词器对应的下采样 Stride 步长')
    parser.add_argument('--vocab_code_shift', type=int, default=128, help='Token ID 映射进模型词表所需的偏移常数')
    parser.add_argument('--boundary_num', type=int, default=1, help='Boundary 提取策略下，首尾各抽取的 Token 个数')
    parser.add_argument('--dynamic_top_n', type=int, default=5, help='Dynamic 提取策略下，基于活跃度抽取的最大 Token 个数')
    parser.add_argument('--weight_sigma', type=float, default=3.0, help='计算高斯加权融合特征向量时的标准差')

    args = parser.parse_args()
    process_nanopore_data(args)

if __name__ == "__main__":
    main()
