import json
import gzip
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from poregpt.utils import get_rsq_vector_from_integer

def calculate_statistics(input_file, kmer_k):
    """
    预先统计数据集中 pattern_ref 的总碱基长度以及 K-mer 的理论数量。
    """
    total_bases = 0
    total_kmers = 0
    valid_rows = 0
    
    print(f"正在预统计数据集特征... ({input_file})")
    with gzip.open(input_file, 'rt') as f:
        for line in f:
            try:
                record = json.loads(line)
                pattern_ref = record.get('pattern_ref', '')
                spans = record.get('base_sample_spans_rel', [])
                
                if len(pattern_ref) != len(spans) or len(pattern_ref) == 0:
                    continue
                
                N = len(pattern_ref)
                total_bases += N
                valid_rows += 1
                
                if N >= kmer_k:
                    total_kmers += (N - kmer_k + 1)
                    
            except Exception:
                continue
                
    print("----------------- 统计报告 -----------------")
    print(f"有效数据行数 (Valid Rows): {valid_rows}")
    print(f"总碱基长度 (Total Reference Bases): {total_bases} bp")
    print(f"在 K={kmer_k} 时，理论 {kmer_k}-mer 总数量: {total_kmers}")
    print("-------------------------------------------")
    return total_bases, total_kmers

def generate_vector_map(levels, num_quantizers):
    total_ids = np.prod(levels)
    vector_map = {}
    print(f"正在预计算码表向量: levels={levels}, num_quantizers={num_quantizers}...")
    for i in range(total_ids):
        vec = get_rsq_vector_from_integer(i, levels, num_quantizers, use_fast=True)
        if hasattr(vec, 'cpu'):
            vec = vec.cpu().detach().numpy()
        vector_map[i] = np.array(vec, dtype=np.float32).flatten()
    return vector_map


def select_boundary_tokens_kmer(spans, layer0_ids, boundary_num, stride):
    """
    针对 K-mer 窗口的边界提取逻辑（完全忠实原始物理顺序）：
    按 spans 原本的物理顺序遍历每一个碱基 span，提取其首尾指定数量的 Token 索引。
    完全不使用 set，也绝对不进行 sort，100% 还原对齐数据中的原始时间流和交错状态。
    """
    if not spans:
        return []

    combined_indices = []

    for span in spans:
        # 将当前碱基的物理坐标转换为 Token 索引
        span_start = span[0] // stride
        span_end = span[1] // stride

        # 如果对应的 token 区间无效，则跳过当前碱基
        if span_start >= span_end:
            continue

        # 1. 严格按时间顺序提取该碱基的头部 boundary_num 个 Token 索引
        head_indices = list(range(span_start, min(span_start + boundary_num, span_end)))
        
        # 2. 严格按时间顺序提取该碱基的尾部 boundary_num 个 Token 索引
        tail_indices = list(range(max(span_start, span_end - boundary_num), span_end))

        # 3. 直接顺次将索引存入列表（即使 span 之间存在物理交错也完全保留）
        combined_indices.extend(head_indices)
        combined_indices.extend(tail_indices)

    # 4. 根据提取出的、忠于原始顺序的索引，直接映射为 Token ID
    selected_tokens = [layer0_ids[idx] for idx in combined_indices if idx < len(layer0_ids)]
    return selected_tokens


def select_dynamic_tokens(full_token_ids, vector_map, top_n):
    """
    基于动态活跃度选择 Token
    """
    if len(full_token_ids) <= 1:
        return full_token_ids

    vectors = np.array([vector_map[tid] for tid in full_token_ids])
    activities = [0.0]

    for i in range(1, len(vectors)):
        dist = np.linalg.norm(vectors[i] - vectors[i-1])
        activities.append(dist)

    activities = np.array(activities)
    k = min(top_n, len(full_token_ids))
    top_indices = np.argsort(activities)[-k:]
    top_indices = sorted(top_indices)

    selected_tokens = [full_token_ids[i] for i in top_indices]
    return selected_tokens

def compute_gaussian_weighted_feature(selected_tokens, vector_map, sigma):
    """
    计算高斯加权特征
    """
    if not selected_tokens:
        return ""

    vectors = np.array([vector_map[tid] for tid in selected_tokens])
    L = len(selected_tokens)
    mid_idx = (L - 1) / 2.0
    offsets = np.arange(L) - mid_idx

    raw_weights = np.exp(-(offsets**2) / (2 * sigma**2))
    weights = raw_weights / np.sum(raw_weights)

    mean_vec = np.sum(vectors * weights[:, np.newaxis], axis=0)

    # 格式化前 4 维
    feature_str = "_".join([f"{v:.6f}" for v in mean_vec[:4]])
    return feature_str

def process_nanopore_data(args):
    # 先行执行全局统计
    calculate_statistics(args.input, args.kmer_k)
    
    vector_map = generate_vector_map(args.levels, args.num_quantizers)
    final_data = []
    global_id = 0

    # 错误过滤原因计数器
    skip_reasons = {
        "length_mismatch": 0,
        "index_out_of_bounds": 0,
        "empty_tokens": 0
    }

    print(f"开始处理输入文件: {args.input}")
    print(f"K-mer 长度: {args.kmer_k} | Token Stride: {args.token_stride}")

    with gzip.open(args.input, 'rt') as f:
        for line_idx, line in enumerate(tqdm(f, desc="Processing rows")):
            try:
                record = json.loads(line)
                pattern_ref = record['pattern_ref']
                layer0_ids = [t[0] for t in record['tokens_layered']]
                spans = record['base_sample_spans_rel']

                # --- 1. 基础长度校验 ---
                if len(pattern_ref) != len(spans):
                    skip_reasons["length_mismatch"] += 1
                    if skip_reasons["length_mismatch"] <= 5:
                        print(f"\n[⚠️ 长度不匹配] 行号 {line_idx}: pattern_ref ({len(pattern_ref)}) != spans ({len(spans)})")
                    continue

                N = len(pattern_ref)
                K = args.kmer_k
                stride = args.token_stride

                # 滑动窗口
                for i in range(N - K + 1):
                    kmer_seq = pattern_ref[i : i + K]
                    kmer_spans = spans[i : i + K]

                    # --- 🔥 引入 Stride 进行索引映射转换 ---
                    pattern_start = kmer_spans[0][0] // stride
                    pattern_end = kmer_spans[-1][1] // stride

                    # --- 2. 确保索引不越界 ---
                    if pattern_start >= len(layer0_ids) or pattern_end > len(layer0_ids):
                        skip_reasons["index_out_of_bounds"] += 1
                        if skip_reasons["index_out_of_bounds"] <= 5:
                            print(f"\n[🚫 越界过滤] 行号 {line_idx}, K-mer索引 {i} ({kmer_seq}): "
                                  f"Token总长={len(layer0_ids)}, 转换后切片范围=[{pattern_start}:{pattern_end}] (原始=[{kmer_spans[0][0]}:{kmer_spans[-1][1]}])")
                        continue

                    # 提取映射后的 Token 列表
                    full_token_ids = layer0_ids[pattern_start : pattern_end]
                    
                    # --- 3. 确保提取出的 Token 不为空 ---
                    if not full_token_ids:
                        skip_reasons["empty_tokens"] += 1
                        if skip_reasons["empty_tokens"] <= 5:
                            print(f"\n[🧩 空Token过滤] 行号 {line_idx}, K-mer索引 {i} ({kmer_seq}): 切片范围 [{pattern_start}:{pattern_end}] 内无Token")
                        continue

                    # --- 策略 1: all ---
                    tokens_all = full_token_ids
                    feature_all = compute_gaussian_weighted_feature(tokens_all, vector_map, args.weight_sigma)

                    # --- 策略 2: dynamic ---
                    tokens_dynamic = select_dynamic_tokens(full_token_ids, vector_map, args.dynamic_top_n)
                    feature_dynamic = compute_gaussian_weighted_feature(tokens_dynamic, vector_map, args.weight_sigma)

                    # --- 策略 3: boundary ---
                    tokens_boundary = select_boundary_tokens_kmer(kmer_spans, layer0_ids, args.boundary_num, stride)
                    feature_boundary = compute_gaussian_weighted_feature(tokens_boundary, vector_map, args.weight_sigma)

                    # 组织输出数据
                    row = {
                        'id': global_id,
                        'kmer_seq': kmer_seq,
                        'tokens_all': "-".join(map(str, tokens_all)),
                        'tokens_dyn': "-".join(map(str, tokens_dynamic)),
                        'tokens_bnd': "-".join(map(str, tokens_boundary)),
                        'feature_all': feature_all,
                        'feature_dyn': feature_dynamic,
                        'feature_bnd': feature_boundary
                    }
                    final_data.append(row)
                    global_id += 1

            except Exception as e:
                continue

    # 输出汇总报告
    print("\n----------------- 过滤原因汇总 -----------------")
    print(f"1. 因 [pattern_ref 与 spans 长度不匹配] 过滤的整行数: {skip_reasons['length_mismatch']}")
    print(f"2. 因 [索引越界 (Token序列不够长)] 过滤的 K-mer 数: {skip_reasons['index_out_of_bounds']}")
    print(f"3. 因 [提取到空 Token 列表] 过滤的 K-mer 数: {skip_reasons['empty_tokens']}")
    print("-------------------------------------------")

    if final_data:
        df = pd.DataFrame(final_data)
        df.to_csv(args.output, index=False)
        print(f"处理完成！实际成功提取并保存了 {len(df)} 条 K-mer 复合特征。")
    else:
        print("处理失败，没有提取到有效数据")

def main():
    parser = argparse.ArgumentParser(description="Nanopore K-mer Feature Extractor (Multi-Strategy)")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_features.csv')
    parser.add_argument('--kmer_k', type=int, default=5, help='滑动窗口 K-mer 的长度')
    
    # 🔥 新增参数：token_stride
    parser.add_argument('--token_stride', type=int, default=4, help='Token 对应的 Stride 下采样步长')

    parser.add_argument('--boundary_num', type=int, default=1,
                        help='当采用 boundary 提取时，K-mer 首尾碱基分别提取的 token 数量')
    parser.add_argument('--dynamic_top_n', type=int, default=5,
                        help='当采用 dynamic 提取时，选择的 Top N 个活跃 token')

    parser.add_argument('--levels', type=int, nargs='+', default=[5, 5, 5, 5])
    parser.add_argument('--num_quantizers', type=int, default=1)
    parser.add_argument('--weight_sigma', type=float, default=2.0, help='高斯加权的标准差')

    args = parser.parse_args()
    process_nanopore_data(args)

if __name__ == "__main__":
    main()
