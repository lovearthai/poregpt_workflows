import json
import gzip
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================================================================================
# 脚本修改说明 (Step280 按照 Step252 方式改造)
# ==================================================================================
# ❌ 【修改1】主循环 (line ~140)
#    - 改为 K-mer 滑动窗口而非 Homopolymer 块
#    - for i in range(N - K + 1)
#
# ❌ 【修改2】数据字段提取 (line ~115)
#    - 从 tokens_based → pattern_ref
#    - 从 groupby(seq) → 直接使用 base_sample_spans_rel
#
# ❌ 【修改3】边界提取函数 (line ~40)
#    - 重写基于 K-mer spans 而非 blocks
#    - 参数改为 (kmer_spans, full_embeddings, boundary_num, stride)
#
# ❌ 【修改4】索引映射 (line ~150)
#    - 加入 stride 参数: pattern_start = span[0] // stride
#
# ❌ 【修改5】策略输出 (line ~170)
#    - 改为三种策略并行 (all, boundary, dynamic)
#
# ❌ 【修改6】特征提取 (line ~30)
#    - 新增 compute_weighted_feature() 函数
#    - 输出完整维度特征
#
# ❌ 【修改7】输出列 (line ~185)
#    - 改为: tokens_all, tokens_bnd, tokens_dyn
#    - 改为: feature_all, feature_bnd, feature_dyn
#
# ❌ 【修改8】参数定义 (line ~210)
#    - --block_count → --kmer_k
#    - 移除 --min_repeat, --select_token_strategy
#    - 新增 --token_stride
# ==================================================================================

# --- 模型处理核心逻辑 ---

def load_hf_model(model_path):
    print(f"正在加载模型及分词器 (CausalLM 模式): {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

def get_full_row_embeddings(model, tokenizer, high_layer_ids, vocab_code_shift):
    """
    通过 vocab_code_shift 偏移量构造 input_ids。
    """
    # 核心逻辑：将原始 ID 加上偏移量进入模型词表
    input_ids = torch.tensor([tid + vocab_code_shift for tid in high_layer_ids]).long().unsqueeze(0)

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        # 获取最后一层隐藏状态 [1, Seq_len, Dim]
        embeddings = outputs.hidden_states[-1][0].cpu().numpy()

    return embeddings

# --- Token 选择策略逻辑 ---

# ❌ 【修改3】重写边界提取函数 - 改为基于 K-mer spans 而非 blocks
# 参考 step252 的 select_boundary_tokens_kmer 逻辑
def select_boundary_tokens_embeddings(
    kmer_spans,
    window_embeddings,
    boundary_num,
    stride,
    pattern_start,
    pattern_end
):
    """
    针对 K-mer 窗口的边界提取逻辑
    按 spans 原本的物理顺序遍历每一个碱基 span，提取其首尾指定数量的 Token 索引
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

        head_indices = list(range(
            local_start,
            min(local_start + boundary_num, local_end)
        ))
        tail_indices = list(range(
            max(local_start, local_end - boundary_num),
            local_end
        ))

        combined_indices.extend(head_indices)
        combined_indices.extend(tail_indices)

    # 提取对应的嵌入向量
    selected_vecs = []

    valid_indices = []

    for idx in combined_indices:

        if 0 <= idx < len(window_embeddings):

            selected_vecs.append(window_embeddings[idx])

            valid_indices.append(idx)

    return valid_indices, np.array(selected_vecs)

def select_dynamic_tokens_embeddings(full_token_embeddings, top_n):

    if len(full_token_embeddings) <= 1:
        return list(range(len(full_token_embeddings))), full_token_embeddings

    activities = [0.0]

    for i in range(1, len(full_token_embeddings)):
        dist = np.linalg.norm(
            full_token_embeddings[i] -
            full_token_embeddings[i - 1]
        )
        activities.append(dist)

    activities = np.array(activities)

    k = min(top_n, len(full_token_embeddings))

    top_indices = np.argsort(activities)[-k:]
    top_indices = sorted(top_indices)

    selected_embeddings = full_token_embeddings[top_indices]

    return top_indices, selected_embeddings

# ❌ 【修改6】特征提取 - 新增高斯加权函数
def compute_weighted_feature(vecs, sigma):
    """
    计算高斯加权特征向量
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

# --- 主处理流水线 ---

def process_nanopore_data(args):
    model, tokenizer = load_hf_model(args.model_path)
    final_data = []
    global_id = 0

    print(f"开始处理: {args.input} (Vocab Shift: {args.vocab_code_shift})")

    with gzip.open(args.input, 'rt') as f:
        for line_idx, line in enumerate(tqdm(f, desc="Processing Rows")):
            try:
                record = json.loads(line)
                
                # ❌ 【修改2】数据字段提取 - 改为 pattern_ref 和 spans
                pattern_ref = record.get('pattern_ref', '')
                # 原始的 high_layer_ids
                high_layer_ids = [t[0] for t in record['tokens_layered']]
                spans = record.get('base_sample_spans_rel', [])

                # ❌ 【修改2】字段校验 - 改为 pattern_ref 和 spans
                if len(pattern_ref) != len(spans) or len(pattern_ref) == 0:
                    continue

                # 模型推理 (应用偏移量)
                full_row_embeddings = get_full_row_embeddings(
                    model, tokenizer, high_layer_ids, args.vocab_code_shift
                )

                if len(full_row_embeddings) != len(high_layer_ids):
                    print(f"CRITICAL: Alignment error at line {line_idx}!")
                    continue

                # ❌ 【修改1】主循环 - 改为 K-mer 滑动窗口（参考 step252）
                N = len(pattern_ref)
                K = args.kmer_k
                stride = args.token_stride
                
                for i in range(N - K + 1):
                    kmer_seq = pattern_ref[i : i + K]
                    kmer_spans = spans[i : i + K]
                    
                    # ❌ 【修改4】索引映射 - 加入 stride 参数
                    pattern_start = kmer_spans[0][0] // stride
                    pattern_end = kmer_spans[-1][1] // stride
                    
                    # 确保索引不越界
                    if pattern_start >= len(high_layer_ids) or pattern_end > len(high_layer_ids):
                        continue
                    
                    full_token_ids = high_layer_ids[pattern_start : pattern_end]
                    
                    if not full_token_ids:
                        continue
                    
                    # 获取对应的嵌入向量
                    window_full_vecs = full_row_embeddings[pattern_start : pattern_end]
                    
                    # ❌ 【修改5】策略输出 - 改为三种策略并行
                    # 策略1: all
                    vecs_all = window_full_vecs
                    mean_all = compute_weighted_feature(vecs_all, args.weight_sigma)
                    feature_all = "_".join([f"{v:.6f}" for v in mean_all]) if len(mean_all) > 0 else ""
                    
                    # 策略2: boundary
                    #vecs_boundary = select_boundary_tokens_embeddings(kmer_spans, full_row_embeddings, args.boundary_num, stride)
                    bnd_indices_local, vecs_boundary = \
                        select_boundary_tokens_embeddings(
                            kmer_spans,
                            window_full_vecs,
                            args.boundary_num,
                            stride,
                            pattern_start,
                            pattern_end
                    )
                    bnd_indices_global = [
                        pattern_start + idx
                        for idx in bnd_indices_local
                    ]
                    tokens_boundary = [
                        high_layer_ids[idx]
                        for idx in bnd_indices_global
                    ]       
                    
                    mean_boundary = compute_weighted_feature(vecs_boundary, args.weight_sigma)
                    feature_boundary = "_".join([f"{v:.6f}" for v in mean_boundary]) if len(mean_boundary) > 0 else ""
                    
                    # 策略3: dynamic
                    #vecs_dynamic = select_dynamic_tokens_embeddings(window_full_vecs, args.dynamic_top_n)
                    dyn_indices_local, vecs_dynamic = select_dynamic_tokens_embeddings(window_full_vecs, args.dynamic_top_n)
                    
                    dyn_indices_global = [
                        pattern_start + idx
                        for idx in dyn_indices_local
                    ]
                    
                    tokens_dynamic = [
                        high_layer_ids[idx]
                        for idx in dyn_indices_global
                    ]
                    
                    mean_dynamic = compute_weighted_feature(vecs_dynamic, args.weight_sigma)
                    feature_dynamic = "_".join([f"{v:.6f}" for v in mean_dynamic]) if len(mean_dynamic) > 0 else ""
                    
                    # ❌ 【修改7】输出列 - 改为三种策略的输出
                    row = {
                        'id': global_id,
                        'kmer_seq': kmer_seq,
                        'tokens_all': "-".join(map(str, full_token_ids)),
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

    if final_data:
        df = pd.DataFrame(final_data)
        df.to_csv(args.output, index=False)
        print(f"\n处理成功。输出文件：{args.output}，总条目：{len(df)}")
    else:
        print("处理失败，没有提取到有效数据")

def main():
    parser = argparse.ArgumentParser(description="PoreGPT Feature Extractor (K-mer based)")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='signal_features.csv')
    parser.add_argument('--model_path', type=str, required=True)

    # ❌ 【修改8】参数定义 - 替换参数名称和新增
    parser.add_argument('--kmer_k', type=int, default=5, help='K-mer 长度 (替代 block_count)')
    parser.add_argument('--token_stride', type=int, default=4, help='Token stride 下采样步长 (新增)')
    parser.add_argument('--vocab_code_shift', type=int, default=128, help="Token ID 的词表偏移量 (保留)")
    # 删除：--block_count 和 --min_repeat
    # 删除：--select_token_strategy（改为并行所有策略）
    
    parser.add_argument('--boundary_num', type=int, default=1)
    parser.add_argument('--dynamic_top_n', type=int, default=5)
    parser.add_argument('--weight_sigma', type=float, default=3.0)

    args = parser.parse_args()
    process_nanopore_data(args)

if __name__ == "__main__":
    main()

