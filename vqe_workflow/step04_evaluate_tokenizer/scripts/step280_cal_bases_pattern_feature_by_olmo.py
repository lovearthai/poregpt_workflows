import json
import gzip
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def select_boundary_tokens_embeddings(window, full_embeddings, boundary_num):
    selected_vecs = []
    for block in window:
        b_start = block['start']
        b_end = b_start + block['len']
        head_idx = list(range(b_start, min(b_start + boundary_num, b_end)))
        tail_idx = list(range(max(b_start, b_end - boundary_num), b_end))
        combined_idx = sorted(list(set(head_idx + tail_idx)))
        for idx in combined_idx:
            selected_vecs.append(full_embeddings[idx])
    return np.array(selected_vecs)

def select_dynamic_tokens_embeddings(full_token_embeddings, top_n):
    if len(full_token_embeddings) <= 1:
        return full_token_embeddings
    activities = [0.0]
    for i in range(1, len(full_token_embeddings)):
        dist = np.linalg.norm(full_token_embeddings[i] - full_token_embeddings[i-1])
        activities.append(dist)
    activities = np.array(activities)
    k = min(top_n, len(full_token_embeddings))
    top_indices = np.argsort(activities)[-k:]
    top_indices = sorted(top_indices)
    return full_token_embeddings[top_indices]

# --- 主处理流水线 ---

def process_nanopore_data(args):
    model, tokenizer = load_hf_model(args.model_path)
    final_data = []
    global_id = 0

    print(f"开始处理: {args.input} (Vocab Shift: {args.vocab_code_shift})")

    with gzip.open(args.input, 'rt') as f:
        for line in tqdm(f, desc="Processing Rows"):
            try:
                record = json.loads(line)
                seq = record['tokens_based']
                # 原始的 high_layer_ids
                high_layer_ids = [t[0] for t in record['tokens_layered']]

                if len(seq) != len(high_layer_ids):
                    continue

                # 1. 提取 Homopolymer 块结构
                blocks = []
                current_pos = 0
                for char, group in groupby(seq):
                    length = len(list(group))
                    blocks.append({'base': char, 'len': length, 'start': current_pos})
                    current_pos += length

                # 2. 模型推理 (应用偏移量)
                full_row_embeddings = get_full_row_embeddings(
                    model, tokenizer, high_layer_ids, args.vocab_code_shift
                )

                if len(full_row_embeddings) != len(high_layer_ids):
                    print(f"CRITICAL: Alignment error!")
                    continue

                # 3. 滑动窗口
                n = args.block_count
                for i in range(len(blocks) - n + 1):
                    window = blocks[i : i + n]

                    if all(b['len'] >= args.min_repeat for b in window):
                        p_start = window[0]['start']
                        p_end = window[-1]['start'] + window[-1]['len']

                        window_full_vecs = full_row_embeddings[p_start : p_end]

                        # 4. 应用选择策略
                        if args.select_token_strategy == "boundary":
                            selected_vecs = select_boundary_tokens_embeddings(window, full_row_embeddings, args.boundary_num)
                        elif args.select_token_strategy == "dynamic":
                            selected_vecs = select_dynamic_tokens_embeddings(window_full_vecs, args.dynamic_top_n)
                        else:
                            selected_vecs = window_full_vecs

                        # 5. 加权求和
                        if len(selected_vecs) > 0:
                            L = len(selected_vecs)
                            mid_idx = (L - 1) / 2.0
                            offsets = np.arange(L) - mid_idx
                            sigma = args.weight_sigma
                            raw_weights = np.exp(-(offsets**2) / (2 * (sigma**2)))
                            weights = raw_weights / np.sum(raw_weights)

                            mean_vec = np.sum(selected_vecs * weights[:, np.newaxis], axis=0)
                            embedding_str = "_".join([f"{v:.6f}" for v in mean_vec])

                            # 记录信息
                            pattern_desc = "".join([f"{b['base']}" * b['len'] for b in window])
                            category = "".join([b['base'] for b in window])

                            final_data.append({
                                'id': global_id,
                                'base_pattern': pattern_desc,
                                'category': category,
                                'embedding': embedding_str
                            })
                            global_id += 1

            except Exception as e:
                print(f"Error: {e}")
                continue

    if final_data:
        df = pd.DataFrame(final_data)
        df.to_csv(args.output, index=False)
        print(f"\n处理成功。输出文件：{args.output}，总条目：{len(df)}")

def main():
    parser = argparse.ArgumentParser(description="PoreGPT Feature Extractor")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='signal_features.csv')
    parser.add_argument('--model_path', type=str, required=True)

    # 新增偏移量参数
    parser.add_argument('--vocab_code_shift', type=int, default=128, help="Token ID 的词表偏移量")

    parser.add_argument('--min_repeat', type=int, default=2)
    parser.add_argument('--block_count', type=int, default=3)
    parser.add_argument('--select_token_strategy', type=str, default='all', choices=['all', 'boundary', 'dynamic'])
    parser.add_argument('--boundary_num', type=int, default=1)
    parser.add_argument('--dynamic_top_n', type=int, default=5)
    parser.add_argument('--weight_sigma', type=float, default=3.0)

    args = parser.parse_args()
    process_nanopore_data(args)

if __name__ == "__main__":
    main()

