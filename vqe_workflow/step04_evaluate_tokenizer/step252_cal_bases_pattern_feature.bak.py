import json
import gzip
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
from poregpt.utils import get_rsq_vector_from_integer

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

def process_nanopore_data(args):
    vector_map = generate_vector_map(args.levels, args.num_quantizers)
    final_data = []
    global_id = 0

    print(f"开始处理输入文件: {args.input}")
    print(f"模式: 寻找连续 {args.block_count} 个碱基块，每块至少重复 {args.min_repeat} 次")

    with gzip.open(args.input, 'rt') as f:
        for line in tqdm(f, desc="Processing rows"):
            try:
                record = json.loads(line)
                seq = record['tokens_based']
                layer0_ids = [t[0] for t in record['tokens_layered']]

                if len(seq) != len(layer0_ids):
                    continue

                # 1. 提取所有的 Homopolymer 块
                # blocks 存储结构: [(碱基, 长度, 起始索引), ...]
                blocks = []
                current_idx = 0
                for char, group in groupby(seq):
                    group_list = list(group)
                    length = len(group_list)
                    blocks.append((char, length, current_idx))
                    current_idx += length

                # 2. 使用滑动窗口寻找复合 Pattern (如 AAAATTTTCCCC)
                # n 是连续块的数量，比如 AAAATTTT 就是 2，AAAATTTTCCCC 就是 3
                n = args.block_count
                for i in range(len(blocks) - n + 1):
                    window = blocks[i : i + n]
                    
                    # 检查窗口内每一块是否都满足最小重复长度
                    if all(b[1] >= args.min_repeat for b in window):
                        # 提取该复合模式对应的所有 Token
                        start_pos = window[0][2]
                        end_pos = window[-1][2] + window[-1][1]
                        
                        # 模式描述，例如 "A(4)T(4)C(4)"
                        pattern_desc = "".join([f"{b[0]}" * b[1] for b in window])
                        # 碱基类别，例如 "ATC"
                        base_category = "".join([b[0] for b in window])
                        
                        # 逻辑：取整个复合模式中间的 Token
                        # 同样去掉复合序列总体的首尾各一个
                        target_tokens = layer0_ids[start_pos + 1 : end_pos - 1]

                        if target_tokens:
                            vectors = [vector_map[tid] for tid in target_tokens]
                            mean_vec = np.mean(vectors, axis=0)

                            row = {
                                'id': global_id,
                                'base_pattern': pattern_desc,
                                'category': base_category, # 用于绘图染色
                                'dim0': float(mean_vec[0]),
                                'dim1': float(mean_vec[1]),
                                'dim2': float(mean_vec[2]),
                                'dim3': float(mean_vec[3])
                            }
                            final_data.append(row)
                            global_id += 1

            except Exception:
                continue

    if final_data:
        df = pd.DataFrame(final_data)
        df.to_csv(args.output, index=False)
        print(f"处理完成！提取了 {len(df)} 条复合特征。")

def main():
    parser = argparse.ArgumentParser(description="Nanopore Complex Homopolymer Pattern Extractor")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_features.csv')
    
    # 核心参数
    parser.add_argument('--min_repeat', type=int, default=4, 
                        help='每个碱基块的最小重复次数，例如 4 代表 AAAA')
    parser.add_argument('--block_count', type=int, default=3, 
                        help='连续的不同碱基块数量，3 代表 AAAATTTTCCCC')
    
    parser.add_argument('--levels', type=int, nargs='+', default=[5, 5, 5, 5])
    parser.add_argument('--num_quantizers', type=int, default=1)

    args = parser.parse_args()
    process_nanopore_data(args)

if __name__ == "__main__":
    main()
