import json
import gzip
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
from poregpt.utils import get_rsq_vector_from_integer


def generate_vector_map(levels, num_quantizers):
    """
    预计算所有可能token_id对应的向量映射表
    levels: 每一层量化的级数，例如 [5, 5, 5, 5]
    num_quantizers: 残差量化的总层数 (Layers)
    """
    # 注意：这里的 total_ids 是单层码本的大小
    # 对于 RSQ 任务，如果你要处理的是 Layer 0 的 token，这里的 total_ids 应该是 np.prod(levels)
    total_ids = np.prod(levels)
    vector_map = {}

    print(f"正在预计算码表向量: levels={levels}, num_quantizers={num_quantizers}, total_ids={total_ids}...")
    
    for i in range(total_ids):
        # 显式传递 num_quantizers 参数，不再依赖 len(levels)
        # 增加 use_fast=True 提升预计算速度
        vec = get_rsq_vector_from_integer(i, levels, num_quantizers, use_fast=True)

        # 确保数据从 GPU 回到 CPU
        if hasattr(vec, 'cpu'):
            vec = vec.cpu().detach().numpy()

        # 这里的 vec 可能是 [1, dim]，我们取第 0 行
        vector_map[i] = np.array(vec, dtype=np.float32).flatten()
        
    return vector_map


def process_nanopore_data(args):
    # 1. 初始化码表映射，传入暴露的 args.num_quantizers
    vector_map = generate_vector_map(args.levels, args.num_quantizers)

    final_data = []
    global_id = 0

    # 2. 读取并处理数据
    print(f"开始处理输入文件: {args.input}")
    with gzip.open(args.input, 'rt') as f:
        for line in tqdm(f, desc="Processing rows"):
            try:
                record = json.loads(line)
                seq = record['tokens_based']
                # 获取 Layer 0 的 token ids
                layer0_ids = [t[0] for t in record['tokens_layered']]

                if len(seq) != len(layer0_ids):
                    continue

                current_idx = 0
                for base, group in groupby(seq):
                    group_list = list(group)
                    length = len(group_list)

                    if length >= args.min_repeat:
                        # 逻辑：去掉首尾，取中间对应的 token
                        start_idx = current_idx + 1
                        end_idx = current_idx + length - 1
                        target_tokens = layer0_ids[start_idx : end_idx]

                        if target_tokens:
                            # 提取向量并计算均值
                            vectors = [vector_map[tid] for tid in target_tokens]
                            mean_vec = np.mean(vectors, axis=0)

                            row = {
                                'id': global_id,
                                'base_pattern': base * length,
                                'token_id_list': "-".join(map(str, target_tokens)),
                                'dim0': float(mean_vec[0]),
                                'dim1': float(mean_vec[1]),
                                'dim2': float(mean_vec[2]),
                                'dim3': float(mean_vec[3])
                            }

                            final_data.append(row)
                            global_id += 1

                    current_idx += length

            except Exception as e:
                continue

    # 3. 导出 CSV
    if final_data:
        print(f"正在写入 CSV: {args.output} ...")
        df = pd.DataFrame(final_data)
        cols = ['id', 'base_pattern', 'token_id_list', 'dim0', 'dim1', 'dim2', 'dim3']
        df.to_csv(args.output, index=False, columns=cols)
        print(f"处理完成！共提取 {len(df)} 条特征。")


def main():
    parser = argparse.ArgumentParser(description="Nanopore Homopolymer Token Vector Extractor")

    # 文件路径参数
    parser.add_argument('--input', type=str, required=True, help='Input .jsonl.gz file path')
    parser.add_argument('--output', type=str, default='output_features.csv', help='Output .csv file path')

    # 算法参数
    parser.add_argument('--min_repeat', type=int, default=3, help='Minimum repeat count')
    parser.add_argument('--levels', type=int, nargs='+', default=[5, 5, 5, 5], help='Quantization levels')
    
    # --- 核心暴露参数 ---
    parser.add_argument('--num_quantizers', type=int, default=1, 
                        help='Number of quantizer layers (num_quantizers in RSQ)')

    args = parser.parse_args()
    process_nanopore_data(args)


if __name__ == "__main__":
    main()
