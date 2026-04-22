import argparse
import torch
import numpy as np
import pandas as pd
from umap import UMAP
import sys
import os
import math
from itertools import product

# 确保能找到 poregpt 模块
sys.path.append(os.getcwd())

try:
    from poregpt.utils import get_rsq_vector_from_integer
except ImportError:
    print("❌ 错误：无法导入 get_rsq_vector_from_integer。请确保在项目根目录运行。")
    sys.exit(1)

def generate_full_token_vectors(levels, num_q):
    """
    全量生成多层 ResidualFSQ 的数据，同时保留 ID、Code 和高维向量。
    """
    single_layer_capacity = math.prod(levels)
    total_tokens = single_layer_capacity ** num_q

    print(f"🧬 配置: Levels={levels}, Num_Q={num_q}")
    print(f"🔢 总计 Token 数: {total_tokens}")

    # 1. 预计算单层：坐标组合 -> (单层ID, 坐标字符串)
    single_layer_coords = list(product(*[range(l) for l in levels]))

    s_layer_data = []
    for coords in single_layer_coords:
        sid, stride = 0, 1
        for i, val in enumerate(coords):
            sid += val * stride
            stride *= levels[i]

        coord_str = "".join(map(str, coords))
        s_layer_data.append((sid, coord_str))

    # 2. 跨层全量组合
    all_combos = product(s_layer_data, repeat=num_q)

    token_ids = []
    layer_ids_storage = [[] for _ in range(num_q)]
    layer_codes_storage = [[] for _ in range(num_q)]
    vectors = []

    print("🚀 开始全量计算向量、分层 ID 与编码...")
    with torch.no_grad():
        for count, combo in enumerate(all_combos):
            final_token_id = 0
            for i, (layer_id, layer_str) in enumerate(combo):
                power = num_q - 1 - i
                final_token_id += layer_id * (single_layer_capacity ** power)
                layer_ids_storage[i].append(layer_id)
                layer_codes_storage[i].append(layer_str)

            token_ids.append(final_token_id)

            # 获取向量并展平
            vec = get_rsq_vector_from_integer(final_token_id, levels, num_q)
            vectors.append(vec.detach().cpu().numpy().flatten())

            if (count + 1) % 20000 == 0:
                print(f"进度: {count + 1} / {total_tokens}")

    return np.array(token_ids), layer_ids_storage, layer_codes_storage, np.stack(vectors)

def process_to_csv(output_csv, levels, num_q):
    token_ids, l_ids, l_codes, X = generate_full_token_vectors(levels, num_q)

    # 1. UMAP 降维
    print(f"🧪 正在进行全量 UMAP 降维 (点数: {len(token_ids)})...")
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=-1)
    embedding = reducer.fit_transform(X)

    # 2. 构造结果集
    print(f"💾 正在构造 DataFrame...")
    data = {
        'token_id': token_ids,
        'x': embedding[:, 0],
        'y': embedding[:, 1]
    }

    # 动态增加 layerX_id 和 layerX_code 列
    for i in range(num_q):
        data[f'layer{i}_id'] = l_ids[i]
        data[f'layer{i}_code'] = l_codes[i]

    # --- 核心修改：动态增加向量维度列 (dim0, dim1, ...) ---
    num_dims = X.shape[1]
    for d in range(num_dims):
        data[f'dim{d}'] = X[:, d]

    df = pd.DataFrame(data)

    print(f"💾 正在保存至: {output_csv}")
    # 工业级实践：对于 39 万行且列数较多的 CSV，建议使用更高的精度或检查文件大小
    df.to_csv(output_csv, index=False,float_format='%.6f')
    print(f"✅ 全量处理完成！文件包含 UMAP 坐标及 {num_dims} 维原始向量。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='token_embedding_complete_map.csv')
    parser.add_argument('--levels', type=int, nargs='+', default=[5, 5, 5, 5])
    parser.add_argument('--num_q', type=int, default=2)

    args = parser.parse_args()
    process_to_csv(args.output, args.levels, args.num_q)
