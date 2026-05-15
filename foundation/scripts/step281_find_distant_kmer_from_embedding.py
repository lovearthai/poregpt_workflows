import pandas as pd
import numpy as np
import argparse
import os
from itertools import product
from scipy.spatial.distance import pdist
from tqdm import tqdm

def parse_embedding(s):
    """解析以 '-' 分隔的 embedding 字符串"""
    return np.fromstring(s, sep='-')

def find_best_mask_flexible(args):
    if not os.path.exists(args.input):
        print(f"❌ 错误: 找不到文件 {args.input}")
        return

    print(f"📖 正在读取并转换 Embedding 数据: {args.input}")
    # 1. 读取 CSV
    df = pd.read_csv(args.input)
    
    # 2. 将字符串 embedding 转换为 numpy 矩阵
    # 使用 tqdm 显示解析进度，因为高维向量解析较慢
    tqdm.pandas(desc="Parsing Embeddings")
    vec_list = np.stack(df['embedding'].progress_apply(parse_embedding).values)
    
    # 3. 创建临时 DataFrame 用于分组计算
    # 只保留 category 并关联解析后的向量
    group_df = pd.DataFrame({'category': df['category']})
    
    # 按 category 分组并计算均值向量
    print("聚类计算各 Category 的均值向量...")
    # 我们直接利用 numpy 的索引来加速，不直接在 DataFrame 里存矩阵
    unique_cats = group_df['category'].unique()
    cat_groups = {}
    for cat in tqdm(unique_cats, desc="Grouping"):
        idx = group_df[group_df['category'] == cat].index
        if len(idx) >= args.min_samples:
            cat_groups[cat] = np.mean(vec_list[idx], axis=0)

    # 4. 设置 K-mer 搜索模式
    bases = ['A', 'G', 'C', 'T']
    context_pairs = ["".join(p) for p in product(bases, repeat=2)]
    # 对称模式指左右侧序列一致
    all_masks = [(c, c) for c in context_pairs] if args.symmetric else list(product(context_pairs, repeat=2))

    results = []

    print(f"🧪 正在评估 {len(all_masks)} 种模式 (寻找区分度最高的中心碱基 X)...")
    for left, right in tqdm(all_masks, desc="Evaluating"):
        pattern = f"{left}X{right}"
        group_vectors = []
        found_bases = []

        # 检查该 Context 下 A, G, C, T 的均值向量是否存在
        for b in bases:
            full_kmer = f"{left}{b}{right}"
            if full_kmer in cat_groups:
                group_vectors.append(cat_groups[full_kmer])
                found_bases.append(b)

        # 只要存在至少 2 个碱基即可计算距离
        if len(group_vectors) < 2:
            continue

        means = np.array(group_vectors)
        # 计算两两之间的欧氏距离
        distances = pdist(means, metric='euclidean')

        results.append({
            'pattern': pattern,
            'avg_dist': np.mean(distances), 
            'total_dist': np.sum(distances),
            'found_count': len(found_bases),
            'found_str': "".join(found_bases),
            'dist_list': [round(d, 4) for d in distances]
        })

    if not results:
        print("\n⚠️ 未找到满足条件的模式。")
        return

    # 按平均距离（Avg Dist）排序，反映该 Context 对中心碱基的敏感度
    results.sort(key=lambda x: x['avg_dist'], reverse=True)

    # 5. 打印结果
    print("\n" + "="*110)
    print(f"{'Rank':<5} | {'Pattern':<8} | {'Bases':<6} | {'Avg Dist':<10} | {'Pairwise Distances (Center X Difference)'}")
    print("-" * 110)

    for i, res in enumerate(results[:args.top_n]):
        dist_str = ", ".join(f"{d:.3f}" for d in res['dist_list'])
        print(f"{i+1:<5} | {res['pattern']:<8} | {res['found_str']:<6} | {res['avg_dist']:<10.4f} | {dist_str}")
    print("="*110)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find most distant center-base patterns from high-dim embeddings")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input CSV file from step280")
    parser.add_argument('--symmetric', action='store_true', help="Only evaluate patterns where left == right context")
    parser.add_argument('--min_samples', type=int, default=3, help="Minimum occurrences of a kmer to consider it")
    parser.add_argument('--top_n', type=int, default=20, help="Number of results to display")
    args = parser.parse_args()
    find_best_mask_flexible(args)
