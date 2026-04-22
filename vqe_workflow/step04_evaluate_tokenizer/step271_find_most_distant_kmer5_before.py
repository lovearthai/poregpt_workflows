import pandas as pd
import numpy as np
import argparse
import os
from itertools import product
from scipy.spatial.distance import pdist
from tqdm import tqdm

def find_best_mask_flexible(args):
    if not os.path.exists(args.input):
        print(f"❌ 错误: 找不到文件 {args.input}")
        return
    
    print(f"📖 正在读取并索引数据: {args.input}")
    df = pd.read_csv(args.input)
    features = ['dim0', 'dim1', 'dim2', 'dim3']

    cat_groups = {cat: group[features].values for cat, group in df.groupby('category')}
    
    bases = ['A', 'G', 'C', 'T']
    context_pairs = ["".join(p) for p in product(bases, repeat=2)]
    all_masks = [(c, c) for c in context_pairs] if args.symmetric else list(product(context_pairs, repeat=2))
    
    results = []

    print(f"🧪 正在评估 {len(all_masks)} 种模式 (要求至少 2 个碱基存在)...")
    for left, right in tqdm(all_masks, desc="Evaluating"):
        pattern = f"{left}X{right}"
        group_vectors = []
        found_bases = []
        counts = []

        for b in bases:
            full_kmer = f"{left}{b}{right}"
            if full_kmer in cat_groups:
                vecs = cat_groups[full_kmer]
                if len(vecs) >= args.min_samples:
                    group_vectors.append(np.mean(vecs, axis=0))
                    found_bases.append(b)
                    counts.append(len(vecs))
        
        # 核心修改：只要 >= 2 个碱基存在即可
        if len(group_vectors) < 2:
            continue

        means = np.array(group_vectors)
        distances = pdist(means, metric='euclidean')
        
        results.append({
            'pattern': pattern,
            'avg_dist': np.mean(distances), # 平均区分度
            'total_dist': np.sum(distances),
            'found_count': len(found_bases),
            'found_str': "".join(found_bases),
            'dist_list': [round(d, 4) for d in distances]
        })

    if not results:
        print("\n⚠️ 未找到满足条件的模式。")
        return

    # 按平均距离排序（衡量区分度最科学）
    results.sort(key=lambda x: x['avg_dist'], reverse=True)

    print("\n" + "="*100)
    print(f"{'Rank':<5} | {'Pattern':<8} | {'Bases':<6} | {'Avg Dist':<10} | {'Pairwise Distances'}")
    print("-" * 100)
    
    for i, res in enumerate(results[:args.top_n]):
        dist_str = ", ".join(f"{d:.3f}" for d in res['dist_list'])
        print(f"{i+1:<5} | {res['pattern']:<8} | {res['found_str']:<6} | {res['avg_dist']:<10.4f} | {dist_str}")
    print("="*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('--symmetric', action='store_true')
    parser.add_argument('--min_samples', type=int, default=3)
    parser.add_argument('--top_n', type=int, default=20)
    args = parser.parse_args()
    find_best_mask_flexible(args)
