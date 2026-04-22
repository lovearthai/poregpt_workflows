import pandas as pd
import numpy as np
import argparse
import os
from scipy.spatial.distance import pdist, squareform, cdist

def find_most_distant_subset(args):
    # 1. 检查并加载数据
    if not os.path.exists(args.input):
        print(f"❌ 错误: 找不到输入文件 {args.input}")
        return

    print(f"正在读取数据: {args.input}")
    df = pd.read_csv(args.input)

    # --- 核心修改：适配 feature 字符串列提取维度 ---
    if 'feature' not in df.columns:
        print("❌ 错误: CSV 中缺少 'feature' 列")
        return

    print("正在解析 feature 字段...")
    # 拆分字符串并转换为浮点数
    features_split = df['feature'].str.split('_', expand=True).astype(float)
    
    # 动态生成维度列名并赋值给 df (dim0, dim1, dim2, dim3...)
    dim_cols = [f'dim{i}' for i in range(features_split.shape[1])]
    df[dim_cols] = features_split
    # --------------------------------------------

    # 3. 按 category 聚合，计算质心 (保持原样，使用动态生成的 dim_cols)
    print(f"正在聚合类别特征... (共有 {len(df['category'].unique())} 个类别)")
    category_means = df.groupby('category')[dim_cols].mean()
    cat_names = category_means.index.tolist()
    cat_values = category_means.values

    if len(cat_names) < args.top_n:
        print(f"⚠️ 警告: 类别总数({len(cat_names)})少于请求的数量({args.top_n})")
        args.top_n = len(cat_names)

    # 4. 贪婪算法寻找互相距离最大的子集 (保持原样)
    dist_matrix = squareform(pdist(cat_values, metric='euclidean'))
    idx1, idx2 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    
    selected_indices = [idx1, idx2]

    for _ in range(args.top_n - 2):
        remaining_indices = [i for i in range(len(cat_names)) if i not in selected_indices]
        sub_dist_matrix = cdist(cat_values[remaining_indices], cat_values[selected_indices], metric='euclidean')
        min_dists = np.min(sub_dist_matrix, axis=1)
        next_idx_in_remaining = np.argmax(min_dists)
        selected_indices.append(remaining_indices[next_idx_in_remaining])

    # 5. 输出结果 (保持原样)
    print("\n" + "="*60)
    print(f"🏆 相互距离最大化的前 {args.top_n} 个类别 (Max-Min 离散策略):")
    print("="*60)
    
    result_cats = [cat_names[i] for i in selected_indices]
    
    print(f"\n字符串格式 (可以直接复制):")
    print(f"\"{' '.join(result_cats)}\"")
    
    print(f"\n详细坐标与平均距离:")
    for i, idx in enumerate(selected_indices, 1):
        cat_name = cat_names[idx]
        vec = cat_values[idx]
        other_indices = [j for j in selected_indices if j != idx]
        if other_indices:
            avg_dist = np.mean(cdist([vec], cat_values[other_indices], metric='euclidean'))
            # 动态适配坐标打印的长度
            coord_str = ", ".join([f"{v:.3f}" for v in vec])
            print(f"   {i}. {cat_name:10} | 坐标: [{coord_str}] | 平均间隔距离: {avg_dist:.6f}")
        else:
            coord_str = ", ".join([f"{v:.3f}" for v in vec])
            print(f"   {i}. {cat_name:10} | 坐标: [{coord_str}]")

    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Find a subset of categories with maximum mutual distance.")
    parser.add_argument('--input', '-i', type=str, required=True, help='输入的特征 CSV 文件路径')
    parser.add_argument('--top_n', '-n', type=int, default=5, help='需要寻找的互相距离最大的类别数量 (默认: 5)')

    args = parser.parse_args()
    find_most_distant_subset(args)

if __name__ == "__main__":
    main()