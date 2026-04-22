import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.cluster import KMeans
from umap import UMAP

def plot_with_clustering(args):
    # 1. 检查并加载数据
    if not os.path.exists(args.input):
        print(f"错误: 找不到文件 {args.input}")
        return

    print(f"读取数据: {args.input}")
    df = pd.read_csv(args.input)
    file_basename = os.path.basename(args.input)

    # 采样以加速计算和绘图渲染
    if len(df) > args.max_points:
        print(f"数据量大，随机采样至 {args.max_points} 行...")
        df = df.sample(n=args.max_points, random_state=42)

    # 2. 提取特征 (直接使用原始标准化特征)
    feature_cols = ['dim0', 'dim1', 'dim2', 'dim3']
    if not all(col in df.columns for col in feature_cols):
        print(f"错误: CSV中缺少特征列 {feature_cols}")
        return
    features = df[feature_cols].values

    # 3. 聚类 (K-Means)
    print(f"正在执行 K-Means 聚类 (K={args.n_clusters})...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features)

    # 4. 降维 (UMAP)
    print("正在使用 UMAP 降维至 2D...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features)
    df['x'], df['y'] = embedding[:, 0], embedding[:, 1]

    # 5. 绘图展示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=150)
    
    # 统一的标题信息：仅保留文件名、方法和聚类数
    info_str = f"File: {file_basename} | Method: UMAP | K={args.n_clusters}"

    # --- 左图：Ground Truth (按 A/C/G/T 着色) ---
    color_map = {'A': '#1f77b4', 'C': '#ff7f0e', 'G': '#2ca02c', 'T': '#d62728'}
    df['base'] = df['base_pattern'].str[0] # 取 homopolymer 的第一个碱基

    for base in ['A', 'C', 'G', 'T']:
        mask = df['base'] == base
        if mask.any():
            ax1.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], 
                        label=base, alpha=0.5, s=3, c=color_map.get(base))
    
    ax1.set_title(f"Base Pattern Labels\n{info_str}", fontsize=11)
    ax1.legend(markerscale=6, loc='upper right')

    # --- 右图：Unsupervised Clustering (按聚类 ID 着色) ---
    scatter2 = ax2.scatter(df['x'], df['y'], c=df['cluster'], 
                           cmap='tab10', alpha=0.5, s=3)
    ax2.set_title(f"K-Means Clustering Results\n{info_str}", fontsize=11)
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID')

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"可视化分析完成，结果已保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanopore Feature Visualization & Clustering")
    
    # 核心输入输出参数
    parser.add_argument('--input', type=str, required=True, help='输入特征CSV文件')
    parser.add_argument('--output', type=str, required=True, help='保存图片的路径')
    
    # 绘图控制参数
    parser.add_argument('--n_clusters', type=int, default=4, help='K-Means聚类数量')
    parser.add_argument('--max_points', type=int, default=100000, help='最大采样点数')

    args = parser.parse_args()
    plot_with_clustering(args)
