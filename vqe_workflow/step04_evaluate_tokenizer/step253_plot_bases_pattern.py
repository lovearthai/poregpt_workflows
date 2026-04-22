import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from umap import UMAP

def plot_distinct_colors(args):
    # 1. 检查并加载数据
    if not os.path.exists(args.input):
        print(f"错误: 找不到文件 {args.input}")
        return

    print(f"读取数据: {args.input}")
    df = pd.read_csv(args.input)
    file_basename = os.path.basename(args.input)

    # 采样以加速计算
    if len(df) > args.max_points:
        print(f"数据量大，随机采样至 {args.max_points} 行...")
        df = df.sample(n=args.max_points, random_state=42)

    # --- 修改部分：从 feature 字段提取特征矩阵 ---
    if 'feature' not in df.columns:
        print("错误: CSV中缺少 'feature' 列")
        return

    print("正在解析 feature 字符串字段...")
    # 将 "0.1_0.2_0.3_0.4" 转换成 numpy 矩阵 (N, 4)
    try:
        features = np.array([
            [float(x) for x in f_str.split('_')] 
            for f_str in df['feature']
        ])
    except Exception as e:
        print(f"解析特征出错: {e}")
        return
    # ------------------------------------------

    # 3. 降维 (UMAP)
    print(f"正在使用 UMAP 降维至 2D (输入维度: {features.shape[1]})...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features)
    df['x'], df['y'] = embedding[:, 0], embedding[:, 1]

    # 4. 核心逻辑：类别筛选与排序
    all_categories = sorted(df['category'].unique().tolist())
    print(f"原始数据中共有 {len(all_categories)} 个类别。")

    required_cats = []
    if args.required_base_patterns:
        potential_required = args.required_base_patterns.split()
        required_cats = [c for c in potential_required if c in all_categories]
        missing_cats = [c for c in potential_required if c not in all_categories]
        if missing_cats:
            print(f"警告: 以下指定类别在数据中未找到: {missing_cats}")
        print(f"已锁定核心类别 ({len(required_cats)}个): {required_cats}")

    if args.max_categories is not None and len(all_categories) > args.max_categories:
        num_to_sample = max(0, args.max_categories - len(required_cats))
        remaining_cats = sorted([c for c in all_categories if c not in required_cats])
        if num_to_sample > 0 and len(remaining_cats) > 0:
            sampled_cats = list(np.random.choice(remaining_cats, size=min(num_to_sample, len(remaining_cats)), replace=False))
            final_categories = required_cats + sorted(sampled_cats)
        else:
            final_categories = required_cats
    else:
        remaining_cats = sorted([c for c in all_categories if c not in required_cats])
        final_categories = required_cats + remaining_cats

    df = df[df['category'].isin(final_categories)]
    print(f"最终绘制类别数: {len(final_categories)}，总点数: {len(df)}")

    # 5. 配色策略
    distinct_cmap = plt.get_cmap('Set1')
    category_colors = {}
    for i, cat in enumerate(final_categories):
        if i < 9:
            category_colors[cat] = distinct_cmap(i)
        else:
            base_color = plt.get_cmap('tab20')(i % 20)
            category_colors[cat] = (base_color[0], base_color[1], base_color[2], 0.3)

    # 6. 绘图展示
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=150)
    info_str = f"File: {file_basename} | Displayed: {len(final_categories)}"

    print(f"正在绘制散点图...")
    # 先画背景类 (9 以后)
    for cat in final_categories[9:]:
        mask = df['category'] == cat
        ax.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'],
                    label=cat, color=category_colors[cat], s=15, edgecolors='none')

    # 再画核心类 (前 9 个)
    for cat in final_categories[:9]:
        mask = df['category'] == cat
        ax.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'],
                    label=cat, color=category_colors[cat], s=50, alpha=0.7, edgecolors='none')

    ax.set_title(f"UMAP Pattern Visualization (SSEM Feature)\n{info_str}", fontsize=15)
    ax.set_xlabel("UMAP_1", fontsize=12)
    ax.set_ylabel("UMAP_2", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.2)

    # 图例优化
    handles, labels = ax.get_legend_handles_labels()
    order = {cat: i for i, cat in enumerate(final_categories)}
    # 过滤掉不在列表里的 labels 并排序
    sorted_indices = sorted([i for i, l in enumerate(labels) if l in order], key=lambda k: order[labels[k]])
    
    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    ax.legend(sorted_handles[:12], sorted_labels[:12],
              markerscale=2.0, loc='upper right', fontsize='small',
              title="Core Patterns", frameon=True, shadow=True, facecolor='white', framealpha=0.8)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"可视化分析完成，结果已保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanopore UMAP (From Merged Feature Column)")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_points', type=int, default=50000)
    parser.add_argument('--max_categories', type=int, default=30)
    parser.add_argument('--required_base_patterns', type=str, default=None)

    args = parser.parse_args()
    plot_distinct_colors(args)
