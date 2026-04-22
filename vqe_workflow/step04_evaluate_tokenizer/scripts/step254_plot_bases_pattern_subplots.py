import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from itertools import combinations

def plot_dimensions_grid(args):
    # 1. 检查并加载数据
    if not os.path.exists(args.input):
        print(f"错误: 找不到文件 {args.input}")
        return

    print(f"读取数据: {args.input}")
    df = pd.read_csv(args.input)
    file_basename = os.path.basename(args.input)

    # 2. 类别筛选逻辑
    all_categories = sorted(df['category'].unique())
    print(f"原始数据中共有 {len(all_categories)} 个类别。")

    required_cats = []
    if args.required_base_patterns:
        potential_required = args.required_base_patterns.split()
        required_cats = [c for c in potential_required if c in all_categories]
        print(f"已锁定必须显示的类别: {required_cats}")

    if args.max_categories is not None and len(all_categories) > args.max_categories:
        num_to_sample = max(0, args.max_categories - len(required_cats))
        remaining_cats = [c for c in all_categories if c not in required_cats]
        sampled_cats = list(np.random.choice(remaining_cats, size=min(num_to_sample, len(remaining_cats)), replace=False))
        selected_categories = required_cats + sampled_cats
    else:
        selected_categories = all_categories

    df = df[df['category'].isin(selected_categories)]
    categories = sorted(selected_categories)

    # 数据量限制
    if len(df) > args.max_points:
        print(f"数据量大，采样至 {args.max_points} 行...")
        df = df.sample(n=args.max_points, random_state=42)

    # 3. 核心修改：直接从 feature 字段提取 vector (NumPy Array)
    if 'feature' not in df.columns:
        print("错误: CSV中缺少 'feature' 列")
        return

    print("正在将 feature 转换为向量...")
    # 转换为 numpy array 列表
    df['vec'] = df['feature'].apply(lambda x: np.fromstring(x.replace('_', ' '), sep=' '))
    
    # 获取第一个向量来确定实际维度
    sample_vec = df['vec'].iloc[0]
    num_dims = len(sample_vec)
    print(f"探测到特征维度: {num_dims}")

    # 4. 动态布局与绘图
    dim_pairs = list(combinations(range(num_dims), 2))
    num_plots = len(dim_pairs)
    
    if num_plots == 0:
        print("维度不足，无法进行组合绘图。")
        return

    cols = 2
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows), dpi=100)
    fig.suptitle(f"Pairwise Vector Distribution (Dims: {num_dims})\nFile: {file_basename}", fontsize=16)

    # 扁平化 axes 方便循环
    axes_flat = axes.flatten() if num_plots > 1 else [axes]

    # 配色方案
    high_contrast_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#999999']
    cmap_tab20 = plt.get_cmap('tab20')
    all_colors = high_contrast_colors + [cmap_tab20(i) for i in range(20)]
    colors = [all_colors[i % len(all_colors)] for i in range(len(categories))]

    # 5. 绘图循环
    for idx, (dim_x, dim_y) in enumerate(dim_pairs):
        ax = axes_flat[idx]
        
        for i, cat in enumerate(categories):
            mask = df['category'] == cat
            cat_data = df[mask]
            
            if len(cat_data) > 0:
                # 直接从 vec 列中通过索引提取 x 和 y
                # 使用 np.stack 将 Series 中的 array 拼成矩阵 [N, Dims]
                coords = np.stack(cat_data['vec'].values)
                ax.scatter(coords[:, dim_x], coords[:, dim_y],
                           label=cat, alpha=0.4, s=120, color=colors[i])

        ax.set_xlabel(f"Vector Index [{dim_x}]", fontsize=10)
        ax.set_ylabel(f"Vector Index [{dim_y}]", fontsize=10)
        ax.set_title(f"Dim {dim_x} vs Dim {dim_y}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 只有第一列或者最后一个子图显示图例，防止遮挡
        if idx % cols == cols - 1 or idx == num_plots - 1:
            ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # 隐藏多余子图
    for j in range(num_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(args.output)
    print(f"✅ 可视化分析完成，保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanopore Vector Visualization")
    parser.add_argument('--input', type=str, required=True, help='输入特征CSV文件')
    parser.add_argument('--output', type=str, required=True, help='保存图片的路径')
    parser.add_argument('--max_points', type=int, default=50000)
    parser.add_argument('--max_categories', type=int, default=None)
    parser.add_argument('--required_base_patterns', type=str, default=None,
                        help='必须包含的类别，空格分隔')

    args = parser.parse_args()
    plot_dimensions_grid(args)
