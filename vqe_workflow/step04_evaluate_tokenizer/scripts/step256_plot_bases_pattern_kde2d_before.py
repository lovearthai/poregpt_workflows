import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from itertools import combinations

def plot_dimensions_2d_kde(args):
    # 1. 检查并加载数据
    if not os.path.exists(args.input):
        print(f"错误: 找不到文件 {args.input}")
        return

    print(f"读取数据: {args.input}")
    df = pd.read_csv(args.input)
    file_basename = os.path.basename(args.input)

    # 2. 类别筛选逻辑
    all_categories = sorted(df['category'].unique())
    
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

    # 采样逻辑（KDE 计算较慢，点数不宜过多）
    if len(df) > args.max_points:
        print(f"采样至 {args.max_points} 行以加速 KDE 计算...")
        df = df.sample(n=args.max_points, random_state=42)

    # 3. 绘图展示 (3行2列布局，对应 4 维度的两两组合)
    dim_pairs = list(combinations(range(4), 2))
    fig, axes = plt.subplots(3, 2, figsize=(22, 25), dpi=100)
    fig.suptitle(f"2D KDE Density Contours per Dimension Pair\nFile: {file_basename}", fontsize=18)

    # 配色方案
    palette = sns.color_palette("bright", len(categories))
    
    print(f"正在绘制 {len(dim_pairs)} 个二维 KDE 子图...")

    for idx, (dim_x, dim_y) in enumerate(dim_pairs):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        x_col, y_col = f'dim{dim_x}', f'dim{dim_y}'

        # 绘制二维 KDE 等高线
        sns.kdeplot(
            data=df,
            x=x_col,
            y=y_col,
            hue='category',
            hue_order=categories,
            ax=ax,
            palette=palette,
            fill=True,      # 填充等高线内部
            alpha=0.3,      # 透明度，方便观察重叠
            levels=5,       # 等高线层数，不宜过多否则太乱
            thresh=0.1,     # 过滤掉最外围的稀疏区域
            common_norm=False
        )

        ax.set_xlabel(f"Dimension {dim_x}", fontsize=12)
        ax.set_ylabel(f"Dimension {dim_y}", fontsize=12)
        ax.set_title(f"Dim {dim_x} vs Dim {dim_y} (Density)", fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.5)

        # 统一处理图例
        if idx == 1:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), title='Category')
        else:
            if ax.get_legend(): ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(args.output)
    print(f"二维 KDE 分析完成，结果已保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanopore 2D KDE Feature Visualization")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_points', type=int, default=10000, help='KDE建议点数设小一点，如10000')
    parser.add_argument('--max_categories', type=int, default=8)
    parser.add_argument('--required_base_patterns', type=str, default=None)

    args = parser.parse_args()
    plot_dimensions_2d_kde(args)
