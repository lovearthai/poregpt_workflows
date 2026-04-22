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

    # --- 核心修改：解析 feature 字段 ---
    if 'feature' not in df.columns:
        print("错误: CSV中缺少 'feature' 列")
        return

    print("正在解析 feature 字段为独立维度...")
    # 拆分字符串并转换为浮点数
    features_split = df['feature'].str.split('_', expand=True).astype(float)
    num_actual_dims = features_split.shape[1]
    
    # 动态创建列名 dim0, dim1, ...
    feature_cols = [f'dim{i}' for i in range(num_actual_dims)]
    features_split.columns = feature_cols
    
    # 合并回原 DataFrame
    df = pd.concat([df, features_split], axis=1)
    # --------------------------------

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
        if len(remaining_cats) > 0:
            sampled_cats = list(np.random.choice(remaining_cats, size=min(num_to_sample, len(remaining_cats)), replace=False))
        else:
            sampled_cats = []
        selected_categories = required_cats + sampled_cats
    else:
        selected_categories = all_categories

    df = df[df['category'].isin(selected_categories)]
    categories = sorted(selected_categories)

    # 采样逻辑（2D KDE 计算非常耗时，点数超过 5000 就会明显变慢）
    if len(df) > args.max_points:
        print(f"采样至 {args.max_points} 行以加速 KDE 计算...")
        df = df.sample(n=args.max_points, random_state=42)

    # 3. 绘图展示 (基于实际探测到的维度进行两两组合)
    # 默认针对前 4 个维度做组合，如果不足 4 维则按实际维度做组合
    plot_dims = min(4, num_actual_dims)
    dim_pairs = list(combinations(range(plot_dims), 2))
    
    # 计算布局
    num_plots = len(dim_pairs)
    ncols = 2
    nrows = (num_plots + 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 8 * nrows), dpi=100)
    fig.suptitle(f"2D KDE Density Contours per Dimension Pair\nFile: {file_basename}", fontsize=18)
    
    # 如果只有一行，axes 会是一维数组，需要统一处理成二维
    if nrows == 1:
        axes = np.array([axes])

    palette = sns.color_palette("bright", len(categories))
    print(f"正在绘制 {num_plots} 个二维 KDE 子图...")

    for idx, (dim_x, dim_y) in enumerate(dim_pairs):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        x_col, y_col = f'dim{dim_x}', f'dim{dim_y}'

        sns.kdeplot(
            data=df,
            x=x_col,
            y=y_col,
            hue='category',
            hue_order=categories,
            ax=ax,
            palette=palette,
            fill=True,
            alpha=0.3,
            levels=5,
            thresh=0.1,
            common_norm=False
        )

        ax.set_xlabel(f"Dimension {dim_x}", fontsize=12)
        ax.set_ylabel(f"Dimension {dim_y}", fontsize=12)
        ax.set_title(f"Dim {dim_x} vs Dim {dim_y} (Density)", fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.5)

        # 仅在特定的图上保留图例
        if idx == 1:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), title='Category')
        else:
            if ax.get_legend(): ax.get_legend().remove()

    # 隐藏多余的子图（如果有）
    for i in range(idx + 1, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(args.output)
    print(f"二维 KDE 分析完成，结果已保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanopore 2D KDE Feature Visualization")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_points', type=int, default=5000, help='KDE建议点数不宜过大')
    parser.add_argument('--max_categories', type=int, default=8)
    parser.add_argument('--required_base_patterns', type=str, default=None)

    args = parser.parse_args()
    plot_dimensions_2d_kde(args)