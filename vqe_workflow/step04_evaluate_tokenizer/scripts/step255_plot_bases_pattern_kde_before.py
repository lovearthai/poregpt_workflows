import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_dimensions_hist(args):
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

    # 3. 绘图展示 (2行2列布局)
    feature_cols = ['dim0', 'dim1', 'dim2', 'dim3']
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=100)
    fig.suptitle(f"Histogram & KDE Distribution per Dimension\nFile: {file_basename}", fontsize=18)

    # 使用高对比度色板
    palette = sns.color_palette("bright", len(categories))

    print(f"正在绘制 {len(categories)} 个类别的柱状图+KDE...")

    for i, col_name in enumerate(feature_cols):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # 使用 sns.histplot 叠加 KDE
        # element="step" 可以让柱状图不互相遮挡，看起来更像轮廓
        sns.histplot(
            data=df,
            x=col_name,
            hue='category',
            hue_order=categories,
            kde=True,
            element="step", 
            palette=palette,
            alpha=0.1,         # 减弱柱子填充颜色
            linewidth=2,       # 加粗线条
            ax=ax,
            common_norm=False  # 每个类别的密度独立归一化，防止样本少的类别被“压扁”
        )
        
        ax.set_title(f"Distribution of {col_name}", fontsize=14)
        ax.set_xlabel("Dimension Value", fontsize=12)
        ax.set_ylabel("Count / Density", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.5)

        # 调整图例位置：只在右上角子图显示总图例
        if i != 1:
            ax.get_legend().remove() if ax.get_legend() else None
        else:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), title='Category')

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(args.output)
    print(f"分析完成，混合分布图已保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanopore Feature Hist+KDE Distribution")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_points', type=int, default=100000)
    parser.add_argument('--max_categories', type=int, default=8)
    parser.add_argument('--required_base_patterns', type=str, default=None)

    args = parser.parse_args()
    plot_dimensions_hist(args)
