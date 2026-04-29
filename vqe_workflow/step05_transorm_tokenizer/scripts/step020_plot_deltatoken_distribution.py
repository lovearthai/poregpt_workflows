import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="PoreGPT BaseCount vs TokenDiff Correlation Plotter")
    parser.add_argument("--csv_path", type=str, required=True, help="统计CSV文件的路径")
    parser.add_argument("--output_png", type=str, default="base_vs_diff_correlation.png", help="输出图片路径")
    parser.add_argument("--cols", type=int, default=4, help="每行显示的子图数量")
    parser.add_argument("--max_dims", type=int, default=None, help="限制绘制的前几维数量")
    # 新增参数：限制绘图行数
    parser.add_argument("--max_plot_lines", type=int, default=None, help="随机采样绘制的最大行数 (如 20000)")

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: 找不到文件: {args.csv_path}")
        return

    print(f"正在加载数据: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    total_rows = len(df)

    # 1. 随机采样逻辑
    if args.max_plot_lines is not None and args.max_plot_lines < total_rows:
        print(f"数据总量 {total_rows} 较大，正在随机采样 {args.max_plot_lines} 行以加速绘图...")
        # random_state 保证每次绘图结果一致
        df = df.sample(n=args.max_plot_lines, random_state=42).reset_index(drop=True)
    else:
        print(f"使用全量数据进行绘图 (共 {total_rows} 行)。")

    # 2. 解析 diff_distribution
    # 使用 str.split 展开并转换为 float
    diff_expanded = df['diff_distribution'].str.split('_', expand=True).astype(float)

    # 3. 确定要绘制的维度
    all_diff_cols = [f'Dist_{i}' for i in range(diff_expanded.shape[1])]

    if args.max_dims is not None:
        num_to_plot = min(args.max_dims, len(all_diff_cols))
        diff_cols = all_diff_cols[:num_to_plot]
        diff_expanded = diff_expanded.iloc[:, :num_to_plot]
    else:
        num_to_plot = len(all_diff_cols)
        diff_cols = all_diff_cols

    diff_expanded.columns = diff_cols
    plot_df = pd.concat([df['base_count'], diff_expanded], axis=1)

    # 4. 计算相关系数 (基于采样后的数据)
    print("正在计算相关系数...")
    correlations = plot_df.corr()['base_count'].drop('base_count')

    # 5. 绘图
    cols = args.cols
    rows = (num_to_plot + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)

    if num_to_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    print(f"正在渲染 {num_to_plot} 个子图...")
    for i, col_name in enumerate(diff_cols):
        sns.regplot(
            data=plot_df,
            x='base_count',
            y=col_name,
            ax=axes[i],
            # 优化点：减小点的大小并提高透明度，方便看清趋势
            scatter_kws={'alpha': 0.15, 's': 6, 'color': '#3498db'},
            line_kws={'color': '#e74c3c', 'lw': 2}
        )

        r_val = correlations[col_name]
        axes[i].set_title(f'Manhattan Dist: {i}\nCorr: {r_val:.4f}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Base Count')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Correlation Analysis: Base Count vs Token Manhattan Distance Distribution\n(Sampled {len(plot_df)} lines from {os.path.basename(args.csv_path)})',
                 fontsize=16, y=1.02)

    # 6. 保存
    plt.savefig(args.output_png, dpi=300, bbox_inches='tight')
    print(f"成功！图表已保存至: {args.output_png}")

if __name__ == "__main__":
    main()
