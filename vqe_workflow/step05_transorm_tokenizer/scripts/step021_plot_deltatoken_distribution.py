import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="PoreGPT BaseCount vs Cumulative TokenDiff Correlation Plotter")
    parser.add_argument("--csv_path", type=str, required=True, help="统计CSV文件的路径")
    parser.add_argument("--output_png", type=str, default="base_vs_cumulative_diff_correlation.png", help="输出图片路径")
    parser.add_argument("--cols", type=int, default=4, help="每行显示的子图数量")
    parser.add_argument("--max_dims", type=int, default=None, help="限制绘制的前几维累积和")
    # 新增参数：随机采样行数
    parser.add_argument("--max_plot_lines", type=int, default=None, help="随机采样绘制的最大行数 (例如 10000)")

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: 找不到文件: {args.csv_path}")
        return

    print(f"正在加载数据: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    total_rows = len(df)

    # 1. 核心修改：随机采样
    if args.max_plot_lines is not None and args.max_plot_lines < total_rows:
        print(f"检测到数据量较大 ({total_rows} 行)，随机采样 {args.max_plot_lines} 行进行绘图...")
        df = df.sample(n=args.max_plot_lines, random_state=42).reset_index(drop=True)
    else:
        print(f"使用全量数据进行绘图 (共 {total_rows} 行)。")

    # 2. 解析 diff_distribution 并计算后缀累积和
    # 获取原始分布值
    diff_values = df['diff_distribution'].str.split('_', expand=True).astype(float)
    
    # 计算后缀累积和: sum(Dist_i, ..., Dist_N)
    cumulative_diffs = diff_values.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    
    # 确定列名
    all_cols = [f'CumSum_Dist_ge_{i}' for i in range(cumulative_diffs.shape[1])]
    cumulative_diffs.columns = all_cols

    # 确定绘制范围
    if args.max_dims is not None:
        num_to_plot = min(args.max_dims, len(all_cols))
        plot_cols = all_cols[:num_to_plot]
        plot_data = cumulative_diffs.iloc[:, :num_to_plot]
    else:
        num_to_plot = len(all_cols)
        plot_cols = all_cols
        plot_data = cumulative_diffs

    # 合并数据
    final_df = pd.concat([df['base_count'], plot_data], axis=1)
    
    # 计算全量采样后的相关系数
    correlations = final_df.corr()['base_count'].drop('base_count')

    # 3. 绘图
    cols = args.cols
    rows = (num_to_plot + cols - 1) // cols

    # 动态调整画布大小
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    
    if num_to_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    print(f"开始渲染 {num_to_plot} 个子图...")
    for i, col_name in enumerate(plot_cols):
        # 绘图逻辑
        sns.regplot(
            data=final_df,
            x='base_count',
            y=col_name,
            ax=axes[i],
            # 增加透明度以应对采样后的点重叠
            scatter_kws={'alpha': 0.1, 's': 6, 'color': '#2ecc71'}, 
            line_kws={'color': '#e74c3c', 'lw': 2}
        )

        r_val = correlations[col_name]
        axes[i].set_title(f'Manhattan Dist $\geq$ {i}\nCorr: {r_val:.4f}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Base Count')
        axes[i].set_ylabel('Cumulative Freq')
        axes[i].grid(True, linestyle='--', alpha=0.4)

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Correlation: Base Count vs Cumulative Token Difference (Sampled {len(final_df)} lines)\nFile: {os.path.basename(args.csv_path)}',
                 fontsize=16, y=1.02)

    plt.savefig(args.output_png, dpi=300, bbox_inches='tight')
    print(f"绘图成功！结果已保存至: {args.output_png}")

if __name__ == "__main__":
    main()
