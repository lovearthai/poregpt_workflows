import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # 服务器后端
import matplotlib.pyplot as plt
import argparse
import os

def analyze_kmer_distribution(csv_path, kmer_len, output_path, bin_num=1000):
    os.makedirs(output_path, exist_ok=True)

    # 1. 加载数据
    print(f"Reading {csv_path}...")
    # 假设 csv 中包含每一行是一个 kmer pattern 及其 count
    df = pd.read_csv(csv_path, usecols=['count'], dtype={'count': np.int64})
    counts = df['count'].values
    
    # --- 数据分箱/预处理逻辑 ---
    # 1. 排序：为了保持 Rank-Abundance 的趋势，必须先排序（从大到小）
    counts_sorted = np.sort(counts)[::-1] 
    num_patterns = len(counts_sorted)

    # 2. 判断是否需要分箱
    # 如果 bin_num 为 0，或者 设定的 bin_num 大于 实际数据量，则不分箱（全量绘制）
    if bin_num == 0 or bin_num >= num_patterns:
        print(f"Data size ({num_patterns}) <= bin_num ({bin_num}). Plotting all patterns directly.")
        binned_counts = counts_sorted
        actual_bins = num_patterns
    else:
        print(f"Binning {num_patterns} patterns into {bin_num} groups...")
        # 分割：将数组分割成 bin_num 个部分
        bins = np.array_split(counts_sorted, bin_num)
        # 聚合：计算每个箱子的平均值
        binned_counts = np.array([b.mean() for b in bins])
        actual_bins = bin_num

    # 更新统计变量用于绘图
    num_observed_kmers = len(binned_counts) 
    total_kmers_sum = counts.sum()          # 总和仍使用原始数据计算
    mu = total_kmers_sum / (4 ** kmer_len)  # 期望值

    # 3. 确定布局：2行1列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3) 
    x_rank = range(num_observed_kmers)

    # --- 子图 1：Log Scale ---
    ax1.bar(x_rank, binned_counts, alpha=0.7, color='#1f77b4')
    ax1.axhline(y=mu, color='#d62728', linestyle='--', label=f'Poisson Exp ($\lambda$={mu:.1f})')
    ax1.set_yscale('log')
    # 标题中显示实际使用的柱子数量
    ax1.set_title(f'Rank-Abundance (Log Scale, N={actual_bins})', fontsize=13)
    ax1.set_ylabel('Count (Log Scale)', fontsize=11)
    ax1.grid(True, which="both", ls="-", alpha=0.15)
    ax1.legend()

    # --- 子图 2：Linear Scale (Y-limited) ---
    ax2.bar(x_rank, binned_counts, alpha=0.7, color='#2ca02c')
    ax2.axhline(y=mu, color='#d62728', linestyle='--')
    ax2.set_ylim(0, mu * 5) 
    ax2.set_title(f'Rank-Abundance (Linear Scale, N={actual_bins})', fontsize=13)
    ax2.set_xlabel('Pattern Rank', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.grid(True, ls="-", alpha=0.15)

    # 保存结果
    output_png = os.path.join(output_path, "kmer_analysis.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--output_dir", type=str, required=True)
    # 默认 1000，如果设为 0 则强制全量绘制（如果数据量允许）
    parser.add_argument("--bin_num", type=int, default=1000, help="Number of bins to merge (0 for no binning)")

    args = parser.parse_args()

    analyze_kmer_distribution(args.input, args.k, args.output_dir, bin_num=args.bin_num)
