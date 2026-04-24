import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # 服务器后端
import matplotlib.pyplot as plt
import argparse
import os
import time

def analyze_kmer_distribution(csv_path, kmer_len, output_path, plot_spectrum=True):
    os.makedirs(output_path, exist_ok=True)
    
    # 1. 加载数据 (优化：只读需要的列)
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['count'], dtype={'count': np.int64})
    counts = df['count'].values
    num_observed_kmers = len(counts)
    total_kmers_sum = counts.sum()
    mu = total_kmers_sum / (4 ** kmer_len)

    # 2. 确定布局
    if plot_spectrum:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    x_rank = range(num_observed_kmers)

    # --- 必画图：Rank-Abundance ---
    # Ax1: Log Scale
    ax1.bar(x_rank, counts, alpha=0.7, color='#1f77b4')
    ax1.axhline(y=mu, color='#d62728', linestyle='--', label=f'Poisson Exp ($\lambda$={mu:.1f})')
    ax1.set_yscale('log')
    ax1.set_title('Rank-Abundance (Log Scale)', fontsize=13)
    ax1.grid(True, which="both", ls="-", alpha=0.15)
    ax1.legend()

    # Ax2: Linear Scale (Y-limited)
    ax2.bar(x_rank, counts, alpha=0.7, color='#2ca02c')
    ax2.axhline(y=mu, color='#d62728', linestyle='--')
    ax2.set_ylim(0, mu * 5) 
    ax2.set_title(f'Rank-Abundance (Linear, Y-limit: 5*$\lambda$)', fontsize=13)
    ax2.grid(True, ls="-", alpha=0.15)

    # --- 选画图：K-mer Spectrum (Ax3, Ax4) ---
    if plot_spectrum:
        print("Calculating spectrum and percentiles...")
        # 计算 99.5% 分位数用于裁剪
        x_limit = np.percentile(counts, 99.5)

        # Ax3: Histogram
        ax3.hist(counts, bins=80, color='#9467bd', alpha=0.7, edgecolor='white')
        ax3.set_yscale('log')
        ax3.set_title('Histogram of Counts (Y-log)', fontsize=13)
        ax3.grid(True, which="both", ls="-", alpha=0.15)

        # Ax4: Binned Species Count
        ax4.hist(counts, bins=100, range=(0, x_limit), color='#e377c2', alpha=0.8, edgecolor='white')
        ax4.set_yscale('log')
        ax4.set_title(f'Binned Species (Clipped at 99.5%)', fontsize=13)
        ax4.set_xlabel(f'Frequency (0 to {x_limit:.0f})', fontsize=11)
        ax4.grid(True, which="both", ls="-", alpha=0.15)
    
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
    # 添加开关参数：默认 True (绘制)，加上 --skip_spectrum 则为 False (不画)
    parser.add_argument("--plot_spectrum", action="store_true", default=False, help="Enable Ax3 and Ax4 (slow for large data)")
    
    args = parser.parse_args()
    
    # 注意这里逻辑：如果命令行带了 --plot_spectrum，则为 True
    analyze_kmer_distribution(args.input, args.k, args.output_dir, plot_spectrum=args.plot_spectrum)
