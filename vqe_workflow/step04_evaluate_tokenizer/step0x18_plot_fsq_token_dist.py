import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import sys

def plot_fsq_token_dist(base_csv, loc_csv, output_png, top_percent=0.1, power=2.0):
    # 1. 加载数据
    print(f"📖 正在加载数据...")
    df_base = pd.read_csv(base_csv)
    df_loc = pd.read_csv(loc_csv, dtype={'layer0_code': str})

    # 2. 筛选前 N% 的高频记录
    num_top = int(len(df_base) * top_percent)
    df_top_base = df_base.head(num_top).copy()

    # 3. 合并坐标
    df_merged = pd.merge(df_top_base, df_loc[['token_id', 'x', 'y']], on='token_id', how='inner')
    
    # 4. 增强半径差异 (Min-Max 缩放)
    counts = df_merged['count'].values
    amplified_values = np.power(counts, power)
    v_min, v_max = amplified_values.min(), amplified_values.max()
    # 映射到 [10, 500] 的面积区间
    df_merged['s_size'] = 10 + (amplified_values - v_min) / (v_max - v_min + 1e-5) * 500

    # 5. 定义颜色与布局
    base_colors = {'A': '#FF4D4D', 'T': '#33FF33', 'G': '#FF9900', 'C': '#3333FF', 'N': '#999999'}
    bases_to_plot = ['A', 'T', 'G', 'C', 'N']
    
    # 创建 2x3 的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(24, 16), dpi=150)
    axes = axes.flatten() # 展开成一维方便遍历

    print("🎨 正在绘制子图...")

    # --- 前 5 个子图：独立碱基 ---
    for i, base in enumerate(bases_to_plot):
        ax = axes[i]
        # 绘制背景
        ax.scatter(df_loc['x'], df_loc['y'], c='#EEEEEE', s=1, alpha=0.1)
        
        # 绘制特定碱基
        subset = df_merged[df_merged['base'] == base]
        if not subset.empty:
            ax.scatter(
                subset['x'], subset['y'], 
                s=subset['s_size'], 
                c=base_colors[base], 
                alpha=0.6, edgecolors='white', linewidth=0.2
            )
        ax.set_title(f"Base: {base}", fontsize=20, fontweight='bold')
        ax.axis('off') # 隐藏坐标轴让画面更清爽

    # --- 第 6 个子图：全部叠加 (All Bases) ---
    ax_all = axes[5]
    ax_all.scatter(df_loc['x'], df_loc['y'], c='#EEEEEE', s=1, alpha=0.1)
    for base, color in base_colors.items():
        subset = df_merged[df_merged['base'] == base]
        if not subset.empty:
            ax_all.scatter(
                subset['x'], subset['y'], 
                s=subset['s_size'], 
                c=color, alpha=0.5, edgecolors='none'
            )
    ax_all.set_title("All Bases Combined", fontsize=20, fontweight='bold', color='darkred')
    ax_all.axis('off')

    # 6. 保存
    plt.suptitle(f"ResidualFSQ Latent Space Base Distribution (Top {top_percent*100}%, Power={power})", 
                 fontsize=28, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print(f"💾 正在保存至: {output_png}")
    plt.savefig(output_png)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_csv', type=str, required=True)
    parser.add_argument('--loc_csv', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--top', type=float, default=0.1)
    parser.add_argument('--power', type=float, default=2.0)

    args = parser.parse_args()
    plot_fsq_token_dist(args.base_csv, args.loc_csv, args.output, args.top, args.power)
