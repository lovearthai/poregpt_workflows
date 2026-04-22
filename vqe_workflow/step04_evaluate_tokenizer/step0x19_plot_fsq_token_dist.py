import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import sys

def plot_base_latent_amplified(base_csv, loc_csv, output_png, top_percent=0.1, power=1.5):
    # 1. 加载数据
    df_base = pd.read_csv(base_csv)
    df_loc = pd.read_csv(loc_csv, dtype={'layer0_code': str})

    # 2. 筛选前 N%
    num_top = int(len(df_base) * top_percent)
    df_top_base = df_base.head(num_top).copy()

    # 3. 合并坐标
    df_merged = pd.merge(df_top_base, df_loc[['token_id', 'x', 'y']], on='token_id', how='inner')
    
    # --- 核心改进：放大半径差异 ---
    # 方法：先计算幂函数，再进行 Min-Max 标准化映射到指定的视觉尺寸范围 (例如 20 到 500)
    counts = df_merged['count'].values
    # 使用幂函数拉开差距 (power 越大，高频点越巨大)
    amplified_values = np.power(counts, power) 
    
    # 将数值缩放到合理的像素面积区间 [20, 800]
    v_min, v_max = amplified_values.min(), amplified_values.max()
    df_merged['s_size'] = 20 + (amplified_values - v_min) / (v_max - v_min + 1e-5) * 800

    # 4. 颜色映射
    base_colors = {'A': '#FF4D4D', 'T': '#33FF33', 'C': '#3333FF', 'G': '#FF9900', 'N': '#999999'}

    # 5. 绘图
    plt.figure(figsize=(20, 15), dpi=150)
    
    # 背景铺底
    plt.scatter(df_loc['x'], df_loc['y'], c='#E0E0E0', s=1, alpha=0.2)

    # 绘制
    for base, color in base_colors.items():
        subset = df_merged[df_merged['base'] == base]
        if not subset.empty:
            # 使用计算出的 s_size (注意：scatter 的 s 参数代表面积)
            plt.scatter(
                subset['x'], subset['y'], 
                s=subset['s_size'], 
                c=color, 
                label=f'Base {base}', 
                alpha=0.6, 
                edgecolors='white', 
                linewidth=0.3
            )

    plt.title(f"Amplified Base Distribution (Power={power}, Top {top_percent*100}%)", fontsize=20)
    lgnd = plt.legend(loc="upper right", title="Bases", fontsize=15)
    for handle in lgnd.legend_handles:
        handle._sizes = [200]

    plt.tight_layout()
    plt.savefig(output_png)
    print(f"✅ 图片已增强差异并保存至: {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_csv', type=str, required=True)
    parser.add_argument('--loc_csv', type=str, required=True)
    parser.add_argument('--output', type=str, default='amplified_base_map.png')
    parser.add_argument('--top', type=float, default=0.1)
    parser.add_argument('--power', type=float, default=2.0, help='放大系数，越大区别越明显')

    args = parser.parse_args()
    plot_base_latent_amplified(args.base_csv, args.loc_csv, args.output, args.top, args.power)
