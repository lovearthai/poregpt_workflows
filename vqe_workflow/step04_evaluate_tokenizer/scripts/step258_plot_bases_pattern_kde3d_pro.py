import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import argparse
import os

def plot_3d_kde_multi_view(args):
    if not os.path.exists(args.input):
        print(f"Error: Cannot find file {args.input}")
        return

    print(f"Reading data: {args.input}")
    df = pd.read_csv(args.input)

    # --- 仅修改：解析 feature 字段适配输入格式 ---
    print("Parsing features...")
    # 将 "val1_val2_val3_val4" 拆分为独立列
    features_split = df['feature'].str.split('_', expand=True).astype(float)
    # 为拆分后的列命名，适配脚本后续使用的 dim0, dim1, dim2, dim3
    for i in range(features_split.shape[1]):
        df[f'dim{i}'] = features_split[i]
    # ------------------------------------------

    # 1. 类别筛选逻辑 (保持原样)
    all_categories = sorted(df['category'].unique())
    required_cats = []
    if args.required_base_patterns:
        potential_required = args.required_base_patterns.split()
        required_cats = [c for c in potential_required if c in all_categories]

    if args.max_categories and len(all_categories) > args.max_categories:
        num_to_sample = max(0, args.max_categories - len(required_cats))
        remaining_cats = [c for c in all_categories if c not in required_cats]
        sampled_cats = list(np.random.choice(remaining_cats, size=min(num_to_sample, len(remaining_cats)), replace=False))
        selected_categories = required_cats + sampled_cats
    else:
        selected_categories = all_categories

    df = df[df['category'].isin(selected_categories)]
    categories = sorted(selected_categories)

    # 3D KDE 采样点数限制 (保持原样)
    if len(df) > args.max_points:
        print(f"Sampling to {args.max_points} points for KDE efficiency...")
        df = df.sample(n=args.max_points, random_state=42)

    # 2. 维度组合 (保持原样)
    view_combinations = [
        ('dim0', 'dim1', 'dim2'),
        ('dim0', 'dim1', 'dim3'),
        ('dim0', 'dim2', 'dim3'),
        ('dim1', 'dim2', 'dim3')
    ]

    # 3. 视角生成 (保持原样)
    num_rows = args.num_rows
    elevations = np.linspace(15, 45, num_rows) 
    azimuths = np.linspace(0, 360, num_rows, endpoint=False)

    # 4. 绘图初始化 (保持原样)
    fig = plt.figure(figsize=(24, 6 * num_rows), dpi=100)
    fig.suptitle(f"3D KDE Feature Space Projections | {os.path.basename(args.input)}", fontsize=24, y=0.98)

    cmap = plt.get_cmap('Set1')
    colors = [cmap(i % 9) for i in range(len(categories))]

    # 预计算每个类别在四种组合下的 KDE 密度 (保持原样)
    print("Pre-calculating 3D KDE density distributions...")
    kde_data = {}
    for combo in view_combinations:
        kde_data[combo] = {}
        for cat in categories:
            subset = df[df['category'] == cat]
            if len(subset) < 5: continue
            
            xyz = subset[list(combo)].values.T
            try:
                kde = gaussian_kde(xyz)
                density = kde(xyz)
                d_min, d_max = density.min(), density.max()
                if d_max > d_min:
                    density_norm = (density - d_min) / (d_max - d_min)
                else:
                    density_norm = np.ones_like(density)
                kde_data[combo][cat] = (subset, density_norm)
            except:
                kde_data[combo][cat] = (subset, np.ones(len(subset)) * 0.5)

    # 5. 渲染绘图矩阵 (保持原样)
    for r in range(num_rows):
        curr_elev = elevations[r]
        curr_azim = azimuths[r]
        print(f"Rendering Row {r+1}/{num_rows} (Elev: {curr_elev:.0f}, Azim: {curr_azim:.0f})...")

        for c, combo in enumerate(view_combinations):
            vx, vy, vz = combo
            ax_idx = r * 4 + c + 1
            ax = fig.add_subplot(num_rows, 4, ax_idx, projection='3d')

            for i, cat in enumerate(categories):
                if cat not in kde_data[combo]: continue
                
                subset, density_norm = kde_data[combo][cat]
                
                base_color = np.array(colors[i])
                rgba_colors = np.zeros((len(subset), 4))
                rgba_colors[:, :3] = base_color[:3]
                rgba_colors[:, 3] = 0.1 + density_norm * 0.7

                ax.scatter(subset[vx], subset[vy], subset[vz],
                           label=cat if r == 0 and c == 0 else "",
                           color=rgba_colors,
                           s=args.point_size * (0.5 + density_norm * 1.5), 
                           edgecolors='none')

            ax.set_xlabel(vx, fontsize=10)
            ax.set_ylabel(vy, fontsize=10)
            ax.set_zlabel(vz, fontsize=10)
            ax.view_init(elev=curr_elev, azim=curr_azim)

            if c == 0:
                ax.text2D(-0.15, 0.85, f"View {r+1}\nElev: {curr_elev:.0f}°\nAzim: {curr_azim:.0f}°", 
                          transform=ax.transAxes, fontsize=12, fontweight='bold', color='darkred')
            
            if r == 0:
                ax.set_title(f"Space: {vx}-{vy}-{vz}", fontsize=16, pad=10)

    # 6. 图例与保存 (保持原样)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95),
                ncol=1, markerscale=4, title="K-mer Patterns", fontsize=14, title_fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(args.output)
    print(f"Successfully saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D KDE Multi-View Visualization")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_points', type=int, default=4000)
    parser.add_argument('--max_categories', type=int, default=5)
    parser.add_argument('--required_base_patterns', type=str, default=None)
    parser.add_argument('--num_rows', type=int, default=3)
    parser.add_argument('--point_size', type=int, default=15)

    args = parser.parse_args()
    plot_3d_kde_multi_view(args)