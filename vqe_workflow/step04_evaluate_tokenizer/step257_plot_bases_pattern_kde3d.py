import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def plot_3d_multi_view(args):
    if not os.path.exists(args.input):
        print(f"错误: 找不到文件 {args.input}")
        return

    print(f"读取数据: {args.input}")
    df = pd.read_csv(args.input)

    # 1. 类别筛选逻辑 (优先保留 required_base_patterns)
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

    # 采样
    if len(df) > args.max_points:
        df = df.sample(n=args.max_points, random_state=42)

    # 2. 维度组合 (4 列)
    view_combinations = [
        ('dim0', 'dim1', 'dim2'),
        ('dim0', 'dim1', 'dim3'),
        ('dim0', 'dim2', 'dim3'),
        ('dim1', 'dim2', 'dim3')
    ]

    # 3. 视角生成 (N 行)
    num_rows = args.num_rows
    # 自动生成不同的角度组合，这里让视角在方位角(azim)上做 360 度均匀分布
    elevations = np.linspace(10, 40, num_rows)  # 仰角在 10 到 40 度之间变化
    azimuths = np.linspace(0, 360, num_rows, endpoint=False) # 方位角旋转一圈

    # 4. 绘图开始
    # figsize 根据行列数动态调整
    fig = plt.figure(figsize=(24, 5 * num_rows), dpi=100)
    fig.suptitle(f"Multi-View 3D Projections | File: {os.path.basename(args.input)}", fontsize=22)

    # 使用高辨识度色板
    cmap = plt.get_cmap('Set1')
    colors = [cmap(i % 9) for i in range(len(categories))]

    for r in range(num_rows):
        curr_elev = elevations[r]
        curr_azim = azimuths[r]

        for c, (vx, vy, vz) in enumerate(view_combinations):
            # 计算子图索引 (1-based)
            ax_idx = r * 4 + c + 1
            ax = fig.add_subplot(num_rows, 4, ax_idx, projection='3d')

            for i, cat in enumerate(categories):
                subset = df[df['category'] == cat]
                if subset.empty: continue

                ax.scatter(subset[vx], subset[vy], subset[vz],
                           label=cat if r == 0 and c == 0 else "", # 只给第一个图加标签
                           color=colors[i], s=args.point_size, alpha=0.6, edgecolors='none')

            # 设置轴标签
            ax.set_xlabel(vx, fontsize=10)
            ax.set_ylabel(vy, fontsize=10)
            ax.set_zlabel(vz, fontsize=10)

            # 设置当前行的视角
            ax.view_init(elev=curr_elev, azim=curr_azim)

            # 每一行的第一列标注当前视角角度
            if c == 0:
                ax.text2D(-0.1, 0.85, f"Row {r+1}\nElev: {curr_elev:.0f}°\nAzim: {curr_azim:.0f}°",
                          transform=ax.transAxes, fontsize=12, fontweight='bold', color='darkred')

            # 设置标题 (仅第一行设置，区分空间)
            if r == 0:
                ax.set_title(f"Space: {vx}-{vy}-{vz}", fontsize=14, pad=10)

    # 图例处理：放在图片顶部中央
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
               ncol=1, markerscale=3, title="Categories", fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(args.output)
    print(f"✅ 多视角 3D 可视化完成: {args.output} (共 {num_rows}行 x 4列)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_points', type=int, default=3000)
    parser.add_argument('--max_categories', type=int, default=5)
    parser.add_argument('--required_base_patterns', type=str, default=None)
    # 新增参数
    parser.add_argument('--num_rows', type=int, default=3, help='观察的角度行数')
    parser.add_argument('--point_size', type=int, default=15, help='点的大小')

    args = parser.parse_args()
    plot_3d_multi_view(args)
