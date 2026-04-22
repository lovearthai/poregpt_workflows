import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from itertools import combinations

def plot_dimensions_grid(args):
    # 1. 检查并加载数据
    if not os.path.exists(args.input):
        print(f"错误: 找不到文件 {args.input}")
        return

    print(f"读取数据: {args.input}")
    df = pd.read_csv(args.input)
    file_basename = os.path.basename(args.input)

    # 2. 类别筛选逻辑
    categories = sorted(df['category'].unique())
    print(f"原始数据中共有 {len(categories)} 个类别。")

    # 采样逻辑
    if len(df) > args.max_points:
        print(f"数据量大，先随机采样至 {args.max_points} 行...")
        df = df.sample(n=args.max_points, random_state=42)

    if args.max_categories is not None and len(categories) > args.max_categories:
        print(f"根据 --max_categories 限制，从 {len(categories)} 个类别中随机抽取 {args.max_categories} 个...")
        selected_categories = np.random.choice(categories, size=args.max_categories, replace=False)
        df = df[df['category'].isin(selected_categories)]
        categories = sorted(selected_categories)
        print(f"筛选后数据量: {len(df)} 行。")
    else:
        categories = sorted(categories)

    # 3. 提取特征
    feature_cols = ['dim0', 'dim1', 'dim2', 'dim3']
    if not all(col in df.columns for col in feature_cols):
        print(f"错误: CSV中缺少特征列 {feature_cols}")
        return

    # 4. 绘图展示 (3行2列布局)
    dim_pairs = list(combinations(range(4), 2))
    
    # 布局：3行2列，画布稍微拉宽一点给图例留空间
    fig, axes = plt.subplots(3, 2, figsize=(24, 24), dpi=100)
    fig.suptitle(f"Pairwise Dimensions Distribution\nFile: {file_basename}", fontsize=16)

    # 颜色映射

    # --- 修改部分：自定义高对比度配色方案 ---
    # 前8个使用高辨识度颜色：红、蓝、绿、紫、橙、褐、粉、灰
    # 后面继续使用 tab20 的剩余颜色
    high_contrast_colors = [
        '#E41A1C', # 红色
        '#377EB8', # 蓝色
        '#4DAF4A', # 绿色
        '#984EA3', # 紫色
        '#FF7F00', # 橙色
        '#A65628', # 褐色
        '#F781BF', # 粉色
        '#999999', # 灰色
    ]
    
    # 获取 tab20 的其他颜色作为补充
    cmap_tab20 = plt.get_cmap('tab20')
    # tab20 的索引 8-19 是另一组颜色
    extra_colors = [cmap_tab20(i) for i in range(8, 20)]
    
    # 合并颜色列表
    all_colors = high_contrast_colors + extra_colors
    
    # 根据类别数量分配颜色（如果超过19个，则循环使用）
    colors = [all_colors[i % len(all_colors)] for i in range(len(categories))]


    print(f"正在绘制 {len(dim_pairs)} 个子图...")

    for idx, (dim_x, dim_y) in enumerate(dim_pairs):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        x_col = f'dim{dim_x}'
        y_col = f'dim{dim_y}'
        
        # 绘制散点
        for i, cat in enumerate(categories):
            mask = df['category'] == cat
            if mask.sum() > 0:
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                           label=cat, alpha=0.7, s=200, color=colors[i])
        
        ax.set_xlabel(f"Dimension {dim_x}", fontsize=12)
        ax.set_ylabel(f"Dimension {dim_y}", fontsize=12)
        ax.set_title(f"Dim {dim_x} vs Dim {dim_y}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # --- 为每个子图添加图例 ---
        # 将图例放在子图的右侧外部
        # bbox_to_anchor=(1.05, 1) 表示在 axes 的右上角外部
        # loc='upper left' 表示图例的左上角对齐 anchor 点
        ax.legend(title="Category", 
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  fontsize='large', 
                  markerscale=1, 
                  borderaxespad=0.)

    # 调整布局，防止图例被截断
    # rect 参数给右侧留出空间，防止图例超出画布边界
    plt.tight_layout(rect=[0, 0, 0.95, 0.98])

    plt.savefig(args.output)
    print(f"可视化分析完成，结果已保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanopore Feature Visualization (Grid View)")
    parser.add_argument('--input', type=str, required=True, help='输入特征CSV文件')
    parser.add_argument('--output', type=str, required=True, help='保存图片的路径')
    parser.add_argument('--max_points', type=int, default=100000, help='最大采样点数')
    
    parser.add_argument('--max_categories', type=int, default=None,
                        help='最大绘制类别数。如果设置，将从所有类别中随机抽取 N 个进行绘制')

    args = parser.parse_args()
    plot_dimensions_grid(args)
