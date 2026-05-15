import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import math

def plot_token_distribution(csv_path, output_png, top_n, tokens_per_row):
    # 1. 加载数据
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 无法读取 CSV 文件: {e}")
        sys.exit(1)

    # 2. 数据预处理
    df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce')
    df = df.dropna(subset=['frequency'])
    total_count = df['frequency'].sum()
    df['percentage'] = (df['frequency'] / total_count) * 100

    # 3. 排序并截取
    df = df.sort_values(by='frequency', ascending=False)
    if top_n > 0:
        plot_data = df.head(top_n)
    else:
        plot_data = df
    
    total_to_plot = len(plot_data)
    if total_to_plot == 0:
        print("❌ 无数据。")
        return

    # 4. 计算布局 (取消 sharey)
    num_rows = math.ceil(total_to_plot / tokens_per_row)
    fig_width = max(14, tokens_per_row * 0.35)
    fig_height = 5 * num_rows # 每行高度稍微收缩，使整体比例协调
    
    # 注意：这里移除了 sharey=True
    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width, fig_height))
    sns.set_theme(style="whitegrid")

    if num_rows == 1:
        axes = [axes]

    # 5. 分段迭代绘制
    for i in range(num_rows):
        start_idx = i * tokens_per_row
        end_idx = min(start_idx + tokens_per_row, total_to_plot)
        row_data = plot_data.iloc[start_idx:end_idx]
        
        ax = axes[i]
        
        if not row_data.empty:
            sns.barplot(x='token', y='percentage', data=row_data, ax=ax, palette='viridis')
            
            # 动态调整当前子图的 Y 轴上限 (留出 15% 的余量给标签)
            current_max = row_data['percentage'].max()
            ax.set_ylim(0, current_max * 1.15) 

            # 子图标题增加排名信息
            ax.set_title(f"Rank {start_idx + 1} to {end_idx} (Local Max: {current_max:.2f}%)", 
                         fontsize=13, fontweight='bold', loc='left')
            
            ax.set_xlabel("")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis='x', rotation=90, labelsize=12)

            # 柱头百分比标注
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(f'{h:.2f}%', 
                                (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', rotation=90,
                                xytext=(0, 5), textcoords='offset points', fontsize=12)
        else:
            ax.axis('off')

    plt.suptitle(f'Token Distribution (Total Top {total_to_plot})', fontsize=20, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 6. 保存
    plt.savefig(output_png, dpi=200)
    print(f"✅ 绘图完成！已根据每行数据动态调整 Y 轴上限。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="token_dist_dynamic.png")
    parser.add_argument("--top_n", type=int, default=150)
    parser.add_argument("--plot_tokens_per_row", type=int, default=50)

    args = parser.parse_args()
    plot_token_distribution(args.input, args.output, args.top_n, args.plot_tokens_per_row)

if __name__ == "__main__":
    main()
