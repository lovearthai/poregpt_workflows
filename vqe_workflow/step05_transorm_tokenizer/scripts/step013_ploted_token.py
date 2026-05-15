import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

def plot_token_distribution(csv_path, output_png, top_n):
    # 1. 加载数据
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 无法读取 CSV 文件: {e}")
        sys.exit(1)

    # 2. 数据预处理
    df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce')
    df = df.dropna(subset=['frequency']) # 剔除无效数据
    total_count = df['frequency'].sum()
    df['percentage'] = (df['frequency'] / total_count) * 100

    # 3. 处理 top_n 逻辑
    # 默认按频次从高到低排序
    df = df.sort_values(by='frequency', ascending=False)
    
    if top_n > 0:
        plot_data = df.head(top_n)
        display_n = top_n
    else:
        plot_data = df
        display_n = len(df)
        print(f"📊 检测到 top_n=0，将绘制全部 {display_n} 个 Token。")

    # 4. 动态调整画布尺寸
    # 每个柱子至少给 0.3 英寸的宽度，最小总宽 12 英寸
    dynamic_width = max(12, display_n * 0.3)
    plt.figure(figsize=(dynamic_width, 10))
    sns.set_theme(style="whitegrid")

    # 5. 绘制柱状图
    ax = sns.barplot(x='token', y='percentage', data=plot_data, palette='viridis')

    # 6. 细节优化
    plt.title(f'Token Distribution (Total: {display_n})', fontsize=20, fontweight='bold')
    plt.xlabel('Token Name', fontsize=14)
    plt.ylabel('Percentage of Total (%)', fontsize=14)

    # X轴 Token 标签：旋转90度
    plt.xticks(rotation=90, fontsize=min(10, max(6, 1000 // display_n if display_n > 0 else 10)))

    # 7. 标注百分比数值，并旋转90度
    # 如果绘制的 Token 太多（比如超过 100 个），标注会非常拥挤，这里做个保护
    if display_n <= 100:
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', 
                            va='bottom', 
                            xytext=(0, 5), 
                            textcoords='offset points',
                            fontsize=9,
                            rotation=90)
    else:
        print("💡 由于 Token 数量较多 (>100)，已自动隐藏柱头百分比标注以保持图表整洁。")

    plt.tight_layout()

    # 8. 保存
    plt.savefig(output_png, dpi=300)
    print(f"✅ 柱状图已成功保存至: {output_png} (画布宽度: {dynamic_width:.1f} inch)")

def main():
    parser = argparse.ArgumentParser(description="根据 Token 频次绘制分布图")
    
    parser.add_argument("--input", type=str, required=True, help="输入 CSV 路径")
    parser.add_argument("--output", type=str, default="token_distribution.png", help="输出图片路径")
    parser.add_argument("--top_n", type=int, default=40, help="绘制前 N 个。设为 0 则绘制全部。")

    args = parser.parse_args()

    plot_token_distribution(args.input, args.output, args.top_n)

if __name__ == "__main__":
    main()
