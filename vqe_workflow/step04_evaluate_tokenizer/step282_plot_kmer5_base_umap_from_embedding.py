import pandas as pd
import numpy as np
import re
import argparse
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_clustering(args):
    # 1. 加载数据
    print(f"📖 正在读取文件: {args.input}")
    if not os.path.exists(args.input):
        print(f"❌ 错误: 找不到文件 {args.input}")
        return

    df = pd.read_csv(args.input)
    csv_name = os.path.basename(args.input)

    # 2. 构造通配符正则
    regex_pattern = f"^{args.cato_pattern.replace('X', '.')}$"

    def match_and_extract(cat):
        match = re.match(regex_pattern, str(cat))
        if match:
            x_index = args.cato_pattern.find('X')
            return cat[x_index] if x_index < len(cat) else None
        return None

    df['sub_category'] = df['category'].apply(match_and_extract)
    filtered_df = df.dropna(subset=['sub_category']).copy()

    if filtered_df.empty:
        print(f"⚠️ 未找到匹配模式 '{args.cato_pattern}' 的行。")
        return

    # 3. 准备特征向量 (核心改动部分)
    print(f"🧬 正在解析 Embedding 字段...")
    # 将 "v1_v2_v3..." 字符串转换为 numpy 数组
    # 注意：如果你的 CSV 实际是以 '-' 分隔，请将 split('_') 改为 split('-')
    X_raw = np.array([np.fromstring(s.replace('_', ' '), sep=' ') for s in filtered_df['embedding']])
    X_scaled = StandardScaler().fit_transform(X_raw)

    # 4. 降维
    print(f"🧪 正在使用 {args.method} 进行降维...")
    if args.method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    else:
        reducer = PCA(n_components=2)

    X_embedded = reducer.fit_transform(X_scaled)

    # 5. 绘图
    plt.figure(figsize=(12, 10), dpi=150)

    # 针对 A/C/G/T 的最高对比度配色
    contrast_colors = {
        'A': '#0072B2', # 蓝色
        'C': '#E69F00', # 橙色
        'G': '#009E73', # 绿色
        'T': '#D55E00', # 朱红色
    }

    sub_cats = sorted(filtered_df['sub_category'].unique())

    for i, sub_cat in enumerate(sub_cats):
        mask = filtered_df['sub_category'] == sub_cat
        color = contrast_colors.get(sub_cat.upper(), plt.cm.tab10(i))

        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            label=f"X = {sub_cat}",
            color=color,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5,
            s=65
        )

    # 在标题中加入 CSV 文件名
    plt.title(f"File: {csv_name}\nPattern: {args.cato_pattern} | Method: {args.method.upper()}",
              fontsize=15, fontweight='bold', pad=20)

    plt.xlabel(f"{args.method.upper()} Component 1", fontsize=12)
    plt.ylabel(f"{args.method.upper()} Component 2", fontsize=12)

    # --- Legend 放置在图片内部 ---
    plt.legend(
        title="Base (X)",
        loc='upper right',
        fontsize=12,
        title_fontsize=13,
        frameon=True,
        framealpha=0.8,
        edgecolor='gray',
        facecolor='white'
    )

    plt.grid(True, linestyle=':', alpha=0.4)

    # 保存结果
    plt.savefig(args.output, bbox_inches='tight')
    print(f"🎉 绘图完成！使用 Embedding 维度。保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--cato_pattern', type=str, default='CGXGA')
    parser.add_argument('--method', type=str, choices=['tsne', 'pca'], default='tsne')
    args = parser.parse_args()
    plot_clustering(args)
