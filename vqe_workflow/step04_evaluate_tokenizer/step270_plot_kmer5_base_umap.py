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

    # --- 核心修改：从 feature 字段提取特征向量 ---
    if 'feature' not in filtered_df.columns:
        print("❌ 错误: CSV中缺少 'feature' 列")
        return

    print("🧪 正在解析 feature 字符串并进行标准化...")
    try:
        # 解析下划线分隔的字符串为浮点数矩阵
        X_raw = np.array([
            [float(x) for x in f_str.split('_')] 
            for f_str in filtered_df['feature']
        ])
    except Exception as e:
        print(f"❌ 解析特征向量失败: {e}")
        return

    # 标准化特征
    X_scaled = StandardScaler().fit_transform(X_raw)
    # ------------------------------------------

    # 4. 降维
    print(f"🧪 正在使用 {args.method} 进行降维 (输入维度: {X_raw.shape[1]})...")
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
    # 方案二：高对比度三原色 + 纯黑
    # 特点：利用红绿蓝的物理原色差异，以及黑色的极致明度对比
    contrast_colors = {
        'A': '#FF0000', # 纯红 - 视觉冲击力最强
        'C': '#00FF00', # 纯绿 - 与红色形成鲜明对比
        'G': '#0000FF', # 纯蓝 - 冷色调的极致
        'T': '#000000', # 纯黑 - 极致的暗色，与前三者形成最大反差
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

    plt.title(f"File: {csv_name}\nPattern: {args.cato_pattern} | Method: {args.method.upper()}",
              fontsize=15, fontweight='bold', pad=20)

    plt.xlabel(f"{args.method.upper()} Component 1", fontsize=12)
    plt.ylabel(f"{args.method.upper()} Component 2", fontsize=12)

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
    print(f"🎉 绘图完成！输入维度为 {X_raw.shape[1]}。保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--cato_pattern', type=str, default='CGXGA')
    parser.add_argument('--method', type=str, choices=['tsne', 'pca'], default='tsne')
    args = parser.parse_args()
    plot_clustering(args)
