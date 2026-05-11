# -*- coding: utf-8 -*-

"""
===============================================================================
脚本名称:
    step300_compare_shift_separability.py

功能:
    对比 shift 前后 representation separability 的变化。

核心分析:
    1. Intra-class Distance
    2. Inter-class Distance
    3. Separation Ratio
    4. Silhouette Score
    5. Cosine Silhouette Score

输入:
    before_shift.csv
    after_shift.csv

输出:
    output_dir/
    ├── metrics_summary.csv
    ├── metrics_summary.txt
    ├── barplot_metrics.png
    ├── before_interclass_heatmap.png
    ├── after_interclass_heatmap.png

作者:
    ChatGPT Engineering Version
===============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cdist


# =============================================================================
# 工具函数
# =============================================================================

def parse_feature_column(df):
    """
    将 feature 字符串解析为 numpy 矩阵

    输入:
        "0.1_0.2_0.3"

    输出:
        ndarray shape = (N, D)
    """

    features = np.array([
        [float(x) for x in f.split('_')]
        for f in df['feature']
    ], dtype=np.float32)

    return features


def calculate_intra_class_distance(features, labels, metric='euclidean'):
    """
    计算类内平均距离

    含义:
        同一个类别内部的平均距离。

    interpretation:
        越小表示:
            - 同类更紧凑
            - representation 更稳定
    """

    unique_labels = sorted(set(labels))

    intra_distances = []

    for label in unique_labels:

        class_features = features[labels == label]

        if len(class_features) < 2:
            continue

        dist_matrix = cdist(class_features, class_features, metric=metric)

        # 去掉对角线
        upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

        intra_distances.extend(upper.tolist())

    return np.mean(intra_distances)


def calculate_inter_class_distance(features, labels, metric='euclidean'):
    """
    计算类别中心之间的平均距离

    含义:
        不同类别之间的 separation 程度

    interpretation:
        越大表示:
            - 类别更容易区分
    """

    unique_labels = sorted(set(labels))

    centroids = []

    for label in unique_labels:
        class_features = features[labels == label]
        centroid = np.mean(class_features, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    dist_matrix = cdist(centroids, centroids, metric=metric)

    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

    return np.mean(upper), dist_matrix, unique_labels


def analyze_dataset(csv_path, metric='euclidean'):
    """
    对单个 CSV 做完整分析
    """

    print("--------------------------------------------------")
    print(f"正在分析: {csv_path}")

    df = pd.read_csv(csv_path)

    features = parse_feature_column(df)

    labels = df['category'].values

    print(f"数据规模: {features.shape}")

    # ==========================================================
    # Intra-class
    # ==========================================================

    intra = calculate_intra_class_distance(
        features,
        labels,
        metric=metric
    )

    # ==========================================================
    # Inter-class
    # ==========================================================

    inter, dist_matrix, unique_labels = calculate_inter_class_distance(
        features,
        labels,
        metric=metric
    )

    # ==========================================================
    # Separation Ratio
    # ==========================================================

    ratio = inter / (intra + 1e-8)

    # ==========================================================
    # Silhouette
    # ==========================================================

    sil_score = silhouette_score(
        features,
        labels,
        metric=metric
    )

    result = {
        'intra_class_distance': intra,
        'inter_class_distance': inter,
        'separation_ratio': ratio,
        'silhouette_score': sil_score
    }

    return result, dist_matrix, unique_labels


# =============================================================================
# 可视化函数
# =============================================================================

def plot_metric_bar(metrics_df, output_path):
    """
    绘制指标柱状图
    """

    metrics = [
        'intra_class_distance',
        'inter_class_distance',
        'separation_ratio',
        'silhouette_score'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes = axes.flatten()

    for i, metric in enumerate(metrics):

        ax = axes[i]

        ax.bar(
            metrics_df['dataset'],
            metrics_df[metric]
        )

        ax.set_title(metric)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_path, dpi=200)

    print(f"柱状图已保存: {output_path}")


def plot_heatmap(dist_matrix, labels, output_path, title):
    """
    绘制类间距离热图
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(dist_matrix)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    ax.set_title(title)

    plt.colorbar(im)

    plt.tight_layout()

    plt.savefig(output_path, dpi=200)

    print(f"热图已保存: {output_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Shift Separability Comparison"
    )

    parser.add_argument('--before_csv', type=str, required=True)

    parser.add_argument('--after_csv', type=str, required=True)

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        choices=['euclidean', 'cosine']
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ==========================================================
    # Before Shift
    # ==========================================================

    before_result, before_matrix, before_labels = analyze_dataset(
        args.before_csv,
        metric=args.metric
    )

    # ==========================================================
    # After Shift
    # ==========================================================

    after_result, after_matrix, after_labels = analyze_dataset(
        args.after_csv,
        metric=args.metric
    )

    # ==========================================================
    # 汇总表
    # ==========================================================

    summary_df = pd.DataFrame([
        {
            'dataset': 'before_shift',
            **before_result
        },
        {
            'dataset': 'after_shift',
            **after_result
        }
    ])

    summary_csv = os.path.join(
        args.output_dir,
        'metrics_summary.csv'
    )

    summary_df.to_csv(summary_csv, index=False)

    print(summary_df)

    # ==========================================================
    # 保存 TXT
    # ==========================================================

    txt_path = os.path.join(
        args.output_dir,
        'metrics_summary.txt'
    )

    with open(txt_path, 'w') as f:

        f.write(summary_df.to_string())

    # ==========================================================
    # 柱状图
    # ==========================================================

    plot_metric_bar(
        summary_df,
        os.path.join(args.output_dir, 'barplot_metrics.png')
    )

    # ==========================================================
    # 热图
    # ==========================================================

    plot_heatmap(
        before_matrix,
        before_labels,
        os.path.join(args.output_dir,
                     'before_interclass_heatmap.png'),
        'Before Shift Inter-class Distance'
    )

    plot_heatmap(
        after_matrix,
        after_labels,
        os.path.join(args.output_dir,
                     'after_interclass_heatmap.png'),
        'After Shift Inter-class Distance'
    )

    print("==================================================")
    print("✅ 分析完成")
    print(f"结果目录: {args.output_dir}")


if __name__ == "__main__":
    main()