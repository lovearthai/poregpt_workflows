import gzip
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def decode_fsq(x, levels):
    """
    通用 FSQ 解码函数
    x: token id
    levels: 每个维度的量化级数，例如 [5,5,5,5] 或 [11,11,11,11]
    返回: 各维度值列表
    """
    dims = []
    for base in reversed(levels):
        dims.append(x % base)
        x //= base
    return list(reversed(dims))


def analyze_fsq_heatmap(input_path, output_path, max_reads, levels):
    base_list = ['A', 'T', 'C', 'G', 'N']

    num_dims = len(levels)
    max_level = max(levels)

    # 初始化矩阵
    matrices = {
        base: np.zeros((max_level, num_dims))
        for base in base_list
    }

    print(f"🚀 正在提取维度特征: {input_path}")
    print(f"📐 Levels: {levels} (Total codebook size ≈ {np.prod(levels)})")

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= max_reads:
                break

            data = json.loads(line)
            layered = data.get('tokens_layered', [])
            based = data.get('tokens_based', "")

            if len(layered) == len(based):
                for pair, char in zip(layered, based):
                    base = char.upper()
                    if base not in matrices:
                        base = 'N'

                    high_val = pair[0] # 只分析第一个维度（高层），如果有多个维度，可以扩展为 pair[i] 对应第 i 个维度

                    dims = decode_fsq(high_val, levels)

                    for dim_idx, level in enumerate(dims):
                        if level < levels[dim_idx]:
                            matrices[base][level, dim_idx] += 1

            count += 1
            if count % 1000 == 0:
                print(f"⏳ 已读取 {count} 条记录...", end='\r')

    # =========================
    # 绘图
    # =========================
    fig, axes = plt.subplots(1, len(base_list), figsize=(5 * len(base_list), 6))

    if len(base_list) == 1:
        axes = [axes]

    for i, base in enumerate(base_list):
        ax = axes[i]
        matrix = matrices[base]

        # 列归一化（每个维度内部归一）
        col_sums = matrix.sum(axis=0, keepdims=True)
        norm_matrix = np.divide(
            matrix,
            col_sums,
            out=np.zeros_like(matrix),
            where=col_sums != 0
        )

        xticklabels = [f"Dim {i+1}" for i in range(num_dims)]
        yticklabels = [f"L{j}" for j in range(max_level)]

        sns.heatmap(
            norm_matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            cbar=(i == len(base_list) - 1),
            xticklabels=xticklabels,
            yticklabels=yticklabels
        )

        ax.set_title(f"Base: {base}", fontsize=14, fontweight='bold')

        if i == 0:
            ax.set_ylabel("Quantization Levels", fontsize=12)

        ax.set_xlabel("FSQ Dimensions", fontsize=12)

    plt.suptitle(
        f"FSQ High-Layer Analysis\nLevels={levels} (Column-Normalized)",
        fontsize=16,
        y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=200)

    print(f"\n✅ 热图分析完成，保存至: {output_path}")

    # =========================
    # 终端分析
    # =========================
    print("\n" + "=" * 60)
    print("📊 维度偏好汇总 (每个维度最常出现的 level)")
    print("-" * 60)

    for base in base_list:
        matrix = matrices[base]
        pref_levels = np.argmax(matrix, axis=0)
        summary = ", ".join([f"D{i+1}:L{lvl}" for i, lvl in enumerate(pref_levels)])
        print(f"{base} | {summary}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FSQ 维度热图分析（通用版本）')

    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='fsq_dim_heatmap.png')
    parser.add_argument('-n', '--max-reads', type=int, default=5000)

    # 🔥 关键新增参数
    parser.add_argument(
        '--levels',
        type=int,
        nargs='+',
        required=True,
        help='FSQ levels，例如: 5 5 5 5 或 11 11 11 11'
    )

    args = parser.parse_args()

    analyze_fsq_heatmap(
        args.input_file,
        args.output_file,
        args.max_reads,
        args.levels
    )