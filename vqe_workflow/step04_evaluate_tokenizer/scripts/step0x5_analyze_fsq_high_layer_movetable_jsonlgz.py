import gzip
import json
import os
import argparse
import matplotlib.pyplot as plt
from collections import Counter

def analyze_x_distribution(input_path, output_path, max_reads, top_k):
    base_list = ['A', 'T', 'C', 'G', 'N']
    base_colors = {
        'A': '#2ca02c',
        'T': '#d62728',
        'C': '#1f77b4',
        'G': '#ff7f0e',
        'N': '#7f7f7f'
    }

    x_stats = {base: [] for base in base_list}

    print(f"🚀 正在分析: {input_path}")

    # =========================
    # 读取数据
    # =========================
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
                    b = char.upper()
                    if b in x_stats:
                        x_stats[b].append(pair[0])
                    else:
                        x_stats['N'].append(pair[0])

            count += 1
            if count % 500 == 0:
                print(f"⏳ 已读取 {count} 条记录...", end='\r')

    # =========================
    # 全局范围计算（关键）
    # =========================
    all_x = []
    for base in base_list:
        all_x.extend(x_stats[base])

    if not all_x:
        print("❌ 没有有效数据，退出")
        return

    x_min, x_max = min(all_x), max(all_x)

    print(f"\n📊 数据范围: X ∈ [{x_min}, {x_max}]")

    # =========================
    # 绘图
    # =========================
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)

    for i, base in enumerate(base_list):
        ax = axes[i]
        data = x_stats[base]

        if not data:
            ax.set_title(f"Base: {base} (No Data)")
            continue

        counts = Counter(data)
        xs = list(counts.keys())
        ys = list(counts.values())

        # 稀疏柱状图（关键改进）
        ax.bar(xs, ys, color=base_colors[base], width=1.0, alpha=0.8)

        # Top-K 标注
        top_common = counts.most_common(top_k)
        for val, freq in top_common:
            if freq == 0:
                continue

            ax.text(
                val, freq,
                f"X={val}\n({freq})",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5)
            )

            ax.axvline(
                x=val,
                color='black',
                linestyle='--',
                alpha=0.15,
                linewidth=0.8
            )

        ax.set_title(
            f"High Layer Frequency - Base: {base} (Top {top_k})",
            fontsize=14,
            fontweight='bold',
            color=base_colors[base],
            pad=20   # 👈 关键：把标题往上推
        )
        ax.set_ylabel("Frequency")

        # 统一坐标范围（关键）
        ax.set_xlim(x_min - 10, x_max + 10)

    plt.xlabel("High Layer Value")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig(output_path, dpi=200)
    print(f"\n✅ 绘图完成: {output_path}")

    # =========================
    # 终端报告
    # =========================
    print("\n" + "="*60)
    print(f"📊 高频 X 坐标汇总报告 (Top {top_k} Peaks)")
    print("-"*60)
    print(f"{'碱基':<5} | {'高频 X 坐标及其出现频次 (X_Value: Count)':<50}")
    print("-"*60)

    for base in base_list:
        common = Counter(x_stats[base]).most_common(top_k)

        formatted_list = [f"X={v}: {c}" for v, c in common if c > 0]
        report_line = ", ".join(formatted_list)

        print(f"{base:<5} | {report_line}")

    print("="*60)
    print("💡 提示：如果多个碱基的高频 X 坐标相同，说明该编码为通用背景特征。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='high_layer_freq.png')
    parser.add_argument('-n', '--max-reads', type=int, default=5000)
    parser.add_argument('-k', '--top-k', type=int, default=5)

    args = parser.parse_args()

    analyze_x_distribution(
        args.input_file,
        args.output_file,
        args.max_reads,
        args.top_k
    )