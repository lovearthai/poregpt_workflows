import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def analyze_dwell_time(input_file, output_png):
    # 初始化统计字典
    base_lengths = {'A': [], 'C': [], 'G': [], 'T': []}

    print(f"读取文件并统计碱基长度分布: {input_file}")

    open_func = gzip.open if input_file.endswith('.gz') else open
    mode = 'rt' if input_file.endswith('.gz') else 'r'

    with open_func(input_file, mode) as f:
        for line in tqdm(f, desc="Processing"):
            try:
                record = json.loads(line)
                pattern = record['pattern']
                spans = record['base_sample_spans_rel']

                for i in range(min(len(pattern), len(spans))):
                    base = pattern[i].upper()
                    if base in base_lengths:
                        start, end = spans[i]
                        length = end - start
                        if length > 0:
                            base_lengths[base].append(length)
            except Exception:
                continue

    # 绘制分布图
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=100)
    axes = axes.flatten()
    colors = {'A': '#0072B2', 'C': '#E69F00', 'G': '#009E73', 'T': '#D55E00'}

    for i, base in enumerate(['A', 'C', 'G', 'T']):
        data = np.array(base_lengths[base])
        if len(data) == 0:
            axes[i].set_title(f"Base {base} - No Data")
            continue

        mean_val = np.mean(data)
        median_val = np.median(data)
        total_count = len(data)
        
        # 设定显示范围（99分位数，过滤极端 Stall 造成的长尾）
        plot_range = (0, np.percentile(data, 99))
        bins_count = 50

        # --- 核心修改：获取 hist 的返回值 ---
        # n: 频次, bins: 边界, patches: 图形对象
        n, bins, patches = axes[i].hist(data, bins=bins_count, range=plot_range,
                                        color=colors[base], alpha=0.7, edgecolor='black')

        # 在每个柱子上方添加百分比文字
        for count, bin_edge, patch in zip(n, bins, patches):
            if count > 0:
                # 计算百分比
                percentage = (count / total_count) * 100
                # 获取柱子的中心位置和高度
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                
                # 绘制文字：为了美观，只对频次较高的柱子标注，或者根据高度调整
                # 这里我们标注所有非 0 的，文字设为垂直以节省空间
                #axes[i].text(x, y, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
                axes[i].text(x, y, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=12)

        # 辅助线
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
        axes[i].axvline(median_val, color='blue', linestyle=':', label=f'Median: {median_val:.1f}')

        axes[i].set_title(f"Dwell Time Distribution: Base {base} (N={total_count})", fontsize=14)
        axes[i].set_xlabel("Signal Length (Samples)", fontsize=12)
        axes[i].set_ylabel("Frequency", fontsize=12)
        
        # 适当增加 Y 轴上限，防止百分比文字超出画面
        axes[i].set_ylim(0, max(n) * 1.15)
        
        axes[i].legend()
        axes[i].grid(axis='y', alpha=0.3)

    plt.suptitle("Nanopore Base Dwell Time Analysis with Percentage Labels", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(output_png)
    print(f"统计完成！图表已保存至: {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help="aligned.jsonl.gz 文件路径")
    parser.add_argument('-o', '--output', type=str, default="base_dwell_time_distribution.png", help="输出图片路径")
    args = parser.parse_args()

    analyze_dwell_time(args.input, args.output)
