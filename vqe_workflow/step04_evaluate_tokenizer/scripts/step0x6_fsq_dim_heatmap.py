import gzip
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def decode_fsq_625(x):
    """
    将 0-624 的索引解码为 4个维度 (d1, d2, d3, d4)，
    每个维度对应 5个档位 (0, 1, 2, 3, 4)。
    """
    d1 = (x // 125) % 5
    d2 = (x // 25) % 5
    d3 = (x // 5) % 5
    d4 = x % 5
    return [d1, d2, d3, d4]

def analyze_fsq_heatmap(input_path, output_path, max_reads):
    base_list = ['A', 'T', 'C', 'G', 'N']
    # 初始化矩阵：纵轴5档位，横轴4维度 -> (5, 4)
    matrices = {base: np.zeros((5, 4)) for base in base_list}

    print(f"🚀 正在提取维度特征: {input_path}")
    
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= max_reads: break
            data = json.loads(line)
            layered = data.get('tokens_layered', [])
            based = data.get('tokens_based', "")
            
            if len(layered) == len(based):
                for pair, char in zip(layered, based):
                    base = char.upper()
                    if base in matrices:
                        high_val = pair[0] # High Layer (0-624)
                        dims = decode_fsq_625(high_val)
                        # dims 是长度为4的列表 [L1, L2, L3, L4]
                        # 遍历 4 个维度
                        for dim_idx, level in enumerate(dims):
                            # 纵轴为档位 (level), 横轴为维度 (dim_idx)
                            matrices[base][level, dim_idx] += 1
            count += 1
            if count % 1000 == 0:
                print(f"⏳ 已读取 {count} 条记录...", end='\r')

    # 绘图逻辑：1行5列子图
    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    
    for i, base in enumerate(base_list):
        ax = axes[i]
        matrix = matrices[base]
        
        # 归一化处理：按“列”（维度）归一化，观察每个维度内部的档位分布
        col_sums = matrix.sum(axis=0, keepdims=True)
        norm_matrix = np.divide(matrix, col_sums, out=np.zeros_like(matrix), where=col_sums!=0)
        
        sns.heatmap(norm_matrix, annot=True, fmt=".2f", cmap="YlOrRd", 
                    ax=ax, cbar=(i == 4),
                    xticklabels=["Dim 1", "Dim 2", "Dim 3", "Dim 4"],
                    yticklabels=["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"])
        
        ax.set_title(f"Base: {base}", fontsize=14, fontweight='bold')
        if i == 0:
            ax.set_ylabel("Quantization Levels (0-4)", fontsize=12)
        ax.set_xlabel("FSQ Dimensions", fontsize=12)

    plt.suptitle(f"FSQ High-Layer Analysis: 4 Dimensions vs 5 Levels\n(Column-Normalized: Distribution per Dimension)", 
                 fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    print(f"\n✅ 热图分析完成，保存至: {output_path}")

    # 终端输出简要汇总
    print("\n" + "="*50)
    print("📊 维度偏好汇总 (各维度最高频档位)")
    print("-" * 50)
    for base in base_list:
        # 找到每一列（维度）最大值所在的行（档位）
        pref_levels = np.argmax(matrices[base], axis=0)
        print(f"碱基 {base} | 维度1:L{pref_levels[0]}, 维度2:L{pref_levels[1]}, 维度3:L{pref_levels[2]}, 维度4:L{pref_levels[3]}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FSQ 维度热图分析')
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='fsq_dim_heatmap.png')
    parser.add_argument('-n', '--max-reads', type=int, default=5000)
    args = parser.parse_args()
    analyze_fsq_heatmap(args.input_file, args.output_file, args.max_reads)