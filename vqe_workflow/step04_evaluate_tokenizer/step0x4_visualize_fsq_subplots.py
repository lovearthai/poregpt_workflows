import gzip
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def visualize_fsq_subplots(input_path, output_path, max_reads=1000):
    # 配置信息
    base_list = ['A', 'T', 'C', 'G', 'N']
    base_colors = {
        'A': '#2ca02c', # 绿
        'T': '#d62728', # 红
        'C': '#1f77b4', # 蓝
        'G': '#ff7f0e', # 橙
        'N': '#7f7f7f'  # 灰
    }
    
    # 初始化数据结构，用于存储每个碱基的坐标
    data_points = {base: {'x': [], 'y': []} for base in base_list}

    print(f"🚀 正在从 {input_path} 提取数据...")
    
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= max_reads:
                break
            
            data = json.loads(line)
            layered = data.get('tokens_layered', [])
            based = data.get('tokens_based', "")
            
            if len(layered) == len(based):
                for (hx, lx), char in zip(layered, based):
                    b = char.upper()
                    if b in data_points:
                        data_points[b]['x'].append(hx)
                        data_points[b]['y'].append(lx)
                    else:
                        data_points['N']['x'].append(hx)
                        data_points['N']['y'].append(lx)
            
            count += 1
            if count % 100 == 0:
                print(f"⏳ 已读取 {count} 条 Read...", end='\r')

    print(f"\n📊 正在生成 2x3 子图布局...")

    # 创建 2x3 画布
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    axes = axes.flatten() # 转为一维数组方便遍历

    # 1-5: 绘制独立碱基子图
    for i, base in enumerate(base_list):
        ax = axes[i]
        ax.scatter(data_points[base]['x'], data_points[base]['y'], 
                   c=base_colors[base], s=0.5, alpha=0.4)
        ax.set_title(f"Base: {base}", fontsize=14, fontweight='bold', color=base_colors[base])
        ax.set_xlim(-10, 635)
        ax.set_ylim(-10, 635)
        ax.grid(True, linestyle='--', alpha=0.3)

    # 6: 绘制混合全景图
    ax_all = axes[5]
    for base in base_list:
        ax_all.scatter(data_points[base]['x'], data_points[base]['y'], 
                       c=base_colors[base], s=0.5, alpha=0.3, label=base)
    
    ax_all.set_title("All Bases Mixed", fontsize=14, fontweight='bold')
    ax_all.set_xlim(-10, 635)
    ax_all.set_ylim(-10, 635)
    ax_all.grid(True, linestyle='--', alpha=0.3)
    # 在最后一个子图添加图例
    leg = ax_all.legend(loc='upper right', markerscale=10)
    for text in leg.get_texts():
        text.set_weight('bold')

    # 公共标签
    fig.suptitle(f"FSQ Latent Space Subplots (Max Reads: {max_reads})\nFile: {os.path.basename(input_path)}", 
                 fontsize=16, y=0.95)
    
    # 为每行/列添加轴标签
    for ax in axes[3:]: # 底行
        ax.set_xlabel("High Layer (Layer 0)")
    for ax in axes[0::3]: # 左列
        ax.set_ylabel("Low Layer (Layer 1)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存结果
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ 子图可视化已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FSQ 潜空间 2x3 子图可视化')
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='fsq_subplots.png')
    parser.add_argument('-n', '--max-reads', type=int, default=1000)
    
    args = parser.parse_args()
    visualize_fsq_subplots(args.input_file, args.output_file, args.max_reads)