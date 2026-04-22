import gzip
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def visualize_fsq_latent_space(input_path, output_path, max_reads=1000):
    # 碱基映射字典
    base_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
    # 定义标准测序颜色: A(绿), T(红), C(蓝), G(橙/黄), N(灰)
    colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#7f7f7f']
    cmap = ListedColormap(colors)
    
    x_coords = []
    y_coords = []
    z_values = []

    print(f"🚀 开始提取绘图数据: {input_path}")
    
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= max_reads:
                break
            
            data = json.loads(line)
            layered = data.get('tokens_layered', [])
            based = data.get('tokens_based', "")
            
            # 校验数据长度一致性
            if len(layered) == len(based):
                for pair, char in zip(layered, based):
                    x_coords.append(pair[0]) # High Layer
                    y_coords.append(pair[1]) # Low Layer
                    # 转换碱基为数字，未知碱基统一归为 N
                    z_values.append(base_map.get(char.upper(), 4))
            
            count += 1
            if count % 100 == 0:
                print(f"⏳ 已读取 {count} 条 Read...", end='\r')

    if not x_coords:
        print("❌ 错误：未提取到有效数据，请检查字段是否存在且匹配。")
        return

    print(f"\n📊 正在生成散点图 (共 {len(x_coords)} 个 points)...")

    # 创建绘图
    plt.figure(figsize=(12, 10), dpi=150)
    
    # 使用 scatter 绘图
    # s 是点的大小，alpha 是透明度（高密度场景建议设小一点）
    scatter = plt.scatter(x_coords, y_coords, c=z_values, cmap=cmap, s=1, alpha=0.5)
    
    # 设置坐标轴范围 (FSQ 范围是 0-625)
    plt.xlim(-10, 635)
    plt.ylim(-10, 635)
    
    # 添加图例
    cbar = plt.colorbar(scatter, ticks=[0.4, 1.2, 2.0, 2.8, 3.6])
    cbar.ax.set_yticklabels(['A', 'T', 'C', 'G', 'N'])
    cbar.set_label('Bases')

    plt.title(f"FSQ Latent Space Visualization (High vs Low Layer)\nFile: {os.path.basename(input_path)}")
    plt.xlabel("High Layer (Layer 0)")
    plt.ylabel("Low Layer (Layer 1)")
    plt.grid(True, linestyle='--', alpha=0.3)

    # 保存图片
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ 可视化结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FSQ 潜空间碱基分布可视化')
    parser.add_argument('-i', '--input-file', type=str, required=True, help='输入的 jsonl.gz 路径')
    parser.add_argument('-o', '--output-file', type=str, default='fsq_latent_plot.png', help='输出图片路径')
    parser.add_argument('-n', '--max-reads', type=int, default=1000, help='处理的最大 Read 数量，默认 1000')
    
    args = parser.parse_args()
    visualize_fsq_latent_space(args.input_file, args.output_file, args.max_reads)