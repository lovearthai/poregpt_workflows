import gzip
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import textwrap

def visualize_single_reads(input_path, output_path, plot_range=(0, 2440)):
    # 创建 3x1 布局，增加子图间距以容纳完整的序列标题
    fig, axes = plt.subplots(3, 1, figsize=(20, 18), dpi=150)
    
    print(f"🚀 开始提取数据: {input_path}")
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    valid_count = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if valid_count >= 3:
                break
            
            try:
                data = json.loads(line)
                signal = data.get('signal', [])
                # 获取指定的 pattern_matched 字段
                full_sequence = data.get('pattern_matched', "N/A")
                
                if not signal:
                    continue
                
                ax = axes[valid_count]
                
                # 1. 截取绘制范围
                display_sig = signal[plot_range[0]:plot_range[1]]
                x_axis = np.arange(len(display_sig)) + plot_range[0]
                
                # 2. 绘制原始信号 (Raw Signal)
                ax.plot(x_axis, display_sig, color=colors[valid_count], linewidth=1.2, label=f"Read {valid_count + 1}")
                
                # 3. 处理标题：由于序列可能非常长，使用 textwrap 自动换行
                # 每 100 个碱基换一行，防止标题超出图片边缘
                wrapped_seq = "\n".join(textwrap.wrap(f"Sequence: {full_sequence}", width=120))
                
                ax.set_title(wrapped_seq, fontsize=10, loc='left', family='monospace', color='#333333')
                ax.set_ylabel("Raw Current (pA)", fontsize=12)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='upper right')
                
                valid_count += 1
                
            except Exception as e:
                print(f"⚠️ 处理第 {valid_count+1} 条数据时出错: {e}")
                continue

    # 设置全局 X 轴
    axes[2].set_xlabel("Signal Point Index (Time Steps)", fontsize=12)
    plt.xlim(plot_range[0], plot_range[1])
    
    plt.suptitle(f"Individual Signal Trace & Pattern Matched Comparison\nFile: {os.path.basename(input_path)}", 
                 fontsize=16, y=0.98, fontweight='bold')
    
    # 调整布局，为多行标题留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    print(f"\n✅ 绘图完成！每个子图已展示单条数据及其完整序列。")
    print(f"💾 结果保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='独立绘制三条数据的信号与 pattern_matched 序列')
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='single_signal_plots.png')
    parser.add_argument('-r', '--range', type=int, nargs=2, default=[0, 2440], help='信号点显示范围')
    
    args = parser.parse_args()
    visualize_single_reads(args.input_file, args.output_file, tuple(args.range))