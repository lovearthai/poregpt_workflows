import gzip
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import textwrap

def visualize_base_alignment(input_path, output_path, plot_range=(0, 2440)):
    # 增加画布高度，适配大字体和多行文本
    fig, axes = plt.subplots(3, 1, figsize=(26, 22), dpi=150)
    
    print(f"🚀 启动碱基级信号对齐分析: {input_path}")
    
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    valid_count = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if valid_count >= 3: break
            try:
                data = json.loads(line)
                signal = data.get('signal', [])
                p_query = data.get('pattern_query', "N/A")
                p_matched = data.get('pattern_matched', "")
                spans = data.get('base_sample_spans_rel', [])
                
                if not signal or not p_matched: continue
                ax = axes[valid_count]

                # 1. 绘制信号波形
                display_sig = signal[plot_range[0]:plot_range[1]]
                x_axis = np.arange(len(display_sig)) + plot_range[0]
                ax.plot(x_axis, display_sig, color=colors[valid_count], linewidth=1.2, alpha=0.9)

                # 2. 顶部信息优化 (调大字体)
                # 使用较宽的换行，确保大字体下也能尽量单行
                header_wrap = 150 
                q_text = textwrap.fill(f"QUERY: {p_query}", width=header_wrap)
                m_text = textwrap.fill(f"MATCH: {p_matched}", width=header_wrap)
                # 字体调大至 14pt，增加 pad 留白
                ax.set_title(f"{q_text}\n{m_text}", fontsize=14, loc='left', family='monospace', pad=25, fontweight='bold')

                # 3. 碱基与区间框对齐绘制
                y_min, y_max = ax.get_ylim()
                # 确定碱基字符打印的基准高度（在信号下方）
                base_y_pos = y_min - (y_max - y_min) * 0.12
                
                # 遍历碱基和对应的信号跨度
                for i, (start, end) in enumerate(spans):
                    if i >= len(p_matched): break
                    
                    # 只处理在当前视觉窗口内的碱基
                    mid_point = (start + end) / 2
                    if max(start, plot_range[0]) < min(end, plot_range[1]):
                        base_char = p_matched[i]
                        
                        # --- 绘制区间辅助线 (Start 和 End) ---
                        # 在每个碱基的起始和结束位置画垂直虚线
                        ax.vlines([start, end], y_min - (y_max-y_min)*0.05, y_max, 
                                  colors='gray', linestyles='--', linewidth=0.5, alpha=0.4)
                        
                        # --- 绘制碱基字符 ---
                        # 交错排布：偶数位在上，奇数位在下，防止字符挤在一起
                        y_offset = 0 if i % 2 == 0 else (y_max - y_min) * 0.08
                        
                        ax.text(mid_point, base_y_pos - y_offset, base_char,
                                fontsize=14, family='monospace', fontweight='bold',
                                color='#222222', ha='center', va='top',
                                bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

                # 4. 坐标轴样式调整
                ax.set_ylabel("Raw Current (pA)", fontsize=13, fontweight='bold')
                ax.set_xlim(plot_range[0], plot_range[1])
                ax.grid(True, linestyle=':', alpha=0.3)
                
                # 动态扩展底部范围以容纳碱基
                ax.set_ylim(base_y_pos - (y_max - y_min) * 0.15, y_max)

                valid_count += 1
            except Exception as e:
                print(f"⚠️ 解析异常: {e}")
                continue

    # 全局修饰
    axes[2].set_xlabel("Signal Sample Index", fontsize=14, fontweight='bold')
    plt.suptitle(f"Base-to-Signal Mapping Analysis (Nanopore Raw Data)\nRange: {plot_range[0]} - {plot_range[1]}", 
                 fontsize=20, y=0.98, fontweight='bold')
    
    # 调整布局，增加 hspace 防止大标题被遮挡
    plt.subplots_adjust(hspace=0.8, top=0.90, bottom=0.05, left=0.06, right=0.96)
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ 绘图成功！已生成碱基对齐分析图。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='base_alignment_debug.png')
    parser.add_argument('-r', '--range', type=int, nargs=2, default=[0, 2440])
    args = parser.parse_args()
    visualize_base_alignment(args.input_file, args.output_file, tuple(args.range))