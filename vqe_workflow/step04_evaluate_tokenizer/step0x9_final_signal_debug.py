import gzip
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import textwrap

def visualize_aligned_analysis(input_path, output_path, plot_range=(0, 2440)):
    # 增加高度以容纳多行对齐的 Token
    fig, axes = plt.subplots(3, 1, figsize=(24, 20), dpi=150)
    
    print(f"🚀 启动精准对齐分析: {input_path}")
    print(f"🔍 信号对齐区间: {plot_range[0]} - {plot_range[1]}")
    
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    valid_count = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if valid_count >= 3: break
            try:
                data = json.loads(line)
                signal = data.get('signal', [])
                p_query = data.get('pattern_query', "N/A")
                p_matched = data.get('pattern_matched', "N/A")
                layered = data.get('tokens_layered', [])
                spans = data.get('base_sample_spans_rel', [])
                
                if not signal: continue
                ax = axes[valid_count]

                # 1. 绘制信号波形
                display_sig = signal[plot_range[0]:plot_range[1]]
                x_axis = np.arange(len(display_sig)) + plot_range[0]
                ax.plot(x_axis, display_sig, color=colors[valid_count], linewidth=1, alpha=0.9)

                # 2. 顶部序列信息 (Query & Match)
                # 尽量单行显示，设宽一点
                header_wrap = 200
                q_text = textwrap.fill(f"Q: {p_query}", width=header_wrap)
                m_text = textwrap.fill(f"M: {p_matched}", width=header_wrap)
                ax.set_title(f"{q_text}\n{m_text}", fontsize=9, loc='left', family='monospace', pad=20)

                # 3. Token 精准横向对齐绘制
                # 我们在 y 轴下方一点的位置绘制 Token
                y_min, y_max = ax.get_ylim()
                # 预留底部空间给文字
                token_y_base = y_min - (y_max - y_min) * 0.15 
                
                for i, (start, end) in enumerate(spans):
                    # 只处理在当前 plot_range 窗口内的 Token
                    mid_point = (start + end) / 2
                    if plot_range[0] <= mid_point <= plot_range[1]:
                        if i < len(layered):
                            token_val = str(layered[i][0])
                            
                            # 交错排布逻辑：如果是偶数索引，稍微往下挪一点，防止横向挤占
                            y_offset = 0 if i % 2 == 0 else (y_max - y_min) * 0.08
                            
                            ax.text(mid_point, token_y_base - y_offset, token_val,
                                    fontsize=8, family='monospace', color='#444444',
                                    ha='center', va='top', rotation=0,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='#cccccc'))
                            
                            # 画一条极细的虚线连接信号和 Token
                            ax.vlines(mid_point, token_y_base, np.min(display_sig), 
                                      colors='gray', linestyles='--', linewidth=0.3, alpha=0.5)

                # 4. 样式调整
                ax.set_ylabel("Raw Current (pA)", fontsize=11)
                ax.set_xlim(plot_range[0], plot_range[1])
                ax.grid(True, linestyle=':', alpha=0.3)
                
                # 调整纵轴范围以容纳底部 Token
                ax.set_ylim(token_y_base - (y_max - y_min) * 0.15, y_max)

                valid_count += 1
            except Exception as e:
                print(f"⚠️ 错误: {e}")
                continue

    axes[2].set_xlabel("Signal Sample Index (Time Steps)", fontsize=12)
    plt.suptitle(f"Aligned Signal-to-Token Mapping (High Layer)\nRange: {plot_range[0]}-{plot_range[1]}", 
                 fontsize=16, y=0.98, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"✅ 绘图成功！Token 已根据其信号跨度自动横向对齐。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='aligned_debug_report.png')
    parser.add_argument('-r', '--range', type=int, nargs=2, default=[0, 2440])
    args = parser.parse_args()
    visualize_aligned_analysis(args.input_file, args.output_file, tuple(args.range))