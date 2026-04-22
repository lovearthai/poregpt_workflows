import gzip
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def get_adaptive_fonts(width, seq_len):
    """
    基于区间宽度和序列长度共同决定字号
    """
    # 基础缩放因子（基于信号区间宽度）
    base_width = 1000
    w_scale = np.clip((base_width / width) ** 0.45, 0.5, 1.8)
    
    # 序列缩放因子（如果序列过长，强行压减标题字号以实现单行展示）
    # 假设 28 英寸画布在 150dpi 下，10pt 字体大约能跑 250 个字符
    s_scale = 1.0
    if seq_len > 150:
        s_scale = max(0.4, 150 / seq_len)
    
    return {
        'title': 14 * min(w_scale, 1.2) * s_scale, 
        'base': 14 * w_scale,
        'token': 14 * w_scale,
        'label': 13 * min(w_scale, 1.1),
        'h_pad': 2.5 * w_scale
    }

def visualize_long_line_alignment(input_path, output_path, plot_range=(0, 2440)):
    view_width = plot_range[1] - plot_range[0]
    
    # 增加宽度 (30英寸)，为单行展示提供物理基础
    fig, axes = plt.subplots(3, 1, figsize=(30, 24), dpi=150)
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    valid_count = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if valid_count >= 3: break
            try:
                data = json.loads(line)
                signal, pattern = data.get('signal', []), data.get('pattern', "")
                spans, layered = data.get('base_sample_spans_rel', []), data.get('tokens_layered', [])
                
                if not signal: continue
                ax = axes[valid_count]

                # 计算该行序列的自适应字体
                fs = get_adaptive_fonts(view_width, len(pattern))

                # --- 1. 绘制信号 ---
                ax.plot(np.arange(len(signal)), signal, color=colors[valid_count], linewidth=1.8, alpha=0.9)

                # --- 2. 核心改进：单行大标题 ---
                # 打印 pattern 字段内容
                full_title = f"Pattern: {pattern}"
                
                # 使用 set_title 的 loc='left'，并稍微向左偏移确保起始位置
                ax.set_title(full_title, fontsize=fs['title'], loc='left', family='monospace', 
                             pad=30, fontweight='bold', linespacing=1.5)

                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                
                # --- 3. 碱基对齐 ---
                base_y = y_min - y_range * 0.09
                for i, (start, end) in enumerate(spans):
                    if i >= len(pattern): break
                    mid = (start + end) / 2
                    if max(start, plot_range[0]) < min(end, plot_range[1]):
                        # 辅助线使用对应子图的signal颜色，保持虚线样式
                        ax.vlines([start, end], y_min - y_range*0.03, y_max, colors=colors[valid_count], linestyles='--', linewidth=1.2, alpha=0.6)
                        ax.text(mid, base_y, pattern[i], fontsize=fs['base'], fontweight='bold', ha='center', va='top', family='monospace',
                                bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='#bbbbbb', alpha=0.85))

                # --- 4. Token 旋转对齐 ---
                visible_tokens = [(i, token_pair) for i, token_pair in enumerate(layered)
                                  if plot_range[0] <= (i * 4 + 1.5) <= plot_range[1]]
                token_count = len(visible_tokens)
                token_font = fs['token']
                if token_count > 16:
                    token_font = max(6, fs['token'] * (300.0 / max(view_width, 300)))
                elif token_count > 10:
                    token_font = max(7, fs['token'] * 0.85)

                single_line_tokens = token_count <= 12 or view_width < 320
                token_y_base = y_min - y_range * (0.24 if single_line_tokens else 0.42)

                # 计算所有token的y位置，用于动态调整y轴下限
                all_token_ys = []
                for idx, (i, token_pair) in enumerate(visible_tokens):
                    if single_line_tokens:
                        y_off = 0
                    else:
                        y_off = 0 if idx % 2 == 0 else y_range * 0.18
                    token_y = token_y_base - y_off
                    all_token_ys.append(token_y)

                # 动态调整y轴下限，确保所有token都在框内
                if all_token_ys:
                    min_token_y = min(all_token_ys)
                    y_axis_bottom = min_token_y - y_range * 0.15  # 额外留白
                else:
                    y_axis_bottom = token_y_base - y_range * 0.22

                for idx, (i, token_pair) in enumerate(visible_tokens):
                    t_mid = i * 4 + 1.5
                    token_val = str(token_pair[0])
                    if single_line_tokens:
                        y_off = 0
                    else:
                        y_off = 0 if idx % 2 == 0 else y_range * 0.18

                    token_y = token_y_base - y_off
                    ax.text(t_mid, token_y, token_val,
                            fontsize=token_font, family='monospace', color='white',
                            ha='center', va='top', fontweight='bold', rotation=90,
                            bbox=dict(boxstyle='round,pad=0.2', fc='#333333', ec='none', alpha=0.85))
                    ax.vlines(t_mid, token_y, y_min - y_range*0.28, colors='blue', linestyles=':', linewidth=0.3, alpha=0.2)

                # --- 5. 坐标轴与布局 ---
                ax.set_ylabel("Current (pA)", fontsize=fs['label'], fontweight='bold')
                ax.set_xlim(plot_range[0], plot_range[1])
                ax.set_ylim(y_axis_bottom, y_max)
                ax.grid(True, linestyle=':', alpha=0.3)

                valid_count += 1
            except Exception as e:
                print(f"⚠️ 跳过错误: {e}")
                continue

    axes[2].set_xlabel("Signal Sample Index", fontsize=13, fontweight='bold')
    
    # 自动布局：调整 rect 留出顶部大标题空间
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94], h_pad=fs['h_pad'])
    
    plt.suptitle(f"Adaptive Single-Line Mapping | View: {plot_range[0]}-{plot_range[1]}", 
                 fontsize=20, y=0.98, fontweight='bold')
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ 成功生成。序列已强制尝试单行展示。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='single_line_report.png')
    parser.add_argument('-r', '--range', type=int, nargs=2, default=[0, 2440])
    args = parser.parse_args()
    visualize_long_line_alignment(args.input_file, args.output_file, tuple(args.range))