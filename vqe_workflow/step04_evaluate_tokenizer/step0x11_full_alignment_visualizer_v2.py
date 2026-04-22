import gzip
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def get_adaptive_fonts(width, seq_len):
    """基于区间宽度和序列长度共同决定字号"""
    base_width = 1000
    w_scale = np.clip((base_width / width) ** 0.45, 0.5, 1.8)
    
    s_scale = 1.0
    if seq_len > 150:
        s_scale = max(0.4, 150 / seq_len)
    
    return {
        'title': 16 * min(w_scale, 1.2) * s_scale, 
        'base': 14 * w_scale,
        'token': 14 * w_scale,
        'label': 14 * min(w_scale, 1.1)
    }

def visualize_single_alignment(input_path, output_path, plot_range=(0, 2440)):
    view_width = plot_range[1] - plot_range[0]
    
    # 修改为 1x1 布局，高度调小一点 (30x10)
    fig, ax = plt.subplots(1, 1, figsize=(30, 10), dpi=150)
    colors = '#1f77b4'

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        found_valid = False
        for line in f:
            try:
                data = json.loads(line)
                signal = data.get('signal', [])
                if not signal: continue
                
                # 提取数据
                p_query = data.get('pattern_query', "N/A")
                p_matched = data.get('pattern_matched', "")
                spans = data.get('base_sample_spans_rel', [])
                layered = data.get('tokens_layered', [])
                
                # 计算自适应字体
                fs = get_adaptive_fonts(view_width, max(len(p_query), len(p_matched)))

                # --- 1. 绘制信号 ---
                ax.plot(np.arange(len(signal)), signal, color=colors, linewidth=1, alpha=0.8)

                # --- 2. 绘制单行大标题 ---
                full_title = f"Q: {p_query}\nM: {p_matched}"
                ax.set_title(full_title, fontsize=fs['title'], loc='left', family='monospace', 
                             pad=40, fontweight='bold', linespacing=1.5)

                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                
                # --- 3. 碱基对齐 ---
                base_y = y_min - y_range * 0.15
                for i, (start, end) in enumerate(spans):
                    if i >= len(p_matched): break
                    mid = (start + end) / 2
                    if max(start, plot_range[0]) < min(end, plot_range[1]):
                        ax.vlines([start, end], y_min - y_range*0.05, y_max, colors='gray', linestyles='--', linewidth=0.5, alpha=0.3)
                        ax.text(mid, base_y, p_matched[i], fontsize=fs['base'], fontweight='bold', 
                                ha='center', va='top', family='monospace',
                                bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='#bbbbbb', alpha=0.8))

                # --- 4. Token 旋转对齐 ---
                token_y_base = y_min - y_range * 0.45
                for i, token_pair in enumerate(layered):
                    t_mid = i * 4 + 1.5
                    if plot_range[0] <= t_mid <= plot_range[1]:
                        token_val = str(token_pair[0])
                        y_off = 0 if i % 2 == 0 else y_range * 0.22
                        ax.text(t_mid, token_y_base - y_off, token_val,
                                fontsize=fs['token'], family='monospace', color='white',
                                ha='center', va='top', fontweight='bold', rotation=90,
                                bbox=dict(boxstyle='round,pad=0.2', fc='#333333', ec='none', alpha=0.8))
                        ax.vlines(t_mid, token_y_base - y_off, y_min - y_range*0.25, colors='blue', linestyles=':', linewidth=0.3, alpha=0.2)

                # --- 5. 坐标轴与布局 ---
                ax.set_ylabel("Current (pA)", fontsize=fs['label'], fontweight='bold')
                ax.set_xlabel("Signal Sample Index", fontsize=fs['label'], fontweight='bold')
                ax.set_xlim(plot_range[0], plot_range[1])
                ax.set_ylim(token_y_base - y_range * 0.65, y_max)
                ax.grid(True, linestyle=':', alpha=0.3)
                
                found_valid = True
                break # 只处理第一条
            except Exception as e:
                print(f"⚠️ 跳过错误: {e}")
                continue

    if found_valid:
        plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.92])
        plt.suptitle(f"Single Entry Alignment | View: {plot_range[0]}-{plot_range[1]}", 
                     fontsize=20, y=0.98, fontweight='bold')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ 成功生成单图: {output_path}")
    else:
        print("❌ 未发现有效数据进行绘制")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='single_alignment_report.png')
    parser.add_argument('-r', '--range', type=int, nargs=2, default=[0, 2440])
    args = parser.parse_args()
    visualize_single_alignment(args.input_file, args.output_file, tuple(args.range))