"""
================================================================================
脚本名称: Nanopore 信号、隐空间动力学(绝对值柱状图)与 Token 详情联合可视化
代码思路:
1. 动力学绝对值: 差值计算采用 np.abs()，仅观察波动强度。
2. 线性对齐标注: 取消错位排列，所有 Token ID:Coord 在同一水平线对齐展示。
3. 布局优化: 自动扩展底部空间，防止旋转文字溢出。
================================================================================
"""

import gzip
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 尝试导入核心工具函数
from poregpt.utils import get_rsq_coords_from_integer, get_rsq_vector_from_integer

def get_adaptive_fonts(width, seq_len):
    base_width = 1000
    w_scale = np.clip((base_width / width) ** 0.45, 0.5, 1.8)
    s_scale = max(0.4, 150 / seq_len) if seq_len > 150 else 1.0
    return {
        'title': 14 * min(w_scale, 1.2) * s_scale,
        'base': 13 * w_scale,
        'token': 8.5 * w_scale, # 进一步微调字体以适应对齐排列
        'label': 13 * min(w_scale, 1.1),
        'h_pad': 3.5 * w_scale
    }



def calculate_vector_diff(layered_tokens, win_size, win_stride):
    RSQ_LEVELS = [5, 5, 5, 5]
    vectors = []
    tids = [] 
    
    for tp in layered_tokens:
        raw_tid = tp[0]
        # 兼容性处理：有些数据可能是 [[tid]] 格式，这里将其拍平
        tid = int(np.array(raw_tid).flatten()[0])
        tids.append(tid)
        v = get_rsq_vector_from_integer(tid, RSQ_LEVELS, num_quantizers=1, use_fast=True)
        # 确保 v 是 (4,) 而不是 (1, 4)
        vectors.append(np.array(v).flatten())

    vec_matrix = np.array(vectors) # 期望 shape: (n_tokens, 4)
    n_tokens = len(vec_matrix)
    
    window_centers, time_indices = [], []

    for start in range(0, n_tokens - win_size + 1, win_stride):
        window_data = vec_matrix[start:start + win_size]
        center_vec = np.mean(window_data, axis=0) 
        window_centers.append(center_vec)
        time_indices.append(start * 4 + 1.5)

    window_centers = np.array(window_centers)
    
    # 计算欧氏距离
    diff_vectors = np.diff(window_centers, axis=0, prepend=[window_centers[0]])
    diff_values = np.linalg.norm(diff_vectors, axis=1)

    # --- 深度调试逻辑 ---
    zero_mask = (diff_values == 0)
    zero_mask[0] = False 
    
    if np.any(zero_mask):
        zero_indices = np.where(zero_mask)[0]
        print(f"\n[🔬 Deep Vector Debug] Found {len(zero_indices)} zero diff points.")
        
        for idx in zero_indices[:3]:
            c_s = idx * win_stride
            p_s = (idx - 1) * win_stride
            
            p_tids = tids[p_s : p_s + win_size]
            c_tids = tids[c_s : c_s + win_size]
            
            p_vecs = vec_matrix[p_s : p_s + win_size]
            c_vecs = vec_matrix[c_s : c_s + win_size]
            
            p_center = window_centers[idx-1]
            c_center = window_centers[idx]
            
            # 核心修复：使用 .item() 确保提取的是 Python 标量
            # 即使 diff_values[idx] 是 array([0.]) 也能安全转为 0.0
            try:
                dist_val = diff_values[idx].item()
            except:
                dist_val = float(diff_values[idx][0]) if hasattr(diff_values[idx], "__len__") else float(diff_values[idx])

            print(f"--- Zero Diff at Sample {time_indices[idx]:.1f} ---")
            print(f"  IDs:    Prev {p_tids} -> Curr {c_tids}")
            np.set_printoptions(precision=4, suppress=True)
            print(f"  Vector (Prev Window Matrix):\n{p_vecs}")
            print(f"  Vector (Curr Window Matrix):\n{c_vecs}")
            print(f"  Window Centroid: Prev={p_center.flatten()}, Curr={c_center.flatten()}")
            print(f"  Euclidean Distance: {dist_val:.6f}")
            print("-" * 40)
            
    return np.array(time_indices), np.array(diff_values)
def visualize_integrated_dynamics(args):
    RSQ_LEVELS = [5, 5, 5, 5]
    ID_PADDING = 3
    plot_range = tuple(args.range)
    view_width = plot_range[1] - plot_range[0]

    fig, axes = plt.subplots(3, 1, figsize=(30, 22), dpi=150)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    valid_count = 0

    with gzip.open(args.input_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if valid_count >= 3: break
            try:
                data = json.loads(line)
                signal, pattern = data.get('signal', []), data.get('pattern', "")
                spans, layered = data.get('base_sample_spans_rel', []), data.get('tokens_layered', [])
                if not signal: continue

                ax = axes[valid_count]
                fs = get_adaptive_fonts(view_width, len(pattern))

                # --- 1. 左轴 (Signal) ---
                ax.plot(np.arange(len(signal)), signal, color=colors[valid_count], linewidth=1.5, alpha=0.6)
                ax.set_ylabel("Current (pA)", color=colors[valid_count], fontsize=fs['label'], fontweight='bold')
                ax.set_title(f"Pattern: {pattern}", fontsize=fs['title'], loc='left', family='monospace', fontweight='bold', pad=30)

                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min

                # --- 2. 右轴 (Latent Dynamics) ---
                ax_twin = ax.twinx()
                t_idx, v_diff = calculate_vector_diff(layered, args.window_size, args.window_stride)
                t_idx, v_diff = np.array(t_idx), np.array(v_diff)
                mask = (t_idx >= plot_range[0]) & (t_idx <= plot_range[1])
                
                bar_width = 4.0 * args.window_stride * 0.7 
                ax_twin.bar(t_idx[mask], v_diff[mask], width=bar_width, color='purple', 
                            alpha=0.25, edgecolor='purple', linewidth=0.4)
                ax_twin.set_ylabel("|Δ Latent Euclidean Distance |", color='purple', fontsize=fs['label'], fontweight='bold')
                ax_twin.tick_params(axis='y', labelcolor='purple')

                # --- 3. 碱基绘制 (Circle Labels) ---
                base_y = y_min - y_range * 0.08
                for i, (start, end) in enumerate(spans):
                    if i >= len(pattern): break
                    mid = (start + end) / 2
                    if max(start, plot_range[0]) < min(end, plot_range[1]):
                        ax.vlines([start, end], y_min - y_range*0.02, y_max, colors=colors[valid_count], linestyles='--', linewidth=1.0, alpha=0.3)
                        ax.text(mid, base_y, pattern[i], fontsize=fs['base'], fontweight='bold', ha='center', va='top', family='monospace',
                                bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='#bbbbbb', alpha=0.9))

                # --- 4. Token ID & Coord 展示 (线性对齐版) ---
                visible_tokens = [(i, tp) for i, tp in enumerate(layered) if plot_range[0] <= (i * 4 + 1.5) <= plot_range[1]]
                
                # 统一所有 Token 的 Y 轴起始高度，不再使用错位逻辑
                token_y_fixed = y_min - y_range * 0.22 

                for i, token_pair in visible_tokens:
                    t_mid = i * 4 + 1.5
                    tid_int = int(token_pair[0][0] if isinstance(token_pair[0], list) else token_pair[0])
                    coords = get_rsq_coords_from_integer(tid_int, RSQ_LEVELS, num_quantizers=1)
                    if isinstance(coords, list) and len(coords) == 1: coords = coords[0]
                    coord_str = "".join(map(str, coords))
                    
                    display_text = f"{str(tid_int).zfill(ID_PADDING)}:{coord_str}"

                    # 统一绘制在 token_y_fixed
                    ax.text(t_mid, token_y_fixed, display_text, fontsize=fs['token'], family='monospace', color='white',
                            ha='center', va='top', fontweight='bold', rotation=90,
                            bbox=dict(boxstyle='round,pad=0.2', fc='#333333', ec='none', alpha=0.4))
                    
                    # 绘制连接虚线
                    ax.vlines(t_mid, token_y_fixed, y_min - y_range*0.02, colors='gray', linestyles=':', linewidth=0.5, alpha=0.3)

                # 5. 最终对齐与边界调整
                ax.set_xlim(plot_range[0], plot_range[1])
                # 增加底部 Y 轴缓冲空间，防止旋转后的长文本被截断
                ax.set_ylim(token_y_fixed - y_range * 0.45, y_max)
                ax.grid(True, axis='x', linestyle=':', alpha=0.2)
                valid_count += 1

            except Exception as e:
                print(f"⚠️ Skip faulty line: {e}")
                continue

    plt.suptitle(f"Nanopore Signal vs Latent Dynamics (Linear Token Alignment)\nWindow Size={args.window_size}, Stride={args.window_stride}", 
                 fontsize=22, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    plt.savefig(args.output_file, bbox_inches='tight')
    print(f"✅ 线性可视化完成: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='dynamics_linear_alignment.png')
    parser.add_argument('-r', '--range', type=int, nargs=2, default=[0, 1000])
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--window-stride', type=int, default=1)
    args = parser.parse_args()
    visualize_integrated_dynamics(args)
