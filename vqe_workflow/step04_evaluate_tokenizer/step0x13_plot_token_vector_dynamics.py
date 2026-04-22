import gzip
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

# 尝试导入核心工具函数
from poregpt.utils import get_rsq_coords_from_integer, get_rsq_vector_from_integer

def get_adaptive_fonts(width, seq_len):
    seq_len = max(1, seq_len)
    base_width = 1000
    w_scale = np.clip((base_width / width) ** 0.45, 0.5, 1.8)
    s_scale = max(0.4, 150 / seq_len) if seq_len > 150 else 1.0
    return {
        'title': 14 * min(w_scale, 1.2) * s_scale,
        'base': 13 * w_scale,
        'token': 9.0 * w_scale,
        'label': 13 * min(w_scale, 1.1),
    }

def calculate_vector_diff(layered_tokens, win_size, win_stride):
    if not layered_tokens or len(layered_tokens) < win_size:
        return np.array([]), np.array([])

    RSQ_LEVELS = [5, 5, 5, 5]
    vectors = []
    for tp in layered_tokens:
        raw_tid = tp[0]
        tid = int(np.array(raw_tid).flatten()[0])
        v = get_rsq_vector_from_integer(tid, RSQ_LEVELS, num_quantizers=1, use_fast=True)
        vectors.append(np.array(v).flatten())

    vec_matrix = np.array(vectors)
    n_tokens = len(vec_matrix)
    window_centers, time_indices = [], []

    for start in range(0, n_tokens - win_size + 1, win_stride):
        window_data = vec_matrix[start:start + win_size]
        center_vec = np.mean(window_data, axis=0)
        window_centers.append(center_vec)
        time_indices.append(start + 0.375)

    window_centers = np.array(window_centers)
    if len(window_centers) == 0:
        return np.array([]), np.array([])

    diff_vectors = np.diff(window_centers, axis=0, prepend=[window_centers[0]])
    diff_values = np.linalg.norm(diff_vectors, axis=1)
    return np.array(time_indices), np.array(diff_values)


def visualize_integrated_dynamics_bak(args):
    RSQ_LEVELS = [5, 5, 5, 5]
    ID_PADDING = 3

    target_data = None
    print(f"🔍 正在读取 Line ID: {args.line_id}...")
    with gzip.open(args.input_file, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == args.line_id:
                target_data = json.loads(line)
                break

    if not target_data:
        print(f"❌ 错误: 未找到 Line ID {args.line_id}")
        return

    # 提取信号数据
    signal = np.array(target_data.get('signal', []))
    recon = np.array(target_data.get('recon', []))
    recon_l1 = np.array(target_data.get('recon_layer1', []))
    layered = target_data.get('tokens_layered', [])
    pattern = target_data.get('pattern', "")
    spans = target_data.get('base_sample_spans_rel', [])

    real_token_count = len(layered)
    safe_token_start = max(0, min(args.token_start, real_token_count - 1))
    safe_token_end = max(safe_token_start + 3, min(args.token_end, real_token_count))
    total_plot_tokens = safe_token_end - safe_token_start

    # --- 逻辑修改：动态计算分段 ---
    TOKENS_PER_PLOT = 100
    if total_plot_tokens <= TOKENS_PER_PLOT:
        num_subplots = 1
        segments = [(safe_token_start, safe_token_end)]
    else:
        num_subplots = int(np.ceil(total_plot_tokens / TOKENS_PER_PLOT))
        segments = []
        for i in range(num_subplots):
            start = safe_token_start + i * TOKENS_PER_PLOT
            end = min(start + TOKENS_PER_PLOT, safe_token_end)
            segments.append((start, end))

    # 动态调整 figsize，每多一个 subplot 增加高度
    fig_height = 8 * num_subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(40, fig_height), dpi=150, squeeze=False)
    axes = axes.flatten() # 确保 axes 总是可迭代的
    
    colors = plt.cm.get_cmap('tab10').colors # 使用循环颜色组
    # -----------------------------

    for i, (t_start, t_end) in enumerate(segments):
        ax = axes[i]
        color_idx = i % len(colors)
        
        s_start, s_end = t_start * args.signal_stride, t_end * args.signal_stride

        # --- 对齐信号长度 ---
        sub_signal = signal[s_start:s_end]
        if len(sub_signal) == 0: continue
        curr_len = len(sub_signal)
        current_x = np.arange(s_start, s_start + curr_len)

        sub_recon = recon[s_start:s_start + curr_len] if len(recon) > 0 else []
        sub_recon_l1 = recon_l1[s_start:s_start + curr_len] if len(recon_l1) > 0 else []

        sub_layered = layered[t_start:t_end]
        fs = get_adaptive_fonts(curr_len, len(sub_layered))

        # A. 绘制信号对比
        ax.plot(current_x, sub_signal, color=colors[color_idx], linewidth=1.5, alpha=0.35, label='Original Signal')

        if len(sub_recon) > 0:
            ax.plot(current_x[:len(sub_recon)], sub_recon, color='darkorange',
                    linestyle='--', linewidth=1.5, alpha=0.9, label='Recon (Full)')

        if len(sub_recon_l1) > 0:
            ax.plot(current_x[:len(sub_recon_l1)], sub_recon_l1, color='magenta',
                    linestyle=':', linewidth=1.5, alpha=0.9, label='Recon (Layer 1)')

        ax.set_ylabel("Current (pA)", fontsize=fs['label'], fontweight='bold')
        ax.legend(loc='upper right', fontsize=fs['token'], frameon=True)

        ax.relim()
        ax.autoscale_view()
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # B. 动力学 (右轴)
        ax_twin = ax.twinx()
        t_idx_rel, v_diff = calculate_vector_diff(sub_layered, args.window_size, args.window_stride)
        if len(v_diff) > 0:
            t_plot = (t_idx_rel + t_start) * args.signal_stride
            ax_twin.bar(t_plot, v_diff, width=args.signal_stride*0.8, color='purple', alpha=0.05, clip_on=False)
            ax_twin.set_ylabel("L2 Distance", color='purple', fontsize=fs['label'], fontweight='bold')
            ax_twin.set_ylim(0, np.max(v_diff) * 1.5)
        else:
            ax_twin.set_visible(False)

        # C. 碱基绘制逻辑
        base_y = y_min - y_range * 0.05
        for b_idx, (b_start, b_end) in enumerate(spans):
            if b_idx >= len(pattern): break
            # 只有当碱基的范围与当前视图有交集时才绘制
            if max(b_start, s_start) < min(b_end, s_end):
                mid = (b_start + b_end) / 2
                if s_start <= b_start <= s_end:
                    ax.vlines(b_start, y_min, y_max, colors=colors[color_idx], linestyles='--', alpha=0.4)
                if s_start <= mid <= s_end:
                    ax.text(mid, base_y, pattern[b_idx], fontsize=fs['base'], fontweight='bold',
                            ha='center', va='top', bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='#bbbbbb', alpha=0.9))

        # D. Token 详情
        token_y_outside = y_min - y_range * 0.15
        for j, tp in enumerate(sub_layered):
            t_mid = (t_start + j) * args.signal_stride + (args.signal_stride / 2)
            tid = int(np.array(tp[0]).flatten()[0])
            coords = get_rsq_coords_from_integer(tid, RSQ_LEVELS, num_quantizers=1)
            coord_str = "".join(map(str, coords[0] if isinstance(coords[0], list) else coords))

            ax.text(t_mid, token_y_outside, f"{str(tid).zfill(ID_PADDING)}:{coord_str}",
                    fontsize=fs['token'], rotation=90, ha='center', va='top',
                    color='black', fontweight='bold', clip_on=False,
                    bbox=dict(boxstyle='round,pad=0.2', fc='#eeeeee', ec='#cccccc', alpha=0.8))

            ax.vlines(t_mid, token_y_outside, y_min, colors='gray', linestyles=':', alpha=0.3, clip_on=False)

        ax.set_xlim(s_start, s_end)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)

    plt.suptitle(f"Line {args.line_id} | Tokens {safe_token_start}-{safe_token_end}\nSignal Reconstruction & Dynamics Alignment",
                 fontsize=22, y=0.98, fontweight='bold')

    # 根据子图数量自动调整间距
    plt.subplots_adjust(hspace=0.6 if num_subplots > 1 else 0.4, bottom=0.1, top=0.92, left=0.05, right=0.95)

    plt.savefig(args.output_file, bbox_inches='tight')
    print(f"✅ 完成！共绘制 {num_subplots} 个片段。保存至: {args.output_file}")


def visualize_integrated_dynamics(args):
    RSQ_LEVELS = [5, 5, 5, 5]
    ID_PADDING = 3
    # 按照你的要求，设定固定的 Y 轴范围
    Y_MIN_FIXED = -3
    Y_MAX_FIXED = 3
    Y_RANGE_FIXED = Y_MAX_FIXED - Y_MIN_FIXED # 等于 6

    target_data = None
    print(f"🔍 正在读取 Line ID: {args.line_id}...")
    with gzip.open(args.input_file, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == args.line_id:
                target_data = json.loads(line)
                break

    if not target_data:
        print(f"❌ 错误: 未找到 Line ID {args.line_id}")
        return

    # 提取数据
    signal = np.array(target_data.get('signal', []))
    recon = np.array(target_data.get('recon', []))
    recon_l1 = np.array(target_data.get('recon_layer1', []))
    layered = target_data.get('tokens_layered', [])
    pattern = target_data.get('pattern', "")
    spans = target_data.get('base_sample_spans_rel', [])

    real_token_count = len(layered)
    safe_token_start = max(0, min(args.token_start, real_token_count - 1))
    safe_token_end = max(safe_token_start + 3, min(args.token_end, real_token_count))
    total_plot_tokens = safe_token_end - safe_token_start

    # --- 分段逻辑：超过 100 tokens 自动切分 ---
    TOKENS_PER_PLOT = 100
    if total_plot_tokens <= TOKENS_PER_PLOT:
        num_subplots = 1
        segments = [(safe_token_start, safe_token_end)]
    else:
        num_subplots = int(np.ceil(total_plot_tokens / TOKENS_PER_PLOT))
        segments = []
        for i in range(num_subplots):
            start = safe_token_start + i * TOKENS_PER_PLOT
            end = min(start + TOKENS_PER_PLOT, safe_token_end)
            segments.append((start, end))

    fig_height = 8 * num_subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(30, fig_height), dpi=150, squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.get_cmap('tab10').colors

    for i, (t_start, t_end) in enumerate(segments):
        ax = axes[i]
        color_idx = i % len(colors)
        
        s_start, s_end = t_start * args.signal_stride, t_end * args.signal_stride
        sub_signal = signal[s_start:s_end]
        if len(sub_signal) == 0: continue
        
        curr_len = len(sub_signal)
        current_x = np.arange(s_start, s_start + curr_len)
        sub_recon = recon[s_start:s_start + curr_len] if len(recon) > 0 else []
        sub_recon_l1 = recon_l1[s_start:s_start + curr_len] if len(recon_l1) > 0 else []

        sub_layered = layered[t_start:t_end]
        fs = get_adaptive_fonts(curr_len, len(sub_layered))

        # A. 信号绘制
        ax.plot(current_x, sub_signal, color=colors[color_idx], linewidth=1.5, alpha=0.35, label='Original Signal')
        if len(sub_recon) > 0:
            ax.plot(current_x[:len(sub_recon)], sub_recon, color='darkorange',
                    linestyle='--', linewidth=1.5, alpha=0.9, label='Recon (Full)')
        if len(sub_recon_l1) > 0:
            ax.plot(current_x[:len(sub_recon_l1)], sub_recon_l1, color='magenta',
                    linestyle=':', linewidth=1.5, alpha=0.9, label='Recon (Layer 1)')

        # --- 强制设定 Y 轴范围 ---
        ax.set_ylim(Y_MIN_FIXED, Y_MAX_FIXED)
        ax.set_ylabel("Current (Normalized)", fontsize=fs['label'], fontweight='bold')
        ax.legend(loc='upper right', fontsize=fs['token'], frameon=True)

        # B. 动力学 (右轴) - 独立缩放不受信号 ylim 影响
        ax_twin = ax.twinx()
        t_idx_rel, v_diff = calculate_vector_diff(sub_layered, args.window_size, args.window_stride)
        if len(v_diff) > 0:
            t_plot = (t_idx_rel + t_start) * args.signal_stride
            ax_twin.bar(t_plot, v_diff, width=args.signal_stride*0.8, color='purple', alpha=0.05, clip_on=False)
            ax_twin.set_ylabel("L2 Distance", color='purple', fontsize=fs['label'], fontweight='bold')
            ax_twin.set_ylim(0, np.max(v_diff) * 1.5 if np.max(v_diff) > 0 else 1)
        else:
            ax_twin.set_visible(False)

        # C. 碱基绘制 (基于固定坐标计算垂直偏移)
        base_y = Y_MIN_FIXED - Y_RANGE_FIXED * 0.05
        for b_idx, (b_start, b_end) in enumerate(spans):
            if b_idx >= len(pattern): break
            if max(b_start, s_start) < min(b_end, s_end):
                mid = (b_start + b_end) / 2
                if s_start <= b_start <= s_end:
                    ax.vlines(b_start, Y_MIN_FIXED, Y_MAX_FIXED, colors=colors[color_idx], linestyles='--', alpha=0.4)
                if s_start <= mid <= s_end:
                    ax.text(mid, base_y, pattern[b_idx], fontsize=fs['base'], fontweight='bold',
                            ha='center', va='top', bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='#bbbbbb', alpha=0.9))

        # D. Token 详情 (基于固定坐标下推)
        token_y_outside = Y_MIN_FIXED - Y_RANGE_FIXED * 0.15
        for j, tp in enumerate(sub_layered):
            t_mid = (t_start + j) * args.signal_stride + (args.signal_stride / 2)
            tid = int(np.array(tp[0]).flatten()[0])
            coords = get_rsq_coords_from_integer(tid, RSQ_LEVELS, num_quantizers=1)
            coord_str = "".join(map(str, coords[0] if isinstance(coords[0], list) else coords))

            ax.text(t_mid, token_y_outside, f"{str(tid).zfill(ID_PADDING)}:{coord_str}",
                    fontsize=fs['token'], rotation=90, ha='center', va='top',
                    color='black', fontweight='bold', clip_on=False,
                    bbox=dict(boxstyle='round,pad=0.2', fc='#eeeeee', ec='#cccccc', alpha=0.8))

            ax.vlines(t_mid, token_y_outside, Y_MIN_FIXED, colors='gray', linestyles=':', alpha=0.3, clip_on=False)

        ax.set_xlim(s_start, s_end)
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)

    plt.suptitle(f"Line {args.line_id} | Tokens {safe_token_start}-{safe_token_end}\nSignal Reconstruction & Dynamics Alignment (Fixed Y-Range [-3, 3])",
                 fontsize=22, y=0.98, fontweight='bold')

    plt.subplots_adjust(hspace=0.7, bottom=0.1, top=0.92, left=0.05, right=0.95)

    plt.savefig(args.output_file, bbox_inches='tight')
    print(f"✅ 完成！Y轴固定为 [-3, 3]，共分 {num_subplots} 个片段。保存至: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='segmented_dynamics.png')
    parser.add_argument('--line-id', type=int, default=0)
    parser.add_argument('--token-start', type=int, default=0)
    parser.add_argument('--token-end', type=int, default=150)
    parser.add_argument('--signal-stride', type=int, default=4)
    parser.add_argument('--window-size', type=int, default=1)
    parser.add_argument('--window-stride', type=int, default=1)
    args = parser.parse_args()
    visualize_integrated_dynamics(args)
