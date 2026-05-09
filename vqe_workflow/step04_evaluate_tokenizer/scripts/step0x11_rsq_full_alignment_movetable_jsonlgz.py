import gzip
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 🔥 导入工具函数（带增强 fallback）
# ===============================
from poregpt.utils import get_rsq_coords_from_integer


def get_adaptive_fonts(width, seq_len):
    base_width = 1000
    w_scale = np.clip((base_width / width) ** 0.45, 0.5, 1.8)
    s_scale = max(0.4, 150 / seq_len) if seq_len > 150 else 1.0
    return {
        'title': 14 * min(w_scale, 1.2) * s_scale,
        'base': 13 * w_scale,
        'token': 12 * w_scale,
        'label': 13 * min(w_scale, 1.1),
        'h_pad': 3.5 * w_scale
    }


def visualize_rsq_integrated(input_path, output_path, plot_range, levels):
    # --- 核心配置 ---
    RSQ_LEVELS = levels
    NUM_Q = 1

    # 🔥 自动 padding（关键改动）
    codebook_size = int(np.prod(RSQ_LEVELS))
    ID_PADDING = len(str(codebook_size - 1))

    view_width = plot_range[1] - plot_range[0]
    fig, axes = plt.subplots(3, 1, figsize=(30, 20), dpi=150)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    valid_count = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if valid_count >= 3:
                break
            try:
                data = json.loads(line)
                signal = data.get('signal', [])
                pattern = data.get('pattern', "")
                spans = data.get('base_sample_spans_rel', [])
                layered = data.get('tokens_layered', [])

                if not signal:
                    continue

                ax = axes[valid_count]
                fs = get_adaptive_fonts(view_width, len(pattern))

                # =========================
                # 1. 信号
                # =========================
                ax.plot(np.arange(len(signal)), signal,
                        color=colors[valid_count], linewidth=1.8, alpha=0.9)

                ax.set_title(
                    f"Pattern: {pattern}",
                    fontsize=fs['title'],
                    loc='left',
                    family='monospace',
                    pad=30,
                    fontweight='bold',
                    linespacing=1.5
                )

                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min

                # =========================
                # 2. 碱基
                # =========================
                base_y = y_min - y_range * 0.09
                for i, (start, end) in enumerate(spans):
                    if i >= len(pattern):
                        break

                    mid = (start + end) / 2

                    if max(start, plot_range[0]) < min(end, plot_range[1]):
                        ax.vlines([start, end],
                                  y_min - y_range * 0.03, y_max,
                                  colors=colors[valid_count],
                                  linestyles='--',
                                  linewidth=1.2,
                                  alpha=0.6)

                        ax.text(mid, base_y, pattern[i],
                                fontsize=fs['base'],
                                fontweight='bold',
                                ha='center',
                                va='top',
                                family='monospace',
                                bbox=dict(boxstyle='circle,pad=0.2',
                                          fc='white', ec='#bbbbbb', alpha=0.85))

                # =========================
                # 3. Token
                # =========================
                visible_tokens = [
                    (i, token_pair)
                    for i, token_pair in enumerate(layered)
                    if plot_range[0] <= (i * 4 + 1.5) <= plot_range[1]
                ]

                token_count = len(visible_tokens)
                token_font = fs['token']

                if token_count > 16:
                    token_font = max(6, fs['token'] * (300.0 / max(view_width, 300)))
                elif token_count > 10:
                    token_font = max(7, fs['token'] * 0.85)

                single_line_tokens = token_count <= 12 or view_width < 320
                token_y_base = y_min - y_range * (0.24 if single_line_tokens else 0.42)

                all_token_ys = []
                for idx, (i, token_pair) in enumerate(visible_tokens):
                    y_off = 0 if (single_line_tokens or idx % 2 == 0) else y_range * 0.18
                    token_y = token_y_base - y_off
                    all_token_ys.append(token_y)

                if all_token_ys:
                    min_token_y = min(all_token_ys)
                    y_axis_bottom = min_token_y - y_range * 0.15
                else:
                    y_axis_bottom = token_y_base - y_range * 0.22

                for idx, (i, token_pair) in enumerate(visible_tokens):
                    t_mid = i * 4 + 1.5

                    raw_tid = token_pair[0]
                    tid_int = int(raw_tid[0] if isinstance(raw_tid, list) else raw_tid)

                    # 🔥 自动 padding
                    tid_padded = str(tid_int).zfill(ID_PADDING)

                    # 🔥 解码（带 fallback）
                    coord_idxs = get_rsq_coords_from_integer(
                        tid_int,
                        RSQ_LEVELS,
                        num_quantizers=NUM_Q,
                        use_fast=True
                    )

                    if isinstance(coord_idxs, list) and len(coord_idxs) == 1 and isinstance(coord_idxs[0], list):
                        coord_idxs = coord_idxs[0]

                    # 👉 保持你原来的紧凑格式
                    #idx_str = "".join(map(str, coord_idxs))
                    # 使用修改后的代码
                    idx_str = "".join([str(x) if x != 10 else 'a' for x in coord_idxs])
                    
                    display_text = f"{tid_padded}:{idx_str}"

                    y_off = 0 if (single_line_tokens or idx % 2 == 0) else y_range * 0.18
                    token_y = token_y_base - y_off

                    ax.text(t_mid, token_y, display_text,
                            fontsize=token_font,
                            family='monospace',
                            color='white',
                            ha='center',
                            va='top',
                            fontweight='bold',
                            rotation=90,
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc='#333333', ec='none', alpha=0.85))

                    ax.vlines(t_mid, token_y,
                              y_min - y_range * 0.28,
                              colors='blue',
                              linestyles=':',
                              linewidth=0.3,
                              alpha=0.2)

                # =========================
                # 4. 坐标轴
                # =========================
                ax.set_ylabel("Current (pA)", fontsize=fs['label'], fontweight='bold')
                ax.set_xlim(plot_range[0], plot_range[1])
                ax.set_ylim(y_axis_bottom, y_max)
                ax.grid(True, linestyle=':', alpha=0.3)

                valid_count += 1

            except Exception as e:
                print(f"⚠️ 跳过错误: {e}")
                continue

    axes[2].set_xlabel("Signal Sample Index", fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94], h_pad=fs['h_pad'])

    plt.suptitle(
        f"Adaptive Single-Line Mapping | View: {plot_range[0]}-{plot_range[1]} | Levels={levels}",
        fontsize=20,
        y=0.98,
        fontweight='bold'
    )

    plt.savefig(output_path, bbox_inches='tight')

    print(f"✅ 成功生成。已适配 levels={levels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='final_view_single_line.png')
    parser.add_argument('-r', '--range', type=int, nargs=2, default=[0, 2440])

    # 🔥 新增 levels
    parser.add_argument('--levels', type=int, nargs='+', required=True)

    args = parser.parse_args()

    visualize_rsq_integrated(
        args.input_file,
        args.output_file,
        tuple(args.range),
        args.levels
    )
