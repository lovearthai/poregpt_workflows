import json
import gzip
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_signal_by_stride():
    parser = argparse.ArgumentParser(description="根据 Token 层级定位信号并保存叠加波形图（不区分碱基）")
    parser.add_argument("--input", type=str, required=True, help="输入的 .jsonl.gz 文件路径")
    parser.add_argument("--target_token", type=int, default=406, help="目标唯一 Token ID")
    parser.add_argument("--stride_factor", type=int, default=4, help="步长因子")
    parser.add_argument("--half_window", type=int, default=25, help="波形切片半径")
    parser.add_argument("--max_plot_lines", type=int, default=100, help="最大绘制线条数")
    parser.add_argument("--alpha", type=float, default=0.2, help="透明度")
    parser.add_argument("--output", type=str, default="", help="保存路径")
    
    # 层级参数
    parser.add_argument("--layer", type=int, default=0, help="0: tokens字段; >0: tokens_layered拼接前N层")
    parser.add_argument("--codebook_size", type=int, default=512, help="每层Codebook大小")

    args = parser.parse_args()

    collected_segments = []

    with gzip.open(args.input, 'rt') as f:
        for line in f:
            if len(collected_segments) >= args.max_plot_lines:
                break
            try:
                data = json.loads(line.strip())
            except:
                continue

            signal = data.get("signal", [])
            
            # 确定当前样本的 Token ID 序列
            current_tokens = []
            if args.layer == 0:
                current_tokens = data.get("tokens", [])
            else:
                tokens_layered = data.get("tokens_layered", [])
                for step in tokens_layered:
                    uni_id = 0
                    for i in range(args.layer):
                        uni_id = uni_id * args.codebook_size + step[i]
                    current_tokens.append(uni_id)

            # 遍历并比对 (去掉了对 base 的 zip 和判断)
            for i, t_id in enumerate(current_tokens):
                if t_id == args.target_token:
                    center = i * args.stride_factor
                    left, right = center - args.half_window, center + args.half_window
                    
                    if left >= 0 and right <= len(signal):
                        collected_segments.append(signal[left:right])
                    
                    if len(collected_segments) >= args.max_plot_lines:
                        break

    if not collected_segments:
        print(f"Skipped: No data for Token {args.target_token} (Mode: Layer {args.layer})")
        return

    # --- 绘图逻辑 ---
    plt.figure(figsize=(10, 5))
    x = np.arange(-args.half_window, args.half_window)
    
    # 绘制所有采集到的线段
    for seg in collected_segments:
        if len(seg) == len(x):
            plt.plot(x, seg, color='tab:blue', alpha=args.alpha, linewidth=0.8)

    # 绘制平均线 (可选，能更清晰地看到该 Token 的特征波形)
    avg_seg = np.mean(collected_segments, axis=0)
    plt.plot(x, avg_seg, color='red', linewidth=2, label='Mean Signal', alpha=0.8)

    title_str = f"Token: {args.target_token} | Total Segments: {len(collected_segments)} | Mode: Layer {args.layer}"
    plt.title(title_str)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5) # 中心对齐线
    plt.xlabel("Offset from Stride Center")
    plt.ylabel("Normalized Signal")
    plt.grid(True, alpha=0.2)
    plt.legend()

    # 保存
    if args.output:
        save_path = args.output
    else:
        save_path = f"t{args.target_token}_l{args.layer}_nobase.png"

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Successfully saved to: {save_path} (Included {len(collected_segments)} segments)")

if __name__ == "__main__":
    plot_signal_by_stride()
