import json
import gzip
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_signal_by_stride():
    parser = argparse.ArgumentParser(description="根据 Token 层级定位信号并保存 PNG")
    parser.add_argument("--input", type=str, required=True, help="输入的 .jsonl.gz 文件路径")
    parser.add_argument("--target_token", type=int, default=406, help="目标唯一 Token ID")
    parser.add_argument("--target_base", type=str, default="A", help="目标碱基")
    parser.add_argument("--stride_factor", type=int, default=4, help="步长因子")
    parser.add_argument("--half_window", type=int, default=50, help="波形切片半径")
    parser.add_argument("--max_plot_lines", type=int, default=100, help="最大绘制线条数")
    parser.add_argument("--alpha", type=float, default=0.2, help="透明度")
    parser.add_argument("--output", type=str, default="", help="保存路径")
    
    # 新增参数
    parser.add_argument("--layer", type=int, default=0, help="如果为0则用tokens字段，>0则从tokens_layered拼接前N层")
    parser.add_argument("--codebook_size", type=int, default=512, help="计算唯一ID时每层Codebook的大小")

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
            tokens_based = data.get("tokens_based", "")
            
            # 确定当前样本的 Token ID 序列
            current_tokens = []
            if args.layer == 0:
                # 模式0：直接取 tokens 字段
                current_tokens = data.get("tokens", [])
            else:
                # 模式 > 0：从 tokens_layered 拼接
                tokens_layered = data.get("tokens_layered", [])
                for step in tokens_layered:
                    # 计算拼接 ID: 类似于进制转换
                    # 例如 layer=2: step[0] * size + step[1]
                    uni_id = 0
                    for i in range(args.layer):
                        uni_id = uni_id * args.codebook_size + step[i]
                    current_tokens.append(uni_id)

            # 遍历并比对
            for i, (t_id, base) in enumerate(zip(current_tokens, tokens_based)):
                if t_id == args.target_token and base == args.target_base:
                    center = i * args.stride_factor
                    left, right = center - args.half_window, center + args.half_window
                    
                    if left >= 0 and right <= len(signal):
                        collected_segments.append(signal[left:right])
                    
                    if len(collected_segments) >= args.max_plot_lines:
                        break

    if not collected_segments:
        print(f"Skipped: No data for Token {args.target_token} (Layer mode: {args.layer})")
        return

    # 绘图
    plt.figure(figsize=(10, 5))
    x = np.arange(-args.half_window, args.half_window)
    for seg in collected_segments:
        if len(seg) == len(x):
            plt.plot(x, seg, color='tab:blue', alpha=args.alpha, linewidth=0.8)

    title_str = f"Token: {args.target_token} | Base: {args.target_base} | Mode: Layer {args.layer}"
    plt.title(title_str)
    plt.axvline(x=0, color='red', linestyle=':', alpha=0.5)
    plt.grid(True, alpha=0.3)

    if args.output:
        save_path = args.output
    else:
        save_path = f"t{args.target_token}_l{args.layer}_s{args.stride_factor}.png"

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path} | Count: {len(collected_segments)}")

if __name__ == "__main__":
    plot_signal_by_stride()
