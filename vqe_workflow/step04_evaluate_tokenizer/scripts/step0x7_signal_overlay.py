import gzip
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def visualize_raw_signal(input_path, output_path, max_reads=50, plot_range=(0, 2000)):
    plt.figure(figsize=(18, 7), dpi=200)
    
    print(f"🚀 正在绘制原始信号叠加图 (Raw Signal): {input_path}")
    print(f"📊 绘制范围: 信号点 {plot_range[0]} 到 {plot_range[1]}")
    
    read_count = 0
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if read_count >= max_reads:
                break
            
            try:
                data = json.loads(line)
                # 直接获取原始信号列表
                raw_signal = data.get('signal', [])
                
                if not raw_signal:
                    continue
                
                # 截取用户指定的展示范围
                # 如果信号长度不足，会自动截断到末尾
                display_sig = raw_signal[plot_range[0]:plot_range[1]]
                
                # 创建对应的 X 轴坐标（信号点索引）
                x_axis = np.arange(len(display_sig)) + plot_range[0]
                
                # 叠加绘制：使用极低的 alpha (0.15) 来观察重合密度
                # 不做任何 Z-Score 归一化，保留原始电流值
                plt.plot(x_axis, display_sig, color='tab:blue', alpha=0.3, linewidth=1.0)
                
                read_count += 1
                if read_count % 10 == 0:
                    print(f"⏳ 已读取 {read_count} 条 Read...", end='\r')
            except Exception as e:
                continue

    if read_count == 0:
        print("❌ 未找到有效信号数据")
        return

    # 设置标题和标签
    plt.title(f"Raw Signal Consistency Check (N={read_count} Reads)\nDataset: {os.path.basename(input_path)}", fontsize=14)
    plt.xlabel("Signal Point Index (Time Step)", fontsize=12)
    plt.ylabel("Raw Signal Value (Current / pA)", fontsize=12)
    
    # 根据你的统计结果优化网格
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xlim(plot_range[0], plot_range[1])
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n✅ 原始信号叠加图已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制原始信号叠加图以验证一致性')
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, default='raw_signal_overlay.png')
    parser.add_argument('-n', '--max-reads', type=int, default=50, help='叠加数量')
    parser.add_argument('-r', '--range', type=int, nargs=2, default=[0, 2400], help='信号点的范围')
    
    args = parser.parse_args()
    visualize_raw_signal(args.input_file, args.output_file, args.max_reads, tuple(args.range))