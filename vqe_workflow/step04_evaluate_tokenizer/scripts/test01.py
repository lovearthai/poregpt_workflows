import gzip
import json
import argparse
import os
import numpy as np

def count_signal_points(input_path, max_lines=None):
    total_points = 0
    lengths = []
    line_count = 0

    print(f"🧐 正在扫描文件: {input_path}")

    try:
        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if max_lines and line_count >= max_lines:
                    break
                
                data = json.loads(line)
                sig_len = len(data.get('signal', []))
                
                lengths.append(sig_len)
                total_points += sig_len
                line_count += 1
                
                if line_count % 1000 == 0:
                    print(f"⏳ 已处理 {line_count} 行...", end='\r')
    except Exception as e:
        print(f"❌ 读取出错: {e}")
        return

    if line_count == 0:
        print("Empty file or no signal field found.")
        return

    # 计算统计指标
    print("\n" + "="*40)
    print("📊 Signal 字段统计报告")
    print("-" * 40)
    print(f"文件总行数:    {line_count}")
    print(f"信号总数值量:  {total_points:,}")
    print(f"单行最大长度:  {max(lengths):,}")
    print(f"单行最小长度:  {min(lengths):,}")
    print(f"平均每行长度:  {np.mean(lengths):.2f}")
    print(f"长度中位数:    {np.median(lengths):.2f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-n', '--max-lines', type=int, default=None, help='限制读取行数，不传则读全表')
    
    args = parser.parse_args()
    count_signal_points(args.input_file, args.max_lines)