# -*- coding: utf-8 -*-
import gzip
import json
import csv
import argparse
from collections import Counter
from tqdm import tqdm

def process_token_counts(input_path, output_path):
    # 使用 Counter 存储 token_id 的频次
    token_counts = Counter()
    total_count = 0  # 用于记录总 token 数以计算 ratio

    print(f"📖 正在读取并统计: {input_path}")

    try:
        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="处理进度"):
                try:
                    data = json.loads(line.strip())
                    
                    # 直接获取 tokens 字段
                    tokens = data.get('tokens', [])

                    # 只统计 token_id，不再与 base 配对
                    for t_id in tokens:
                        token_counts[t_id] += 1
                        total_count += 1 
                except (json.JSONDecodeError, TypeError, KeyError):
                    # 略过格式不正确的行
                    continue
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {input_path}")
        return

    if total_count == 0:
        print("⚠️ 警告：没有统计到任何有效的 token。")
        return

    # 按照 count 降序排列
    print(f"📊 正在进行排序并计算比例 (Total Tokens: {total_count})...")
    sorted_items = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"💾 正在写入 CSV: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头，移除了 base 字段
        writer.writerow(['id', 'token_id', 'count', 'ratio'])
        for idx, (t_id, count) in enumerate(sorted_items, 1):
            # 计算占比
            ratio = count / total_count
            # 使用 format 确保即使数值很小也能以非科学计数法完整输出 9 位
            ratio_str = "{:.9f}".format(ratio)

            writer.writerow([idx, t_id, count, ratio_str])

    print(f"✅ 处理完成！总计 Token 数: {total_count}, 不同 Token 种类: {len(token_counts)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='统计 tokens 字段的频次分布并计算占比')
    parser.add_argument('-i', '--input', type=str, required=True, help='输入的 jsonl.gz 文件')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出的 CSV 文件')

    args = parser.parse_args()
    process_token_counts(args.input, args.output)
