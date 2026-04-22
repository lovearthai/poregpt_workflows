# -*- coding: utf-8 -*-
import gzip
import json
import csv
import argparse
from collections import Counter
from tqdm import tqdm

def process_token_base_counts(input_path, output_path):
    # 使用 Counter 存储 (token_id, base) 组合的频次
    pair_counts = Counter()
    total_count = 0  # 用于记录总频次以计算 ratio

    print(f"📖 正在读取并统计: {input_path}")

    try:
        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="处理进度"):
                try:
                    data = json.loads(line.strip())
                    
                    # --- 修改处：直接获取 tokens 字段 ---
                    # 不再处理 tokens_layered 的每一层，直接取 token ID 列表
                    tokens = data.get('tokens', [])

                    # 提取对应的碱基
                    bases = data.get('tokens_based', [])

                    # 使用 zip 将两者配对并更新计数器
                    # 确保只配对 tokens 和 bases 长度相等的部分
                    for t_id, base in zip(tokens, bases):
                        pair_counts[(t_id, base)] += 1
                        total_count += 1 # 累加总数
                except (json.JSONDecodeError, TypeError, KeyError, IndexError):
                    # 略过格式不正确的行
                    continue
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {input_path}")
        return

    if total_count == 0:
        print("⚠️ 警告：没有统计到任何有效的 token-base 组合。")
        return

    # 按照 count 降序排列
    print(f"📊 正在进行排序并计算比例 (Total: {total_count})...")
    sorted_items = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"💾 正在写入 CSV: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['id', 'token_id', 'base', 'count', 'ratio'])

        for idx, ((t_id, base), count) in enumerate(sorted_items, 1):
            # 计算占比并保留 9 位小数
            ratio = count / total_count
            # 使用 format 确保即使数值很小也能以非科学计数法完整输出 9 位
            ratio_str = "{:.9f}".format(ratio)

            writer.writerow([idx, t_id, base, count, ratio_str])

    print(f"✅ 处理完成！总计样本数: {total_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='统计 tokens 字段与碱基的组合频次并计算占比')
    parser.add_argument('-i', '--input', type=str, required=True, help='输入的 jsonl.gz 文件')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出的 CSV 文件')

    args = parser.parse_args()
    process_token_base_counts(args.input, args.output)
