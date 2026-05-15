import os
import gzip
import json
import csv
import glob
import multiprocessing as mp
import argparse
import re
from collections import Counter
from tqdm import tqdm

def process_file(file_path):
    """处理单个文件并返回该文件的 Token 计数器"""
    local_counter = Counter()
    # 预编译正则以提高多行处理效率
    token_pattern = re.compile(r'<\|[^>|]+\|>')
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    # 提取所有符合 <|...|> 格式的 token
                    tokens = token_pattern.findall(text)
                    local_counter.update(tokens)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"\n[Error] 无法读取文件 {file_path}: {e}")
    return local_counter

def main():
    parser = argparse.ArgumentParser(description="多进程统计 jsonl.gz 文件中的 Token 频次")
    parser.add_argument("--input_dir", type=str, required=True, help="包含 jsonl.gz 文件的根目录")
    parser.add_argument("--output_csv", type=str, default="token_counts.csv", help="输出 CSV 文件路径")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="并行进程数，默认使用全部 CPU")
    
    args = parser.parse_args()

    # 1. 递归扫描文件
    print(f"正在扫描目录: {args.input_dir}")
    files = glob.glob(os.path.join(args.input_dir, "**/*.jsonl.gz"), recursive=True)
    
    if not files:
        print("未找到任何 .jsonl.gz 文件，请检查路径。")
        return

    print(f"找到 {len(files)} 个文件。准备使用 {args.workers} 个进程进行处理...")

    # 2. 多进程并行处理
    global_counter = Counter()
    with mp.Pool(args.workers) as pool:
        # 使用 imap_unordered 配合 tqdm 显示进度
        pbar = tqdm(total=len(files), desc="进度", unit="file")
        for result_counter in pool.imap_unordered(process_file, files):
            global_counter.update(result_counter)
            pbar.update(1)
        pbar.close()

    # 3. 结果排序并写入 CSV
    print(f"正在按频次倒序排列并写入: {args.output_csv}")
    # most_common() 返回按 count 排序的 list
    sorted_counts = global_counter.most_common()

    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["token", "frequency"])
            writer.writerows(sorted_counts)
        print(f"✅ 统计完成！总计不同 Token 数量: {len(sorted_counts)}")
    except Exception as e:
        print(f"写入 CSV 失败: {e}")

if __name__ == "__main__":
    main()
