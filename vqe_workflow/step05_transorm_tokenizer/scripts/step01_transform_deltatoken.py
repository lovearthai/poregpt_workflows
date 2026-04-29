import os
import gzip
import json
import re
import argparse
import csv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# 严格遵守要求的接口引入
from poregpt.utils import get_rsq_coords_from_integer

def extract_token_ids(text):
    """从 <|bwav:ID|> 格式中提取 ID 列表"""
    if not text: return []
    pattern = re.compile(r'<\|bwav:(\d+)\|>')
    return [int(tid) for tid in pattern.findall(text)]

def process_row_statistics(token_ids, levels, num_quantizers, max_vocab, max_sum_diff):
    """计算一行中相邻 Token 坐标的曼哈顿距离分布"""
    dist_counts = np.zeros(max_sum_diff + 1, dtype=int)
    if not token_ids or len(token_ids) < 2:
        return "_".join(["0"] * (max_sum_diff + 1))

    coords_seq = []
    for tid in token_ids:
        if 0 <= tid < max_vocab:
            res = get_rsq_coords_from_integer(tid, levels, num_quantizers=num_quantizers)
            if res and len(res) > 0:
                coords_seq.append(res[0])

    if len(coords_seq) < 2:
        return "_".join(["0"] * (max_sum_diff + 1))

    coords_array = np.array(coords_seq)
    diffs = np.sum(np.abs(np.diff(coords_array, axis=0)), axis=1)

    for d in diffs:
        dist_idx = int(round(float(d)))
        if 0 <= dist_idx < len(dist_counts):
            dist_counts[dist_idx] += 1

    return "_".join(map(str, dist_counts))

def process_single_file(file_path, levels, num_quantizers, max_vocab, max_sum_diff):
    """并行处理单个文件的 worker 函数"""
    file_results = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data = json.loads(line)

                row_id = data.get('read_id', data.get('id', 'unknown'))
                
                # 兼容性获取序列字段
                raw_seq = data.get('bases') or data.get('query') or data.get('sequence') or ""
                base_count = len(raw_seq)

                text_content = data.get('text', "")
                token_ids = extract_token_ids(text_content)
                token_count = len(token_ids)

                diff_str = process_row_statistics(
                    token_ids, levels, num_quantizers, max_vocab, max_sum_diff
                )
                file_results.append([row_id, base_count, token_count, diff_str])
    except Exception as e:
        # 错误信息通过 print 输出，不会阻塞主进程
        print(f"\n[Error] {file_path}: {e}")
    
    return file_results

def main():
    parser = argparse.ArgumentParser(description="PoreGPT FSQ Manhattan Distance Analyzer (Parallel)")
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录")
    parser.add_argument("--output_csv", type=str, required=True, help="输出CSV路径")
    parser.add_argument("--levels", type=int, nargs='+', required=True, help="FSQ levels")
    parser.add_argument("--num_quantizers", type=int, default=1, help="Quantizer 数量")
    parser.add_argument("--processes", type=int, default=cpu_count(), help="进程数")

    args = parser.parse_args()

    # 1. 计算词表上限和最大可能差异
    max_vocab = 1
    max_sum_diff = 0
    for l in args.levels:
        max_vocab *= l
        max_sum_diff += (l - 1)

    # 2. 获取文件列表
    all_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith('.jsonl.gz'):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print("未找到 .jsonl.gz 文件。")
        return

    print(f"开始并行处理: {len(all_files)} 个文件，使用 {args.processes} 个进程...")

    # 3. 准备输出
    out_dir = os.path.dirname(args.output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 使用 partial 封装参数
    worker = partial(
        process_single_file, 
        levels=args.levels, 
        num_quantizers=args.num_quantizers, 
        max_vocab=max_vocab, 
        max_sum_diff=max_sum_diff
    )

    # 4. 多进程执行并配合 tqdm 显示进度
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'base_count', 'token_count', 'diff_distribution'])

        with Pool(processes=args.processes) as pool:
            # imap_unordered 可以让 tqdm 实时更新
            # chunksize=1 适合文件大小不一的情况
            pbar = tqdm(pool.imap_unordered(worker, all_files), total=len(all_files), desc="Overall Progress")
            for file_data in pbar:
                if file_data:
                    writer.writerows(file_data)

    print(f"\n任务处理完成，结果保存在: {args.output_csv}")

if __name__ == "__main__":
    main()
