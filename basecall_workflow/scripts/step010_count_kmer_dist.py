import gzip
import json
import glob
import csv
import argparse
import multiprocessing as mp
from collections import Counter
from tqdm import tqdm

def process_file(file_path, k):
    """处理单个文件并返回 k-mer 计数"""
    local_counter = Counter()
    try:
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                data = json.loads(line)
                bases = data.get('bases', "")
                # 统计当前行的 k-mer
                for i in range(len(bases) - k + 1):
                    kmer = bases[i:i+k]
                    local_counter[kmer] += 1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return local_counter

def main():
    parser = argparse.ArgumentParser(description="K-mer frequency statistics")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to jsonl.gz files")
    parser.add_argument("--output_csv", type=str, default="kmer_stats.csv", help="Output CSV filename")
    parser.add_argument("--kmer_len", type=int, default=5, help="Length of K-mer")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    # 1. 获取所有文件
    files = glob.glob(f"{args.input_dir}/*.jsonl.gz")
    print(f"Found {len(files)} files. Using {args.workers} workers.")

    # 2. 并行统计
    global_counter = Counter()
    with mp.Pool(args.workers) as pool:
        # 使用 partial 传递固定参数 k
        import functools
        func = functools.partial(process_file, k=args.kmer_len)
        
        # 结果汇总
        for local_res in tqdm(pool.imap_unordered(func, files), total=len(files), desc="Processing"):
            global_counter.update(local_res)

    # 3. 计算总数和比例
    total_count = sum(global_counter.values())
    print(f"Total k-mers processed: {total_count}")

    # 4. 写入 CSV
    # 按频次降序排列
    sorted_kmers = global_counter.most_common()
    
    with open(args.output_csv, 'w', newline='') as csvfile:
        fieldnames = ['id', 'kmer', 'count', 'ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, (kmer, count) in enumerate(sorted_kmers, 1):
            writer.writerow({
                'id': idx,
                'kmer': kmer,
                'count': count,
                'ratio': f"{(count / total_count):.8f}"
            })

    print(f"Statistics saved to {args.output_csv}")

if __name__ == "__main__":
    main()
