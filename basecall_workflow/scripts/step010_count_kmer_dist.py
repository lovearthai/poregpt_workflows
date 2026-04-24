import gzip
import json
import glob
import csv
import argparse
import multiprocessing as mp
import math
import functools
import time
from collections import Counter
from tqdm import tqdm

def process_file(file_path, k):
    """
    处理单个文件
    返回: (local_total_counter, local_row_presence_counter, row_count)
    """
    local_total_counter = Counter()
    local_row_presence_counter = Counter()
    line_count = 0
    
    try:
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                line_count += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    
                bases = data.get('bases', "")
                
                # 提取当前行的 unique k-mers 用于计算 IDF
                unique_kmers_in_row = set()
                # 遍历提取所有 k-mer
                for i in range(len(bases) - k + 1):
                    kmer = bases[i:i+k]
                    local_total_counter[kmer] += 1
                    unique_kmers_in_row.add(kmer)
                
                # 更新行频率计数
                local_row_presence_counter.update(unique_kmers_in_row)
                
    except Exception as e:
        print(f"\n[Error] Processing {file_path}: {e}")
    
    return local_total_counter, local_row_presence_counter, line_count

def main():
    parser = argparse.ArgumentParser(description="K-mer TF-IDF statistics (Row-based)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to jsonl.gz files")
    parser.add_argument("--output_csv", type=str, default="kmer_tfidf_row_based.csv", help="Output CSV filename")
    parser.add_argument("--kmer_len", type=int, default=5, help="Length of K-mer")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    # --- 阶段 1: 扫描文件 ---
    print(f"[*] Step 1: Scanning directory for files...")
    files = glob.glob(f"{args.input_dir}/*.jsonl.gz")
    num_files = len(files)
    if num_files == 0:
        print("[!] No .jsonl.gz files found. Exit.")
        return
    print(f"[+] Found {num_files} files. Using {args.workers} workers.")

    # --- 阶段 2: 并行读取与局部统计 ---
    print(f"[*] Step 2: Extracting k-mers and row statistics (Parallel)...")
    global_total_counter = Counter()
    global_row_presence_counter = Counter()
    total_rows = 0

    start_time = time.time()
    with mp.Pool(args.workers) as pool:
        func = functools.partial(process_file, k=args.kmer_len)
        
        # 使用 tqdm 监控文件处理进度
        for l_total, l_row_presence, l_count in tqdm(pool.imap_unordered(func, files), total=num_files, desc="Processing files"):
            global_total_counter.update(l_total)
            global_row_presence_counter.update(l_row_presence)
            total_rows += l_count

    if total_rows == 0:
        print("[!] No rows found in documents.")
        return

    # --- 阶段 3: 计算 TF-IDF ---
    print(f"[*] Step 3: Calculating TF-IDF scores...")
    total_kmers_count = sum(global_total_counter.values())
    results = []
    
    # 为了提速，预计算常量
    # log 使用自然对数，total_rows 作为 N
    log_total_rows = total_rows 
    
    # 使用 tqdm 监控词频计算过程
    for kmer, count in tqdm(global_total_counter.items(), desc="Calculating TF-IDF"):
        # TF: 总频次占比
        tf = count / total_kmers_count
        
        # IDF: log(N / (df + 1)) + 1
        rows_with_kmer = global_row_presence_counter[kmer]
        idf = math.log(total_rows / (rows_with_kmer + 1)) + 1
        
        tfidf = tf * idf
        
        results.append({
            'kmer': kmer,
            'count': count,
            'row_count': rows_with_kmer,
            'tf': tf,
            'idf': idf,
            'tfidf': tfidf
        })

    # --- 阶段 4: 排序 ---
    print(f"[*] Step 4: Sorting {len(results)} patterns by TF-IDF (Descending)...")
    results.sort(key=lambda x: x['tfidf'], reverse=True)

    # --- 阶段 5: 写入文件 ---
    print(f"[*] Step 5: Writing results to {args.output_csv}...")
    try:
        with open(args.output_csv, 'w', newline='') as csvfile:
            fieldnames = ['id', 'kmer', 'count', 'row_count', 'tf', 'idf', 'tfidf']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for idx, row in enumerate(tqdm(results, desc="Writing to CSV"), 1):
                row['id'] = idx
                # 格式化以保证 CSV 易读且保留精度
                row['tf'] = f"{row['tf']:.10f}"
                row['idf'] = f"{row['idf']:.6f}"
                row['tfidf'] = f"{row['tfidf']:.10f}"
                writer.writerow(row)
    except IOError as e:
        print(f"[!] Error writing CSV: {e}")

    duration = time.time() - start_time
    print(f"\n[√] Done! Total time: {duration:.2f}s")
    print(f"[i] Total Rows processed: {total_rows}")
    print(f"[i] Total Unique k-mers: {len(results)}")

if __name__ == "__main__":
    main()
