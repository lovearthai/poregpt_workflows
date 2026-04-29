import numpy as np
import pandas as pd
import json
import gzip
import glob
import argparse
import os
import multiprocessing as mp
import time
from tqdm import tqdm

# 碱基映射
BASE_MAP = {ord('1'): 0, ord('2'): 1, ord('3'): 2, ord('4'): 3}

def has_long_homopolymer(bases_str, max_len):
    """
    检测序列中是否存在长度超过 max_len 的连续相同碱基。
    如果有，返回 True。
    """
    if not bases_str:
        return False
    
    current_char = bases_str[0]
    count = 1
    
    # 从第二个字符开始遍历
    for char in bases_str[1:]:
        if char == current_char:
            count += 1
            # 提前退出优化：一旦超过阈值立即返回
            if count > max_len:
                return True
        else:
            current_char = char
            count = 1
            
    return False

def get_kmer_indices(bases_str, k):
    """
    将 DNA 序列字符串转换为 K-mer 的整数索引列表。
    """
    if not isinstance(bases_str, str) or len(bases_str) < k:
        return []

    indices = []
    current_idx = 0
    window_len = 0
    mask = (1 << (2 * k)) - 1

    for char in bases_str:
        val = BASE_MAP.get(ord(char))
        if val is not None:
            current_idx = ((current_idx << 2) | val) & mask
            window_len += 1
            if window_len >= k:
                indices.append(current_idx)
        else:
            # 遇到无效字符，重置滑动窗口
            window_len = 0
            current_idx = 0
    return indices

def worker_task(file_path, k, target, current_counts_snap, output_dir, proc_id, ratio, least_repeat_n, exclude_poly_len):
    """
    核心工作单元
    """
    output_filename = os.path.basename(file_path)
    final_output = os.path.join(output_dir, output_filename)

    saved_in_file = 0
    original_count = 0
    local_increment = np.zeros(4**k, dtype=np.uint32)
    fname = os.path.basename(file_path)

    update_interval = 500
    local_counter = 0

    pbar = tqdm(
        unit="lines", desc=f"Core-{proc_id:02d} | {fname[:10]}",
        position=proc_id + 1, leave=False,
        bar_format='{l_bar}{bar}| {n_fmt} lines'
    )

    try:
        with gzip.open(file_path, 'rt') as f, gzip.open(final_output, 'wt') as out_f:
            for line in f:
                original_count += 1
                local_counter += 1

                if local_counter >= update_interval:
                    pbar.update(local_counter)
                    local_counter = 0

                data = json.loads(line)
                bases_seq = data.get('bases', "")
                
                # --- 修改点：优先判断长同聚物 ---
                # 如果设置了阈值 (>0) 且发现超长同聚物，直接跳过该行
                if exclude_poly_len > 0 and has_long_homopolymer(bases_seq, exclude_poly_len):
                    continue

                indices = get_kmer_indices(bases_seq, k)
                if not indices:
                    continue

                # --- 核心筛选逻辑 ---
                needed_count = 0
                force_accept = False

                for idx in indices:
                    count = current_counts_snap[idx]

                    # 规则 A: 极稀有策略
                    if count < least_repeat_n:
                        force_accept = True
                        break

                    # 规则 B: 稀缺性统计
                    if count < target:
                        needed_count += 1

                # 最终判定
                is_valuable = force_accept or (needed_count > len(indices) * ratio)

                if is_valuable:
                    out_f.write(line)
                    saved_in_file += 1
                    # 只有被保留的序列才更新本地计数增量
                    for idx in indices:
                        local_increment[idx] += 1

            if local_counter > 0:
                pbar.update(local_counter)

    except Exception as e:
        print(f"\n[ERROR] Core-{proc_id} failed on {fname}: {e}")
    finally:
        pbar.close()

    return final_output, saved_in_file, original_count, local_increment

def run_balancing_mp(args):
    """
    主控流程
    """
    start_time = time.time()
    k = args.k

    # --- 1. 加载统计数据 ---
    print(f"[*] Step 1: Loading K-mer stats from {args.csv}...")
    df = pd.read_csv(args.csv, usecols=['kmer', 'count'])

    global_counts = np.zeros(4**k, dtype=np.uint64)

    # 使用快速映射填充全局计数器
    for kmer_str, count_val in tqdm(zip(df['kmer'], df['count']), total=len(df), desc="Mapping CSV to Memory"):
        idx = 0
        for char in str(kmer_str):
            idx = (idx << 2) | (ord(char) - 49) 
        global_counts[idx] = count_val

    del df 

    # --- 2. 准备文件列表 ---
    files = sorted(glob.glob(args.input))
    if not files:
        print(f"[!] No files found matching pattern: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)
    print(f"[*] Step 2: Processing {len(files)} files using {args.threads} workers...")

    # --- 3. 并行分发 ---
    pool = mp.Pool(processes=args.threads)
    total_saved = 0
    total_original = 0
    stats_list = []

    main_pbar = tqdm(total=len(files), desc="Overall Progress", position=0)

    results = []
    for i, f in enumerate(files):
        proc_id = i % args.threads
        # 传入 exclude_poly_len 参数
        res = pool.apply_async(worker_task, args=(
            f, k, args.target, global_counts.copy(), args.output, proc_id, args.ratio, args.least_repeat_n, args.exclude_poly_len
        ))
        results.append(res)

    # --- 4. 结果汇总 ---
    for res in results:
        output_path, saved_count, orig_count, local_inc = res.get()

        total_saved += saved_count
        total_original += orig_count
        global_counts += local_inc 

        retention_rate = (saved_count / orig_count * 100) if orig_count > 0 else 0.0
        stats_list.append({
            "file_name": os.path.basename(output_path),
            "original_lines": orig_count,
            "filtered_lines": saved_count,
            "retention_rate(%)": round(retention_rate, 2)
        })

        main_pbar.update(1)
        main_pbar.set_postfix({"Saved": f"{total_saved:,}"})

    pool.close()
    pool.join()
    main_pbar.close()

    # --- 5. 输出报告 ---
    duration = time.time() - start_time
    print(f"\n[✔] Done in {duration:.2f}s! Total saved: {total_saved:,} / {total_original:,} lines.")

    stats_df = pd.DataFrame(stats_list)
    stats_csv_path = os.path.join(args.output, "filter_summary.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"[📊] Detailed summary saved to: {stats_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-mer Based Corpus Filter")
    parser.add_argument("--input", type=str, required=True, help="Input glob pattern (e.g. 'data/*.jsonl.gz')")
    parser.add_argument("--csv", type=str, required=True, help="K-mer stats CSV from step030")
    parser.add_argument("--k", type=int, default=9, help="K-mer length")
    parser.add_argument("--target", type=int, default=8000, help="Target frequency threshold")
    parser.add_argument("--least_repeat_n", type=int, default=10, help="Ultra-rare threshold")
    parser.add_argument("--ratio", type=float, default=0.4, help="Retention ratio threshold")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--threads", type=int, default=mp.cpu_count(), help="Number of threads")
    
    # --- 新增参数 ---
    parser.add_argument("--exclude_poly_len", type=int, default=-1, 
                        help="Filter out lines containing homopolymers longer than this value (e.g. 10). Set to -1 to disable.")

    args = parser.parse_args()
    run_balancing_mp(args)
