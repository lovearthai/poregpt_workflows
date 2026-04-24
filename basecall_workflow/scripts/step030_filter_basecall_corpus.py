import numpy as np
import pandas as pd
import json
import gzip
import glob
import argparse
import os
import multiprocessing as mp
from tqdm import tqdm

# 碱基映射：将字符 '1', '2', '3', '4' 映射为 0, 1, 2, 3，用于后续的二进制位运算
BASE_MAP = {ord('1'): 0, ord('2'): 1, ord('3'): 2, ord('4'): 3}

def get_kmer_indices(bases_str, k):
    """
    将 DNA 序列字符串转换为 K-mer 的整数索引列表。
    使用位运算（左移2位）进行编码，效率极高。
    """
    if not isinstance(bases_str, str) or len(bases_str) < k:
        return []
    # 预处理：将字符转换为数字，过滤掉无效字符
    encoded = [BASE_MAP.get(ord(c)) for c in bases_str if ord(c) in BASE_MAP]
    if len(encoded) < k: return []

    indices = []
    current_idx = 0
    # 计算第一个 K-mer 的索引
    for i in range(k):
        current_idx = (current_idx << 2) | encoded[i]
    indices.append(current_idx)
    
    # 滑动窗口计算后续 K-mer
    # mask 用于保留低 2k 位，去除高位溢出
    mask = (1 << (2 * k)) - 1
    for i in range(k, len(encoded)):
        # 左移2位（相当于乘以4），加上新碱基，并与 mask 进行与运算以维持长度
        current_idx = ((current_idx << 2) | encoded[i]) & mask
        indices.append(current_idx)
    return indices

def worker_task(file_path, k, target, current_counts_snap, output_dir, proc_id, ratio):
    """
    【核心工作单元】
    负责处理单个文件。
    
    关键逻辑说明：
    1. 接收 current_counts_snap：这是“当前时刻”的全局 K-mer 计数快照。
       - 它初始于 CSV 文件（全量统计），
       - 但在处理多个文件时会动态累加。
       - 作用：判断某个 K-mer 是否已经“达标”（不再稀缺）。
    2. 筛选策略：优先保留包含“稀缺” K-mer 的序列，以实现数据均衡。
    """
    # 确定输出路径：保持原文件名，保存到指定输出目录
    output_filename = os.path.basename(file_path)
    final_output = os.path.join(output_dir, output_filename)

    saved_in_file = 0
    original_count = 0  # 新增：统计原文件行数
    # 用于统计本文件中被选中的 K-mer 数量，稍后返回给主进程以更新全局状态
    local_increment = np.zeros(4**k, dtype=np.uint32)
    fname = os.path.basename(file_path)

    update_interval = 1000
    local_counter = 0

    pbar = tqdm(
        unit="lines", desc=f"Core-{proc_id:02d} | {fname[:12]}",
        position=proc_id + 1, leave=False,
        bar_format='{l_bar}{bar}| {n_fmt} lines [{elapsed}]'
    )

    try:
        with gzip.open(file_path, 'rt') as f, gzip.open(final_output, 'wt') as out_f:
            for line in f:
                original_count += 1 # 新增：计数原文件行数
                local_counter += 1

                if local_counter >= update_interval:
                    pbar.update(local_counter)
                    local_counter = 0

                data = json.loads(line)
                indices = get_kmer_indices(data['bases'], k)
                if not indices: continue

                # --- 核心筛选逻辑 ---
                # 统计这条序列中，有多少个 K-mer 是“当前仍稀缺”的（计数 < target）
                # 注意：这里使用的是 current_counts_snap，它反映了 CSV 初始值 + 之前文件已筛选的增量
                needed = sum(1 for idx in indices if current_counts_snap[idx] < target)

                # 【修复】先计算比例
                ratio_needed = needed / len(indices) if len(indices) > 0 else 0
                # --- 诊断代码：打印前 10 行的详细情况 ---
                if original_count <= 10:
                    print(f"\n[DEBUG] Line {original_count}:")
                    print(f"  Total K-mers: {len(indices)}")
                    print(f"  Needed (<{target}): {needed}")
                    print(f"  Ratio: {ratio_needed:.2f} (Threshold: {ratio})")
                    # 打印几个 K-mer 的实际计数看看
                    sample_idx = indices[0]
                    print(f"  Example K-mer count: {current_counts_snap[sample_idx]}")
                # ---------------------------------------

                # 判定规则：如果一条序列中超过 ratio (默认40%) 的 K-mer 都是稀缺的，则认为该序列具有高价值，予以保留
                if needed > len(indices) * ratio:
                    out_f.write(line)
                    saved_in_file += 1
                    # 记录这些 K-mer 被选中了，用于后续更新全局计数，防止后续文件重复选择同类 K-mer
                    for idx in indices:
                        local_increment[idx] += 1

            if local_counter > 0:
                pbar.update(local_counter)

    except Exception as e:
        pass
    finally:
        pbar.close()

    # 修改返回值：增加 original_count
    return final_output, saved_in_file, original_count, local_increment

def run_balancing_mp(input_pattern, csv_path, k, target, output_dir, threads, ratio):
    """
    【主控流程】
    """
    # --- 1. 加载统计数据 (CSV) ---
    print(f"[*] Loading K-mer stats from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'kmer': str}) # 确保 kmer 列读作字符串
    
    # 初始化全局计数器 (全 0)
    global_counts = np.zeros(4**k, dtype=np.uint64) 
    
    # 【核心修复】将 CSV 中的统计值填入 global_counts
    # CSV 格式: id, kmer(如111111111), count, ratio
    for _, row in df.iterrows():
        kmer_str = row['kmer']
        count_val = row['count']
        
        # 将 '111111111' 这种字符串转换回数字索引
        # 逻辑与 get_kmer_indices 中的编码部分一致
        current_idx = 0
        valid = True
        for c in kmer_str:
            if c == '1': val = 0
            elif c == '2': val = 1
            elif c == '3': val = 2
            elif c == '4': val = 3
            else: 
                valid = False # 遇到非法字符
                break
            current_idx = (current_idx << 2) | val
            
        if valid and len(kmer_str) == k:
            global_counts[current_idx] = count_val
            
    print(f"[✔] CSV Loaded. Total unique K-mers: {len(df)}")
    print(f"    Max count in CSV: {global_counts.max()}")
    print(f"    Min count in CSV (non-zero): {global_counts[global_counts > 0].min()}")
    
    # 2. 并行准备
    files = sorted(glob.glob(input_pattern))
    os.makedirs(output_dir, exist_ok=True)

    pool = mp.Pool(processes=threads)
    total_saved = 0
    stats_list = [] # 用于收集统计信息

    main_pbar = tqdm(total=len(files), desc="Overall Progress", position=0, leave=True)

    results = []
    for i, f in enumerate(files):
        proc_id = i % threads
        # 传入 ratio 参数
        res = pool.apply_async(worker_task, args=(f, k, target, global_counts.copy(), output_dir, proc_id, ratio))
        results.append(res)

    # 等待所有结果并汇总统计信息
    for res in results:
        output_path, saved_count, original_count, local_inc = res.get()
        
        total_saved += saved_count
        # 更新全局计数（用于下一轮迭代或日志，虽然这里主要是为了统计）
        global_counts += local_inc
        
        # 计算比例并记录统计信息
        retention_rate = (saved_count / original_count * 100) if original_count > 0 else 0.0
        stats_list.append({
            "file_name": os.path.basename(output_path),
            "original_lines": original_count,
            "filtered_lines": saved_count,
            "retention_rate(%)": round(retention_rate, 2)
        })

        main_pbar.update(1)
        main_pbar.set_postfix({"Saved": f"{total_saved:,}"})

    pool.close()
    pool.join()
    main_pbar.close()

    # --- 3. 输出统计报告 ---
    print(f"\n[✔] Done! Total saved: {total_saved:,} lines.")
    print(f"Output files are in: {output_dir}")
    
    # 将统计列表转换为 DataFrame 并保存为 CSV
    stats_df = pd.DataFrame(stats_list)
    stats_csv_path = os.path.join(output_dir, "stats_summary.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"[📊] Statistics saved to: {stats_csv_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--target", type=int, default=8000)
    parser.add_argument("--output", type=str, required=True, help="Output directory for balanced files")
    parser.add_argument("--threads", type=int, default=8)
    # 暴露 ratio 参数，默认值为 0.4，保持原有行为
    parser.add_argument("--ratio", type=float, default=0.4, help="Threshold ratio for filtering (default: 0.4)")
    
    args = parser.parse_args()

    run_balancing_mp(args.input, args.csv, args.k, args.target, args.output, args.threads, args.ratio)
