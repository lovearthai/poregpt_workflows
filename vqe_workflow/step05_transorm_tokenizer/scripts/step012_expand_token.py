import os
import gzip
import json
import glob
import argparse
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from poregpt.utils import get_rsq_coords_from_integer, get_rsq_vector_from_integer

def process_single_file(args_tuple):
    """
    处理单个 jsonl.gz 文件的核心逻辑
    功能：流式处理、保持目录结构、自动创建目录
    """
    file_path, output_dir, input_dir, levels, num_quantizers = args_tuple
    
    # 1. 计算目标文件路径 (保持目录结构)
    # 计算相对路径，例如: "sub_dir/train_001.jsonl.gz"
    relative_path = os.path.relpath(file_path, input_dir)
    # 拼接完整输出路径，并修改后缀
    output_file = os.path.join(output_dir, relative_path.replace('.jsonl.gz', '_processed.jsonl.gz'))

    # 定义维度标签 (A, B, C, D...)
    dim_labels = [chr(ord('A') + i) for i in range(len(levels))]
    
    # 编译正则表达式
    pattern = re.compile(r'<\|bwav:(\d+)\|>')

    # 2. 预统计行数 (仅为了显示进度条)
    # 这一步只读不存，内存占用极小
    total_lines = 0
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total_lines += 1

    # 3. 【关键修复】确保目标文件的父目录存在
    # 如果 output_file 在子目录中，这里会自动创建该子目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 4. 流式读取与写入
    with gzip.open(file_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_file, 'wt', encoding='utf-8') as f_out:
        
        with tqdm(total=total_lines, desc=f"处理 {os.path.basename(file_path)}", unit="行") as pbar:
            for line in f_in:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                
                try:
                    data = json.loads(line)
                    text = data.get("text", "")

                    # 使用正则找到所有 bwav token_id
                    token_ids_str = pattern.findall(text)
                    tokens = [int(t) for t in token_ids_str]

                    # 重新构建 text 字段
                    new_text_parts = []

                    # 遍历每一个 token，将其拆解
                    for token_id in tokens:
                        # 获取坐标 (加上 [0] 提取内部列表)
                        coords = get_rsq_coords_from_integer(token_id, levels=levels, num_quantizers=num_quantizers, use_fast=True)[0]
                        
                        # 遍历每一层维度，生成新的 token
                        for i, val in enumerate(coords):
                            label = dim_labels[i]
                            new_text_parts.append(f"<|bwav:{label}{val}|>")

                    # 将处理后的 parts 重新拼成 text
                    data["text"] = "".join(new_text_parts)
                    
                    # 处理完一行，立刻写入硬盘，并换行
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"处理行出错: {e}, 文件: {file_path}")
                
                # 更新进度条
                pbar.update(1)

    return f"完成处理: {os.path.basename(file_path)}"

def main():
    # ================= 命令行参数配置 =================
    parser = argparse.ArgumentParser(description='多进程并行处理 jsonl.gz 文件，将 token 拆解为多维度 token。')

    parser.add_argument('--input_dir', type=str, required=True, help='输入的根目录 (递归查找 jsonl.gz)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出文件保存的目录')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help=f'并行处理的进程数 (默认: CPU核心数 {cpu_count()})')
    parser.add_argument('--levels', type=int, nargs='+', required=True, help='量化级别列表 (例如: 5 5 5 5)')
    parser.add_argument('--num_quantizers', type=int, default=1, help='量化器的层数 (默认: 1)')

    args = parser.parse_args()
    # ==================================================

    # 1. 递归查找 INPUT_DIR 下所有的 jsonl.gz 文件
    search_pattern = os.path.join(args.input_dir, "**", "*.jsonl.gz")
    file_list = glob.glob(search_pattern, recursive=True)

    if not file_list:
        print(f"在目录 {args.input_dir} 中未找到任何 jsonl.gz 文件！")
        return

    print(f"共找到 {len(file_list)} 个文件，准备使用 {args.num_workers} 个进程并行处理...")
    print(f"Levels: {args.levels}, Num_quantizers: {args.num_quantizers}")

    # 2. 准备多进程的参数元组列表
    # 注意：这里传入了 args.input_dir，用于在 worker 中计算相对路径
    worker_args = [(f, args.output_dir, args.input_dir, args.levels, args.num_quantizers) for f in file_list]

    # 3. 使用多进程池进行并行计算，并配合 tqdm 显示总进度
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap(process_single_file, worker_args), total=len(file_list), desc="文件处理总进度"):
            pass

    print("所有文件处理完毕！")

if __name__ == "__main__":
    main()
