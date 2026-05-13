import os
import gzip
import json
import glob
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from poregpt.utils import get_rsq_coords_from_integer, get_rsq_vector_from_integer

def calculate_deltatoken(coord_prev, coord_curr):
    """
    计算 deltatoken 的占位函数。
    coord_prev: 前一个 token 的坐标列表
    coord_curr: 当前 token 的坐标列表
    """
    # TODO: 由你后续补充具体的 deltatoken 计算代码
    pass

def manhattan_distance(coord1, coord2):
    """计算两个坐标列表的曼哈顿距离"""
    return sum(abs(a - b) for a, b in zip(coord1, coord2))

def process_single_file(args_tuple):
    """
    处理单个 jsonl.gz 文件的核心逻辑
    为了多进程传参方便，这里接收一个包含所有参数的元组
    """
    file_path, output_dir, levels, num_quantizers = args_tuple
    output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.jsonl.gz', '_processed.jsonl.gz'))
    
    processed_lines = []
    
    # 读取并处理文件
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
                
                # 提取所有的 bwav token_id
                tokens = []
                parts = text.split('><')
                for part in parts:
                    part = part.strip('<>').strip()
                    if part.startswith("bwav:"):
                        token_id = int(part.split(':'))
                        tokens.append(token_id)
                
                if len(tokens) < 2:
                    processed_lines.append(json.dumps(data, ensure_ascii=False))
                    continue<websource>source_group_web_1</websource>

                # 重新构建 text 字段
                new_text_parts = []
                # 第一个 token 保持原样
                new_text_parts.append(f"<|bwav:{tokens}|>")
                
                # 遍历后续的 token，与前一个进行比较
                for i in range(1, len(tokens)):
                    prev_id = tokens[i-1]
                    curr_id = tokens[i]
                    
                    # 获取坐标 (提取第一层坐标列表)
                    coord_prev = get_rsq_coords_from_integer(prev_id, levels=levels, num_quantizers=num_quantizers, use_fast=True)
                    coord_curr = get_rsq_coords_from_integer(curr_id, levels=levels, num_quantizers=num_quantizers, use_fast=True)
                    
                    # 计算曼哈顿距离
                    dist = manhattan_distance(coord_prev, coord_curr)
                    
                    if dist <= 2:
                        # 距离小于等于2，调用你的函数计算 deltatoken
                        delta_result = calculate_deltatoken(coord_prev, coord_curr)
                        if delta_result is not None:
                            new_text_parts.append(f"<|deltatoken:{delta_result}|>") 
                        else:
                            # 如果函数还没写好，暂时保持原样
                            new_text_parts.append(f"<|bwav:{curr_id}|>")
                    else:
                        # 距离大于2，保持原样
                        new_text_parts.append(f"<|bwav:{curr_id}|>")
                
                # 将处理后的 parts 重新拼成 text
                data["text"] = "".join(new_text_parts)
                processed_lines.append(json.dumps(data, ensure_ascii=False))
                
            except Exception as e:
                print(f"处理行出错: {e}, 文件: {file_path}")
                continue

    # 保存处理后的文件
    os.makedirs(output_dir, exist_ok=True)
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        f.write("\n".join(processed_lines) + "\n")
    
    return f"完成处理: {os.path.basename(file_path)}"

def main():
    # ================= 命令行参数配置 =================
    parser = argparse.ArgumentParser(description='多进程并行处理 jsonl.gz 文件，计算曼哈顿距离并标记 deltatoken。')
    
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
    # 因为 pool.imap 只能传一个参数，所以把所有需要的配置打包成元组传给 worker
    worker_args = [(f, args.output_dir, args.levels, args.num_quantizers) for f in file_list]

    # 3. 使用多进程池进行并行计算，并配合 tqdm 显示总进度
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap(process_single_file, worker_args), total=len(file_list), desc="文件处理总进度"):
            pass 

    print("所有文件处理完毕！")

if __name__ == "__main__":
    main()
