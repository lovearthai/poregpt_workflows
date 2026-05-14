import os
import gzip
import json
import glob
import argparse
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# 引入对应的逆向转换函数（计算 token_id）
from poregpt.utils import get_rsq_coords_from_integer, get_integer_from_rsq_coords

def calculate_delta_coords(prev_coords, curr_coords, levels):
    """
    核心变换函数：计算当前 coords 相对于上一个 coords 的差分档位
    """
    new_coords = []
    for i, curr_L in enumerate(curr_coords):
        prev_L = prev_coords[i]
        level_num = levels[i]  # 当前维度的基础 level 数
        
        diff = curr_L - prev_L
        
        if diff == 0:
            new_val = level_num + 1
        elif diff == -1:
            new_val = level_num + 2
        elif diff == 1:
            new_val = level_num + 3
        else:
            # 差值绝对值大于 1，保持原样
            new_val = curr_L
            
        new_coords.append(new_val)
    return new_coords

def process_single_file(args_tuple):
    """
    处理单个 jsonl.gz 文件的核心逻辑
    功能：流式处理、保持目录结构、差分 Token 变换、打印样例供人工检查
    """
    file_path, output_dir, input_dir, levels, num_quantizers = args_tuple

    # 1. 计算目标文件路径 (保持目录结构)
    relative_path = os.path.relpath(file_path, input_dir)
    output_file = os.path.join(output_dir, relative_path.replace('.jsonl.gz', '_processed.jsonl.gz'))

    # 编译正则表达式
    pattern = re.compile(r'<\|bwav:(\d+)\|>')

    # 2. 预统计行数 (仅为了显示进度条)
    total_lines = 0
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total_lines += 1

    # 3. 确保目标文件的父目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 控制人工检查的样例打印数量
    sample_print_limit = 3
    printed_count = 0

    # 计算差分后每个维度实际的量化级数上限（因为新增了 +1, +2, +3）
    # 用于后面的 get_integer_from_rsq_coords 还原新的 token_id
    delta_levels = [L + 4 for L in levels]

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
                    original_text = text  # 保存原始文本用于对比

                    # 使用正则找到所有 bwav token_id
                    token_ids_str = pattern.findall(text)
                    tokens = [int(t) for t in token_ids_str]

                    # 重新构建新的 token 序列
                    new_tokens = []
                    prev_coords = None

                    # 遍历每一个 token 级别进行差分变换
                    for token_id in tokens:
                        # 获取当前原始坐标
                        curr_coords = get_rsq_coords_from_integer(
                            token_id, levels=levels, num_quantizers=num_quantizers, use_fast=True
                        )[0]

                        if prev_coords is None:
                            # 1. 如果是第一个 token，保持原始 coords 不变
                            final_coords = curr_coords
                            # 第一个 token 依然属于原始空间，但为了统一处理，我们用 delta_levels 或者原始 levels 转换。
                            # 建议统一使用原始 levels 转换，或者确保逆向函数支持 final_coords 的范围
                            new_id = get_integer_from_rsq_coords([final_coords], levels=levels, num_quantizers=num_quantizers)
                        else:
                            # 2. 非第一个 token，计算差分档位
                            final_coords = calculate_delta_coords(prev_coords, curr_coords, levels)
                            # 使用扩充后的 delta_levels 重新打包成新的 token_id
                            new_id = get_integer_from_rsq_coords([final_coords], levels=delta_levels, num_quantizers=num_quantizers)

                        # 更新上一个坐标指针
                        prev_coords = curr_coords
                        
                        # 生成变换后的标准文本 Token
                        new_tokens.append(f"<|bwav:{new_id}|>")

                    # 替换原文本（如果原始文本包含其他非 bwav 的非结构化文本，此步需根据实际业务微调。
                    # 这里沿用你原代码的逻辑：直接将所有转换后的 bwav 拼接）
                    data["text"] = "".join(new_tokens)

                    # 打印人工检查信息
                    if printed_count < sample_print_limit:
                        print(f"\n" + "="*40 + " 人工检查样例 (Delta 编码) " + "="*40)
                        print(f"当前处理文件: {os.path.basename(file_path)}")
                        print(f"原始文本片段 (前200字): {original_text[:200]}...")
                        print(f"变换文本片段 (前200字): {data['text'][:200]}...")
                        print(f"统计数据: 原始Token数 = {len(tokens)} -> 变换后Token数 = {len(new_tokens)}")
                        print("="*94 + "\n")
                        printed_count += 1

                    # 写入硬盘
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"处理行出错: {e}, 文件: {file_path}")

                pbar.update(1)

    return f"完成处理: {os.path.basename(file_path)}"

def main():
    parser = argparse.ArgumentParser(description='多进程并行处理 jsonl.gz 文件，执行 Token 相邻差分 Delta 变换。')

    parser.add_argument('--input_dir', type=str, required=True, help='输入的根目录 (递归查找 jsonl.gz)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出文件保存的目录')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help=f'并行处理的进程数 (默认: CPU核心数 {cpu_count()})')
    parser.add_argument('--levels', type=int, nargs='+', required=True, help='原始量化级别列表 (例如: 5 5 5 5)')
    parser.add_argument('--num_quantizers', type=int, default=1, help='量化器的层数 (默认: 1)')

    args = parser.parse_args()

    search_pattern = os.path.join(args.input_dir, "**", "*.jsonl.gz")
    file_list = glob.glob(search_pattern, recursive=True)

    if not file_list:
        print(f"在目录 {args.input_dir} 中未找到任何 jsonl.gz 文件！")
        return

    print(f"共找到 {len(file_list)} 个文件，准备使用 {args.num_workers} 个进程并行处理...")
    print(f"Original Levels: {args.levels}, Num_quantizers: {args.num_quantizers}")

    worker_args = [(f, args.output_dir, args.input_dir, args.levels, args.num_quantizers) for f in file_list]

    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap(process_single_file, worker_args), total=len(file_list), desc="文件处理总进度"):
            pass

    print("所有文件处理完毕！")

if __name__ == "__main__":
    main()
