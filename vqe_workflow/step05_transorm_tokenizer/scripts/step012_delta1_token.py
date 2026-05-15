import os
import gzip
import json
import glob
import argparse
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# 引入对应的逆向转换函数
from poregpt.utils import get_rsq_coords_from_integer, get_rsq_vector_from_integer,get_integer_from_rsq_coords
def calculate_delta_coords(prev_coords, curr_coords, levels):
    """
    计算当前 coords 相对于上一个 coords 的差分档位
    """
    new_coords = []
    for i, curr_L in enumerate(curr_coords):
        prev_L = prev_coords[i]
        level_num = levels[i]  # 当前维度的基础 level 数
        diff = curr_L - prev_L

        if diff == 0:
            new_val = level_num + 0
        elif diff == -1:
            new_val = level_num + 1
        elif diff == 1:
            new_val = level_num + 2
        else:
            new_val = curr_L
        new_coords.append(new_val)
    return new_coords

def transform_token_sequence(token_ids, levels, delta_levels, num_quantizers):
    """
    核心转换函数：输入原始 ID 列表，输出 Delta 变换后的新 ID 列表
    """
    new_ids = []
    prev_coords = None

    for token_id in token_ids:
        # 获取当前原始坐标
        curr_coords = get_rsq_coords_from_integer(
            token_id, levels=levels, num_quantizers=num_quantizers, use_fast=True
        )[0]

        if prev_coords is None:
            # 第一个 token：保持原始 coords，使用原始 levels 转换
            new_id = get_integer_from_rsq_coords([curr_coords], levels=levels, num_quantizers=num_quantizers)
        else:
            # 非第一个 token：计算差分
            final_coords = calculate_delta_coords(prev_coords, curr_coords, levels)
            # 使用扩充后的 delta_levels 转换
            new_id = get_integer_from_rsq_coords([final_coords], levels=delta_levels, num_quantizers=num_quantizers)
        
        new_ids.append(new_id)
        prev_coords = curr_coords
    
    return new_ids


def run_preflight_check(levels, num_quantizers):
    """
    增强版预检函数：详细对比原始序列与变换序列的每一个步骤
    """
    print("\n" + "="*35 + " 核心变换预检 (Detailed Comparison) " + "="*35)
    delta_levels = [L + 3 for L in levels]

    # 模拟一个具有代表性的 token_id 序列
    # 10 -> 10 (差分为0), 10 -> 11 (差分为+1), 11 -> 15 (跳跃)
    test_token_ids = [10, 10, 11, 15]

    print(f"配置信息: Levels={levels}, Num_Quantizers={num_quantizers}")
    print(f"差分空间 Levels: {delta_levels}\n")

    try:
        new_ids = []
        prev_coords = None

        # 打印表头
        header = f"{'步骤':<5} | {'原始ID':<8} | {'原始坐标':<18} | {'变换逻辑':<15} | {'差分/最终坐标':<18} | {'新Token ID':<10}"
        print(header)
        print("-" * len(header))

        for i, tid in enumerate(test_token_ids):
            # 1. 获取当前原始坐标
            curr_coords = get_rsq_coords_from_integer(
                tid, levels=levels, num_quantizers=num_quantizers, use_fast=True
            )[0]

            logic_desc = ""
            final_coords = []

            if prev_coords is None:
                # 第一个 Token
                logic_desc = "Origin (First)"
                final_coords = curr_coords
                new_id = get_integer_from_rsq_coords([final_coords], levels=levels, num_quantizers=num_quantizers)
            else:
                # 计算差分
                logic_desc = "Delta Encode"
                final_coords = calculate_delta_coords(prev_coords, curr_coords, levels)
                new_id = get_integer_from_rsq_coords([final_coords], levels=delta_levels, num_quantizers=num_quantizers)

            # 格式化输出每一行
            print(f"{i:<6} | {tid:<8} | {str(curr_coords):<18} | {logic_desc:<15} | {str(final_coords):<18} | {new_id:<10}")

            new_ids.append(new_id)
            prev_coords = curr_coords

        print("-" * len(header))
        print(f"\n结论:")
        print(f"原始序列: {test_token_ids}")
        print(f"变换序列: {new_ids}")

    except Exception as e:
        print(f"\n❌ 预检运行中出错: {e}")
        import traceback
        traceback.print_exc()

    print("="*94 + "\n")


def process_single_file(args_tuple):
    file_path, output_dir, input_dir, levels, num_quantizers = args_tuple
    relative_path = os.path.relpath(file_path, input_dir)
    output_file = os.path.join(output_dir, relative_path.replace('.jsonl.gz', '_processed.jsonl.gz'))
    pattern = re.compile(r'<\|bwav:(\d+)\|>')
    delta_levels = [L + 3 for L in levels]

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with gzip.open(file_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_file, 'wt', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                token_ids = [int(t) for t in pattern.findall(data.get("text", ""))]
                
                # 调用核心转换函数
                new_ids = transform_token_sequence(token_ids, levels, delta_levels, num_quantizers)
                
                data["text"] = "".join([f"<|bwav:{nid}|>" for nid in new_ids])
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"处理行出错: {e}, 文件: {file_path}")
    return f"完成: {os.path.basename(file_path)}"

def main():
    parser = argparse.ArgumentParser(description='相邻 Token 相邻差分 Delta 变换。')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--levels', type=int, nargs='+', required=True)
    parser.add_argument('--num_quantizers', type=int, default=1)
    args = parser.parse_args()

    # 1. 运行预检
    run_preflight_check(args.levels, args.num_quantizers)

    # 2. 查找文件
    search_pattern = os.path.join(args.input_dir, "**", "*.jsonl.gz")
    file_list = glob.glob(search_pattern, recursive=True)

    if not file_list:
        print("未找到文件！")
        return

    # 3. 并行处理
    worker_args = [(f, args.output_dir, args.input_dir, args.levels, args.num_quantizers) for f in file_list]
    with Pool(processes=args.num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_file, worker_args), total=len(file_list), desc="总进度"):
            pass

if __name__ == "__main__":
    main()
