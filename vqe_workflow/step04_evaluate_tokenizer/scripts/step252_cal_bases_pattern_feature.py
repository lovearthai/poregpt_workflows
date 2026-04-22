import json
import gzip
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
from poregpt.utils import get_rsq_vector_from_integer

def generate_vector_map(levels, num_quantizers):
    total_ids = np.prod(levels)
    vector_map = {}
    print(f"正在预计算码表向量: levels={levels}, num_quantizers={num_quantizers}...")
    for i in range(total_ids):
        vec = get_rsq_vector_from_integer(i, levels, num_quantizers, use_fast=True)
        if hasattr(vec, 'cpu'):
            vec = vec.cpu().detach().numpy()
        vector_map[i] = np.array(vec, dtype=np.float32).flatten()
    return vector_map

def select_boundary_tokens(window, layer0_ids, boundary_num):
    """
    封装原有的边界提取逻辑
    通过 window 获取全局起止位置，从 layer0_ids 中提取 token
    """
    selected_tokens = []
    # 遍历窗口中的每一个 block
    for b_idx, block in enumerate(window):
        # 计算当前 block 在 layer0_ids 中的全局起止索引
        b_start = block['start']
        b_end = b_start + block['len']

        # 提取该 block 的头部 N 个和尾部 N 个
        # 头部：[b_start, b_start + boundary_num)
        # 尾部：[b_end - boundary_num, b_end)
        head_indices = list(range(b_start, min(b_start + boundary_num, b_end)))
        tail_indices = list(range(max(b_start, b_end - boundary_num), b_end))

        # 合并并去重（防止 block 太短导致头尾索引重叠）
        combined_indices = sorted(list(set(head_indices + tail_indices)))
        for idx in combined_indices:
            selected_tokens.append(layer0_ids[idx])

    return selected_tokens

def select_dynamic_tokens(full_token_ids, vector_map, top_n):
    """
    基于动态活跃度选择 Token
    原理：计算相邻 token 向量的欧氏距离，距离越大活跃度越高
    """
    if len(full_token_ids) <= 1:
        return full_token_ids
    
    # 获取所有 token 的向量
    vectors = np.array([vector_map[tid] for tid in full_token_ids])
    
    # 计算活跃度 (Activity)
    activities = []
    
    # 第一个 token 活跃度为 0
    activities.append(0.0)
    
    # 后续 token 计算与前一个的距离
    for i in range(1, len(vectors)):
        dist = np.linalg.norm(vectors[i] - vectors[i-1])
        activities.append(dist)
    
    activities = np.array(activities)
    
    # 选择 Top-N 活跃度的索引
    k = min(top_n, len(full_token_ids))
    # 获取活跃度最大的 k 个元素的索引
    top_indices = np.argsort(activities)[-k:]
    # 保持原始顺序
    top_indices = sorted(top_indices)
    
    selected_tokens = [full_token_ids[i] for i in top_indices]
    
    return selected_tokens
def process_nanopore_data(args):
    vector_map = generate_vector_map(args.levels, args.num_quantizers)
    final_data = []
    global_id = 0

    print(f"开始处理输入文件: {args.input}")
    print(f"模式: 寻找连续 {args.block_count} 个块.")

    with gzip.open(args.input, 'rt') as f:
        for line in tqdm(f, desc="Processing rows"):
            try:
                record = json.loads(line)
                seq = record['tokens_based']
                layer0_ids = [t[0] for t in record['tokens_layered']]

                if len(seq) != len(layer0_ids):
                    continue

                # 1. 提取所有的 Homopolymer 块
                blocks = []
                current_idx = 0
                for char, group in groupby(seq):
                    length = len(list(group))
                    blocks.append({'base': char, 'len': length, 'start': current_idx})
                    current_idx += length

                # 2. 滑动窗口处理
                n = args.block_count
                for i in range(len(blocks) - n + 1):
                    window = blocks[i : i + n]

                    # 检查每个 block 长度是否达标
                    if all(b['len'] >= args.min_repeat for b in window):
                        # 确定复合序列在原始 layer0_ids 中的起止范围
                        pattern_start = window[0]['start']
                        pattern_end = window[-1]['start'] + window[-1]['len']
                        
                        # 提取完整的 token 列表 (对应需求 2: 删掉去掉首尾的代码)
                        full_token_ids = layer0_ids[pattern_start : pattern_end]
                      
                        selected_tokens = []

                        # 3. 根据策略选择 Token
                        if args.select_token_strategy == "all":
                            selected_tokens = full_token_ids
                        
                        elif args.select_token_strategy == "boundary":
                            # 调用封装好的函数，传入 window 和 layer0_ids
                            selected_tokens = select_boundary_tokens(window, layer0_ids, args.boundary_num)
                        
                        elif args.select_token_strategy == "dynamic":
                            # 调用动态选择函数
                            selected_tokens = select_dynamic_tokens(full_token_ids, vector_map, args.dynamic_top_n)
                        
                        else:
                            # 默认 fallback 到 all
                            selected_tokens = full_token_ids

                           
                        if selected_tokens:
                            # 1. 准备向量矩阵 (L, Dim)
                            vectors = np.array([vector_map[tid] for tid in selected_tokens])
                            L = len(selected_tokens)

                            # 2. 确定中心点坐标 (以索引为单位)
                            # 比如 L=5, mid=2.0; L=6, mid=2.5
                            mid_idx = (L - 1) / 2.0

                            # 3. 计算每个 Token 距离中心的绝对物理偏移
                            # 无论序列总长度是多少，距离中心第 n 个位置的偏移始终是 n
                            offsets = np.arange(L) - mid_idx

                            # 4. 应用静态高斯分布函数 (正态分布)
                            # sigma 决定了权重的“集中度”：
                            # sigma 越小，权重越集中在中心；sigma 越大，周围 Token 参与度越高
                            sigma = args.weight_sigma if hasattr(args, 'weight_sigma') else 2.0

                            # Gauss 公式: exp(-x^2 / (2 * sigma^2))
                            # 这里不需要前面的系数，因为最后会统一归一化
                            raw_weights = np.exp(-(offsets**2) / (2 * sigma**2))

                            # 5. 归一化处理 (确保总权重和为 1.0)
                            weights = raw_weights / np.sum(raw_weights)

                            # 6. 执行加权求和得到复合特征向量
                            # mean_vec 是一个 numpy 数组
                            mean_vec = np.sum(vectors * weights[:, np.newaxis], axis=0)

                            # 模式信息
                            pattern_desc = "".join([f"{b['base']}" * b['len'] for b in window])
                            base_category = "".join([b['base'] for b in window])

                            # --- 修改部分：合并 dim 为 feature 字符串 ---
                            # 格式化每个维度为 6 位小数，并用下划线连接
                            feature_str = "_".join([f"{v:.6f}" for v in mean_vec[:4]])
                            row = {
                                'id': global_id,
                                'base_pattern': pattern_desc,
                                'category': base_category,
                                'token_id_list': "-".join(map(str, full_token_ids)),
                                'selected_token_id_list': "-".join(map(str, selected_tokens)),
                                'feature': feature_str  # 新的合并字段
                            }
                            final_data.append(row)
                            global_id += 1

            except Exception:
                continue

    if final_data:
        df = pd.DataFrame(final_data)
        df.to_csv(args.output, index=False)
        print(f"处理完成！提取了 {len(df)} 条复合特征。")
    else:
        print("处理失败，没有数据")


# 权重分布参考示例 (若 $\sigma = 2.0$):距离中心 0 步: 原始权重 $1.000$ (最大)
# 距离中心 1 步: 原始权重 $0.882$
# 距离中心 2 步: 原始权重 $0.606$
# 距离中心 4 步: 原始权重 $0.135$
# 距离中心 6 步: 原始权重 $0.011$ (基本接近 0)
def main():
    parser = argparse.ArgumentParser(description="Nanopore Boundary-aware Feature Extractor")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_features.csv')
    parser.add_argument('--min_repeat', type=int, default=4)
    parser.add_argument('--block_count', type=int, default=3)
   
       # 策略选择参数
    parser.add_argument('--select_token_strategy', type=str, default='all',
                        choices=['all', 'boundary', 'dynamic'],
                        help='选择 token 的策略: all(全部), boundary(边界), dynamic(动态活跃度)')

    # 边界提取数量参数
    parser.add_argument('--boundary_num', type=int, default=1,
                        help='当策略为 boundary 时，每个 block 边界提取的 token 数量')

    # 动态选择数量参数
    parser.add_argument('--dynamic_top_n', type=int, default=5,
                        help='当策略为 dynamic 时，选择的 Top N 个活跃 token')
    # 边界提取宽度

    parser.add_argument('--levels', type=int, nargs='+', default=[5, 5, 5, 5])
    parser.add_argument('--num_quantizers', type=int, default=1)
    # 权重分布的宽度，控制多少个 token 能够贡献核心特征
    # 建议值：1.0 - 3.0。如果 boundary_tokens 较大，建议适当增大 sigma
    parser.add_argument('--weight_sigma', type=float, default=2.0,help='高斯加权的标准差，控制特征提取的集中度')

    args = parser.parse_args()
    process_nanopore_data(args)

if __name__ == "__main__":
    main()
