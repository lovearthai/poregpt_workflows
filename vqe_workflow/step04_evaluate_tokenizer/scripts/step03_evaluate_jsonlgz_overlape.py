import gzip
import json
import math
import argparse
from collections import Counter
from typing import List, Dict, Tuple

def load_all_tokens(filepath: str) -> List[List[int]]:
    """读取文件中所有的 tokens"""
    all_tokens = []
    print(f"正在全量读取: {filepath} ...")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'tokens' in data and isinstance(data['tokens'], list):
                        all_tokens.append(data['tokens'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {filepath}")
        exit(1)
        
    print(f"✅ 成功读取 {len(all_tokens)} 行数据")
    return all_tokens

def get_ngrams(sequence: List[int], n: int) -> set:
    """生成 N-Gram 集合"""
    if len(sequence) < n:
        return {tuple(sequence)}
    return {tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)}

def calculate_metrics(tokens_a: List[int], tokens_b: List[int], n: int) -> Dict[str, float]:
    """计算三个核心指标"""
    
    # 1. 杰卡德相似系数
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    union = set_a | set_b
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # 2. Token 重叠度 (基于数量的相对重叠率)
    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    overlap_count = sum((counter_a & counter_b).values())
    total_len = len(tokens_a) + len(tokens_b)
    token_overlap_ratio = (2 * overlap_count) / total_len if total_len else 0.0
    
    # 3. N-Gram 序列重叠率
    ngrams_a = get_ngrams(tokens_a, n)
    ngrams_b = get_ngrams(tokens_b, n)
    ng_intersection = ngrams_a & ngrams_b
    ng_union = ngrams_a | ngrams_b
    ngram_overlap = len(ng_intersection) / len(ng_union) if ng_union else 0.0
    
    return {
        "jaccard": jaccard,
        "token_overlap": token_overlap_ratio,
        "ngram_overlap": ngram_overlap
    }

def calculate_stats(values: List[float]) -> Tuple[float, float, float, float]:
    """计算最大值、最小值、平均值、标准差"""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    
    n = len(values)
    mean_val = sum(values) / n
    variance_val = sum((x - mean_val) ** 2 for x in values) / n
    std_dev_val = math.sqrt(variance_val)
    
    return max(values), min(values), mean_val, std_dev_val

def print_stats_table(title: str, values: List[float]):
    """格式化打印统计结果"""
    max_v, min_v, avg_v, std_v = calculate_stats(values)
    print(f"{title:<25} | {max_v:>8.4f} | {min_v:>8.4f} | {avg_v:>8.4f} | {std_v:>8.4f}")

def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description='分析 Token 重叠率统计指标')
    
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='输入的 jsonl.gz 文件路径')
    parser.add_argument('-n', '--ngram-size', type=int, default=2,
                        help='N-Gram 的大小，默认为 2')
    
    args = parser.parse_args()

    # 1. 加载所有数据
    tokens_list = load_all_tokens(args.input_file)
    
    if len(tokens_list) < 2:
        print("数据量不足。")
        return

    # 2. 全量两两比对
    total_pairs = len(tokens_list) * (len(tokens_list) - 1) // 2
    print(f"开始全量比对 (共 {len(tokens_list)} 行，预计 {total_pairs} 次计算)...")
    
    # 用于存储所有比对结果的列表
    jaccard_scores = []
    token_overlap_scores = []
    ngram_overlap_scores = []
    
    count = 0

    for i in range(len(tokens_list)):
        for j in range(i + 1, len(tokens_list)):
            metrics = calculate_metrics(tokens_list[i], tokens_list[j], args.ngram_size)
            
            jaccard_scores.append(metrics['jaccard'])
            token_overlap_scores.append(metrics['token_overlap'])
            ngram_overlap_scores.append(metrics['ngram_overlap'])
            
            count += 1
            if count % 1000 == 0:
                print(f"进度: {count} / {total_pairs} ...", end='\r')

    # 3. 输出统计报表
    print("\n" + "="*80)
    print(f"📊 全量统计结果 (基于 {count} 对样本)")
    print("="*80)
    print(f"{'指标名称':<25} | {'最大值':>8} | {'最小值':>8} | {'平均值':>8} | {'标准差':>8}")
    print("-" * 80)
    
    print_stats_table("1. 杰卡德相似系数", jaccard_scores)
    print_stats_table("2. Token 重叠度", token_overlap_scores)
    print_stats_table("3. N-Gram 序列重叠率", ngram_overlap_scores)
    
    print("="*80)
    print("注：标准差越大，说明样本间的相似度差异越大（分布越分散）。")

if __name__ == "__main__":
    main()
