import gzip
import json
import os
import argparse

def get_overlap(a_start, a_end, b_start, b_end):
    """计算两个区间的重叠长度"""
    return max(0, min(a_end, b_end) - max(a_start, b_start))

def process_mapping(input_path, output_path, factor):
    """
    根据信号映射关系，计算每个 token 对应的碱基
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"🚀 开始映射处理: {input_path}")
    print(f"📏 下采样因子 (Factor): {factor} (1 token = {factor} signals)")

    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', encoding='utf-8') as f_out:

        count = 0
        for line in f_in:
            data = json.loads(line)
            
            # 提取必要字段
            tokens = data.get('tokens', [])
            seq = data.get('pattern', "")
            spans = data.get('base_sample_spans_rel', [])
            
            if tokens and seq and spans:
                token_bases = []
                base_idx = 0
                num_bases = len(seq)
                
                # 遍历每个 token
                for i in range(len(tokens)):
                    # 当前 token 对应的信号区间 [t_start, t_end)
                    t_start = i * factor
                    t_end = (i + 1) * factor
                    
                    max_overlap = -1
                    best_base = "N" # 默认值，防止未匹配
                    
                    # 优化搜索：从上一次匹配的碱基附近开始找
                    # 考虑到 Nanopore 信号是有序的，我们不需要每次都遍历整个序列
                    temp_idx = base_idx
                    while temp_idx < num_bases:
                        b_start, b_end = spans[temp_idx]
                        
                        # 如果当前碱基区间已经完全超过了 token 区间，停止搜索
                        if b_start >= t_end:
                            break
                            
                        overlap = get_overlap(t_start, t_end, b_start, b_end)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_base = seq[temp_idx]
                            # 更新 base_idx 以缩小下次搜索范围
                            if overlap > (factor / 2): # 如果重叠超过一半，下次从这里开始
                                base_idx = temp_idx
                        
                        temp_idx += 1
                    
                    token_bases.append(best_base)
                
                # 将结果转为字符串并存入新字段
                # 按照你的需求，插入在 tokens_layered 之后（如果存在的话）
                new_data = {}
                inserted = False
                for key, value in data.items():
                    new_data[key] = value
                    if key == 'tokens_layered':
                        new_data['tokens_based'] = "".join(token_bases)
                        inserted = True
                
                # 如果没有 tokens_layered 字段，就放在 tokens 后面或最后
                if not inserted:
                    if 'tokens' in new_data:
                        # 重新构造以保证顺序
                        temp_data = {}
                        for k, v in new_data.items():
                            temp_data[k] = v
                            if k == 'tokens':
                                temp_data['tokens_based'] = "".join(token_bases)
                        new_data = temp_data
                    else:
                        new_data['tokens_based'] = "".join(token_bases)

                data = new_data

            f_out.write(json.dumps(data) + '\n')
            
            count += 1
            if count % 1000 == 0:
                print(f"⏳ 已处理 {count} 行记录...", end='\r')

    print(f"\n✅ 处理完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='根据信号跨度将 tokens 映射为碱基序列')
    parser.add_argument('-i', '--input-file', type=str, required=True, help='输入的 jsonl.gz 路径')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='输出的 jsonl.gz 路径')
    parser.add_argument('-f', '--factor', type=int, default=4, help='Token 与信号的比例关系，默认为 4')
    
    args = parser.parse_args()
    process_mapping(args.input_file, args.output_file, args.factor)
