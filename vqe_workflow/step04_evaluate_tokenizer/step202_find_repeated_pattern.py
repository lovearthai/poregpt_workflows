# -*- coding: utf-8 -*-
import gzip
import json
import re
import argparse
from tqdm import tqdm

def find_sequence_pattern(args):
    # 构建正则表达式：匹配 pattern 中每个碱基重复至少 min_repeat 次
    # 例如 pattern="ATAT", min=5 -> "A{5,}T{5,}A{5,}T{5,}"
    regex_parts = [f"{char}{{{args.min_repeat},}}" for char in args.pattern]
    pattern_string = "".join(regex_parts)
    pattern_regex = re.compile(pattern_string)

    print(f"🔍 搜索模式: {args.pattern} (每种碱基重复 >= {args.min_repeat} 次)")
    print(f"🔬 匹配逻辑: {pattern_string}")
    print("-" * 80)

    try:
        with gzip.open(args.input_file, 'rt', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    text = data.get("tokens_based", "")
                    
                    if not text:
                        continue

                    # 在 tokens_based 文本中寻找匹配项
                    for match in pattern_regex.finditer(text):
                        start_pos = match.start()
                        end_pos = match.end()
                        length = end_pos - start_pos
                        
                        # 获取具体的碱基片段
                        matched_segment = match.group()
                        
                        # 格式化输出：行号 | 起始 | 结束 | 长度 | 具体的碱基片段
                        print(f"Line: {line_idx:<6} | Start: {start_pos:<5} | End: {end_pos:<5} | Len: {length:<4} | Seq: {matched_segment}")
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error at line {line_idx}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {args.input_file}")
    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="寻找碱基重复模式并输出匹配的碱基段")
    parser.add_argument('-i', '--input-file', type=str, required=True, help='输入 jsonl.gz 路径')
    parser.add_argument('-p', '--pattern', type=str, required=True, help='模式字符串，如 ATAT')
    parser.add_argument('-m', '--min-repeat', type=int, default=5, help='每个碱基最少重复次数')

    args = parser.parse_args()
    find_sequence_pattern(args)
