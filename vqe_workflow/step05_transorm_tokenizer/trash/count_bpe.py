import os
import gzip
import json
import re
from collections import Counter

INPUT_PATH = "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example"

def hard_count_check():
    # 存储相邻对的计数器
    pair_counts = Counter()
    
    import glob
    files = glob.glob(os.path.join(INPUT_PATH, "**/*.jsonl.gz"), recursive=True)
    
    print(f"正在扫描 {len(files)} 个文件...")
    
    for f in files:
        with gzip.open(f, 'rt') as g:
            for line in g:
                data = json.loads(line)
                # 正则精确提取 Token
                tokens = re.findall(r'<\|bwav:\d+\|>', data.get('text', ''))
                if len(tokens) < 2: continue
                
                # 统计相邻对
                pairs = zip(tokens[:-1], tokens[1:])
                pair_counts.update(pairs)

    # 打印前 20 个
    print("\n" + "="*40)
    print("【硬计数结果】数据中真实的 Token 对频次：")
    top_pairs = pair_counts.most_common(20)
    if not top_pairs:
        print("竟然真的没有相邻对！请检查 text 字段内容。")
    else:
        for (p1, p2), count in top_pairs:
            print(f"{p1} + {p2} : 出现了 {count} 次")
    print("="*40)

if __name__ == "__main__":
    hard_count_check()
