import os
import glob
import gzip
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ================= 配置区 =================
INPUT_PATH = "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example_min"
VOCAB_SIZE = 1024
UNICODE_START = 0xE000  # Unicode 私有区 (PUA) 起始位置
# ==========================================

def id_to_unicode(token_id):
    """将信号 ID (0-624) 转换为单个不可分割的 Unicode 字符"""
    return chr(UNICODE_START + token_id)

def raw_text_to_unicode(text):
    """
    将原始语料中的 <|bwav:123|> 替换为单字符
    注意：这里使用简单的 replace 即可，因为我们要的是 1:1 的原子替换
    """
    import re
    # 匹配 <|bwav:数字|> 的正则
    pattern = r"<\|bwav:(\d+)\|>"
    
    def replace_func(match):
        idx = int(match.group(1))
        return id_to_unicode(idx)
    
    return re.sub(pattern, replace_func, text)

def get_training_corpus():
    search_path = os.path.join(INPUT_PATH, "**/*.jsonl.gz")
    files = glob.glob(search_path, recursive=True)
    print(f"找到 {len(files)} 个文件...")
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    # 将长字符串转为单字符流，BPE 将无法拆分这些字符
                    yield raw_text_to_unicode(text)
                except:
                    continue

def train_bpe_unicode_final():
    # 1. 初始化模型
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. 【核心修正】禁用 ByteLevel，改用 Whitespace 或直接不设置
    # 既然我们要保护 Unicode 原子的完整性，不要让它经过 ByteLevel 转换
    # 我们可以通过这个简单的 PreTokenizer 来确保字符不被拆分
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 3. 构造原子 Alphabet
    base_alphabet = [id_to_unicode(i) for i in range(625)] 
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # 4. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        limit_alphabet=True  # 强制锁死 Alphabet，只允许合并，不允许拆分
    )

    # --- 调试测试：原子性校验 ---
    # 我们在测试字符串里加个空格，因为用了 WhitespaceSplit
    test_id_text = f"{id_to_unicode(405)} {id_to_unicode(407)} {id_to_unicode(405)}"
    print(f"test_id_text:{test_id_text}")
    test_res = tokenizer.pre_tokenizer.pre_tokenize_str(test_id_text)

    print(f"\n[调试] 现在的分词预览:")
    # 预期结果应该是 [(char1, (0,1)), (char2, (2,3)), (char3, (4,5))]
    print(f"{test_res}")

    if len(test_res) != 3:
        print("\n" + "!"*50)
        print(f"错误：预期 3 个 Token，实际得到 {len(test_res)} 个。")
        print("!"*50 + "\n")
        return
if __name__ == "__main__":
    train_bpe_unicode_final()
