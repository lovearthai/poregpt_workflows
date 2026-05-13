import os
import glob
import gzip
import json
import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# ================= 配置区 =================
INPUT_PATH = "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example_min"
VOCAB_SIZE = 1024
# ==========================================

def get_training_corpus_with_placeholders():
    """
    读取语料，并将 <|bwav:xxx|> 替换为安全的占位符 __AUDIO_xxx__
    """
    search_path = os.path.join(INPUT_PATH, "**/*.jsonl.gz")
    files = glob.glob(search_path, recursive=True)
    print(f"找到 {len(files)} 个文件...")
    
    # 匹配 <|bwav:数字|> 的正则
    audio_pattern = re.compile(r'<\|bwav:(\d+)\|>')
    
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    
                    # 核心替换逻辑：把 <|bwav:405|> 替换为 __AUDIO_405__
                    def replace_with_placeholder(match):
                        num = match.group(1)
                        return f"__AUDIO_{num}__"
                    
                    processed_text = audio_pattern.sub(replace_with_placeholder, text)
                    yield processed_text
                    
                except Exception as e:
                    continue

def train_bpe_final():
    # 1. 初始化 Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>",continuing_subword_prefix="##" ))

    # 2. 预分词器：现在可以放心大胆地只用 WhitespaceSplit 了
    # 因为我们的占位符 __AUDIO_xxx__ 中间没有空格，会被当作一个整体
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 3. 构造初始字母表：使用占位符
    # 这样 BPE 拿到的基础积木就是完整的 __AUDIO_xxx__，绝对不会去拆
    base_alphabet = [f"__AUDIO_{i}__" for i in range(625)]
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # 4. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        continuing_subword_prefix="##", # 必须和模型里的保持一致！
        limit_alphabet=630
    )

    # --- 调试测试 ---
    test_text = "__AUDIO_405__ __AUDIO_407__ __AUDIO_405__"
    test_res = tokenizer.pre_tokenizer.pre_tokenize_str(test_text)
    print(f"\n[调试] 现在的分词预览 (占位符版):")
    print(f"{test_res}")
    
    if len(test_res) == 3 and all(len(t[0]) > 10 for t in test_res):
        print("✅ Pre-tokenizer 校验通过，正在进入正式训练阶段...")
    else:
        print("❌ 调试失败，请检查占位符逻辑。")
        return
    # ----------------------------------

    # 5. 训练 (喂进去的是被替换成占位符的文本)
    tokenizer.train_from_iterator(get_training_corpus_with_placeholders(), trainer=trainer)

    # 6. 【关键步骤】训练后还原：把占位符换回真实的 <|bwav:xxx|>
    print("\n正在将占位符还原为真实的音频 Token...")
    vocab = tokenizer.get_vocab()
    new_vocab = {}
    
    for token, token_id in vocab.items():
        # 如果词汇表里的 Token 是占位符，就把它换回原本的 <|bwav:xxx|>
        if token.startswith("__AUDIO_") and token.endswith("__"):
            # 提取中间的数字
            num = token[10:-2] 
            real_token = f"<|bwav:{num}|>"
            new_vocab[real_token] = token_id
        else:
            new_vocab[token] = token_id

    # 将替换好的词汇表重新加载回模型中
    # 7. 保存
    # 【最终修复】提取合并规则，并将列表格式转换为元组格式
    import json
    tokenizer_json = json.loads(tokenizer.to_str())
    # 提取出的 merges 格式已经是 [["A", "B"], ["C", "D"]]
    raw_merges = tokenizer_json.get("model", {}).get("merges", [])
    # 直接转换为库要求的元组格式 [("A", "B"), ("C", "D")]
    formatted_merges = [tuple(m) for m in raw_merges]
    
    # 用新词汇表和格式化后的合并规则重新构建 BPE 模型
    tokenizer.model = models.BPE(vocab=new_vocab, merges=formatted_merges)
    # 7. 保存
    tokenizer.save("bwav_bpe_tokenizer.json")
    print("\n" + "="*50)
    print(f"Top 20 合并结果:")
    for i, m in enumerate(merges[:20]):
        print(f"Top {i+1}: {m}")
    print("="*50)

if __name__ == "__main__":
    train_bpe_final()
