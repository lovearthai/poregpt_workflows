import os
import glob
import gzip
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# ================= 配置区 =================
INPUT_PATH = "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example"
VOCAB_SIZE = 65536
# ==========================================

def get_training_corpus():
    search_path = os.path.join(INPUT_PATH, "**/*.jsonl.gz")
    files = glob.glob(search_path, recursive=True)
    print(f"\n[状态] 找到 {len(files)} 个文件，启动语料流...")
    
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    # 【核心】手动注入空格，彻底破坏单字符合并的环境
                    yield text.replace('><', '> <')
                except:
                    continue

def train_bpe_final():
    # 1. 构造模型
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. 【核心】由于语料里加了空格，WhitespaceSplit 是最稳健的选择
    # 它会确保 <|bwav:405|> 作为一个整体进入统计阶段
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 3. 【核心】把完整的标签作为 Alphabet 传给 Trainer
    # 这一步是告诉训练器：基础“字母”是 <|bwav:0|>，而不是单字符 < 或 |
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)] + [" "]
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # 4. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        # 【关键】设为 True，强制不许学习 Alphabet 之外的字符（如单字符 <）
        limit_alphabet=True 
    )

    print("\n[状态] 启动自动化训练。Alphabet 已锁死为长标签。")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 5. 保存
    tokenizer.save("bwav_bpe_tokenizer.json")

    # 6. 验证
    model_json = json.loads(tokenizer.to_str())
    vocab = model_json.get("model", {}).get("vocab", {})
    merges = model_json.get("model", {}).get("merges", [])
    
    print("\n" + "="*50)
    print(f"自动化训练总结:")
    print(f"- 最终词表大小: {len(vocab)}")
    print(f"- 合并规则总数: {len(merges)}")
    
    if len(merges) > 0:
        print(f"✅ 搞定！自动化合并已生效。")
        print(f"Top 10 合并示例: {merges[:10]}")
    else:
        print("❌ 依然没有合并。请检查 min_frequency 设定。")
    print("="*50)

if __name__ == "__main__":
    train_bpe_final()
