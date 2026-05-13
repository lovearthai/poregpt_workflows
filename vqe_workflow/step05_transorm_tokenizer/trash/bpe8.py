import os, glob, gzip, json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# ================= 配置区 =================
INPUT_PATH = "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example"
VOCAB_SIZE = 65536
# ==========================================

def get_training_corpus():
    files = glob.glob(os.path.join(INPUT_PATH, "**/*.jsonl.gz"), recursive=True)
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    text = json.loads(line).get('text', '')
                    # 【核心】手动加空格，从源头切断 > 和 < 的粘连
                    yield text.replace('><', '> <')
                except: continue

def train_bpe_final():
    # 1. 依然使用 BPE，但我们这次要彻底锁死 Pre-tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. 【关键】改用 WhitespaceSplit。既然我们手动加了空格，这是最稳的。
    # 它会严格按空格切分，切出来的就是一个个完整的 <|bwav:405|>
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 3. 【核心修正】initial_alphabet 必须包含完整的标签字符串
    # 这样 BPE 才会把整个标签当成一个基础单位，而不是去拆解里面的 >
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)] + [" "]
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        # 【关键】必须为 True。强制它只许用我们给的长标签盖楼，不许乱合并 >
        limit_alphabet=True 
    )

    print("\n[状态] 启动自动化训练。这次我们锁死了 Alphabet，不许它合并单字符。")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 4. 保存并检查
    tokenizer.save("bwav_bpe_tokenizer.json")
    
    model_json = json.loads(tokenizer.to_str())
    merges = model_json.get("model", {}).get("merges", [])
    vocab = model_json.get("model", {}).get("vocab", {})
    
    print("\n" + "="*50)
    print(f"最终结果：词表大小 {len(vocab)}，合并规则 {len(merges)}")
    if merges:
        print(f"Top 5 合并: {merges[:5]}")
    else:
        print("❌ 依然没有合并？请检查 get_training_corpus 是否真的吐出了数据。")
    print("="*50)

if __name__ == "__main__":
    train_bpe_final()
