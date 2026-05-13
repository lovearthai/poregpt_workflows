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
    print(f"\n[状态] 找到 {len(files)} 个文件...")
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    # 必须加空格，配合 WhitespaceSplit
                    yield text.replace('><', '> <')
                except:
                    continue

def train_bpe_final():
    # 1. 构造基础词表
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)]
    
    # 物理构建初始词表映射
    full_init_vocab = {token: i for i, token in enumerate(special_tokens + base_alphabet)}
    # 加入空格，因为 WhitespaceSplit 会产生空格 Token
    if " " not in full_init_vocab:
        full_init_vocab[" "] = len(full_init_vocab)

    print(f"--- 初始词表构建完成，基础原子数: {len(full_init_vocab)} ---")

    # 2. 直接初始化 BPE 模型，并装载初始词表
    tokenizer = Tokenizer(models.BPE(full_init_vocab, merges=[], unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 3. 配置 BpeTrainer
    # 关键点：不要传 initial_alphabet，因为我们已经手动塞进模型里了
    # 关键点：limit_alphabet 必须为 False，否则它会去检查不存在的单字符字母表
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        limit_alphabet=False 
    )

    print("\n--- 启动 BPE 训练 (Merges 训练) ---")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 4. 保存
    tokenizer.save("bwav_bpe_tokenizer.json")

    # 5. 结果检查
    model_json = json.loads(tokenizer.to_str())
    res_vocab = model_json.get("model", {}).get("vocab", {})
    res_merges = model_json.get("model", {}).get("merges", [])
    
    print("\n" + "="*50)
    print(f"最终统计:")
    print(f"- 初始原子数: {len(full_init_vocab)}")
    print(f"- 训练后词表总数: {len(res_vocab)}")
    print(f"- 合并规则总数: {len(res_merges)}")
    
    if len(res_merges) > 0:
        print(f"✅ 搞定！Top 10 合并:")
        for i, m in enumerate(res_merges[:10]):
            print(f" {m}")
    else:
        print("❌ 依然没有合并。这说明 BPE 在 train 过程中可能重置了 vocab。")
        print("尝试最后的一招：减小 min_frequency 到 1。")
    print("="*50)

if __name__ == "__main__":
    train_bpe_final()
