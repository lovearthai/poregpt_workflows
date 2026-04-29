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
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                data = json.loads(line)
                # 必须加空格！
                yield data.get('text', '').replace('><', '> <')

def train_bpe_final():
    # 1. 初始化模型
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 2. 关键：预先构造所有的基础 Token 作为初始 Alphabet
    # 假设你的 ID 范围是 0-511，如果有变化请调整
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)]
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # 3. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,          # 极低阈值，确保 10万次的对子必进
        special_tokens=special_tokens,
        # 【最核心修改】显式传入 alphabet，并禁止它自己寻找其他字符
        initial_alphabet=base_alphabet,
        # 修改为 False：允许算法在 alphabet 之外创建新 Token
        limit_alphabet=False
    )

    # --- 调试测试 ---
    test_text = "<|bwav:405|> <|bwav:407|>"
    test_res = tokenizer.pre_tokenizer.pre_tokenize_str(test_text)
    print(f"\n[调试] 测试分词结果: {test_res}")
    # ----------------

    # 4. 训练
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 5. 导出结果
    model_json = json.loads(tokenizer.to_str())
    merges = model_json.get("model", {}).get("merges", [])

    print("\n" + "="*50)
    print(f"真正的高频 Token ID 合并结果 (Top 20):")
    
    # 过滤掉可能的字符级干扰（保险起见）
    clean_merges = [m for m in merges if " " in m and "<|bwav:" in m]

    if not clean_merges:
        print("未发现有效的 Token 级合并。")
        print("可能原因：VOCAB_SIZE 太小，或者训练数据量太少（目前只找到9个文件）。")
    else:
        for i, m in enumerate(clean_merges[:20]):
            print(f"Top {i+1}: {m}")
    print("="*50)

    tokenizer.save("bwav_bpe_tokenizer.json")

if __name__ == "__main__":
    train_bpe_final()
