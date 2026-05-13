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
    print(f"\n[状态] 找到 {len(files)} 个文件，开始准备语料流...")
    
    line_count = 0
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    # 为了确保万无一失，我们在语料中依然注入空格作为物理隔离
                    processed = text.replace('><', '> <')
                    yield processed
                    line_count += 1
                except:
                    continue
    print(f"[状态] 语料加载完毕，总计行数: {line_count}")

def train_bpe_final():
    # ---------------------------------------------------------
    # 第一阶段：使用 WordLevel 焊死基础原子
    # ---------------------------------------------------------
    print("\n--- 第一阶段：固定 625 个基础信号原子 ---")
    
    # 初始模型用 WordLevel，它绝不会拆分字符串
    tokenizer = Tokenizer(models.WordLevel(unk_token="<|unk|>"))
    
    # 使用 WhitespaceSplit，配合我们 replace 出来的空格
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)] + [" "]
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # WordLevel 训练器：强制只识别我们给定的原子
    word_trainer = trainers.WordLevelTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        show_progress=True
    )

    tokenizer.train_from_iterator(get_training_corpus(), trainer=word_trainer)
    
    # 此时，词表里已经有了这 625 个完整的标签
    print(f"基础词表构建完成，当前词表大小: {tokenizer.get_vocab_size()}")

    # ---------------------------------------------------------
    # 第二阶段：无损转换为 BPE 模式进行合并训练
    # ---------------------------------------------------------
    print("\n--- 第二阶段：切换至 BPE 模式训练合并规则 (Merges) ---")
    
    # 获取第一步生成的词表，以此作为 BPE 的基础（底座）
    current_vocab = tokenizer.get_vocab()
    print(current_vocab)
    # 强制将模型更换为 BPE，但保持词表不变
    tokenizer.model = models.BPE(current_vocab, merges=[], unk_token="<|unk|>")

    # BPE 训练器：专门用来寻找那 11 万个“406+406”对子
    bpe_trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2, # 只要出现2次以上就合并
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        limit_alphabet=False # 关键：此时必须为 False，允许学习合并后的新词
    )

    # 再次跑一遍语料，这次只做 Merges 计算
    tokenizer.train_from_iterator(get_training_corpus(), trainer=bpe_trainer)

    # ---------------------------------------------------------
    # 第三阶段：保存与验证
    # ---------------------------------------------------------
    # 6. 保存
    tokenizer.save("bwav_bpe_tokenizer.json")

    # 打印 Merge 结果
    model_json = json.loads(tokenizer.to_str())
    merges = model_json.get("model", {}).get("merges", [])
    vocab = model_json.get("model", {}).get("vocab", {})
    
    print("\n" + "="*50)
    print(f"训练总结:")
    print(f"- 最终词表总大小: {len(vocab)}")
    print(f"- 学习到的合并规则数: {len(merges)}")
    
    if len(merges) > 0:
        print(f"✅ 成功！前 20 个高频合并规则:")
        for i, m in enumerate(merges[:20]):
            print(f"Top {i+1}: {m}")
    else:
        print("❌ 失败：未发现任何合并规则。请检查语料重复度。")
    print("="*50)

if __name__ == "__main__":
    train_bpe_final()
