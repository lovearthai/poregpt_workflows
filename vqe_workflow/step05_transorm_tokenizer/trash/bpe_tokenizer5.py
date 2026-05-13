import os
import glob
import gzip
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

# ================= 配置区 =================
INPUT_PATH = "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example"
VOCAB_SIZE = 65536
# ==========================================

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
                    yield text
                except:
                    continue

def train_bpe_final():
    # 1. 使用 BPE 模型
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 【关键修正】内存级空格注入，满足“不改原始语料”要求
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace(pattern="><", content="> <")
    ])

    # 2. 【关键修正】单一 Pre-tokenizer 确保物理切分
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 3. 构造原子 Alphabet
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)] + [" "]
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]
    
    # --- 调试点 1: 初始原子检查 ---
    print("\n" + "-"*30 + " [调试点 1: 初始 Alphabet] " + "-"*30)
    print(f"预设原子总数: {len(base_alphabet)}")
    print(f"前 5 个原子: {base_alphabet[:5]}")
    print(f"特殊 Token: {special_tokens}")

    # 4. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        limit_alphabet=True  # 强制锁死 Alphabet
    )

    # --- 调试点 2: 原子切分强约束检测 ---
    test_text = "<|bwav:405|><|bwav:407|><|bwav:405|>"
    normalized_test = tokenizer.normalizer.normalize_str(test_text)
    test_res = tokenizer.pre_tokenizer.pre_tokenize_str(normalized_test)

    print("\n" + "-"*30 + " [调试点 2: Pre-tokenize 预览] " + "-"*30)
    print(f"原始输入: {test_text}")
    print(f"标准化后: {normalized_test}")
    print(f"分词预览: {test_res}")

    if len(test_res) != 3:
        print("\n" + "!"*50)
        print(f"错误：预期 3 个 Token，实得 {len(test_res)} 个！切分逻辑依然失效。")
        print("!"*50 + "\n")
        return 

    if any(len(t[0]) < 5 for t in test_res if t[0] != " "):
        print("\n" + "!"*50)
        print("警告：检测到短字符碎片！原子可能被拆解了。")
        print("!"*50 + "\n")
        return

    print("✅ Pre-tokenizer 校验通过，正在进入正式训练阶段...")
    
    # 5. 训练
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 6. 保存
    tokenizer.save("bwav_bpe_tokenizer.json")

    # --- 调试点 3: 最终 Vocab 深度分析 ---
    model_json = json.loads(tokenizer.to_str())
    final_vocab = model_json.get("model", {}).get("vocab", {})
    merges = model_json.get("model", {}).get("merges", [])
    
    print("\n" + "-"*30 + " [调试点 3: 最终 Vocab 分析] " + "-"*30)
    print(f"1. 最终词表总大小: {len(final_vocab)}")
    print(f"2. 最终合并规则数: {len(merges)}")
    
    # 统计词表成分
    single_chars = [t for t in final_vocab if len(t) == 1 and t != " "]
    long_tokens = [t for t in final_vocab if len(t) > 15] # 长度大于单个标签的，说明是合并后的
    base_tokens = [t for t in final_vocab if "<|bwav:" in t and len(t) <= 15]

    print(f"3. 词表成分统计:")
    print(f"   - 基础单字符碎片 (异常项): {len(single_chars)} 个 {'(存在风险！)' if single_chars else '(干净)'}")
    if single_chars: print(f"     详情: {single_chars[:10]}")
    print(f"   - 基础原子标签 (<|bwav:i|>): {len(base_tokens)} 个")
    print(f"   - 合并后的复合 Token (级联结果): {len(long_tokens)} 个")

    print("\n" + "="*50)
    print(f"Top 20 合并结果 (Merges):")
    if not merges:
        print("❌ 警告：未发现任何合并规则！BPE 训练未生效。")
    for i, m in enumerate(merges[:20]):
        print(f"  [{i+1}] {m}")
    print("="*50)

if __name__ == "__main__":
    train_bpe_final()
