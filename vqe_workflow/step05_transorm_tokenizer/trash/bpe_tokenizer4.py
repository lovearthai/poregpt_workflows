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
                    # 【核心策略】保持原始语料，不再手动在 Python 层 replace
                    # 我们将处理逻辑下放到 Tokenizer 的 Normalizer 层
                    yield text
                except:
                    continue

def train_bpe_final():
    # 1. 使用 BPE 模型
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 【关键修正 A】使用 Normalizer 替代手动 replace
    # 这会在分词前，在内存中临时将 "><" 替换为 "> <"，从而制造出分割点
    # 这样既不修改原始语料，又能让 WhitespaceSplit 生效
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace(pattern="><", content="> <")
    ])

    # 2. 【关键修正 B】统一 Pre-tokenizer
    # 删掉了之前互相覆盖的三行，只保留 WhitespaceSplit
    # 因为 Normalizer 已经帮我们把标签分开了，WhitespaceSplit 是最稳健的物理隔离
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 3. 构造原子 Alphabet
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)] + [" "]
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # 4. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        # 强制不允许拆解 <|bwav:xxx|> 内部字符
        limit_alphabet=True
    )

    # 1. 深度纠偏：limit_alphabet 的真实内幕
    # limit_alphabet=True (推荐方案)：强制使用 625 个原子作为“原始砖块”

    # --- 调试测试：原子切分强约束检测 ---
    test_text = "<|bwav:405|><|bwav:407|><|bwav:405|>"
    
    # 【测试逻辑修正】手动测试时需要先经过 normalizer
    normalized_test = tokenizer.normalizer.normalize_str(test_text)
    test_res = tokenizer.pre_tokenizer.pre_tokenize_str(normalized_test)

    print(f"\n[调试] 原始输入: {test_text}")
    print(f"[调试] 标准化后: {normalized_test}")
    print(f"[调试] 现在的分词预览: {test_res}")

    # 检查逻辑：预期的原子数量应该是 3
    if len(test_res) != 3:
        print("\n" + "!"*50)
        print("错误：Pre-tokenizer 切分失败！")
        print(f"预期得到 3 个独立 Token，但实际得到 {len(test_res)} 个。")
        print("这说明正则匹配或空格切分逻辑有误，请检查 pre_tokenizer 配置。")
        print("!"*50 + "\n")
        return 

    # 进一步检查：确保没有被拆成字符
    if any(len(t[0]) < 10 for t in test_res):
        print("\n" + "!"*50)
        print("警告：检测到 Token 内部被拆碎！")
        print("虽然数量对，但内容可能被拆成了字符（如 '|' 或 '>'）。")
        print("!"*50 + "\n")
        return

    print("✅ Pre-tokenizer 校验通过，正在进入正式训练阶段...")
    
    # 5. 训练
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 6. 保存
    tokenizer.save("bwav_bpe_tokenizer.json")

    # 打印 Merge 结果
    model_json = json.loads(tokenizer.to_str())
    merges = model_json.get("model", {}).get("merges", [])
    print("\n" + "="*50)
    print(f"Top 20 合并结果:")
    for i, m in enumerate(merges[:20]):
        print(f"Top {i+1}: {m}")
    print("="*50)

if __name__ == "__main__":
    train_bpe_final()
