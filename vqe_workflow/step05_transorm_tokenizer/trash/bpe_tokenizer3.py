import os
import glob
import gzip
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ================= 配置区 =================
INPUT_PATH = "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example_min"
VOCAB_SIZE = 1024
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
                    # 【核心策略】我们在训练语料里加空格，方便 Pre-tokenization 分割
                    # BPE 会在后续步骤中把这些原子重新“粘”起来
                    yield text.replace('><', '> <')
                except:
                    continue

def train_bpe_final():
    # 1. 使用 BPE 模型，并设置特殊的处理逻辑
    # 我们不需要 subword 标记，因为我们要的是完整的标签合并
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. 【关键修正】使用特定的 Pre-tokenizer 组合
    # 既然正则容易失败，我们直接用 WhitespaceSplit，因为我们在语料里手动加了空格
    
    # 【终极修正】参考你提供的 Regex 逻辑
    # 在 Python 原生字符串中，我们需要对 | 进行转义
    # 使用 [^>]+ 可以完美匹配 <|bwav:405|> 这种结构
    # 【核心修正】强制原子分割，且不留空格

    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=r"<\|[^>]+\|>",
        behavior="isolated"
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=r"<\|bwav:\d+\|>", 
        behavior="isolated"
    )

    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    # 1. 先把 <|bwav:xxx|> 这种特殊格式隔离出来，当作一个整体
    pre_tokenizers.Split(pattern=r"(<\|bwav:\d+\|>)", behavior="isolated"),
    ])
    # 2. 【关键】如果你希望以后分词时不需要空格，你需要给 Tokenizer 加一个 Decoder
    # 这样模型输出结果时，会自动把这些标签连起来
    tokenizer.decoder = decoders.BPEDecoder()
    # 3. 构造原子 Alphabet
    #base_alphabet = [f"<|bwav:{i}|>" for i in range(625)] +[" "]# 加上空格
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)] # 加上空格
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # 4. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        # 强制不允许拆解 <|bwav:xxx|> 内部字符
        limit_alphabet=630
    )

    # 1. 深度纠偏：limit_alphabet 的真实内幕
    # 在 tokenizers 库中，BPE 的合并逻辑是自底向上的：
    # limit_alphabet=True (推荐方案)：
    # 逻辑： 强制 BPE 只能使用你在 initial_alphabet 中提供的 625 个原子（<|bwav:0|> 到 <|bwav:624|>）作为“原始砖块”。

    # 结果： BPE 会把这些原子两两合并，产生类似 <|bwav:405|><|bwav:407|> 的新 Token。它绝对不会去拆开原子内部的字符。
    # limit_alphabet=False (风险方案)：
    # 逻辑： 允许 BPE 扫描语料，如果它发现某些字符（比如单个的 | 或 >）出现频率极高，它会绕过你给的 initial_alphabet，自己去提取这些字符作为原始原子。
    # 结果： 这就是为什么你之前会看到 ['|', '|'] 的原因。因为它觉得合并两个 | 比合并两个巨大的信号标签“划算”。
    # 结论： 既然你已经确定 Nanopore 的物理信号只有 625 种，你就必须锁死 alphabet。要把 limit_alphabet 设为 True，强制它只在这 625 个“砖块”上盖大楼。
    
    # --- 调试测试 ---
    # --- 调试测试：原子切分强约束检测 ---
    test_text = "<|bwav:405|><|bwav:407|><|bwav:405|>"
    test_text = "<|bwav:405|> <|bwav:407|> <|bwav:405|>"
    # 注意：如果你的语料里换成了 "> <"，这里也要对齐
    test_res = tokenizer.pre_tokenizer.pre_tokenize_str(test_text)
    
    print(f"\n[调试] 现在的分词预览:")
    print(f"{test_res}")

    # 检查逻辑：预期的原子数量应该是 3
    if len(test_res) != 3:
        print("\n" + "!"*50)
        print("错误：Pre-tokenizer 切分失败！")
        print(f"预期得到 3 个独立 Token，但实际得到 {len(test_res)} 个。")
        print("这说明正则匹配或空格切分逻辑有误，请检查 pre_tokenizer 配置。")
        print("!"*50 + "\n")
        return # 立刻终止训练
    
    # 进一步检查：确保没有被拆成字符（每个元组的字符串长度应该 > 10）
    if any(len(t[0]) < 10 for t in test_res):
        print("\n" + "!"*50)
        print("警告：检测到 Token 内部被拆碎！")
        print("虽然数量对，但内容可能被拆成了字符（如 '|' 或 '>'）。")
        print("!"*50 + "\n")
        return
        
    print("✅ Pre-tokenizer 校验通过，正在进入正式训练阶段...")
    # ----------------------------------
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
