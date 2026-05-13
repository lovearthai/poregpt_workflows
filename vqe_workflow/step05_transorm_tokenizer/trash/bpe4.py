import os
import glob
import gzip
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ================= 配置区 =================
INPUT_PATH = "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example"
VOCAB_SIZE = 65536
# ==========================================

def get_training_corpus():
    search_path = os.path.join(INPUT_PATH, "**/*.jsonl.gz")
    files = glob.glob(search_path, recursive=True)
    print(f"\n[1/3 状态] 找到 {len(files)} 个文件，准备开始迭代...")
    
    line_count = 0
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    # 【核心策略】我们在训练语料里加空格，方便 Pre-tokenization 分割
                    # BPE 会在后续步骤中把这些原子重新“粘”起来
                    processed_text = text.replace('><', '><')
                    
                    # 调试打印：每处理1000行打印一次进度，或者打印前3行
                    if line_count < 3:
                        print(f"[调试-语料流] 样本 {line_count} 注入后预览: {processed_text[:100]}...")
                    
                    yield processed_text
                    line_count += 1
                except Exception as e:
                    print(f"[错误] 解析行失败: {e}")
                    continue
    print(f"[2/3 状态] 语料迭代完毕，总计发送行数: {line_count}")

def train_bpe_final():
    # 1. 使用 BPE 模型，并设置特殊的处理逻辑
    # 我们不需要 subword 标记，因为我们要的是完整的标签合并
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. 【关键修正】使用特定的 Pre-tokenizer 组合
    # 既然正则容易失败，我们直接用 WhitespaceSplit，因为我们在语料里手动加了空格
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 【终极修正】参考你提供的 Regex 逻辑
    # 在 Python 原生字符串中，我们需要对 | 进行转义
    # 使用 [^>]+ 可以完美匹配 <|bwav:405|> 这种结构
    # 【核心修正】强制原子分割，且不留空格
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=r"<\\|bwav:\d+\\|>",
        behavior="isolated"
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=r"<\|[^>]+\|>",
        behavior="isolated"
    )


    # 3. 构造原子 Alphabet
    # base_alphabet = [f"<|bwav:{i}|>" for i in range(625)] + [" "] # 加上空格
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)] # 加上空格
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)]# 加上空格
    print(base_alphabet)
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # 4. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        # 强制不允许拆解 <|bwav:xxx|> 内部字符
        limit_alphabet=False
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
    test_text = "<|bwav:405|> <|bwav:407|> <|bwav:405|>"
    test_text = "<|bwav:405|><|bwav:407|><|bwav:405|>"
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
    
    # [新增调试] 在正式进入训练前，最后确认一次 Alphabet 包含空格
    if " " not in base_alphabet:
        print("⚠️ 警告: WhitespaceSplit 已启用，但 Alphabet 中不含空格，可能导致合并失败。")
    
    # ----------------------------------
    # 5. 训练
    print(f"\n[3/3 状态] 启动 train_from_iterator...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 6. 保存
    tokenizer.save("bwav_bpe_tokenizer.json")

    # 打印 Merge 结果
    model_json = json.loads(tokenizer.to_str())
    merges = model_json.get("model", {}).get("merges", [])
    
    print("\n" + "="*50)
    print(f"训练总结: 最终词表大小: {len(model_json['model']['vocab'])}")
    print(f"合并规则(Merges)总数: {len(merges)}")
    print(f"Top 20 合并结果:")
    if not merges:
        print("❌ 未发现任何合并！请检查 min_frequency 是否过高，或 Alphabet 冲突。")
    else:
        for i, m in enumerate(merges[:20]):
            print(f"Top {i+1}: {m}")
    print("="*50)

if __name__ == "__main__":
    train_bpe_final()
