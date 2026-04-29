import os
import glob
import gzip
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# ================= 配置区 =================
# 建议先用 example 跑通逻辑，正式训练时建议覆盖更多文件
INPUT_PATH = "/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe340s147000l1_split1536_overlap256/example"
VOCAB_SIZE = 65536  # 目标词表大小
# ==========================================

def get_training_corpus():
    search_path = os.path.join(INPUT_PATH, "**/*.jsonl.gz")
    files = glob.glob(search_path, recursive=True)
    if not files:
        print(f"错误：在 {INPUT_PATH} 未找到 .jsonl.gz 文件")
        return
    
    print(f"找到 {len(files)} 个文件，开始解析语料...")
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    data = json.loads(line)
                    # 【关键点 1】不再手动加空格，保持原始紧凑序列
                    # 只有紧凑序列，BPE 才能识别出相邻 Token 的合并机会
                    yield data.get('text', '')
                except json.JSONDecodeError:
                    continue

def train_bpe_final():
    # 1. 初始化 BPE 模型
    # unk_token 设置为特殊 token 之一
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 【关键点 2】使用正则 Split 代替 WhitespaceSplit
    # pattern: 匹配完整的信号标签
    # behavior="isolated": 将匹配到的内容切分为独立的单元，且不丢弃任何内容
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=r"<\|bwav:\d+\|>", 
        behavior="isolated"
    )

    # 2. 构造基础 Alphabet 和特殊 Token
    # initial_alphabet 告诉 BPE 哪些是“原子”
    base_alphabet = [f"<|bwav:{i}|>" for i in range(625)]
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # 3. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,  # 在 Nanopore 数据中，由于重复性高，可以适当调高此值（如 5-10）
        special_tokens=special_tokens,
        initial_alphabet=base_alphabet,
        # 【关键点 3】limit_alphabet=True
        # 强制 BPE 只能使用 initial_alphabet 里的原子进行合并
        # 严禁它把 <|bwav:405|> 拆成字符去合并 '|' 或 '>'
        limit_alphabet=True
    )

    # --- 调试测试 Pre-tokenizer ---
    test_text = "<|bwav:405|><|bwav:407|><|bwav:405|>"
    test_res = tokenizer.pre_tokenizer.pre_tokenize_str(test_text)
    print(f"\n[调试] Pre-tokenize 预览 (应该看到独立的标签列表):")
    print(f"{test_res}")
    # ----------------------------

    # 4. 执行训练
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 5. 保存并分析结果
    tokenizer.save("bwav_bpe_tokenizer.json")
    
    # 提取合并规则进行展示
    model_json = json.loads(tokenizer.to_str())
    merges = model_json.get("model", {}).get("merges", [])

    print("\n" + "="*50)
    print(f"真正的高频信号 Token 合并结果 (Top 20):")
    if not merges:
        print("未发现任何合并项。")
        print("原因分析：")
        print("1. 数据量太小，导致没有重复出现的 Token 对。")
        print("2. min_frequency 设置过高。")
    else:
        for i, m in enumerate(merges[:20]):
            print(f"Top {i+1}: {m}")
    print("="*50)
    print(f"词表训练完成，最终规模: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    train_bpe_final()
