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
    print(f"\n[1/3 状态] 找到 {len(files)} 个文件，准备开始自动化训练流...")
    
    line_count = 0
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as g:
            for line in g:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    # 自动化模式下，我们不需要手动加空格，让 Pre-tokenizer 处理
                    yield text
                    line_count += 1
                except:
                    continue
    print(f"[2/3 状态] 语料迭代完毕，总计发送行数: {line_count}")

def train_bpe_automation_final():
    # 1. 初始化空的 BPE 模型
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # 2. 【核心点】配置强力 Pre-tokenizer 保护原子
    # 使用正则直接匹配完整的标签，behavior="isolated" 会将其作为不可拆分的整体
    # 这样 BPE 就没法把 <|bwav:405|> 拆成字符了
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=r"<\|bwav:\d+\|>",
        behavior="isolated"
    )

    # 3. 【核心点】提取所有基础字符，满足 BPE 的“字符级”执念
    # 我们把标签中可能出现的每一个字符都提取出来作为 Alphabet
    all_chars = list(set("".join([f"<|bwav:{i}|>" for i in range(625)])))
    print(f"提取基础字符集完成，共 {len(all_chars)} 个基础字符。")

    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

    # 4. 配置自动化训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        min_frequency=2,
        special_tokens=special_tokens,
        # 必须提供字符级字母表，否则 BPE 会清空非字母字符
        initial_alphabet=all_chars,
        limit_alphabet=True
    )

    # 5. 执行自动化训练
    print("\n[3/3 状态] 启动 train_from_iterator (自动化路径)...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 6. 保存
    tokenizer.save("bwav_bpe_tokenizer.json")

    # ----------------------------------
    # 结果校验
    # ----------------------------------
    model_json = json.loads(tokenizer.to_str())
    vocab = model_json.get("model", {}).get("vocab", {})
    merges = model_json.get("model", {}).get("merges", [])
    
    print("\n" + "="*50)
    print(f"自动化训练总结:")
    print(f"- 最终词表大小: {len(vocab)}")
    print(f"- 成功学习到的合并规则数: {len(merges)}")
    
    if len(merges) > 0:
        print(f"✅ 自动化道路走通了！")
        print(f"Top 10 合并示例: {merges[:10]}")
    else:
        print("❌ 依然为 0。请检查语料中是否存在相邻的相同标签组合。")
    print("="*50)

if __name__ == "__main__":
    train_bpe_automation_final()
