import os
import glob
import gzip
import json
import argparse
import re
from typing import Iterator
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ================= 知识库：踩坑记录与解决方案 (保留项) =================
# 1. 字符拆分坑：使用 <|bwav:123|> 作为原始字符时，BPE 算法会感知其内部的 '|', ':', 'b' 等基础字符。
#    解决方案：将每个标签映射为单个 Unicode 字符（私有区），从物理上消除拆分的可能。
# 2. ByteLevel 坑：pre_tokenizers.ByteLevel 会将单字符 Unicode 拆解为 UTF-8 字节流（îĨķ 等）。
#    解决方案：弃用 ByteLevel，改用 WhitespaceSplit 并配合在语料中手动插入空格，确保原子独立性。
# 3. limit_alphabet 参数坑：limit_alphabet 是 int 而非 bool。它决定了初始 Alphabet 的容量。
#    解决方案：精确设置为 (信号范围 + 特殊Token数)，防止语料中的杂质字符混入基础原子表。
# 4. 翻译需求：模型内部用 Unicode 效率高，但生成的 json 无法直观分析。
#    解决方案：增加后处理环节，将 vocab 和 merges 中的 Unicode 映射回 <|bwav:xxx|>。
# =============================================================

def setup_args():
    parser = argparse.ArgumentParser(description="Nanopore Signal BPE Tokenizer Trainer")
    parser.add_argument("--input_dir", type=str, required=True, help="输入 jsonl.gz 语料目录")
    parser.add_argument("--output_file", type=str, default="bwav_bpe_tokenizer.json", help="输出文件名")
    parser.add_argument("--vocab_size", type=int, default=1024, help="最终词表大小 (包含合并后的复合 Token)")
    parser.add_argument("--unicode_start", type=int, default=0xE000, help="Unicode 私有区起始地址")
    return parser.parse_args()

class SignalTokenizerTrainer:
    def __init__(self, args):
        self.args = args
        self.unicode_start = args.unicode_start
        self.signal_range = 625
        # 预编译正则，提升百万行处理速度
        self.pattern = re.compile(r"<\|bwav:(\d+)\|>")

    def _id_to_unicode(self, token_id: int) -> str:
        """映射物理 ID 到 Unicode 私有区"""
        return chr(self.unicode_start + token_id)

    def _unicode_to_tag(self, char: str) -> str:
        """将单个 Unicode 字符还原为标签"""
        if len(char) != 1: return char
        code = ord(char)
        if self.unicode_start <= code < self.unicode_start + self.signal_range:
            return f"<|bwav:{code - self.unicode_start}|>"
        return char

    def _raw_to_unicode_stream(self, text: str) -> str:
        """
        核心预处理：将长标签转为单 Unicode 字符，并在字符间插入空格。
        插入空格是为了配合 WhitespaceSplit，确保 BPE 在训练时将每个信号视为独立实体。
        """
        def replace_func(match):
            idx = int(match.group(1))
            return self._id_to_unicode(idx)

        # 1. 替换标签为 Unicode
        converted = self.pattern.sub(replace_func, text)
        # 2. 移除可能存在的旧空格并重新以空格分隔，强制原子化单元
        return " ".join(list(converted.replace(" ", "")))

    def get_training_corpus(self) -> Iterator[str]:
        """迭代读取压缩语料，节省内存"""
        search_path = os.path.join(self.args.input_dir, "**/*.jsonl.gz")
        files = glob.glob(search_path, recursive=True)
        print(f"[*] 启动预训练语料迭代器，找到 {len(files)} 个压缩包...")

        for f in files:
            with gzip.open(f, 'rt', encoding='utf-8') as g:
                for line in g:
                    try:
                        data = json.loads(line)
                        raw_text = data.get('text', '')
                        yield self._raw_to_unicode_stream(raw_text)
                    except Exception:
                        continue

    def _translate_to_human_readable(self, source_json_path, target_json_path):
        """核心翻译逻辑：将生成的 JSON 中的 Unicode 映射回原始标签"""
        print(f"[*] 正在将词表从 Unicode 翻译回人类可读标签...")
        with open(source_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 翻译词表 (Vocab)
        new_vocab = {}
        for token, token_id in data['model']['vocab'].items():
            # 这里需要处理复合 Token，如 "\uE001\uE002"
            readable_token = "".join([self._unicode_to_tag(c) for c in token])
            new_vocab[readable_token] = token_id
        data['model']['vocab'] = new_vocab

        # 2. 翻译合并记录 (Merges)
        new_merges = []
        for merge in data['model']['merges']:
            parts = merge.split(" ")
            readable_merge = " ".join(["".join([self._unicode_to_tag(c) for c in p]) for p in parts])
            new_merges.append(readable_merge)
        data['model']['merges'] = new_merges

        # 3. 写回文件，不使用 ASCII 转义，保证直接可见
        with open(target_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run_train(self):
        # 1. 实例化 BPE 模型
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

        # 2. 配置 Pre-tokenizer (使用 WhitespaceSplit 避开字节拆分坑)
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        # 3. 构造基础字母表
        base_alphabet = [self._id_to_unicode(i) for i in range(self.signal_range)]
        special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|ph_0|>"]

        # 精确计算 limit_alphabet
        expected_alphabet_size = len(base_alphabet) + len(special_tokens)

        # 4. 配置训练器
        trainer = trainers.BpeTrainer(
            vocab_size=self.args.vocab_size,
            show_progress=True,
            min_frequency=2,
            special_tokens=special_tokens,
            initial_alphabet=base_alphabet,
            limit_alphabet=expected_alphabet_size+100,
            max_token_length=5
        )

        # 5. 原子性安全检查
        print("[*] 正在进行原子性安全校验...")
        test_raw = "<|bwav:405|><|bwav:407|><|bwav:405|>"
        test_processed = self._raw_to_unicode_stream(test_raw)
        test_res = tokenizer.pre_tokenizer.pre_tokenize_str(test_processed)
        
        print(f"\n[调试] 现在的分词预览 (Unicode 字符流):")
        print(f"{test_res}")

        if len(test_res) != 3 or any(len(t[0]) != 1 for t in test_res):
            print("CRITICAL ERROR: 原子化切分失败！请检查 Pre-tokenizer 或 Unicode 映射逻辑。")
            return

        print("✅ 原子性校验通过。进入预训练阶段...")

        # 6. 开始训练
        tokenizer.train_from_iterator(self.get_training_corpus(), trainer=trainer)

        # 7. 配置解码器
        tokenizer.decoder = decoders.BPEDecoder()

        # 8. 保存与后处理翻译
        # 先存一个临时文件，翻译后再重命名/替换
        temp_path = self.args.output_file + ".tmp"
        tokenizer.save(temp_path)
        self._translate_to_human_readable(temp_path, self.args.output_file)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        print(f"\n[+] 训练完成。翻译后的可读模型已保存至: {self.args.output_file}")

        # 9. 打印合并结果预览
        self._inspect_merges(self.args.output_file)

    def _inspect_merges(self, json_path):
        """读取翻译后的 JSON 直接打印高频合并记录"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        merges = data.get("model", {}).get("merges", [])
        print("\n" + "="*60)
        print(" 高频信号组合分析 (Top 20 Merges):")
        for i, m in enumerate(merges[:20]):
            print(f"  Rank {i+1:02d}: {m}")
        print("="*60)

if __name__ == "__main__":
    args = setup_args()
    trainer = SignalTokenizerTrainer(args)
    trainer.run_train()
