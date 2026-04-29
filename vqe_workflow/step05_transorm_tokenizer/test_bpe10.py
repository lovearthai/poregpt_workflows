from tokenizers import pre_tokenizers

# --- 核心改动开始 ---
# 我们在 Tokenizer 内部构建一个处理流水线
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    # 1. 运行时替换：在内部将连在一起的标签分开，但不改变原始文件
    pre_tokenizers.Replace(pattern="><", content="> <"),
    # 2. 物理切割：现在有空格了，WhitespaceSplit 绝对能把它们切成 3 个
    pre_tokenizers.WhitespaceSplit()
])
# --- 核心改动结束 ---

# 再次运行校验
test_text = "<|bwav:405|><|bwav:407|><|bwav:405|>"
test_res = tokenizer.pre_tokenizer.pre_tokenize_str(test_text)
print(f"\n[新方案调试] 结果: {test_res}")
print(f"数量: {len(test_res)}")
