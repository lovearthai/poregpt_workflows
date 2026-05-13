from tokenizers import pre_tokenizers

test_text = "<|bwav:405|><|bwav:407|><|bwav:405|>"

# --- 方案 A: 极简非贪婪匹配 (最推荐，容错率最高) ---
# 逻辑：匹配以 < 开始，以 > 结束，中间不含 > 的最小段落
regex_a = r"<[^>]+>"

# --- 方案 B: 严格转义匹配 ---
# 逻辑：精准匹配标签格式，对 | 进行 Rust 风格转义
regex_b = r"<\|bwav:\d+\|>"

# --- 方案 C: 捕获组匹配 ---
# 逻辑：显式定义边界
regex_c = r"(<\|bwav:\d+\|>)"

def test_regex(pattern, name):
    splitter = pre_tokenizers.Split(pattern=pattern, behavior="isolated")
    res = splitter.pre_tokenize_str(test_text)
    print(f"方案 {name} [{pattern}]:")
    print(f"  结果: {res}")
    print(f"  数量: {len(res)} {'✅ 成功' if len(res)==3 else '❌ 失败'}\n")

test_regex(regex_a, "A (非贪婪)")
test_regex(regex_b, "B (严格转义)")
test_regex(regex_c, "C (捕获组)")
