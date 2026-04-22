import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/zzbnew/rnamodel/model/signalDNAmodel/HF_150m_DNA595G_RSQ542_C625_CNN12_V340S147000/base"

# 1. 加载，必须设置 trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

# 2. 构造模型“唯一识别”的字符串格式
# 假设我们要测试连续的信号 token
test_signal = "<|bwav:243|><|bwav:143|><|bwav:512|><|bwav:10|>"

# 3. 验证 Tokenizer 是否能正确切分
input_ids = tokenizer.encode(test_signal, return_tensors="pt").to(model.device)
print(f"输入字符串: {test_signal}")
print(f"Tokenizer 转换后的 ID: {input_ids}")
# 如果输出的 ID 长度与 <|bwav:xxx|> 的个数一致，说明 Tokenizer 配置正确

# 4. 提取 Embedding
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    # 获取最后一层的隐藏状态
    last_hidden_state = outputs.hidden_states[-1] 
    print(f"\n--- Embedding 验证 ---")
    print(f"Embedding 维度: {last_hidden_state.shape}") # 应该是 [1, 4, hidden_size]
    
# 5. 验证推理（预测下一个信号 Token）
with torch.no_grad():
    gen_outputs = model.generate(input_ids, max_new_tokens=5)
    gen_text = tokenizer.decode(gen_outputs[0])
    print(f"\n--- 续写验证 ---")
    print(f"模型预测的后续信号: {gen_text}")
