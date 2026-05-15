import json
import gzip
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设你的工具函数可以这样导入
# 如果路径不对，请手动将 get_rsq_vector_from_integer 的实现粘贴到此处
try:
    from poregpt.utils import get_rsq_vector_from_integer
except ImportError:
    print("Warning: Could not import poregpt.utils. Please ensure the path is correct.")

def load_model_and_tokenizer(model_path):
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

def get_vector(tid):
    """将 token_id 转换为 RSQ 物理空间向量"""
    RSQ_LEVELS = [5, 5, 5, 5]
    # 调用你的工具函数
    v = get_rsq_vector_from_integer(int(tid), RSQ_LEVELS, num_quantizers=1, use_fast=True)
    return np.array(v).flatten()

def predict_next_token(args):
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # 1. 定位到指定行
    target_record = None
    print(f"正在读取文件并定位到第 {args.line_id} 行...")
    with gzip.open(args.input, 'rt') as f:
        for i, line in enumerate(f):
            if i == args.line_id:
                target_record = json.loads(line)
                break
    
    if not target_record:
        print(f"错误: 未能找到第 {args.line_id} 行。")
        return

    # 2. 准备输入数据 (Prompt)
    # tokens_layered 结构通常是 [[tid, ...], [tid, ...]]
    all_layered_ids = [t[0] for t in target_record['tokens_layered']]
    
    if len(all_layered_ids) <= args.prompt_len:
        print(f"错误: 样本长度 ({len(all_layered_ids)}) 不足以提供 {args.prompt_len} 的 prompt。")
        return

    # 截取 Prompt 和对应的真实 Next Token
    prompt_ids = all_layered_ids[:args.prompt_len]
    ground_truth_id = all_layered_ids[args.prompt_len]
    
    # 转换为物理向量用于后续计算距离
    gt_vector = get_vector(ground_truth_id)

    # 3. 模型推理
    # 获取词表偏移量
    base_token_str = "<|bwav:0|>"
    base_id = tokenizer.convert_tokens_to_ids(base_token_str)
    print(f"base_id:{base_id}")
    #if base_id is None or (hasattr(tokenizer, 'unk_token_id') and base_id == tokenizer.unk_token_id):
    print("prompot ids:",prompt_ids)
    input_ids = torch.tensor([prompt_ids]).long()

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    print(f"🚀 正在推理... Prompt 长度: {args.prompt_len}")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        # 取最后一个位置的 Logits: [1, Seq, Vocab] -> [Vocab]
        next_token_logits = outputs.logits[0, -1, :]
        
        # 处理可能的词表偏移（如果 logits 对应的索引是绝对索引，需要减去 base_id）
        # 这里假设输出的概率分布对应于词表中的 <bwav:0> 开始的部分
        # 如果模型输出是全词表，我们需要截取对应的部分
        probs = F.softmax(next_token_logits, dim=-1)
        
    # 4. 选出 Top 100
    top_k = 100
    top_probs, top_indices = torch.topk(probs, k=top_k)
    
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    # 5. 计算距离并打印结果
    print("\n" + "="*80)
    print(f"{'Rank':<6} | {'TokenID':<10} | {'Prob':<10} | {'Euclidean Dist (to GT)':<20}")
    print("-" * 80)

    results = []
    for rank in range(top_k):
        abs_id = top_indices[rank]
        # 转换回相对 ID (即你的 RSQ 整数)
        relative_id = abs_id - base_id if base_id else abs_id
        
        prob = top_probs[rank]
        
        # 计算该预测 Token 的 RSQ 向量
        pred_vector = get_vector(relative_id)
        
        # 计算欧氏距离
        dist = np.linalg.norm(pred_vector - gt_vector)
        
        results.append((rank+1, relative_id, prob, dist))
        print(f"{rank+1:<6} | {relative_id:<10} | {prob:.6f} | {dist:.6f}")

    print("-" * 80)
    print(f"真实 Next Token ID: {ground_truth_id}")
    print(f"真实向量: {gt_vector}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="jsonl.gz 文件路径")
    parser.add_argument('--model_path', type=str, required=True, help="HF 模型路径")
    parser.add_argument('--line_id', type=int, default=0, help="处理第几行数据")
    parser.add_argument('--prompt_len', type=int, default=128, help="用于推理的上下文长度")
    
    args = parser.parse_args()
    predict_next_token(args)
