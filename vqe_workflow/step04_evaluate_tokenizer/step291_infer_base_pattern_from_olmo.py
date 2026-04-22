import matplotlib
matplotlib.use('Agg') 
import json
import gzip
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.lines import Line2D

# 核心工具函数导入
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
    if tid < 0: return np.zeros(4)
    RSQ_LEVELS = [5, 5, 5, 5]
    v = get_rsq_vector_from_integer(int(tid), RSQ_LEVELS, num_quantizers=1, use_fast=True)
    return np.array(v).flatten()

def predict_next_token(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_basename = os.path.basename(args.input)
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # 1. 定位数据
    target_record = None
    with gzip.open(args.input, 'rt') as f:
        for i, line in enumerate(f):
            if i == args.line_id:
                target_record = json.loads(line)
                break
    if not target_record: return

    # 2. 准备数据 (Prev 10 + GT + Future 5)
    all_raw_ids = [t[0] for t in target_record['tokens_layered']]
    split_idx = args.prompt_len
    
    prev_10 = all_raw_ids[max(0, split_idx-10):split_idx]
    gt_raw_id = all_raw_ids[split_idx]
    future_5 = all_raw_ids[split_idx+1 : split_idx+6]
    
    prompt_raw_ids = all_raw_ids[:split_idx]
    gt_vector = get_vector(gt_raw_id)

    # 3. 推理
    input_ids = torch.tensor([[tid + args.vocab_code_shift for tid in prompt_raw_ids]]).long()
    gt_vocab_id = gt_raw_id + args.vocab_code_shift
    if torch.cuda.is_available(): input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

    gt_rank = (torch.argsort(logits, descending=True) == gt_vocab_id).nonzero(as_tuple=True)[0].item() + 1
    gt_prob = probs[gt_vocab_id].item()

    # 4. Top K 准备
    top_k_val = max(args.top_k, 20)
    top_probs, top_indices = torch.topk(probs, k=top_k_val)
    top_probs, top_indices = top_probs.cpu().numpy(), top_indices.cpu().numpy()

    plot_probs, plot_dists, plot_raw_ids = [], [], []
    for rank_idx in range(top_k_val):
        v_id = top_indices[rank_idx]
        r_id = v_id - args.vocab_code_shift
        if v_id >= args.vocab_code_shift:
            plot_probs.append(top_probs[rank_idx])
            plot_dists.append(np.linalg.norm(get_vector(r_id) - gt_vector))
            plot_raw_ids.append(r_id)

    # 5. 绘图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output_dir, f"infer_fix_line{args.line_id}_{timestamp}.png")

    fig = plt.figure(figsize=(18, 6))
    plt.scatter(plot_probs, plot_dists, color='gray', alpha=0.15, s=25)

    # 标注 T1-T9
    for i in range(min(9, len(plot_probs))):
        current_rank = i + 1
        v_id_at_rank = top_indices[i]
        if v_id_at_rank == gt_vocab_id: continue

        color = 'blue' if current_rank == 1 else ('darkgreen' if current_rank < gt_rank else 'gray')
        plt.annotate(f"T{current_rank}:{plot_raw_ids[i]}", xy=(plot_probs[i], plot_dists[i]),
                     xytext=(random.uniform(8, 20), random.uniform(-20, 20)), textcoords='offset points', 
                     fontsize=9, color=color, arrowprops=dict(arrowstyle='->', color=color, alpha=0.3))
        plt.scatter(plot_probs[i], plot_dists[i], color=color, s=70 if current_rank==1 else 40)

    # 标注 GT
    plt.scatter([gt_prob], [0], color='red', s=180, zorder=5, edgecolors='black')
    plt.annotate(f"GT ID:{gt_raw_id}\nRank:{gt_rank}", xy=(gt_prob, 0), xytext=(0, -50),
                 textcoords='offset points', ha='center', color='red', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='red'))

    plt.title(f"Prediction Analysis | Line: {args.line_id} | PromptLen: {args.prompt_len} | Rank: {gt_rank}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(handles=[Line2D([0], [0], color='blue', marker='o', linestyle='None', label='Top 1'),
                        Line2D([0], [0], color='darkgreen', marker='o', linestyle='None', label='Better than GT'),
                        Line2D([0], [0], color='red', marker='o', linestyle='None', label='Ground Truth')], loc='upper right')

    # 修复重叠的序列展示区
    seq_str = "Sequence: " + " | ".join([str(x) for x in prev_10])
    seq_str += f" | >>> {gt_raw_id} <<< | " # 红色 Next Token 的替代方案：用 >>> <<< 包裹
    seq_str += " | ".join([str(x) for x in future_5])

    info_line = f"File: {input_basename} | PromptLen: {args.prompt_len} | Line: {args.line_id}"
    
    plt.gcf().text(0.01, 0.06, info_line, fontsize=10, family='monospace', fontweight='bold')
    plt.gcf().text(0.01, 0.02, seq_str, fontsize=11, family='monospace', color='black',
                   bbox=dict(facecolor='wheat', alpha=0.3, edgecolor='gray'))

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ 图片保存完成，序列重叠已修复。路径: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--line_id', type=int, default=0)
    parser.add_argument('--prompt_len', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=150)
    parser.add_argument('--vocab_code_shift', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default="./infer_results")
    args = parser.parse_args()
    predict_next_token(args)
