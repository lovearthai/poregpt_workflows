import json
import gzip
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """将 token_id 转换为 RSQ 物理空间向量"""
    if tid < 0: return np.zeros(4) 
    RSQ_LEVELS = [5, 5, 5, 5]
    v = get_rsq_vector_from_integer(int(tid), RSQ_LEVELS, num_quantizers=1, use_fast=True)
    return np.array(v).flatten()

def predict_next_token(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_basename = os.path.basename(args.input)
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # 1. 定位数据行
    target_record = None
    with gzip.open(args.input, 'rt') as f:
        for i, line in enumerate(f):
            if i == args.line_id:
                target_record = json.loads(line)
                break

    if not target_record:
        print(f"错误: 未能找到第 {args.line_id} 行。")
        return

    # 2. 准备数据
    all_raw_ids = [t[0] for t in target_record['tokens_layered']]
    prompt_raw_ids = all_raw_ids[:args.prompt_len]
    
    print("prompt",prompt_raw_ids)
    gt_raw_id = all_raw_ids[args.prompt_len]
    gt_vector = get_vector(gt_raw_id)

    # 3. 推理
    input_ids = torch.tensor([[tid + args.vocab_code_shift for tid in prompt_raw_ids]]).long()
    gt_vocab_id = gt_raw_id + args.vocab_code_shift

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

    # 找到 GT 的排名 (在全词表中)
    # argsort 是升序，[::-1] 转为降序
    all_sorted_indices = torch.argsort(logits, descending=True)
    gt_rank = (all_sorted_indices == gt_vocab_id).nonzero(as_tuple=True)[0].item() + 1
    
    gt_logit = logits[gt_vocab_id].item()
    gt_prob = probs[gt_vocab_id].item()

    # 4. 获取 Top K 用于绘图
    top_probs, top_indices = torch.topk(probs, k=args.top_k)
    top_logits = logits[top_indices].cpu().numpy()
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    # 5. 过滤并计算列表数据
    plot_logits, plot_probs, plot_dists = [], [], []
    print("\n" + "="*110)
    print(f"{'Rank':<6} | {'RawID':<10} | {'VocabID':<10} | {'Logit':<10} | {'Prob':<10} | {'Dist':<12}")
    print("-" * 110)

    for rank in range(args.top_k):
        v_id = top_indices[rank]
        r_id = v_id - args.vocab_code_shift 
        logit, prob = top_logits[rank], top_probs[rank]
        
        is_wav = (v_id >= args.vocab_code_shift)
        if is_wav:
            dist = np.linalg.norm(get_vector(r_id) - gt_vector)
            plot_logits.append(logit)
            plot_probs.append(prob)
            plot_dists.append(dist)
            dist_str = f"{dist:.4f}"
        else:
            dist_str = "N/A"

        tag = ""
        if v_id == gt_vocab_id: tag += " (GT)"
        if rank == 0: tag += " (Top1/Inf)"
        print(f"{rank+1:<6} | {r_id:<10} | {v_id:<10} | {logit:<10.4f} | {prob:.6f} | {dist_str:<12}{tag}")

    print("-" * 110)
    print(f"📊 SUMMARY: GT RawID: {gt_raw_id} | GT Rank: {gt_rank} | Prob: {gt_prob:.6e}")
    print("="*110)

    # 6. 绘图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output_dir, f"infer_analysis_line{args.line_id}_{timestamp}.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top 1 逻辑
    top1_v_id = top_indices[0]
    top1_is_wav = (top1_v_id >= args.vocab_code_shift)
    
    # 基础点
    ax1.scatter(plot_probs, plot_dists, color='gray', alpha=0.3, edgecolors='none', label='Wave Tokens')
    ax2.scatter(plot_logits, plot_dists, color='gray', alpha=0.3, edgecolors='none', label='Wave Tokens')

    # Top 1 点 (蓝色)
    if top1_is_wav:
        t1_dist = np.linalg.norm(get_vector(top1_v_id - args.vocab_code_shift) - gt_vector)
        ax1.scatter([top_probs[0]], [t1_dist], color='blue', s=100, edgecolors='none', zorder=4, label='Top 1 (Inf)')
        ax2.scatter([top_logits[0]], [t1_dist], color='blue', s=100, edgecolors='none', zorder=4, label='Top 1 (Inf)')

    # GT 点 (红色)
    ax1.scatter([gt_prob], [0], color='red', s=100, edgecolors='none', zorder=5, label='Ground Truth')
    ax2.scatter([gt_logit], [0], color='red', s=100, edgecolors='none', zorder=5, label='Ground Truth')

    # 在图上指明 GT 排名
    # 我们选择在 ax2 (Logits 图) 的红色点上方或固定位置标注
    ax2.annotate(f"GT Rank: {gt_rank}", 
                 xy=(gt_logit, 0), 
                 xytext=(0, 15), 
                 textcoords='offset points',
                 ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color='red',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='red'))

    ax1.set_title('Probability vs Distance')
    ax2.set_title('Logits vs Distance')
    for ax in [ax1, ax2]:
        ax.set_ylabel('Euclidean Distance to GT')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()

    plt.suptitle(f"Prediction Analysis - Line {args.line_id}\nGT Rank: {gt_rank} in all vocab", fontsize=16, fontweight='bold')
    
    info_text = (f"File: {input_basename}\nShift: {args.vocab_code_shift}\n"
                 f"Prompt: {args.prompt_len}\nGT Rank: {gt_rank}")
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace', backgroundcolor='#f0f0f0')

    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.savefig(save_path)
    print(f"\n✅ 图片保存成功，GT 排名为: {gt_rank}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--line_id', type=int, default=0)
    parser.add_argument('--prompt_len', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--vocab_code_shift', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default="./infer_results")
    args = parser.parse_args()
    predict_next_token(args)
