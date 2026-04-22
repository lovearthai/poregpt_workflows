# -*- coding: utf-8 -*-
import os
import gzip
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
# 确保你的路径下可以找到 vqe_tokenizer
from poregpt.tokenizers.vqe_tokenizer import VQETokenizer

def plot_comparison(original, reconstructed, tokens, save_path, sample_id):
    """ 绘制原始波形与重建波形的对比图 """
    plt.figure(figsize=(15, 8))
    
    # 子图1：波形对比
    plt.subplot(2, 1, 1)
    plt.plot(original, label='Original (Signal)', color='gray', alpha=0.5, linewidth=1)
    plt.plot(reconstructed, label='Reconstructed (Recon)', color='blue', linestyle='--', linewidth=1)
    plt.title(f"Reconstruction Comparison | ID: {sample_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：Token 序列可视化
    plt.subplot(2, 1, 2)
    plt.step(range(len(tokens)), tokens, where='mid', color='red', linewidth=0.8)
    plt.title(f"Tokenized Sequence (Total: {len(tokens)} tokens)")
    plt.xlabel("Token Position")
    plt.ylabel("Token ID")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_jsonl_recon(input_jsonl, tokenizer, batch_size, output_dir, max_plots=50, target_len=1200):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_model = tokenizer.model.module if hasattr(tokenizer.model, 'module') else tokenizer.model
    plot_count = 0
    buffer_signals = []
    buffer_ids = []

    def process_buffer(sigs, ids, current_plot_count):
        # 对齐信号长度
        processed_sigs = []
        for s in sigs:
            s_np = np.array(s, dtype=np.float32)
            if len(s_np) > target_len:
                processed_sigs.append(s_np[:target_len])
            else:
                processed_sigs.append(np.pad(s_np, (0, target_len - len(s_np))))
        
        x = torch.from_numpy(np.array(processed_sigs)).unsqueeze(1).to(tokenizer.device)
        
        with torch.no_grad():
            recon, level_indices, _ = raw_model(x)
            tokens_tensor = raw_model.tokenize_indices(level_indices, layer=0)
            
        recon_np = recon.squeeze(1).cpu().numpy()
        tokens_np = tokens_tensor.cpu().numpy()

        for j in range(len(sigs)):
            if current_plot_count + j >= max_plots:
                return current_plot_count + j
            
            save_path = output_dir / f"recon_{ids[j].replace('/', '_')}.png"
            plot_comparison(processed_sigs[j], recon_np[j], tokens_np[j], save_path, ids[j])
            
        return current_plot_count + len(sigs)

    # 流式读取 jsonl.gz
    with gzip.open(input_jsonl, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing JSONL"):
            if plot_count >= max_plots:
                break
                
            try:
                data = json.loads(line)
                signal = data.get("signal", [])
                read_id = data.get("id", f"idx_{plot_count}")
                
                buffer_signals.append(signal)
                buffer_ids.append(read_id)
                
                if len(buffer_signals) == batch_size:
                    plot_count = process_buffer(buffer_signals, buffer_ids, plot_count)
                    buffer_signals, buffer_ids = [], []
            except Exception as e:
                print(f"Error skipping line: {e}")
                continue

        # 处理剩余的 buffer
        if buffer_signals and plot_count < max_plots:
            process_buffer(buffer_signals, buffer_ids, plot_count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-jsonl', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, default='eval_recon_jsonl')
    parser.add_argument('--model-ckpt', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-plots', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    tokenizer = VQETokenizer(model_ckpt=args.model_ckpt, device=args.device)
    evaluate_jsonl_recon(args.input_jsonl, tokenizer, args.batch_size, args.output_dir, args.max_plots)

if __name__ == "__main__":
    main()
