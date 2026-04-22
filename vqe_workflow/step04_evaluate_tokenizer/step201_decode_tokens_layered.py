# -*- coding: utf-8 -*-
import gzip
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# 注意：请确保 vqe_tokenizer.py 在同一目录下或已正确安装
from poregpt.tokenizers import VQETokenizer

def run_reconstruction(args):
    # 1. 初始化 Tokenizer（自动加载模型并处理设备分配）
    tokenizer = VQETokenizer(model_ckpt=args.model_ckpt, device=args.device)
    
    # 提取底层模型实例（处理可能存在的 DDP 包装）
    raw_model = tokenizer.model.module if hasattr(tokenizer.model, 'module') else tokenizer.model
    raw_model.eval()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    print(f"🚀 开始重构: {input_path.name}")
    print(f"🛠  层级模式: {'全部 (Layer 0)' if args.layer == 0 else f'Layer {args.layer}'}")

    # 2. 处理 jsonl.gz
    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Decoding"):
            try:
                data = json.loads(line)
                
                # 提取 tokens_layered: [[L0, L1], [L0, L1], ...]
                layered_list = data.get("tokens_layered", [])
                if not layered_list:
                    continue
                
                # --- 核心转换 ---
                # 将 list 转换为 Tensor 并增加 Batch 维度: [B=1, N=Steps, K=Layers]
                indices_tensor = torch.tensor([layered_list], dtype=torch.long, device=tokenizer.device)
                
                # 3. 调用模型的 decode_indices 进行信号重建
                with torch.no_grad():
                    # recon 形状: [1, 1, T]
                    recon = raw_model.decode_indices(indices_tensor, layer=args.layer)
                
                # 4. 后处理信号
                # 压缩维度得到 [T]
                recon_np = recon.squeeze().cpu().numpy().astype(np.float32)
                
                # 格式化信号：保留3位小数
                recon_rounded = [round(float(val), 3) for val in recon_np]
                
                # --- 动态键名逻辑 ---
                if args.layer == 0:
                    key_name = "recon"
                else:
                    key_name = f"recon_layer{args.layer}"
                
                # 将重构信号存入原字典
                data[key_name] = recon_rounded
                
                # 写入输出文件
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"⚠️ 处理行时出错: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从 tokens_layered 重构信号')
    parser.add_argument('-i', '--input-file', type=str, required=True, help='输入 .jsonl.gz')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='输出 .jsonl.gz')
    parser.add_argument('--model-ckpt', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--layer', type=int, default=0, help='使用的层级 (0=全部, 1=仅第一层...)')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    args = parser.parse_args()
    run_reconstruction(args)
