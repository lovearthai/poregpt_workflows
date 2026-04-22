# -*- coding: utf-8 -*-
"""
对 jsonl.gz 文件中的 signal 字段进行 token 化
注意：此版本假设 signal 已经是预处理过的数据，直接读取并进行 VQE token 化
"""
import os
import gzip
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import torch
from poregpt.tokenizers.vqe_tokenizer import VQETokenizer

def process_jsonl_gz_with_tokenization(
    input_file: str,
    output_file: str,
    tokenizer,
    layer: int = 0
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    processed_lines = 0
    failed_lines = 0

    print("正在统计输入文件行数...")
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        for _ in f_in:
            total_lines += 1
    print(f"总共需要处理 {total_lines} 行数据")

    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_file, 'wt', encoding='utf-8') as f_out:

        pbar = tqdm(total=total_lines, desc="信号 Token 化中", unit="line")

        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                signal_raw = data.get("signal", [])

                if not signal_raw or not isinstance(signal_raw, list):
                    failed_lines += 1
                    pbar.update(1)
                    continue

                # --- 逻辑精简：不再进行预处理，直接转为 Tensor ---
                # 形状转换为 [Batch=1, Channel=1, Length] 以符合 CNN 输入要求
                signal_tensor = torch.tensor(signal_raw, dtype=torch.float32).view(1, 1, -1).to(tokenizer.device)

                # 使用 tokenizer 进行 token 化
                with torch.no_grad():
                    raw_model = tokenizer.model.module if hasattr(tokenizer.model, 'module') else tokenizer.model

                    # 1. 模型推理
                    # recon: [1, 1, T], level_indices: [1, N, K]
                    _, level_indices, _ = raw_model(signal_tensor)

                    # 2. 处理 tokens 字段 (根据指定的 layer 计算综合 ID)
                    tokens_tensor = raw_model.tokenize_indices(level_indices, layer=layer)
                    data["tokens"] = [int(x) for x in tokens_tensor[0].cpu().numpy()]
                    data["token_count"] = len(data["tokens"])

                    # 3. 处理 tokens_layered 字段 (保存多层原始索引)
                    indices_np = level_indices[0].cpu().numpy() # [N, K]
                    data["tokens_layered"] = [
                        [int(layer_val) for layer_val in step]
                        for step in indices_np
                    ]
                
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_lines += 1

            except Exception as e:
                print(f"错误: 处理第{line_num}行时发生异常: {e}")
                failed_lines += 1

            pbar.update(1)
        pbar.close()

    print(f"\n处理完成！成功: {processed_lines}, 失败: {failed_lines}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='对预处理过的 jsonl.gz 信号进行 VQE Token 化'
    )

    parser.add_argument('-i', '--input-file', type=str, required=True, help='输入的 jsonl.gz 文件路径')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='输出的 jsonl.gz 文件路径')
    parser.add_argument('--model-ckpt', type=str, required=True, help='VQ tokenizer 模型检查点路径')
    parser.add_argument('--tokenize-layer', type=int, required=True, help='使用的 Token 层级')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备 (cuda/cpu)')

    args = parser.parse_args()

    # 验证
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")
    if not Path(args.model_ckpt).exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model_ckpt}")

    # 初始化
    print(f"正在加载模型并部署至 {args.device}...")
    tokenizer = VQETokenizer(model_ckpt=args.model_ckpt, device=args.device)

    print(f"开始 Token 化流程:")
    print(f"  输入: {args.input_file}")
    print(f"  输出: {args.output_file}")
    print(f"  层级: {args.tokenize_layer}")
    print("-" * 60)

    try:
        process_jsonl_gz_with_tokenization(
            input_file=args.input_file,
            output_file=args.output_file,
            tokenizer=tokenizer,
            layer=args.tokenize_layer
        )
        print("\n✅ Token 化处理成功完成！")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        raise

if __name__ == "__main__":
    main()
