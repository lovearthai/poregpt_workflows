# -*- coding: utf-8 -*-
"""
对jsonl.gz文件进行过滤
使用nanopore_process_signal处理数据后，检查数据是否在指定范围内
如果处理后的数据中有超出-clip_value到clip_value范围的数据，整条数据抛弃掉
并将处理后（保留三位小数）的信号覆盖原始 signal 字段
"""
import os
import gzip
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
from poregpt.utils import nanopore_process_signal

def filter_jsonl_gz(
    input_file: str,
    output_file: str,
    clip_value: float = 3.0,
    nanopore_signal_process_strategy: str = "mongo"
):
    """
    过滤jsonl.gz文件：处理signal字段并检查数值范围，覆盖原信号并保留3位小数
    """
    # 创建输出目录
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 统计处理进度
    total_lines = 0
    processed_lines = 0
    filtered_lines = 0
    failed_lines = 0

    # 首先计算总行数用于进度条显示
    print("正在统计输入文件行数...")
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        for _ in f_in:
            total_lines += 1

    print(f"总共需要处理 {total_lines} 行数据")

    # 读取输入文件并处理每一行
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_file, 'wt', encoding='utf-8') as f_out:

        # 使用tqdm显示处理进度
        pbar = tqdm(total=total_lines, desc="过滤并覆盖信号", unit="line")

        for line_num, line in enumerate(f_in, 1):
            try:
                # 解析JSON行
                line_content = line.strip()
                if not line_content:
                    continue
                data = json.loads(line_content)

                # 获取原始信号数据
                signal_raw = data.get("signal", [])

                if not signal_raw or not isinstance(signal_raw, list):
                    print(f"警告: 第{line_num}行无效的signal字段，跳过")
                    failed_lines += 1
                    pbar.update(1)
                    continue

                # 将信号转换为numpy数组
                signal_array = np.array(signal_raw, dtype=np.float32)

                # 调用nanopore_process_signal进行信号处理
                signal_processed = nanopore_process_signal(
                    signal_array,
                    nanopore_signal_process_strategy
                )

                # 确保返回的是数组格式
                if isinstance(signal_processed, list):
                    signal_processed = np.array(signal_processed, dtype=np.float32)

                if isinstance(signal_processed, np.ndarray):
                    # 检查数据范围
                    if np.any((signal_processed < -clip_value) | (signal_processed > clip_value)):
                        filtered_lines += 1
                    else:
                        # --- 核心修改：覆盖信号并保留三位有效数字 ---
                        # np.round 会处理四舍五入，.tolist() 转换回 JSON 可序列化格式
                        # --- 核心修改：覆盖信号并保留三位有效数字 ---
                        # 逻辑：先格式化为字符串确保只有3位，再转回 float 以便 JSON 序列化
                        # 这样可以彻底去除 IEEE 754 浮点数带来的冗余尾数
                        data["signal"] = [round(float(f"{x:.3f}"), 3) for x in signal_processed]

                        # 覆盖原字段
                        # 添加/更新辅助信息
                        data["clip_value"] = clip_value
                        # 写入文件
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        processed_lines += 1
                else:
                    print(f"警告: 第{line_num}行信号处理返回格式错误")
                    failed_lines += 1

            except json.JSONDecodeError as e:
                print(f"错误: 第{line_num}行JSON解析失败: {e}")
                failed_lines += 1
            except Exception as e:
                print(f"错误: 处理第{line_num}行时发生异常: {e}")
                failed_lines += 1

            pbar.update(1)

        pbar.close()

    # 打印处理结果统计
    print(f"\n" + "="*40)
    print(f"处理完成统计:")
    print(f"  总输入行数: {total_lines}")
    print(f"  通过并覆盖: {processed_lines}")
    print(f"  数值超限过滤: {filtered_lines}")
    print(f"  解析失败: {failed_lines}")
    print(f"  精度设定: 保留3位小数")
    print(f"  输出文件: {output_file}")
    print("="*40)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='过滤并覆盖jsonl.gz文件中的signal字段'
    )

    parser.add_argument('-i', '--input-file', type=str, required=True, help='输入的jsonl.gz路径')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='输出的jsonl.gz路径')
    parser.add_argument('--clip-value', type=float, default=3.0, help='过滤阈值(默认±3.0)')
    parser.add_argument('--process-strategy', type=str, default='mongo', help='信号处理策略')

    args = parser.parse_args()

    # 验证输入文件
    input_path = Path(args.input_file)
    if not input_path.exists() or not input_path.is_file():
        print(f"❌ 错误: 输入文件不存在: {input_path}")
        return

    try:
        filter_jsonl_gz(
            input_file=args.input_file,
            output_file=args.output_file,
            clip_value=args.clip_value,
            nanopore_signal_process_strategy=args.process_strategy
        )
        print("\n✅ 处理成功！")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")

if __name__ == "__main__":
    main()
