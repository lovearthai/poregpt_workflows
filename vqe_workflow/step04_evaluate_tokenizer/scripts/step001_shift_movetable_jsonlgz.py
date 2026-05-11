import argparse
import gzip
import json
import os
import sys
from tqdm import tqdm

def open_text(path, mode):
    """
    自动识别压缩格式的文本读取函数
    """
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")

def main():
    # 严格保持参数接口一致
    parser = argparse.ArgumentParser(description="Nanopore 数据平移对齐与信号裁剪工具")
    parser.add_argument("--input", required=True, help="输入 jsonl.gz 路径")
    parser.add_argument("--output", required=True, help="输出 jsonl.gz 路径")
    parser.add_argument("--shift", type=int, default=-4, help="位移量 (负数左移，正数右移，0不移动)")
    args = parser.parse_args()

    # 1. 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    shift = args.shift
    count = 0

    print(f"[*] 开始处理: {args.input}")
    print(f"[*] 设定 shift = {shift}")

    with open_text(args.input, "r") as fin, open_text(args.output, "w") as fout:
        for line_idx, line in enumerate(tqdm(fin, desc="Processing", unit=" reads"), 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                read_id = data.get("read_id", "unknown")
                pattern = data.get("pattern", "")
                spans = data.get("base_sample_spans_rel", [])
                signal = data.get("signal", [])

                # --- 核心处理逻辑：合并位移与信号裁剪 ---

                if shift < 0:
                    # 【左移逻辑】
                    s = abs(shift)
                    new_pattern = pattern[s:]
                    new_spans = spans[:-s] if s > 0 else spans
                    
                    if not new_spans: continue
                    
                    # C. 信号裁剪：根据新的最后一个 Span 的结束坐标裁剪信号
                    last_coord = new_spans[-1][1]
                    new_signal = signal[:last_coord]

                elif shift > 0:
                    # 【右移逻辑】
                    s = shift
                    new_pattern = pattern[:-s]
                    new_spans = spans[s:]
                    
                    if not new_spans: continue

                    # C. 信号裁剪与坐标重置
                    # 右移删除了开头的 Span，需将所有 Span 坐标平移回 0，并切除信号头部
                    offset = new_spans[0][0]
                    new_spans = [[p[0] - offset, p[1] - offset] for p in new_spans]
                    
                    # 获取新信号并根据最后一个 span 裁剪尾部
                    last_coord = new_spans[-1][1]
                    new_signal = signal[offset : offset + last_coord]
                
                else:
                    # 【不移动】
                    new_pattern = pattern
                    new_spans = spans
                    new_signal = signal

                # 基础合法性检查
                if not new_pattern or not new_spans:
                    sys.exit(f"\n[ERROR] 行 {line_idx}: 位移量 {shift} 导致数据为空。Read ID: {read_id}")

                # --- 严格验证逻辑 (一旦失败立刻终止) ---
                
                last_coord = new_spans[-1][1]

                # 验证 1: pattern 字符数必须等于 spans 列表项数
                if len(new_pattern) != len(new_spans):
                    error_msg = (
                        f"\n[CRITICAL ERROR] 验证失败(1): 碱基数与Span数不一致！\n"
                        f"Read ID: {read_id} (行: {line_idx})\n"
                        f"Pattern长度: {len(new_pattern)}, Spans项数: {len(new_spans)}"
                    )
                    sys.exit(error_msg)

                # 验证 2: 最后一个 span 的结束坐标必须等于裁剪后信号的实际长度
                if last_coord != len(new_signal):
                    error_msg = (
                        f"\n[CRITICAL ERROR] 验证失败(2): 信号裁剪长度与Span坐标不匹配！\n"
                        f"Read ID: {read_id} (行: {line_idx})\n"
                        f"最后坐标: {last_coord}, 裁剪后信号长度: {len(new_signal)}"
                    )
                    sys.exit(error_msg)

                # 更新数据字典
                data["pattern"] = new_pattern
                data["base_sample_spans_rel"] = new_spans
                data["signal"] = new_signal
                data["shift"] = args.shift

                # 写入结果
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1

            except json.JSONDecodeError:
                sys.exit(f"\n[ERROR] 第 {line_idx} 行 JSON 解析失败。")
            except Exception as e:
                sys.exit(f"\n[ERROR] 处理第 {line_idx} 行时发生未知错误: {str(e)}")

    print(f"\n[+] 处理完成！有效记录数: {count}")
    print(f"[+] 结果保存路径: {args.output}")

if __name__ == "__main__":
    main()
