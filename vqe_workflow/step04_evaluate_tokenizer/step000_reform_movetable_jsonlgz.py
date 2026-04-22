import gzip
import json
import os
import argparse

def process_signal_precision(input_path, output_path, precision):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"开始处理: {input_path}")
    print(f"精度设置: 保留 {precision} 位小数")

    # 使用 gzip.open 以文本模式 ('rt' 和 'wt') 打开压缩文件
    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', encoding='utf-8') as f_out:

        count = 0
        for line in f_in:
            # 解析每一行的 JSON 数据
            data = json.loads(line)

            # 处理 signal 字段
            if 'signal' in data and isinstance(data['signal'], list):
                # 使用动态精度保留小数
                # 注意：这里使用 round(x, precision) 进行四舍五入
                data['signal'] = [round(float(x), precision) for x in data['signal']]

            # 将修改后的字典写回新文件，并添加换行符
            f_out.write(json.dumps(data) + '\n')

            count += 1
            if count % 1000 == 0:
                print(f"已处理 {count} 行...", end='\r')

    print(f"\n处理完成！新文件已保存至: {output_path}")

if __name__ == "__main__":
    # --- 参数解析部分 ---
    parser = argparse.ArgumentParser(description='处理jsonl.gz文件信号精度')
    
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='输入的jsonl.gz文件路径')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='输出的jsonl.gz文件路径')
    parser.add_argument('-p', '--precision', type=int, default=1,
                        help='保留的小数位数，默认为1')
    
    args = parser.parse_args()
    
    # 调用函数
    process_signal_precision(args.input_file, args.output_file, args.precision)
