# -*- coding: utf-8 -*-
import pandas as pd
import argparse

def token_id_to_fsq_string(
    token_id: int,
    codebook_size: int,
    codebook_nqtz: int,
    codebook_fsqn: int,
    codebook_fsqd: int,
    debug: bool = False
) -> str:
    if debug:
        print(f"\n[Debug] Processing Token ID: {token_id}")

    total_vocab_size = codebook_size ** codebook_nqtz
    if not (0 <= token_id < total_vocab_size):
        raise ValueError(f"token_id 必须在 [0, {total_vocab_size}) 范围内")

    # --- 步骤1: 分解为各 Residual 层的子 token ID ---
    # 逻辑：从 Layer 0 到 Layer N，依次通过 // 权重获取 sub_id，再通过 % 更新余额
    sub_tokens = []
    remaining_token = token_id
    for i in range(codebook_nqtz):
        # 权重计算：如果是 2 层，Layer 0 的权重是 codebook_size^1，Layer 1 是 codebook_size^0
        power = codebook_nqtz - 1 - i
        weight = codebook_size ** power
        sub_id = remaining_token // weight
        sub_tokens.append(sub_id)
        if debug:
            print(f"  Layer {i}: sub_id = {remaining_token} // {weight} = {sub_id} remain {remaining_token%weight}")
        remaining_token %= weight # 更新余额

    # --- 步骤2: 逐层转换为 FSQ 档位字符串 ---
    layer_strings = []
    for i, sub_token in enumerate(sub_tokens):
        digits = []
        remaining_sub = sub_token
        if debug:
            print(f"  --- Layer {i} FSQ Detail ---")
        for j in range(codebook_fsqn):
            # 权重计算：FSQ 内部同理，高位在前
            fsq_power = codebook_fsqn - 1 - j
            fsq_weight = codebook_fsqd ** fsq_power
            digit = remaining_sub // fsq_weight
            digits.append(digit)
            if debug:
                print(f"    FSQ Digit {j}: {digit} (from {remaining_sub} // {fsq_weight})")
            remaining_sub %= fsq_weight # 更新余额

        layer_str = ''.join(str(d) for d in digits)
        layer_strings.append(layer_str)

    final_res = '-'.join(layer_strings)
    if debug:
        print(f"  Final Result: {final_res}")
    return final_res

def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Convert Token IDs to FSQ codes.")
    parser.add_argument("--input-csv", type=str, default="token_frequencies_sorted_by_count_desc.csv",
                        help="输入的 CSV 文件路径 (默认: token_frequencies_sorted_by_count_desc.csv)")
    parser.add_argument("--output-csv", type=str, default="token_frequencies_with_fsqcode.csv",
                        help="输出的 CSV 文件路径 (默认: token_frequencies_with_fsqcode.csv)")
    parser.add_argument("--codebook-fsqd", type=int, default=5,
                        help="FSQ 的档位数 (D) (默认: 5)")
    parser.add_argument("--codebook-fsqn", type=int, default=4,
                        help="每个量化层的维数 (N) (默认: 4)")
    parser.add_argument("--codebook-nqtz", type=int, default=2,
                        help="残差层数 (R) (默认: 2)")
    parser.add_argument("--debug", action="store_true", help="显示前 10 个 token 的详细计算过程")
    args = parser.parse_args()

    # --- 从参数计算派生常量 ---
    codebook_size = args.codebook_fsqd ** args.codebook_fsqn

    print(f"Using parameters: FSQD={args.codebook_fsqd}, FSQN={args.codebook_fsqn}, NQTZ={args.codebook_nqtz}")
    print(f"Calculated codebook_size = FSQD^FSQN = {args.codebook_fsqd}^{args.codebook_fsqn} = {codebook_size}")

    INPUT_CSV = args.input_csv
    OUTPUT_CSV = args.output_csv

    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"找不到输入文件: {INPUT_CSV}")
        return

    print(f"正在处理 {len(df)} 行数据...")

    fsq_codes = []
    for idx, row in df.iterrows():
        token_id = int(row['token_id'])
        # 仅在开启 debug 且是前 10 条时触发打印
        do_debug = args.debug and idx < 10

        fsq_str = token_id_to_fsq_string(
            token_id=token_id,
            codebook_size=codebook_size,
            codebook_nqtz=args.codebook_nqtz,
            codebook_fsqn=args.codebook_fsqn,
            codebook_fsqd=args.codebook_fsqd,
            debug=do_debug
        )
        fsq_codes.append(fsq_str)

    df['fsqcode'] = fsq_codes
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\n✅ 处理完成，结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
