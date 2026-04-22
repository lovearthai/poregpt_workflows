import pandas as pd
import argparse
import sys
import os

def merge_ratios(count_csv, vector_csv, output_csv):
    print(f"📖 正在加载数据...")
    
    # 读取碱基占比数据 (长表)
    # id, token_id, base, count, ratio
    df_base = pd.read_csv(count_csv)
    
    # 读取向量与坐标数据 (主表)
    # token_id, x, y, layer0_id, layer0_code, dim0...
    df_vec = pd.read_csv(vector_csv, dtype={'layer0_code': str})

    print(f"🔄 正在进行透视变换 (Pivot)...")
    # 将 df_base 从长表转为宽表
    # 每一行是一个 token_id，每一列是 ratio_A, ratio_G 等
    df_ratio_wide = df_base.pivot(index='token_id', columns='base', values='ratio').reset_index()
    
    # 重命名列名，增加 ratio_ 前缀，方便识别
    # 转换后列名可能为: token_id, A, C, G, N, T
    rename_cols = {col: f'ratio_{col}' for col in df_ratio_wide.columns if col != 'token_id'}
    df_ratio_wide = df_ratio_wide.rename(columns=rename_cols)

    # 确保 A, G, C, T, N 五列都存在，如果某个 Token 没有对应的碱基统计，填充为 0
    expected_bases = ['ratio_A', 'ratio_G', 'ratio_C', 'ratio_T', 'ratio_N']
    for col in expected_bases:
        if col not in df_ratio_wide.columns:
            df_ratio_wide[col] = 0.0
    
    # 填充空值为 0 (某些 Token 可能完全没有对应某些碱基)
    df_ratio_wide = df_ratio_wide.fillna(0.0)

    print(f"🔗 正在合并主表 (Total Tokens in Vec: {len(df_vec)})...")
    # 使用左连接，确保保留所有的 Token 位置信息，即使它没有碱基统计
    df_final = pd.merge(df_vec, df_ratio_wide[['token_id'] + expected_bases], on='token_id', how='left')

    # 再次填充合并后可能产生的缺失值（针对那些在 count 表里完全没出现的 token_id）
    df_final[expected_bases] = df_final[expected_bases].fillna(0.0)

    print(f"💾 正在保存至: {output_csv}")
    # 保持 ratio 的高精度输出
    df_final.to_csv(output_csv, index=False, float_format='%.9f')
    print(f"✅ 合并完成！新表列数: {len(df_final.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将碱基占比合并至 Token 向量表')
    parser.add_argument('--count_csv', type=str, required=True, help='step0x14 输出的 CSV')
    parser.add_argument('--vec_csv', type=str, required=True, help='step0x15 输出的 CSV')
    parser.add_argument('-o', '--output', type=str, default='step0x16_token_vector_with_ratio.csv')

    args = parser.parse_args()
    merge_ratios(args.count_csv, args.vec_csv, args.output)
