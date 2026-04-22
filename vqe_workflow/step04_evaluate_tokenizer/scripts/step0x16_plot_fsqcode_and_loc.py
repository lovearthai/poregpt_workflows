import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import random

def plot_fsq_map(csv_file, output_png, num_samples=100):
    print(f"📖 正在读取 CSV 文件: {csv_file}...")
    try:
        # --- 核心修改 1：强制指定 dtype ---
        # 显式指定 'layer0_code' 为字符串类型，
        # 彻底防止 Pandas 将其误判为数字 (如把 0000 变成 0)
        df = pd.read_csv(csv_file, dtype={'layer0_code': str})
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 读取 CSV 时出现错误: {e}")
        sys.exit(1)

    # 检查必要的列是否存在
    required_cols = ['token_id', 'x', 'y', 'layer0_code']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ 错误: CSV 文件缺少必要的列: {required_cols}")
        sys.exit(1)

    total_points = len(df)
    print(f"📊 成功读取 {total_points} 个点。")

    # --- 核心修改 2：确认 Code 格式 ---
    # 在采样前，我们确保 layer0_code 都是字符串且长度一致
    # 这里我们只取前几个样本做个验证打印
    sample_codes = df['layer0_code'].dropna().unique()[:5]
    print(f"🔍 验证 Layer0 Code 格式 (前几个唯一值): {sample_codes.tolist()}")
    if len(sample_codes) > 0 and len(sample_codes[0]) != 4:
        print(f"⚠️ 警告: 检测到 layer0_code 长度不为 4 (如: '{sample_codes[0]}')，请检查数据源。")

    # --- 采样逻辑 ---
    # 如果点数过多，全部标记会导致图片不可读。
    if total_points > num_samples:
        print(f"⚠️ 采样逻辑已启动：正在从全量数据中随机采样 {num_samples} 个点进行标记以保证清晰度...")
        # 采样用于标记的索引
        sample_indices = random.sample(range(total_points), num_samples)
        df_labeled = df.iloc[sample_indices].copy()
    else:
        print(f"⚠️ 数据点数小于采样上限，将标记所有 {total_points} 个点。")
        df_labeled = df.copy()

    # --- 绘图设置 ---
    # 使用中文字体 (防止在某些系统上显示乱码，如果不需要可以删掉这一行)
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure(figsize=(18, 14), dpi=150) # 高分辨率大图
    sns.set_style("whitegrid") # 设置干净的背景

    # 1. 绘制所有的点 (作为背景，淡淡的灰色)
    plt.scatter(df['x'], df['y'], c='lightgrey', s=3, alpha=0.4, label='All Tokens')

    # 2. 绘制被采样的点 (稍微大一点，蓝色)
    plt.scatter(df_labeled['x'], df_labeled['y'], c='royalblue', s=25, alpha=0.8, edgecolors='white', linewidth=0.5, label='Labeled Samples')

    # 3. --- 核心：添加文字标记 ---
    print("✍️ 正在添加文字标记 (Token ID + Layer0 Code)...")
    
    # 使用整数索引遍历采样的数据集
    for i in range(len(df_labeled)):
        row = df_labeled.iloc[i]
        
        # 处理可能存在的空值
        token_id_val = int(row['token_id']) if pd.notna(row['token_id']) else "N/A"
        
        # --- 核心修改 3：强制字符串格式化 ---
        # 我们使用 f-string 强制 layer0_code 为字符串。
        # 即使之前的 dtype 设置失效（极低概率），这里的字符串强转也能保证输出 4 个字符（如 0000）。
        layer0_code_raw = str(row['layer0_code'])
        # 再次确认长度 (用于防御)
        layer0_code_str = layer0_code_raw.zfill(4) if len(layer0_code_raw) < 4 else layer0_code_raw

        # 构造标签文本：T表示TokenID, C表示Code
        label_text = f"T{token_id_val}\nC{layer0_code_str}"
        
        # 添加文本
        plt.text(
            row['x'],             # X轴位置
            row['y'],             # Y轴位置
            label_text,          # 文本内容
            fontsize=8,          # 字号 (需要比较小)
            ha='left',           # 水平对齐：左对齐
            va='bottom',         # 垂直对齐：底部对齐
            alpha=0.95,          # 透明度
            color='black',       # 颜色
            fontweight='bold',   # 加粗
            # bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1') # 可选：加个白色底
        )

    # --- 完善图表 ---
    plt.title(f"ResidualFSQ Token Embedding Map (Total:{total_points}, Labeled:{len(df_labeled)})", fontsize=18, fontweight='bold')
    plt.xlabel("UMAP Dimension X", fontsize=14)
    plt.ylabel("UMAP Dimension Y", fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    
    # 防止标签超出边界
    plt.tight_layout()

    print(f"💾 正在保存图片至: {output_png}...")
    plt.savefig(output_png)
    plt.close()
    print("✅ 全量 FSQ 可视化绘图完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制带定长 Layer0 Code 标记的 ResidualFSQ 全量降维图')
    parser.add_argument('-i', '--input', type=str, default='token_embedding_full_map.csv', help='输入的 CSV 文件')
    parser.add_argument('-o', '--output', type=str, default='token_map_full_labeled.png', help='输出的 PNG 文件')
    parser.add_argument('-n', '--num_samples', type=int, default=120, help='随机采样的标记数量 (默认120)')

    args = parser.parse_args()
    
    # 确保采样数不为负
    if args.num_samples < 0:
        print("❌ 错误: 采样数量不能为负。")
        sys.exit(1)
        
    plot_fsq_map(args.input, args.output, args.num_samples)
