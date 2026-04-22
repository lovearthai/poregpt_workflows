import sys

def main():
    # --- FSQ 配置 ---
    levels = [5,5,5,5] 
    # 计算最大可能的 token_id
    max_possible_id = 1
    for l in levels:
        max_possible_id *= l
    max_possible_id -= 1 # ID 从 0 开始

    # --- 检查命令行参数 ---
    if len(sys.argv) != 2:
        print(f"使用方法: python {sys.argv[0]} <token_id>")
        print(f"示例: python {sys.argv[0]} 457")
        print(f"注意: token_id 必须在 0 和 {max_possible_id} 之间")
        sys.exit(1)

    try:
        token_id = int(sys.argv[1])
    except ValueError:
        print(f"错误: '{sys.argv[1]}' 不是一个有效的整数。")
        sys.exit(1)

    if not (0 <= token_id <= max_possible_id):
        print(f"错误: token_id {token_id} 超出了有效范围 [0, {max_possible_id}]。")
        sys.exit(1)

    # --- 执行解码 ---
    indices = []
    remainder = token_id
    products = []
    num_levels = len(levels)

    # 预计算每个维度的“基数乘积”
    # 例如 levels=，则 products= [5*5*5, 5*5, 5*1, 1] = [125, 25, 5, 1]
    for i in range(num_levels):
        product = 1
        for j in range(i + 1, num_levels):
            product *= levels[j]
        products.append(product)

    for i in range(num_levels):
        base = products[i]
        index = remainder // base
        remainder = remainder % base
        indices.append(index)

    # --- 输出结果 ---
    print(f"Token ID: {token_id}")
    print("四个维度的编码值:")
    for i, idx in enumerate(indices):
        print(f"  维度 {i}: {idx}")

if __name__ == "__main__":
    main()
