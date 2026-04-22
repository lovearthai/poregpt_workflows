import torch

def get_vector_from_fsq_id(fsq_id, levels):
    """
    fsq_id: 索引 (可以是 scalar 或 tensor)
    levels: 各个维度的能级列表, 例如 [8, 5, 5, 5]
    """
    device = fsq_id.device if isinstance(fsq_id, torch.Tensor) else "cpu"
    levels_tensor = torch.tensor(levels, device=device)
    
    # 1. 计算基数 (Basis) 用于解算索引
    # basis = [1, L1, L1*L2, L1*L2*L3, ...]
    basis = torch.cumprod(torch.tensor([1] + levels[:-1], device=device), dim=0)
    
    # 2. 将 ID 还原为各维度的整数坐标 (Indices)
    # indices[i] = (id // basis[i]) % levels[i]
    indices = (fsq_id.unsqueeze(-1) // basis) % levels_tensor
    
    # 3. 将整数坐标映射回连续空间 (通常映射到 [-1, 1])
    # 公式: output = 2 * (indices / (levels - 1)) - 1
    # 注意: 如果某个维度 level 是 1，公式需要特殊处理，通常 FSQ level 至少为 2
    fsq_vectors = 2 * (indices / (levels_tensor - 1)) - 1
    
    return fsq_vectors

# --- 示例 ---
# 假设你的 FSQ 配置是 [8, 5, 5, 5] (总 codebook 大小 1000)
my_levels = [5, 5, 5, 5]
target_id = torch.tensor([123, 456]) # 假设我们要查 ID 为 123 和 456 的向量

vectors = get_vector_from_fsq_id(target_id, my_levels)
print(f"ID 123 对应的向量:\n{vectors[0]}")
print(f"ID 456 对应的向量:\n{vectors[1]}")
