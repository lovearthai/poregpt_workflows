import torch
from vector_quantize_pytorch import ResidualFSQ
from poregpt.utils import get_rsq_vector_from_integer,get_rsq_vector_from_indices,get_fsq_vector_from_indices_via_math 



# --- 校验与使用示例 ---
if __name__ == "__main__":
    levels = [5, 5, 5, 5]
    num_q = 2 # 测试多层残差
    my_token_id = 62502

    # 1. 测试单整数接口
    # ⚠️ 修正：你原代码 main 里的函数名写错了，这里已修正
    result_vector = get_rsq_vector_from_integer(my_token_id, levels, num_q,debug=True)

    print(f"输入 ID: {my_token_id}")
    print(f"输出向量形状: {result_vector.shape}")
    print(f"输出向量前 4 维: {result_vector[0, :4]}")

    # 2. 对比校验 (API vs Math)
    mock_indices = torch.tensor([[10, 20]], device=result_vector.device) # 随机模拟两层 ID
    v_api = get_rsq_vector_from_indices(mock_indices, levels, num_q)
    v_math = get_fsq_vector_from_indices_via_math(mock_indices, levels, num_q)

    diff = torch.abs(v_api - v_math).max().item()
    print(f"\nAPI 与数学公式的一致性误差: {diff:.2e}")


