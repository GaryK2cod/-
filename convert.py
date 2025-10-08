import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


# 层名称映射关系：原始名称 -> (新名称, 分割维度)
mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    转换并保存模型检查点文件到指定格式。

    参数:
        hf_ckpt_path (str): 输入检查点文件所在目录的路径。
        save_path (str): 转换后的检查点文件保存目录的路径。
        n_experts (int): 模型中的专家总数。
        mp (int): 模型并行度因子。
        
    返回:
        无
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp  # 每个并行分区的本地专家数
    state_dicts = [{} for _ in range(mp)]  # 为每个并行分区创建空状态字典

    # 遍历所有safetensors文件
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                # 跳过第61层的参数（特定模型可能需要）
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                
                # 统一参数名称格式
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                
                # 根据映射表转换键名
                key = name.split(".")[-2]
                assert key in mapping, f"键 {key} 在映射表中未找到"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                
                # 为每个并行分区处理参数
                for i in range(mp):
                    new_param = param
                    # 处理专家参数的分区
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        # 只保留属于当前分区的专家
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    # 处理需要分割的维度
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"维度 {dim} 必须能被 {mp} 整除"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 保存每个并行分区的状态字典
    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    # 复制tokenizer相关文件
    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True, help="HuggingFace检查点路径")
    parser.add_argument("--save-path", type=str, required=True, help="保存路径")
    parser.add_argument("--n-experts", type=int, required=True, help="专家数量")
    parser.add_argument("--model-parallel", type=int, required=True, help="模型并行度")
    args = parser.parse_args()
    # 验证专家数能被并行度整除
    assert args.n_experts % args.model_parallel == 0, "专家数量必须能被模型并行度整除"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)