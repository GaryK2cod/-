import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant

def main(fp8_path, bf16_path):
    """
    将FP8权重转换为BF16格式并保存转换后的权重。

    此函数从指定目录读取FP8权重，将其转换为BF16格式，
    并将转换后的权重保存到另一个指定目录。同时更新模型索引文件以反映更改。

    参数:
        fp8_path (str): 包含FP8权重和模型索引文件的目录路径。
        bf16_path (str): 转换后的BF16权重保存目录的路径。

    异常:
        KeyError: 如果缺少权重所需的scale_inv张量。

    说明:
        - 函数假设FP8权重存储在safetensor文件中。
        - 函数缓存已加载的safetensor文件以优化内存使用。
        - 函数更新模型索引文件以移除对scale_inv张量的引用。
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    # 已加载safetensor文件的缓存
    loaded_files = {}
    fp8_weight_names = []

    # 辅助函数：从正确的文件中获取张量
    def get_tensor(tensor_name):
        """
        从缓存的safetensor文件中检索张量，如果未缓存则从磁盘加载。

        参数:
            tensor_name (str): 要检索的张量名称。

        返回:
            torch.Tensor: 检索到的张量。

        异常:
            KeyError: 如果safetensor文件中不存在该张量。
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8权重
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # 从正确的文件获取scale_inv
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(f"警告: 缺少 {weight_name} 的scale_inv张量，跳过转换")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # 内存管理：只保留最近使用的2个文件
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()
    
    # 更新模型索引
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
        

if __name__ == "__main__":
    parser = ArgumentParser(description="将FP8模型权重转换为BF16格式")
    parser.add_argument("--input-fp8-hf-path", type=str, required=True, help="输入FP8模型路径")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True, help="输出BF16模型路径")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)