import os
import re
import argparse
import copy
from tqdm import tqdm
import shutil
import torch
from peft import  get_peft_model, PeftConfig
import json
from safetensors.torch import save_file
from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration

def save_model_sharded(state_dict, save_path, max_shard_size=5*1024*1024*1024):  # 5GB per shard
    """
    将模型参数分片保存为主流格式（model-00001-of-00012.safetensors）
    
    Args:
        state_dict: 模型状态字典
        save_path: 保存目录路径
        max_shard_size: 每个分片的最大大小（字节），默认5GB
    """
    # 计算每个参数的大小
    param_sizes = {}
    total_size = 0
    for name, param in state_dict.items():
        size = param.numel() * param.element_size()
        param_sizes[name] = size
        total_size += size
    
    # 按参数大小排序，优先放置大参数
    sorted_params = sorted(param_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # 分片逻辑
    shards = []
    current_shard = {}
    current_shard_size = 0
    
    for param_name, param_size in sorted_params:
        # 如果当前分片加上这个参数会超过最大大小，且当前分片不为空，则开始新分片
        if current_shard_size + param_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0
        
        current_shard[param_name] = state_dict[param_name]
        current_shard_size += param_size
    
    # 添加最后一个分片
    if current_shard:
        shards.append(current_shard)
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 保存分片文件
    total_shards = len(shards)
    weight_map = {}
    
    for i, shard in enumerate(shards):
        shard_filename = f"model-{i+1:05d}-of-{total_shards:05d}.safetensors"
        shard_path = os.path.join(save_path, shard_filename)
        
        # 添加必要的元数据，确保与Transformers库兼容
        metadata = {
            "format": "pt",  # PyTorch格式
            "shard_id": str(i + 1),
            "total_shards": str(total_shards)
        }
        save_file(shard, shard_path, metadata=metadata)
        
        # 记录参数到文件的映射
        for param_name in shard.keys():
            weight_map[param_name] = shard_filename
    
    # 生成index.json文件
    index_data = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }
    
    index_path = os.path.join(save_path, "model.safetensors.index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"模型已保存为 {total_shards} 个分片，总大小: {total_size / (1024**3):.2f} GB")
    print(f"分片文件: model-00001-of-{total_shards:05d}.safetensors 到 model-{total_shards:05d}-of-{total_shards:05d}.safetensors")
    print(f"索引文件: model.safetensors.index.json")

def aggregation(checkpoint_path, source_ep_num=4):
    """聚合DeepSpeed专家并行检查点的函数
    
    Args:
        checkpoint_path: 检查点文件路径
        source_ep_num: 源专家并行数量，默认为8
    
    Returns:
        target: 聚合后的模型状态字典列表
    """
    # 定义专家文件的正则表达式模式，用于匹配layer_X_expert_Y_mp_rank_00_model_states.pt格式的文件
    expert_pattern = r"layer_(\d+)_expert_(\d+)_mp_rank_00_model_states.pt"
    # 定义MLP层名称的正则表达式模式，用于匹配模型中专家层的参数名称
    mlp_name_pattern = r"model\.layers\.(\d+)\.mlp\.dynamic_real_moe\.deepspeed_moe\.experts\.deepspeed_experts\.(\d+)"
    # 定义MLP层重命名的格式字符串，用于生成新的参数名称
    mlp_rename_format = "model.layers.{layer_id}.mlp.dynamic_real_moe.deepspeed_moe.experts.deepspeed_experts.{new_expert_id}{rest}"
    # 计算专家组数量，即每个目标专家包含多少个源专家
    ep_group_num = source_ep_num
    # 加载主模型状态字典，从mp_rank_00_model_states.pt文件中读取
    module = torch.load(os.path.join(checkpoint_path, "mp_rank_00_model_states.pt"), map_location="cpu")["module"]
    # 创建目标模型列表，每个目标专家都是主模块的深拷贝
    target = copy.deepcopy(module)
    # 获取检查点目录下的所有文件列表
    files = os.listdir(checkpoint_path)
    # 遍历所有文件，显示进度条
    for file in tqdm(files):
        # 使用正则表达式匹配专家文件名
        match = re.match(expert_pattern, file)
        # 如果匹配成功，说明这是一个专家文件
        if match:
            # 提取层ID（第一个捕获组）
            layer_id = int(match.group(1))
            if layer_id > 27:
                continue
            # 提取专家ID（第二个捕获组）
            expert_id = int(match.group(2))
            # 加载专家参数字典
            param_dict = torch.load(os.path.join(checkpoint_path, file), map_location="cpu")
            # 遍历专家参数字典中的每个参数
            for name, param in param_dict.items():
                # 断言：确保参数名不在目标模型中（避免重复）
                assert name not in target
                # 使用正则表达式匹配参数名称
                name_match = re.match(mlp_name_pattern, name)
                # 断言：确保参数名匹配成功
                assert name_match, f"lora_name_match failed for {name}"
                # 断言：确保参数名中的层ID与文件名中的层ID一致
                assert int(name_match.group(1)) == layer_id
                # 断言：确保参数名中的专家ID与文件名中的专家ID一致
                assert int(name_match.group(2)) == expert_id
                # 提取参数名的剩余部分（去掉匹配的前缀）
                rest = name[len(name_match.group(0)) :]
                # 生成新的MLP参数名，将专家ID重新映射到目标专家组内的ID
                new_mlp_name = mlp_rename_format.format(**{"layer_id": layer_id, "new_expert_id": expert_id % ep_group_num, "rest": rest})
                # 断言：确保新参数名不在目标模型中（避免冲突）
                assert new_mlp_name not in target
                # 将参数添加到对应的目标专家模型中sad
                target[new_mlp_name] = param

    return target


def aggregation_lora(checkpoint_path, source_ep_num=4):
    """聚合DeepSpeed专家并行检查点的函数
    
    Args:
        checkpoint_path: 检查点文件路径
        source_ep_num: 源专家并行数量，默认为8
    
    Returns:
        target: 聚合后的模型状态字典列表
    """
    # 定义专家文件的正则表达式模式，用于匹配layer_X_expert_Y_mp_rank_00_model_states.pt格式的文件
    expert_pattern = r"layer_(\d+)_expert_(\d+)_mp_rank_00_model_states.pt"
    # 定义MLP层名称的正则表达式模式，用于匹配模型中专家层的参数名称
    mlp_name_pattern = r"base_model\.model\.model\.layers\.(\d+)\.mlp\.dynamic_real_moe\.deepspeed_moe\.experts\.deepspeed_experts\.(\d+)"
    # 定义MLP层重命名的格式字符串，用于生成新的参数名称
    mlp_rename_format = "base_model.model.model.layers.{layer_id}.mlp.dynamic_real_moe.deepspeed_moe.experts.deepspeed_experts.{new_expert_id}{rest}"
    # 计算专家组数量，即每个目标专家包含多少个源专家
    ep_group_num = source_ep_num
    # 加载主模型状态字典，从mp_rank_00_model_states.pt文件中读取
    module = torch.load(os.path.join(checkpoint_path, "mp_rank_00_model_states.pt"), map_location="cpu")["module"]
    # 创建目标模型列表，每个目标专家都是主模块的深拷贝
    target = copy.deepcopy(module)
    # 获取检查点目录下的所有文件列表
    files = os.listdir(checkpoint_path)
    # 遍历所有文件，显示进度条
    for file in tqdm(files):
        # 使用正则表达式匹配专家文件名
        match = re.match(expert_pattern, file)
        # 如果匹配成功，说明这是一个专家文件
        if match:
            # 提取层ID（第一个捕获组）
            layer_id = int(match.group(1))
            if layer_id > 27:
                continue
            # 提取专家ID（第二个捕获组）
            expert_id = int(match.group(2))
            # 加载专家参数字典
            param_dict = torch.load(os.path.join(checkpoint_path, file), map_location="cpu")
            # 遍历专家参数字典中的每个参数
            for name, param in param_dict.items():
                # print(name)
                # 断言：确保参数名不在目标模型中（避免重复）
                assert name not in target
                # 使用正则表达式匹配参数名称
                name_match = re.match(mlp_name_pattern, name)
                # 断言：确保参数名匹配成功
                assert name_match, f"lora_name_match failed for {name}"
                # 断言：确保参数名中的层ID与文件名中的层ID一致
                assert int(name_match.group(1)) == layer_id
                # 断言：确保参数名中的专家ID与文件名中的专家ID一致
                assert int(name_match.group(2)) == expert_id
                # 提取参数名的剩余部分（去掉匹配的前缀）
                rest = name[len(name_match.group(0)) :]
                # 生成新的MLP参数名，将专家ID重新映射到目标专家组内的ID
                new_mlp_name = mlp_rename_format.format(**{"layer_id": layer_id, "new_expert_id": expert_id % ep_group_num, "rest": rest})
                # 断言：确保新参数名不在目标模型中（避免冲突）
                assert new_mlp_name not in target
                # 将参数添加到对应的目标专家模型中sad
                target[new_mlp_name] = param

    return target

def find_moe_all_linear(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['vision_tower', 'vision_aligner', 'audio_tower', 'audio_aligner', 'speech_generator', 'image_generator', 'image_hidden_fcs', 'task_hidden_fcs']
    experts_keywords = ['fixed_real_moe', 'dynamic_real_moe', 'self_attn']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if any(e_keyword in name for e_keyword in experts_keywords): # only MoE layers
            if isinstance(module, cls):
                lora_module_names.add(name)

    return list(lora_module_names)


def main():
    parser = argparse.ArgumentParser(description='模型提取和合并工具')
    parser.add_argument('--model_path', type=str, required=True, )
    parser.add_argument('--ckpt_path', type=str, required=True,)
    parser.add_argument('--lora_enable', type=bool, default=True,)
    parser.add_argument('--save_path', type=str, required=True,)
    
    args = parser.parse_args()
    
    lora_enable = args.lora_enable
    
    os.makedirs(args.save_path, exist_ok=True)
    for filename in os.listdir(args.model_path):
        if "model" in filename:
            continue
        filepath = os.path.join(args.model_path, filename)
        if os.path.isfile(filepath):
            shutil.copy(filepath, args.save_path)
        elif os.path.isdir(filepath):
            save_dir = os.path.join(args.save_path, filename)
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            shutil.copytree(filepath, save_dir)
            
    
    if lora_enable:
        base_model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(
            args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        lora_config = PeftConfig.from_pretrained(args.ckpt_path)

        lora_config.target_modules = find_moe_all_linear(base_model)

        peft_model = get_peft_model(base_model, lora_config)
        
        pattern = re.compile(r'^global_step\d+$')
        for dir_name in os.listdir(args.ckpt_path):
            dir_path = os.path.join(args.ckpt_path, dir_name)
            if os.path.isdir(dir_path) and pattern.match(dir_name):
                global_step_dir = dir_path
                break

        lora_prams_dict = aggregation_lora(global_step_dir)

        peft_model.load_state_dict(lora_prams_dict, strict=False)

        merged_model = peft_model.merge_and_unload()

        save_model_sharded(merged_model.state_dict(), args.save_path)
    else:
        pattern = re.compile(r'^global_step\d+$')
        for dir_name in os.listdir(args.ckpt_path):
            dir_path = os.path.join(args.ckpt_path, dir_name)
            if os.path.isdir(dir_path) and pattern.match(dir_name):
                global_step_dir = dir_path
                break
        
        prams_dict = aggregation(global_step_dir)
        save_model_sharded(prams_dict, args.save_path)


if __name__ == "__main__":
    main()