# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import re
import logging
import pathlib
import torch
import transformers
from uni_moe.train.data import ModelArguments,DataArguments,TrainingArguments,make_supervised_data_module
from uni_moe.train.training_utils import rank0_print, rank0_pprint, MYEpochSaveCallback, set_trainable, compress_strings_set
from uni_moe.train.moe_trainer import MoETrainer
from uni_moe.model.modeling_qwen_grin_moe import GrinQwen2VLForConditionalGeneration, GrinQwen2VLConfig
from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor

local_rank = None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_all_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    # audio projector/aligner
    if trainer.args.tune_audio_aligner or trainer.args.tune_audio_projector or trainer.args.tune_audio_module:
        keys_to_match = ['audio_aligner','audio_tower']
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        weight_to_save = {(k[11:] if k.startswith('base_model.') else k): v for k, v in weight_to_save.items()}
        if any(k.startswith('model.model.') for k in weight_to_save):
            weight_to_save = {(k[6:] if k.startswith('model.') else k): v for k, v in weight_to_save.items()}

        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, f'audio_module.bin'))

    if trainer.args.tune_vision_aligner:
        vision_keys_to_match = ['vision_aligner']
        vision_weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), vision_keys_to_match)
        vision_weight_to_save = {(k[11:] if k.startswith('base_model.') else k): v for k, v in vision_weight_to_save.items()}
        if any(k.startswith('model.model.') for k in vision_weight_to_save):
            vision_weight_to_save = {(k[6:] if k.startswith('model.') else k): v for k, v in vision_weight_to_save.items()}

        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            torch.save(vision_weight_to_save, os.path.join(output_dir, f'vision_aligner.bin'))
    
    # image generator
    keys_to_match = ['image_generator']
    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    weight_to_save = {(k[11:] if k.startswith('base_model.') else k): v for k, v in weight_to_save.items()}
    if any(k.startswith('model.model.') for k in weight_to_save):
        weight_to_save = {(k[6:] if k.startswith('model.') else k): v for k, v in weight_to_save.items()}

    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        trainer.model.config.save_pretrained(output_dir)
        torch.save(weight_to_save, os.path.join(output_dir, f'image_generator.bin'))
    
    # mlp gate
    mlp_gates = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), ["mlp.gate."])
    if len(mlp_gates.items())>0:
        mlp_gates = {(k[11:] if k.startswith('base_model.') else k): v for k, v in mlp_gates.items()}
        if any(k.startswith('model.model.') for k in mlp_gates):
            mlp_gates = {(k[6:] if k.startswith('model.') else k): v for k, v in mlp_gates.items()}
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)
            torch.save(mlp_gates, os.path.join(output_dir, f'mlp_gates.bin'))

    if trainer.args.tune_speech_generator:
        embed = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), ["speech_gen"])
        embed = {(k[11:] if k.startswith('base_model.') else k): v for k, v in embed.items()}
        if any(k.startswith('model.model.') for k in embed):
            embed = {(k[6:] if k.startswith('model.') else k): v for k, v in embed.items()}
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)
            torch.save(embed, os.path.join(output_dir, f'speech_gen.bin'))


    # model
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


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
                # names = name.split('.')
                # lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    
    
    
    # expert parallelism
    if model_args.ep_size != 1:
        config = GrinQwen2VLConfig.from_pretrained(
            model_args.model_name_or_path,
        )
        
        config.token_drop = model_args.token_drop
        config.frames_upbound = model_args.frames_upbound
        
        # Complete Model Load from Checkpoint, for expert Loading
        model_complete = GrinQwen2VLOutForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            attn_implementation=training_args.attn_implementation
        )
        
        # expert parallelism model, for really training
        config.ep_size = model_args.ep_size
        model = GrinQwen2VLOutForConditionalGeneration._from_config(
            config,
            torch_dtype=compute_dtype,
            attn_implementation=training_args.attn_implementation
        )
        
        cur_model_state_dict = model.state_dict()
        complete_model_state_dict = model_complete.state_dict()
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        mlp_pattern = r"dynamic_real_moe\.deepspeed_moe\.experts\.deepspeed_experts\.(\d+)"
        for n in cur_model_state_dict.keys():
            match = re.search(mlp_pattern, n)
            if match:
                expert = int(match.group(1))
                
                ep_group_rank = local_rank % config.ep_size * (config.mlp_dynamic_expert_num // config.ep_size) + expert
                target_mlp = n.replace(f"dynamic_real_moe.deepspeed_moe.experts.deepspeed_experts.{expert}", f"dynamic_real_moe.deepspeed_moe.experts.deepspeed_experts.{ep_group_rank}")
                cur_model_state_dict[n] = complete_model_state_dict[target_mlp]
            else:
                cur_model_state_dict[n] = complete_model_state_dict[n]
        model.load_state_dict(cur_model_state_dict) 
        del model_complete
        

    else:
        config = GrinQwen2VLConfig.from_pretrained(
            model_args.model_name_or_path,
        )
        
        config.token_drop = model_args.token_drop
        config.frames_upbound = model_args.frames_upbound
        # Direct Load Model from Checkpoint
        model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation=training_args.attn_implementation,
        )
        
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path)
    model.config.use_cache = False
    processor.data_args = model.config
    data_args.processor = processor
    
    if training_args.gradient_checkpointing:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

            if model_args.tune_image_generator:
            
                img_token_ids = [151672 + i for i in range(32)]
                task_token_ids = [151704, 151705, 151706]
                all_trainable_token_ids = torch.tensor(
                    img_token_ids + task_token_ids, device=output.device
                )
                #print("all trainable token ids OPEN")

                def grad_hook(grad):
                    # grad: [batch, seq_len, hidden_dim]
                    token_ids = input[0]  # [batch, seq_len]
                    mask = torch.isin(token_ids, all_trainable_token_ids).to(dtype=grad.dtype, device=grad.device)
                    mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
                    return grad * mask

                output.register_hook(grad_hook)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    data_module = make_supervised_data_module(data_args=data_args)

    if hasattr(training_args, "freeze_prefix") and training_args.freeze_prefix:
        frozen_count = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if any(name.startswith(prefix) for prefix in training_args.freeze_prefix):
                param.requires_grad = False
                frozen_count += 1
        print(
            f"Froze {frozen_count}/{total_params} parameters based on prefixes: {training_args.freeze_prefix}"
        )
    
        
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")
        
        # freeze base model
        for p in model.model.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_ = False
            
        requires_grad_params = []
        for k, p in model.named_parameters():
            if p.requires_grad:
                requires_grad_params.append(k)
        
        # print(find_moe_all_linear(model))
        lora_config = LoraConfig(
            r=training_args.lora_r or 8,
            lora_alpha=training_args.lora_alpha or 16,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=find_moe_all_linear(model),  # MoE Layer of Uni-MoE 2.0
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        
        for k, p in model.named_parameters():
            ori_name = k[len("base_model.model."):]
            if ori_name in requires_grad_params:
                p.requires_grad = True
        
        # print(model.base_model.model)
        for l in range(len(model.base_model.model.model.layers)):
            for expert in model.base_model.model.model.layers[l].mlp.dynamic_real_moe.deepspeed_moe.experts.deepspeed_experts:
                for param in expert.parameters():
                    param.allreduce = False
                    param.group_name = model.base_model.model.model.layers[l].mlp.dynamic_real_moe.expert_group_name
        
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(f"{training_args.output_dir}/requires_grad_params.txt", "w") as f:
        for k, v in model.named_parameters():
            if v.requires_grad:
                f.write(k + "\n")
    
    trainer = MoETrainer(
        model=model, tokenizer=processor.tokenizer, callbacks=[MYEpochSaveCallback(save_processor=processor, save_dir=training_args.output_dir)], args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_all_for_hf_trainer(trainer=trainer,
                                    output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
