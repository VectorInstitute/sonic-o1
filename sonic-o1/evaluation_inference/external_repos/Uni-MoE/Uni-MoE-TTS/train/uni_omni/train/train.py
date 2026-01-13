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
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Any, Union

import torch
import torch.nn as nn
import transformers


# from torchstat import stat
# import torchsummary

from uni_omni.train.llava_trainer import LLaVATrainer
from uni_omni.train.data import ModelArguments,DataArguments,TrainingArguments,make_supervised_data_module
from uni_omni import conversation as conversation_lib
from uni_omni.model import *
import deepspeed
import torch.distributed as dist


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

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

def find_all_llm_linear_names(model, training_args):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['tower','aligner','mm_projector', 'vision_tower', 'vision_resampler', "generator"]
    if training_args.enable_deepspeed_moe:
        multimodal_keywords.append('mlp.deepspeed_moe.gate.')
        multimodal_keywords.append('mlp.deepspeed_moe.experts.deepspeed_experts.')
    else:
        multimodal_keywords.append('mlp.gate.')
    if training_args.lora_only_mlp:
        multimodal_keywords.append('self_attn.')
        
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def safe_save_all_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str, training_args: TrainingArguments):
    """Collects the state dict and dump to disk."""
    
    if getattr(training_args, 'tune_speech_generator', False):
        keys_to_match = ["speech_generator"]
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        weight_to_save = {(k[11:] if k.startswith('base_model.') else k): v for k, v in weight_to_save.items()}
        if any(k.startswith('model.model.') for k in weight_to_save):
            weight_to_save = {(k[6:] if k.startswith('model.') else k): v for k, v in weight_to_save.items()}

        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir, f'speech_generator.bin'))

    embed = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), ["model.embed_tokens.","lm_head"])
    embed = {(k[11:] if k.startswith('base_model.') else k): v for k, v in embed.items()}
    if any(k.startswith('model.model.') for k in embed):
        embed = {(k[6:] if k.startswith('model.') else k): v for k, v in embed.items()}
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        trainer.model.config.save_pretrained(output_dir)
        torch.save(embed, os.path.join(output_dir, f'embeds.bin'))

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


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))


    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    model = UOmniQwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        ignore_mismatched_sizes=True,
        # attn_implementation=attn_implementation,
        **bnb_model_from_pretrained_args
    )

    model.config.num_experts = 4
    model.config.ep_size = 1
    model.config.capacity_factor = 1.5
    model.config.moe_dp = True

    expert_dir = model_args.expert_dir

    model.config.use_cache = False
    model.requires_grad_(False)
    
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    data_args.local_rank = training_args.local_rank
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
    # resize embedding
    model.model = None
    model.cuda()

    training_args.tune_speech_generator = model_args.tune_speech_generator
    if model_args.tune_speech_generator:
        model_args.audio_mode = data_args.audio_mode
        model.initialize_speech_generator(model_args=model_args)
        speech_generator = model.speech_generator
        speech_generator.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.is_multimodal = True

    if model_args.load_weight_from_qwen:
        print("loading ckpt")
        qwen_weights = torch.load(model_args.load_weight_from_qwen, map_location='cpu')
        qwen_weights = {k: v.to(torch.float16) for k, v in qwen_weights.items()}
        model.load_state_dict(qwen_weights, strict=False)

    
    if model.config.moe_dp:
        print("moe mode")
        local_expert_num = model.config.num_experts // model.config.ep_size
        new_format_version = False
        for state_dict_bias in range(local_expert_num):
            state_dict_num = (state_dict_bias + dist.get_rank() * local_expert_num) % model.config.num_experts
            # if previous formart, only contain the expert weight, and use name to locate
            all_experts = os.listdir(expert_dir)
            now_expert = all_experts[0]
            for edir in all_experts:
                if f"num_{state_dict_num}" in edir:
                    now_expert = edir
            expert_path = os.path.join(expert_dir, now_expert)
            state_dict = torch.load(expert_path, map_location=torch.device('cpu'))

            print("Rank: ", dist.get_rank(), "Load expert from: ", expert_path, " to deepspeed_experts." + str(state_dict_bias))
            moe_dict = {}
            for key, value in state_dict.items():
                # need change
                name = "speech_generator.layers."+key.split(".")[2]+".mlp.deepspeed_moe.experts.deepspeed_experts." + str(state_dict_bias) +  "."+key.split(".")[-2]+".weight"
                moe_dict[name] = value
            model.load_state_dict(moe_dict, strict=False)

    model.requires_grad_(False)

    if model_args.tune_speech_generator:
        for n,p in model.speech_generator.named_parameters():
            p.requires_grad = True
            if "embed_tokens" in n:
                p.requires_grad = False
            if "qwenvl_embed_tokens" in n:
                p.requires_grad = False

    if model_args.tune_qwen25vl_proj_only:
        for p in model.speech_generator.parameters():
            p.requires_grad = False
        for n,p in model.speech_generator.named_parameters():
            if "in_fnn" in n:
                p.requires_grad = True

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total number of parameters: {total_num}, trained: {trainable_num}, ratio: {trainable_num/total_num:.2f}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)


    training_args.lora_enable = training_args.llm_lora_enable

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)


    audio_data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                              data_args=data_args)

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **audio_data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # deepspeed.utils.set_z3_leaf_modules(model, [MoeLayer])
    model.config.use_cache = True

    if training_args.llm_lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        safe_save_all_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir, training_args=training_args)
    else:
        safe_save_all_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir, training_args=training_args)
