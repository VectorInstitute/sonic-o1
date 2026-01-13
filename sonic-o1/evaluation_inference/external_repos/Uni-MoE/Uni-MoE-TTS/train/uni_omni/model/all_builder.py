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
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from uni_omni.model import *
from uni_omni.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_all_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", vison_tower_path = None, audio_tower_path = None):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'uni_omni' in model_name.lower():
        # Load Uni-MoE model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if vison_tower_path is not None:
                lora_cfg_pretrained.mm_vision_tower = vison_tower_path
            if audio_tower_path is not None:
                lora_cfg_pretrained.mm_audio_tower = audio_tower_path
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            print('Loading Uni-MoE from base model...')
            lora_cfg_pretrained.vocab_size = 32000
            model = UOmniLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            # model = UOmniLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            token_num, tokem_dim = len(tokenizer), model.lm_head.in_features
            print(len(tokenizer))
            if model.lm_head.weight.shape[0] != token_num:
                print("reshape")
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.resize_token_embeddings(len(tokenizer),mean_resizing=False)
                print(model.vocab_size)

            # print('Loading additional Uni-MoE weights...')
            # if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            #     non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            # else:
            #     # this is probably from HF Hub
            #     from huggingface_hub import hf_hub_download
            #     def load_from_hf(repo_id, filename, subfolder=None):
            #         cache_file = hf_hub_download(
            #             repo_id=repo_id,
            #             filename=filename,
            #             subfolder=subfolder)
            #         return torch.load(cache_file, map_location='cpu')
            #     non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            # non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            # if any(k.startswith('model.model.') for k in non_lora_trainables):
            #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            # model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
            if hasattr(lora_cfg_pretrained, "whisper_tower"):
                print("loading whisper_tower!")
                for k,v in model.named_parameters():
                    if "whisper_aligner.decoder.audio_decoder.layer_norm.weight" in k:
                        print(k,v)
                whisper_aligner_weights = torch.load(os.path.join(model_path, 'whisper_aligner.bin'), map_location='cpu')
                for k,v in whisper_aligner_weights.items():
                    # print(k)
                    if "decoder.audio_decoder.layer_norm.weight" in k:
                        print(k,v)
                whisper_aligner_weights = {(k[11:] if k.startswith('base_model.') else k): v for k, v in whisper_aligner_weights.items()}
                # if any(k.startswith('model.model.') for k in whisper_aligner_weights):
                whisper_aligner_weights = {(k[6:] if k.startswith('model.') else k): v for k, v in whisper_aligner_weights.items()}
                model.load_state_dict(whisper_aligner_weights, strict=False)
                for k,v in model.named_parameters():
                    if "whisper_aligner.decoder.audio_decoder.layer_norm.weight" in k:
                        print(k,v)
            if hasattr(lora_cfg_pretrained, "speech_generator_type"):
                print("loading speech_generator!")
                for k,v in model.named_parameters():
                    if "speech_generator.layers.11.mlp.gate_proj.weight" in k:
                        print(v)
                speech_generator_weights = torch.load(os.path.join(model_path, 'speech_generator.bin'), map_location='cpu')
                speech_generator_weights = {(k[11:] if k.startswith('base_model.') else k): v for k, v in speech_generator_weights.items()}
                # if any(k.startswith('model.model.') for k in speech_generator_weights):
                speech_generator_weights = {(k[6:] if k.startswith('model.') else k): v for k, v in speech_generator_weights.items()}
                model.load_state_dict(speech_generator_weights, strict=False)
                for k,v in model.named_parameters():
                    if "speech_generator.layers.11.mlp.gate_proj.weight" in k:
                        print(v)
            if os.path.exists(os.path.join(model_path, 'embeds.bin')):
                print("loading embeds!")
                for k,v in model.model.embed_tokens.named_parameters():
                    print(k,v)
                embed_weights = torch.load(os.path.join(model_path, 'embeds.bin'), map_location='cpu')
                embed_weights = {(k[11:] if k.startswith('base_model.') else k): v for k, v in embed_weights.items()}
                # if any(k.startswith('model.model.') for k in embed_weights):
                embed_weights = {(k[6:] if k.startswith('model.') else k): v for k, v in embed_weights.items()}
                for k,v in embed_weights.items():
                    print(k,v)
                model.load_state_dict(embed_weights, strict=False)
                for k,v in model.model.embed_tokens.named_parameters():
                    print(k,v)
        elif model_base is not None:
            # this may be mm projector only
            print('Loading Uni-hear from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            cfg_pretrained.use_flash_attn = False
            model = UOmniLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True , config=cfg_pretrained, **kwargs) # , config=cfg_pretrained
            if hasattr(cfg_pretrained, "whisper_tower"):
                whisper_aligner_weights = torch.load(os.path.join(model_path, 'whisper_all.bin'), map_location='cpu')
                whisper_aligner_weights = {k: v.to(torch.float16) for k, v in whisper_aligner_weights.items()}
                model.load_state_dict(whisper_aligner_weights, strict=False)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'uni_omni' in model_name.lower():
        model.resize_token_embeddings(len(tokenizer))

        audio_processor =None
        whisper_processor = None
        whisper_tower = model.whisper_tower
        if whisper_tower is not None:
            if not whisper_tower.is_loaded:
                print("reload!!!!!!!!!!!!!!!!!!!!!!!!")
                whisper_tower.load_model()
            whisper_tower.to(device=device, dtype=torch.float16)
            whisper_processor = whisper_tower.audio_processor
        audio_processor = {
            "whisper_processor":whisper_processor,
        }

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, audio_processor, context_len
