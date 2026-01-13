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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen2ForCausalLM, Qwen2Model, Qwen2Config

from uni_omni.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, AUDIO_TOKEN_INDEX, VIDEO_TOKEN_INDEX


from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.utils import ModelOutput
import torch.nn.functional as F

from uni_omni.model.speech_generator_AR_ori_v2.builder import build_ar_ori_v2_speech_generator



IGNORE_INDEX = -100
DO_GATE = True
TRAIN = True

sidx = 0

import warnings

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from dataclasses import dataclass


@dataclass
class UOmniCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    gate_info: Optional[Tuple[torch.FloatTensor]] = None


class UOmniConfig(Qwen2Config):
    model_type = "UOmni_qwen"


class UOmniQwen2Model(Qwen2Model):
    config_class = UOmniConfig

    def __init__(self, config: Qwen2Config):
        super(UOmniQwen2Model, self).__init__(config)


class UOmniQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = UOmniConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = UOmniQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # self.model = None
        

        self.speech_generator = None
        if hasattr(config, "speech_generator_type"):
            self.audio_mode = "tts_sft"
            self.tune_speech_generator_only = True
            if config.speech_generator_type == 'ar_ori_v2':
                self.speech_generator = build_ar_ori_v2_speech_generator(config)
                print("finish spgen ar ori v2!")
            if config.speech_generator_type == 'ar_ori_v2_new':
                from uni_omni.model.speech_generator_AR_ori_v2_new.builder import build_ar_ori_v2_new_speech_generator
                self.speech_generator = build_ar_ori_v2_new_speech_generator(config)
                print("finish spgen ar ori v2 new!")
            self.prefix_len = 10

    def initialize_speech_generator(self, model_args):
        self.config.speech_generator_type = getattr(model_args, 'speech_generator_type', 'ar')
        self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', False)
        self.audio_mode = getattr(model_args, 'audio_mode', False)
        self.prefix_len = 10
        if self.config.speech_generator_type == 'ar_ori_v2':
            self.config.idim = getattr(model_args, 'idim', 4) # 896 输入的维度 llama decoder的特征维度
            self.config.odim = getattr(model_args, 'odim', 4096) # 1024 词表维度 tokenizer的词表大小
            self.config.encoder_pre_norm_type = getattr(model_args, 'encoder_pre_norm_type', "ln") # "ln" 似乎没用到
            self.config.encoder_drop_rate = getattr(model_args, 'encoder_drop_rate', 0.1 ) # 0.1 drop out的程度 因为没有NAR所以用不到
            self.config.encoder_criterion = getattr(model_args, 'encoder_criterion', "ce") # "ce" 损失函数类型，不改动
            self.config.encoder_upsample_rate = getattr(model_args, 'encoder_upsample_rate', 9) # 9 没有做upsample所以也用不到
            self.config.transformer_attention_dim = getattr(model_args, 'transformer_attention_dim', 4096) # 896 llama hidden_size 4096 
            self.config.transformer_linear_units = getattr(model_args, 'transformer_linear_units', 11008) # 4864 llama intermediate_size 11008
            self.config.transformer_num_blocks = getattr(model_args, 'transformer_num_blocks', 8) # 4 llama num_hidden_layers 32 
            self.config.transformer_attention_heads = getattr(model_args, 'transformer_attention_heads', 32) # 14 llama num_attention_heads 32
            self.config.transformer_kv_heads = getattr(model_args, 'transformer_kv_heads', 2) 
            self.config.transformer_dropout_rate = getattr(model_args, 'transformer_dropout_rate', 0.1 ) # 0.1 
            self.config.encoder_output_dim = getattr(model_args, 'encoder_output_dim', 896) # 896 输出词表大小
            self.config.llm_vocab_size = getattr(model_args, 'llm_vocab_size', 151936) # qwen 词表大小
            if getattr(self, "speech_generator", None) is None:
                self.speech_generator = build_ar_ori_v2_speech_generator(self.config)

        if self.config.speech_generator_type == 'ar_ori_v2_new':
            from uni_omni.model.speech_generator_AR_ori_v2_new.builder import build_ar_ori_v2_new_speech_generator
            self.config.idim = 2048 # getattr(model_args, 'idim', 2048) # 896 输入的维度 llama decoder的特征维度
            self.config.odim = getattr(model_args, 'odim', 4096) # 1024 词表维度 tokenizer的词表大小
            self.config.encoder_pre_norm_type = getattr(model_args, 'encoder_pre_norm_type', "ln") # "ln" 似乎没用到
            self.config.encoder_drop_rate = getattr(model_args, 'encoder_drop_rate', 0.1 ) # 0.1 drop out的程度 因为没有NAR所以用不到
            self.config.encoder_criterion = getattr(model_args, 'encoder_criterion', "ce") # "ce" 损失函数类型，不改动
            self.config.encoder_upsample_rate = getattr(model_args, 'encoder_upsample_rate', 9) # 9 没有做upsample所以也用不到
            self.config.transformer_attention_dim = getattr(model_args, 'transformer_attention_dim', 4096) # 896 llama hidden_size 4096 
            self.config.transformer_linear_units = getattr(model_args, 'transformer_linear_units', 11008) # 4864 llama intermediate_size 11008
            self.config.transformer_num_blocks = getattr(model_args, 'transformer_num_blocks', 8) # 4 llama num_hidden_layers 32 
            self.config.transformer_attention_heads = getattr(model_args, 'transformer_attention_heads', 32) # 14 llama num_attention_heads 32
            self.config.transformer_kv_heads = getattr(model_args, 'transformer_kv_heads', 2) 
            self.config.transformer_dropout_rate = getattr(model_args, 'transformer_dropout_rate', 0.1 ) # 0.1 
            self.config.encoder_output_dim = getattr(model_args, 'encoder_output_dim', 896) # 896 输出词表大小
            self.config.llm_vocab_size = getattr(model_args, 'llm_vocab_size', 151936) # qwen 词表大小
            if getattr(self, "speech_generator", None) is None:
                self.speech_generator = build_ar_ori_v2_new_speech_generator(self.config)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        prefix_ids: Optional[torch.FloatTensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        codes: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
        if self.config.speech_generator_type == 'ar_ori_v2':
            batch = {}
            batch['y'] = codes
            batch['y_lens'] = torch.sum(codes!=IGNORE_INDEX,dim=-1)
            batch['x'] = input_ids
            batch['x_lens'] = torch.sum(input_ids!=0,dim=-1)
            batch['prompt'] = prompt_ids
            batch['prompt_lens'] = torch.sum(prompt_ids!=IGNORE_INDEX,dim=-1)
            print(batch['y'].shape)
            if prefix_ids != None:
                batch['prefix'] = prefix_ids
                batch['prefix_lens'] = torch.sum(prefix_ids!=IGNORE_INDEX,dim=-1)
            loss = self.speech_generator(batch)

        elif self.config.speech_generator_type == 'ar_ori_v2_new':
            batch = {}
            batch['y'] = codes
            batch['y_lens'] = torch.sum(codes!=IGNORE_INDEX,dim=-1) # codes_lens
            batch['x'] = input_ids
            batch['x_lens'] = torch.sum(input_ids!=0,dim=-1) # input_ids_lens
            batch['prompt'] = prompt_ids
            batch['prompt_lens'] = torch.sum(prompt_ids!=IGNORE_INDEX,dim=-1) # prompt_ids_lens
            preflen = 0
            if prefix_ids != None:
                batch['prefix'] = prefix_ids
                batch['prefix_lens'] = torch.sum(prefix_ids!=IGNORE_INDEX,dim=-1) # prefix_ids_lens
                preflen = batch['prefix'].shape[1]
            loss = self.speech_generator(batch)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        logits = None

        return UOmniCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "features_mask": kwargs.get("features_mask", None),
                "whisper_features": kwargs.get("whisper_features",None),
            }
        )
        return model_inputs

AutoConfig.register("UOmni_qwen", UOmniConfig)
AutoModelForCausalLM.register(UOmniConfig, UOmniQwen2ForCausalLM)
