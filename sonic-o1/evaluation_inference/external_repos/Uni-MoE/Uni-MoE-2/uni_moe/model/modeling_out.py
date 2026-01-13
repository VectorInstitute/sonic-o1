# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2-VL model."""

import os
import re
import pdb
import math
import random
from dataclasses import dataclass
from safetensors.torch import load_file
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers import AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None

from uni_moe.model.modeling_qwen_grin_moe import GrinQwen2VLConfig, GrinQwen2VLForConditionalGeneration, MoEQwen2VLCausalLMOutputWithPast

from uni_moe.model.visual_gen_projector.layers import TextFcLayer
from uni_moe.model.image_generator.image_generator import image_generator

from uni_moe.utils import rank0_print

IGNORE_INDEX = -100

class GrinQwen2VLOutConfig(GrinQwen2VLConfig):
    model_type = "grin_qwen2_vl_out"



class GrinQwen2VLOutForConditionalGeneration(GrinQwen2VLForConditionalGeneration):
    config_class = GrinQwen2VLOutConfig

    def __init__(self, config):
        super().__init__(config)
        self.speech_generator = None
        if hasattr(config, "speech_generator_type"):
            self.audio_mode = "tts_sft"
            self.prefix_len = 10
            self.tune_speech_generator_only = False
            if config.speech_generator_type == 'ar_end':
                self.speech_generator = build_ar_end_speech_generator(config)
                # print("finish spgen ar end!")
            if config.speech_generator_type == 'ar_ori_v2':
                self.speech_generator = build_ar_ori_v2_speech_generator(config)
                # print("finish spgen ar ori v2!")
            if config.speech_generator_type == 'ar_ori_v2_new':
                from uni_moe.model.speech_generator_AR_ori_v2_new.builder import build_ar_ori_v2_new_speech_generator
                # print("varsin:",vars(config))
                self.speech_generator = build_ar_ori_v2_new_speech_generator(config)
                # print("finish spgen ar ori v2 new!")
        # if hasattr(config, "tune_image_generator_stage"):
        #     self.tune_image_generator_stage = config.tune_image_generator_stage
            
        # visual gen part
        if config.tune_image_generator:
            self.image_generator = image_generator()
            layer_num = 1
            input_visual_tokens_num = 576
            input_img_tokens_num = 32
            input_task_tokens_num = 4
            output_visual_tokens_num = 576 # [3,576,1024] - [576,1024]
            #output_img_tokens_num = 256 # 参考Vitron # [3,256,2048] - [256,2048]
            output_img_tokens_num = 32
            output_task_tokens_num = 3 # [3,768]
            
            self.visual_hidden_fcs = nn.ModuleList([])
            for layer_idx in range(layer_num):
                self.visual_hidden_fcs.append(
                    #TextFcLayer(in_dim=1536, out_dim=1024, 
                    TextFcLayer(in_dim=3584, out_dim=1024, 
                                num_input_tokens=input_visual_tokens_num,
                                num_output_tokens=output_visual_tokens_num,
                                mode="linear"))
            self.image_hidden_fcs = nn.ModuleList([])
            for layer_idx in range(layer_num):
                self.image_hidden_fcs.append(
                    #TextFcLayer(in_dim=1536, out_dim=2048,
                    TextFcLayer(in_dim=3584, out_dim=2048, 
                                num_input_tokens=input_img_tokens_num, # 32
                                num_output_tokens=output_img_tokens_num, # 32
                                #mode="linear-binary"))
                                mode="transformer"))
            self.task_hidden_fcs = nn.ModuleList([])
            for layer_idx in range(layer_num): # imggen s2
                self.task_hidden_fcs.append(
                    #TextFcLayer(in_dim=1536, out_dim=768,
                    TextFcLayer(in_dim=3584, out_dim=768,
                                num_input_tokens=input_task_tokens_num , # 4
                                num_output_tokens=output_task_tokens_num, # 3
                                mode="transformer"))

        self.post_init()
    
    def get_speech_generator(self):
        speech_generator = getattr(self, "speech_generator", None)
        return speech_generator
    
    def initialize_speech_generator(self, model_args):
        self.config.speech_generator_type = getattr(model_args, 'speech_generator_type', 'ar_end')
        self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', False)
        self.audio_mode = getattr(model_args, 'audio_mode', False)
        self.prefix_len = 10
        if self.config.speech_generator_type == 'ar_end':
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
            self.config.do_cross_attention = getattr(model_args, 'do_cross_attention', True ) # 0.1
            self.config.cross_attention_layer_num = getattr(model_args, 'cross_attention_layer_num', 6) # 896 输出词表大小
            if getattr(self, "speech_generator", None) is None:
                self.speech_generator = build_ar_end_speech_generator(self.config)
        if self.config.speech_generator_type == 'ar_ori_v2':
            print("building ar_ori_v2 speech gen...")
            self.config.idim = getattr(model_args, 'idim', 2048) # 896 输入的维度 llama decoder的特征维度
            self.config.odim = getattr(model_args, 'odim', 4096) # 1024 词表维度 tokenizer的词表大小
            self.config.encoder_pre_norm_type = getattr(model_args, 'encoder_pre_norm_type', "ln") # "ln" 似乎没用到
            self.config.encoder_drop_rate = getattr(model_args, 'encoder_drop_rate', 0.1 ) # 0.1 drop out的程度 因为没有NAR所以用不到
            self.config.encoder_criterion = getattr(model_args, 'encoder_criterion', "ce") # "ce" 损失函数类型，不改动
            self.config.encoder_upsample_rate = getattr(model_args, 'encoder_upsample_rate', 9) # 9 没有做upsample所以也用不到
            self.config.transformer_attention_dim = getattr(model_args, 'transformer_attention_dim', 896) # 896 llama hidden_size 4096 
            self.config.transformer_linear_units = getattr(model_args, 'transformer_linear_units', 4864) # 4864 llama intermediate_size 11008
            self.config.transformer_num_blocks = getattr(model_args, 'transformer_num_blocks', 24) # 4 llama num_hidden_layers 32 
            self.config.transformer_attention_heads = getattr(model_args, 'transformer_attention_heads', 14) # 14 llama num_attention_heads 32
            self.config.transformer_kv_heads = getattr(model_args, 'transformer_kv_heads', 2) 
            self.config.transformer_dropout_rate = getattr(model_args, 'transformer_dropout_rate', 0.1 ) # 0.1 
            self.config.encoder_output_dim = getattr(model_args, 'encoder_output_dim', 896) # 896 输出词表大小
            # self.config.llm_vocab_size = getattr(model_args, 'llm_vocab_size', 151936) # qwen 词表大小
            if getattr(self, "speech_generator", None) is None:
                self.speech_generator = build_ar_ori_v2_speech_generator(self.config)
        if self.config.speech_generator_type == 'ar_ori_v2_new':
            from uni_moe.model.speech_generator_AR_ori_v2_new.builder import build_ar_ori_v2_new_speech_generator
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
                # print("varsne:",vars(self.config))
                self.speech_generator = build_ar_ori_v2_new_speech_generator(self.config)
    

    def initialize_image_generator(self, model_args):
        self.tune_image_generator = getattr(model_args, 'tune_image_generator', False)
        if getattr(self, "image_generator", None) is None:
            self.image_generator = image_generator() 

    def extract_tts_pos(self,input_ids):
        # <sosp> 32000 <eop> 32001 <eosp> 32002
        sosp_pos = []
        eop_pos = []
        eosp_pos = []
        #print(self.config.speech_start_token_id,self.config.speech_prompt_token_id,self.config.speech_end_token_id)
        for batch_input_ids in input_ids:
            sosp_pos.append(torch.where(batch_input_ids==self.config.speech_start_token_id)[0]) # to change --llama: 32000 32001 32002
            eop_pos.append(torch.where(batch_input_ids==self.config.speech_prompt_token_id)[0])
            eosp_pos.append(torch.where(batch_input_ids==self.config.speech_end_token_id)[0])
        return sosp_pos, eop_pos, eosp_pos

    def pad_hidden(self,hiddens,ref_hidden):
        # pad all
        hidden_lens = [int(x.shape[0]) if x != None else 0 for x in hiddens]
        max_len = max(hidden_lens)

        new_hiddens_align = []
        for cur_hidden in hiddens:
            if cur_hidden != None:
                cur_hidden = torch.cat((cur_hidden, torch.zeros((max_len - cur_hidden.shape[0], cur_hidden.shape[-1]), dtype=cur_hidden.dtype, device=cur_hidden.device)), dim=0)
            else:
                cur_hidden = torch.zeros((max_len, ref_hidden.shape[-1]), dtype=ref_hidden.dtype, device=ref_hidden.device)
            new_hiddens_align.append(cur_hidden)
        new_hiddens = torch.stack(new_hiddens_align, dim=0)
        return new_hiddens, hidden_lens
    
    def extract_speech_ids(self,sosp_pos, eop_pos, eosp_pos, speech_splits, input_ids, codes, split_tokens):
        all_prompt = []
        all_prefix = []
        all_text = []
        code_pos = 0
        for b_sosp_pos, b_eop_pos, b_eosp_pos, b_speech_splits, b_input_ids in zip(sosp_pos, eop_pos, eosp_pos, speech_splits, input_ids):
            assert len(b_sosp_pos) == len(b_eop_pos) == len(b_eosp_pos)
            speech_idx = 0
            split_idx = 0
            for speech_idx in range(len(b_sosp_pos)):
                prompt = b_input_ids[b_sosp_pos[speech_idx]+1:b_eop_pos[speech_idx]]
                split_chunk = []
                split_pos = []
                while len(b_speech_splits)>0 and split_idx<len(b_speech_splits) and b_speech_splits[split_idx] > b_eop_pos[speech_idx] and b_speech_splits[split_idx] < b_eosp_pos[speech_idx]:
                    split_pos.append(b_speech_splits[split_idx])
                    split_idx+=1
                    if split_idx>=len(b_speech_splits):
                        break
                # first cut
                all_prefix.append(None)
                all_prompt.append(prompt)
                code_pos+=1
                if len(split_pos):
                    all_text.append(b_input_ids[b_eop_pos[speech_idx]+1:split_pos[0]])
                else:
                    all_text.append(b_input_ids[b_eop_pos[speech_idx]+1:b_eosp_pos[speech_idx]])
                # other cuts
                for cut_idx,_ in enumerate(split_pos):
                    all_prompt.append(prompt)
                    if codes != None:
                        last_code_pos = 0
                        for scid,sc in enumerate(codes[code_pos-1]):
                            if sc!= IGNORE_INDEX:
                                last_code_pos = scid
                        last_code_pos+=1
                        if last_code_pos-self.prefix_len>=0:
                            all_prefix.append(codes[code_pos-1][last_code_pos-self.prefix_len:last_code_pos])
                        else:
                            all_prefix.append(torch.cat([codes[code_pos-1][:last_code_pos],torch.LongTensor([IGNORE_INDEX]*(self.prefix_len-last_code_pos)).to(device=codes.device)],dim=0))
                    else:
                        all_prefix.append(None)
                    if cut_idx==len(split_pos)-1:
                        all_text.append(b_input_ids[split_pos[cut_idx]:b_eosp_pos[speech_idx]])
                    else:
                        all_text.append(b_input_ids[split_pos[cut_idx]:split_pos[cut_idx+1]])
                    code_pos+=1
        batch = {}
        tmpd = {}
        batch["prompt_ids"] = all_prompt
        prompt = torch.nn.utils.rnn.pad_sequence(all_prompt,
                                                batch_first=True,
                                                padding_value=0)
        prompt = prompt.to(device=input_ids.device)
        prompt_len = [int(x.shape[0]) if x != None else 0 for x in all_prompt]
        tmpd["prompt_lens"] = torch.LongTensor(prompt_len).to(device=prompt.device)
        batch["prompt"] = self.speech_generator.qwenvl_embed_tokens(prompt).to(device=prompt.device)
        batch["prompt_lens"] = torch.LongTensor(prompt_len).to(device=prompt.device)

        if codes != None and sum([pr!=None for pr in all_prefix]):
            batch["prefix"] = torch.stack([pref if pref != None else torch.LongTensor([IGNORE_INDEX]*self.prefix_len).to(device=codes.device) for pref in all_prefix],dim=0)
            batch["prefix_lens"] = torch.LongTensor([self.prefix_len if pref != None else 0 for pref in all_prefix]).to(device=codes.device)
            tmpd["prefix"] = batch["prefix"]
            tmpd["prefix_lens"] = batch["prefix_lens"]
        # pad text
        all_text = [torch.cat([sep_text[:-2],torch.LongTensor([13]).to(device=prompt.device)]) if sep_text[-1] == torch.LongTensor([291]).to(device=prompt.device) and sep_text[-1] == torch.LongTensor([5102]).to(device=prompt.device) else sep_text for sep_text in all_text]
        all_text = [sep_text if sep_text[-1] in split_tokens else torch.cat([sep_text,torch.LongTensor([13]).to(device=prompt.device)])  for sep_text in all_text]
        text = torch.nn.utils.rnn.pad_sequence(all_text,
                                                batch_first=True,
                                                padding_value=0)
        text_len = [int(x.shape[0]) if x != None else 0 for x in all_text]
        tmpd["x_lens"] = torch.LongTensor(text_len).to(device=text.device)
        batch["x"] = self.speech_generator.qwenvl_embed_tokens(text).to(device=text.device)
        batch["x_ids"] = all_text
        batch["x_lens"] = torch.LongTensor(text_len).to(device=text.device)

        return batch,tmpd

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        audio_grid_thw: Optional[torch.LongTensor] = None,
        speech_splits: Optional[torch.LongTensor] = None,
        codes: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        golden_task_embedding: Optional[torch.Tensor] = None,
        golden_caption_embedding: Optional[torch.Tensor] = None,
        golden_visual_embedding: Optional[torch.Tensor] = None,
        img_path: Optional[str] = None,
        tar_path: Optional[str] = None,
        aux_balance_weight: Optional[torch.LongTensor] = None,
        padding_token_mask: Optional[torch.Tensor] = None,
        output_router_logits_and_topk: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        lm_output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                audio_features=audio_features,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                audio_grid_thw=audio_grid_thw,
                rope_deltas=rope_deltas,
                aux_balance_weight=aux_balance_weight,
                padding_token_mask=padding_token_mask,
                output_router_logits_and_topk=output_router_logits_and_topk
        )
        loss = lm_output.loss

        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        #device = torch.device(f"cuda:{local_rank}")
        device = lm_output.hidden_states[0].device

        visual_hidden_embeddings = None
        if tar_path is not None and pixel_values is not None:
            pixel_values = pixel_values.type(self.model.dtype)
            image_embeds = self.vision_tower(pixel_values)
            image_embeds = self.vision_aligner(image_embeds)
            
            height = width = self.vision_tower.num_patches_per_side
            num_patches, num_tokens, num_dim = image_embeds.shape
            image_embeds = image_embeds.view(num_patches, height, width, -1)
            image_embeds = image_embeds.permute(0, 3, 1, 2).contiguous()
            
            scaled_shape = (self.config.vision_spatial_pool_stride, self.config.vision_spatial_pool_stride)
            adaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(scaled_shape)   
            image_embeds = adaptiveAvgPool2d(image_embeds)

            image_embeds = image_embeds.permute(0, 2, 3, 1)
            image_embeds = image_embeds.view(num_patches, -1, num_dim).reshape(-1, num_dim)
            
            for idx, visual_fc_layer in zip([-1], self.visual_hidden_fcs):
                visual_hidden_embeddings = visual_fc_layer(image_embeds.unsqueeze(0)).to(device)

        if tar_path is not None:
            # last_hidden = lm_output['hidden_states'][-1]
            # last_hidden = last_hidden.detach()
            # print("last_hidden detached")

            img_token_ids = [151672 + i for i in range(32)]
            task_token_ids = [151704, 151705, 151706]

            text_emb_layers = [-1]  # the layer index of LLM hidden states
            img_hidden_fcs = self.image_hidden_fcs.to(device)  # alignment modules for image tokens
            task_hidden_fcs = self.task_hidden_fcs.to(device)  # alignment modules for task tokens
            img_start_pos = (labels == img_token_ids[0]).nonzero(as_tuple=False)[:, 1][:1].tolist()
            img_end_pos = (labels == img_token_ids[-1]).nonzero(as_tuple=False)[:, 1][-1:].tolist()
            task_start_pos = (labels == task_token_ids[0]).nonzero(as_tuple=False)[:, 1].tolist()
            task_end_pos = (labels == task_token_ids[-1]).nonzero(as_tuple=False)[:, 1].tolist()
            #print(labels)
            #print(img_start_pos,img_end_pos,task_start_pos,task_end_pos)

            for idx, img_fc_layer, task_fc_layer in zip(text_emb_layers, img_hidden_fcs, task_hidden_fcs):
                hidden_embedding = []
                caption_hidden_embedding = []
                for b, (s, e) in enumerate(zip(img_start_pos, img_end_pos)):
                    assert e - s + 1 == 64, (s, e)
                    hidden_embedding.append(lm_output.hidden_states[idx][b, s:e + 1, :])
                hidden_embedding = torch.stack(hidden_embedding, dim=0).to(device)
                caption_hidden_embedding.append(img_fc_layer(hidden_embedding).to(device))

                hidden_embedding = []
                task_hidden_embedding = []
                for b, (s, e) in enumerate(zip(task_start_pos, task_end_pos)):
                    assert e - s + 1 == 3, (s, e)
                    hidden_embedding.append(lm_output.hidden_states[idx][b, s:e + 1, :])
                hidden_embedding = torch.stack(hidden_embedding, dim=0).to(device)
                task_hidden_embedding.append(task_fc_layer(hidden_embedding).to(device))
        
            # if hasattr(self.config, "tune_image_generator_stage"):
            latent_loss = self.image_generator.forward(img_path, tar_path, prompt_feats=caption_hidden_embedding[0].to(device), prompt_pooled_feats=task_hidden_embedding[0].to(device), input_clip_x=visual_hidden_embeddings.to(device))
            print(f"loss: {loss.item()}, latent_loss: {latent_loss.to(device).item()}")
            loss = loss + latent_loss.to(loss.device)


        return MoEQwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=lm_output.logits,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
            rope_deltas=rope_deltas,
        )
    
    def select_split(self, out_seq, split_tokens, number_tokens,maxcutlen):
        minlen = 5
        maxlen = maxcutlen
        split_idx = []
        for b_os in out_seq:
            split_b = []
            now_prompt = -1
            now_start = -1
            for idx,i in enumerate(b_os):
                if i == self.config.speech_prompt_token_id: #151667:
                    now_prompt = idx
                    now_start = idx
                    now_end = now_start
                if i == self.config.speech_end_token_id: #151666:
                    now_prompt = -1
                    now_start = -1
                if now_prompt>0 and now_start>0:
                    if idx - now_start > maxlen and b_os[idx+1] not in number_tokens and b_os[idx+1] not in split_tokens: 
                        now_end=idx+1
                        split_b.append(now_end)
                        now_start = now_end
                    if i in split_tokens and b_os[idx+1] not in number_tokens and b_os[idx+1] != self.config.speech_end_token_id:
                        now_end=idx+1
                        if now_end - now_start > minlen:
                            split_b.append(now_end)
                            now_start = now_end
            split_bt = torch.LongTensor(split_b)
            split_bt = split_bt.to(device=out_seq.device)
            split_idx.append(split_bt)
        return split_idx

    @torch.no_grad()
    def generate_from_tokens(
        self,
        out_seq: Optional[torch.Tensor] = None,
        split_tokens = None,
        number_tokens = None,
        maxtoklen = None,
        maxcutlen = None,
        **kwargs,
    ):
        o_device = out_seq.device
        if out_seq.shape[-1]>=maxtoklen and int(self.config.speech_end_token_id) not in out_seq[0].tolist():
            print("too long cut...")
            out_seq = out_seq[0].tolist()
            bad = out_seq[-1]
            i = len(out_seq)-1
            while out_seq[i]==bad:
                i-=1
            out_seq = out_seq[:i+2]+[13, int(self.config.speech_end_token_id), 151645]
            out_seq = torch.LongTensor([out_seq]).to(device=o_device)

        sosp_pos, eop_pos, eosp_pos = self.extract_tts_pos(out_seq)
        split_idx = self.select_split(out_seq,split_tokens,number_tokens,maxcutlen)
        batch,tmpd = self.extract_speech_ids(sosp_pos, eop_pos, eosp_pos, split_idx, out_seq, None, split_tokens)
        tmpd['prompt'] = batch["prompt"]
        tmpd['x'] = batch["x"]
        tmpd['prompt_ids'] = batch["prompt_ids"]
        tmpd['x_ids'] = batch["x_ids"]
        ar_preds = []
        tmp_pred = None
        for i in range(tmpd["x"].shape[0]):
            tmp_pred = self.speech_generator.infer(tmpd["x_ids"][i].unsqueeze(0), tmpd["prompt_ids"][i].unsqueeze(0), None, top_k=1, penalty_window_size=-1, penalty=1.0)
            ar_preds.append(tmp_pred)
        
        return ar_preds,split_idx

    @torch.no_grad()
    def generate_visualgen(
        self,
        model_path: str = None,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        outputs = self.generate(
            input_ids = input_ids,
            pixel_values = pixel_values,
            image_grid_thw = image_grid_thw,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample = False,
            num_beams = 1,
            temperature = 0,
            max_new_tokens=512,
            use_cache=True
        )
        # odict_keys(['sequences', 'hidden_states', 'past_key_values'])
        print("out_text:",outputs['sequences'])

        visual_hidden_embeddings = None
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.model.dtype)
            image_embeds = self.vision_tower(pixel_values)
            image_embeds = self.vision_aligner(image_embeds)
            
            height = width = self.vision_tower.num_patches_per_side
            num_patches, num_tokens, num_dim = image_embeds.shape
            image_embeds = image_embeds.view(num_patches, height, width, -1)
            image_embeds = image_embeds.permute(0, 3, 1, 2).contiguous()
            
            scaled_shape = (self.config.vision_spatial_pool_stride, self.config.vision_spatial_pool_stride)
            adaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(scaled_shape)   
            image_embeds = adaptiveAvgPool2d(image_embeds)

            image_embeds = image_embeds.permute(0, 2, 3, 1)
            image_embeds = image_embeds.view(num_patches, -1, num_dim).reshape(-1, num_dim)

            for idx, visual_fc_layer in zip([-1], self.visual_hidden_fcs):
                visual_hidden_embeddings = visual_fc_layer(image_embeds.unsqueeze(0))
            # print(image_embeds.shape, visual_hidden_embeddings.shape)

        hidden_states = outputs['hidden_states']
        generated_ids = outputs.sequences
        # print(hidden_states[0][-1].shape,hidden_states[1][-1].shape,outputs.sequences)
        
        output_embeddings = []
        for _hidden_states in outputs.hidden_states[1:]:
            for idx in [-1]:
                output_embeddings.append(_hidden_states[idx])
        output_embeddings = torch.cat(output_embeddings, dim=1) #input_ids 36 output_embeddings 62 generate_ids 99

        img_idx = [i for i, x in enumerate(generated_ids[0, :] == 151672) if x] #extract [IMG0] from generated_ids
        task_idx = [i for i, x in enumerate(generated_ids[0, :] == 151704) if x] #extract [TASK0] from generated_ids

        if len(img_idx) > 0 or len(task_idx) > 0:
            img_embeddings = output_embeddings[:, -64:, :]
            task_embeddings = output_embeddings[:, -67:-64, :]
            aligned_img_embeddings = self.image_hidden_fcs[0](img_embeddings)
            aligned_task_embeddings = self.task_hidden_fcs[0](task_embeddings).squeeze(0)


            img_path = kwargs.pop("image_path", None)
            if img_path is None:
                img_path = "examples/assets/visual_gen/input_images/white.png"
            model_path = kwargs.pop("model_path", None)

            #self.image_generator = image_generator()
            save_path = kwargs.pop("save_path", None)
            if save_path is None:
                save_path = self.image_generator.image_save_path

            golden_caption_emb = kwargs.pop("golden_caption_emb", None)
            if golden_caption_emb is not None:
                print(aligned_img_embeddings.shape)
                aligned_img_embeddings = golden_caption_emb
                print(aligned_img_embeddings.shape)
            golden_task_emb = kwargs.pop("golden_task_emb", None)
            if golden_task_emb is not None:
                print(aligned_task_embeddings.shape)
                aligned_task_embeddings = golden_task_emb
                print(aligned_task_embeddings.shape)
            golden_visual_emb = kwargs.pop("golden_visual_emb", None)
            if golden_visual_emb is not None:
                print(visual_hidden_embeddings.shape)
                visual_hidden_embeddings = golden_visual_emb
                print(visual_hidden_embeddings.shape)

            print(f"img_path: {img_path}")

            self.image_generator.generate_image(
                img_path=img_path,
                prompt_feats=aligned_img_embeddings,
                prompt_pooled_feats=aligned_task_embeddings,
                input_clip_x=visual_hidden_embeddings,
                save_path=save_path,              
            )
        
        return outputs['sequences']


AutoConfig.register("grin_qwen2_vl_out", GrinQwen2VLOutConfig)
AutoModelForCausalLM.register(GrinQwen2VLOutConfig, GrinQwen2VLOutForConditionalGeneration)
