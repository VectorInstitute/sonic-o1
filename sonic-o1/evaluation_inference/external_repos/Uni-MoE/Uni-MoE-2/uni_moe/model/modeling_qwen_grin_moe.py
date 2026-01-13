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

# Standard library imports
import os
import re
import pdb
import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm
from torch import Tensor
from safetensors.torch import load_file
import matplotlib.pyplot as plt

# DeepSpeed imports
import deepspeed
from deepspeed import comm as dist
from deepspeed.moe.mappings import drop_tokens, gather_tokens
from deepspeed.moe.sharded_moe import FIRST_ALLTOALL_TIMER, MOE_TIMER, SECOND_ALLTOALL_TIMER, _AllToAll, einsum, gumbel_rsample
from deepspeed.utils import groups, log_dist
from deepspeed.utils.timer import SynchronizedWallClockTimer

# Transformers imports
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

# Qwen2VL specific imports
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    QWEN2_VL_ATTENTION_CLASSES,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2VLRotaryEmbedding,
)

# Flash attention (conditional import)
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None

# Local imports
from uni_moe.model.visual_encoder.builder import build_vision_tower
from uni_moe.model.visual_projector.builder import build_vision_aligner
from uni_moe.model.visual_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionModel
from transformers import WhisperConfig
from uni_moe.model.modeling_audio_module import build_whisper_tower, build_whisper_aligner
from uni_moe.model.generation import GenerationUni
from uni_moe.model.grinmoe_utils import compress_matrix, decompress_matrix
from uni_moe.utils import rank0_print
from uni_moe.model.visual_gen_projector.layers import TextFcLayer
from uni_moe.model.image_generator.image_generator import image_generator

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen2VLConfig"
IGNORE_INDEX = -100

"""
copy from qwen2_vl of transformer 4.45.1
相比GrinQwenVL 添加share expert和动态moe
相比GrinQwenVl v2, 添加token drop, shared expert size指定, null expert, token-level的aux loss weight
相比GrinQwenVl v3, 主要改了aux balance loss, 现在在路由函数直接得到, 并且尝试解决shared expert的token_per_expert权重过大的情况
相比GrinQwenVL v4, 将MLP改成了Deepspeed MOE类型, 方便专家并行, 修改了很多deepspeed moe的bug, 
包括token drop和aux loss没有考虑padding token影响, gumbel noise稳定性, 非token drop时候的数据长度不一致导致all reduce问题
实验性的修改了sparsemixer的factor策略
"""

logger = logging.get_logger(__name__)

FAST_INIT = True
if FAST_INIT:
    logger.warning(f"using FAST initial for Grin Qwen2_vl !!!")


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

class GrinQwen2VLConfig(Qwen2VLConfig):
    model_type = "grin_qwen2_vl"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"vision_config": SigLipVisionConfig, "audio_config": WhisperConfig}
    def __init__(
        self,
        vision_config=None,
        audio_config=None,
        mlp_dynamic_expert_num=4,
        mlp_dynamic_null_expert_num=0,
        mlp_dynamic_top_p=0.7,
        mlp_dynamic_top_k=2,
        mlp_fixed_expert_num=2,
        dynamic_intermediate_size=8960,
        shared_intermediate_size=8960,
        # grin moe router
        ignore_differentiable_router=False,
        # deepspeed moe ep
        enable_expert_tensor_parallelism: bool = False,
        ep_size=1,
        fixed_ep_size=1,
        # jitter_noise
        router_jitter_noise=0.01,
        input_jitter_noise=0.01,
        # token drop
        token_drop=False,
        drop_policy: str = "probs",  # probs, position
        min_capacity: int = 8,
        capacity_factor: float = 1.0,
        # others
        fp32_gate=True,
        avg_hidden_states_last=False,
        drop_token_num_print=True,
        # training args, it should not be set here by the way
        l_aux_weight=0,
        min_l_aux_weight=0,
        l_aux_weight_decay_steps=1,
        **kwargs,
    ):
        
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        if isinstance(audio_config, dict):
            self.audio_config = self.sub_configs["audio_config"](**audio_config)
        elif audio_config is None:
            self.audio_config = self.sub_configs["audio_config"]()
        
        self.mlp_dynamic_expert_num = mlp_dynamic_expert_num
        self.mlp_dynamic_top_p = mlp_dynamic_top_p
        self.mlp_dynamic_top_k = mlp_dynamic_top_k
        self.mlp_fixed_expert_num = mlp_fixed_expert_num
        self.mlp_dynamic_null_expert_num = mlp_dynamic_null_expert_num

        self.dynamic_intermediate_size = dynamic_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size

        self.ignore_differentiable_router = ignore_differentiable_router

        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        self.ep_size = ep_size
        self.fixed_ep_size = fixed_ep_size

        self.input_jitter_noise = input_jitter_noise
        self.router_jitter_noise = router_jitter_noise

        self.token_drop = token_drop
        self.drop_policy = drop_policy
        self.min_capacity = min_capacity
        self.capacity_factor = capacity_factor

        self.fp32_gate = fp32_gate
        self.avg_hidden_states_last = avg_hidden_states_last
        self.drop_token_num_print = drop_token_num_print

        self.l_aux_weight = l_aux_weight
        self.min_l_aux_weight = min_l_aux_weight
        self.l_aux_weight_decay_steps = l_aux_weight_decay_steps

        super().__init__(**kwargs)


@dataclass
class MoEQwen2VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    all_router_logits: Tuple = None
    all_router_top_k: Tuple = None
    all_router_expert_mask: Tuple = None
    all_router_weight: Tuple = None
    aux_balance_loss: torch.FloatTensor = None


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_router_logits: Tuple = None
    all_router_top_k: Tuple = None
    all_router_weight: Tuple = None
    all_router_expert_mask: Tuple = None
    all_aux_loss: Tuple = None


class SharedExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.shared_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class DynamicExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.dynamic_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class NULLExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_state):
        # return hidden_state * 0
        return torch.zeros_like(hidden_state, dtype=hidden_state.dtype, device=hidden_state.device)


#
# ---------------------------- copy from Grin ----------------------------
#


class mp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        multiplier: torch.Tensor,
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one

    @staticmethod
    def backward(
        ctx,
        grad_at_output: torch.Tensor,
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors

        grad_at_output = grad_at_output * multiplier

        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )

        return (
            grad_at_scores_expaned,
            None,
            None,
            None,
            None,
        )


def sparsemixer(scores, top_k, jitter_eps, training):
    masked_scores = scores
    multiplier_list = []
    selected_experts_list = []

    for _ in range(top_k):
        with torch.no_grad():
            # compute mask for sparsity
            mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
            # factor = scores.abs().clamp(min=mask_logits_threshold)
            factor = scores.abs().clamp(min=mask_logits_threshold.abs())  # Todo, 这里我主动加了一个.abs() 我认为factor这样才算比较好的尺度
            mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)  # jitter_noise

        # apply mask
        masked_gates = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))

        if training:
            # 在训练期间使用 Gumbel 采样替代传统的 Top-k 选择，确保梯度的稳健性和随机性
            # Gumbel noise 使用 exponent() 和 log() 函数生成，用于从 masked_gates 中选择专家
            # V5 tip: 这里坑死了, 源代码的exponent和log生成会很大概率生成-inf, new_masked_gates全为-inf, 这里直接用了deepspeed的gumbel_rsample, Todo: 有时间再检查一下gumbel_rsample的原理
            noise = gumbel_rsample(masked_gates.shape, device=masked_gates.device)
            assert not torch.isnan(noise).any()
            new_masked_gates = masked_gates + noise
            selected_experts = (new_masked_gates).max(dim=-1)[1].unsqueeze(-1)  # gumbel sampling, more robust than than the multinomial method
        else:
            selected_experts = max_ind

        # compute scores for gradients
        masked_gates = torch.softmax(masked_gates, dim=-1)
        multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

        if training:
            # compute midpoint mask
            max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
            mask_for_one = torch.logical_or(
                selected_experts == max_ind,
                torch.rand_like(max_scores) > 0.75,  # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
            )
            # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
            mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

            multiplier = mp.apply(
                scores,
                multiplier_o,
                selected_experts,
                masked_gates,
                mask_for_one,
            )
        else:
            multiplier = multiplier_o

        # masked out first expert
        masked_scores = torch.scatter(
            masked_scores,
            -1,
            selected_experts,
            float("-inf"),
        )

        multiplier_list.append(multiplier)
        selected_experts_list.append(selected_experts)

    multiplier = torch.concat(multiplier_list, dim=-1)
    selected_experts = torch.concat(selected_experts_list, dim=-1)

    # print(multiplier.shape)
    # if not ((multiplier != 0).sum(1) == top_k).all():
    #     iidx = ((multiplier != 0).sum(0) != top_k).nonzero().tolist()[:3]
    #     raise ValueError(f"top_k: {top_k} iidx: {iidx}\n multiplier: {[multiplier[xx[0]] for xx in iidx]}")

    # if torch.any(torch.sum(selected_experts.unsqueeze(-1) == selected_experts.unsqueeze(-2), dim=-1) > 1):
    #     iidx = (torch.sum(selected_experts.unsqueeze(-1) == selected_experts.unsqueeze(-2), dim=-1) > 1).nonzero().tolist()
    #     raise AssertionError(f"iidx: {iidx}, selected_experts: {[selected_experts[xx[0]] for xx in iidx]}, masked_scores: {[masked_scores[xx[0]] for xx in iidx]}")

    return (
        multiplier,
        selected_experts,
    )


def dynamic_expert_selection(logits, top_p):
    # logits (batch * sequence_length, dynamic_expert_num)
    dynamic_scores = torch.softmax(logits, dim=-1)
    dynamic_scores_sorted, _ = torch.sort(dynamic_scores, dim=-1, descending=True)
    dynamic_scores_cumsum = dynamic_scores_sorted.cumsum(dim=-1)
    dynamic_top_k = (~(dynamic_scores_cumsum >= top_p)).sum(dim=-1)
    dynamic_top_k = dynamic_top_k + 1
    return dynamic_top_k


def _capacity(num_tokens, num_experts, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


def cal_global_weight(
    expert_mask: torch.Tensor,
    full_router_logits: torch.Tensor,
    mlp_dynamic_expert_num: int,
    routing_weights: torch.Tensor,
):
    global_weight = torch.softmax(full_router_logits.masked_fill(expert_mask == 0, float("-inf")), dim=-1)
    # Todo: 这里的global weight和routing_weights 有多个融合方法
    # 1. 用scaling factor * routing_weights, 然后拼接上 fix_routing_weight (当前采用)
    # 2. 指定dynamic和fix的权重分配, 然后根据routing_weights和fix_routing_weight各自分配, 这个不太好
    global_dynamic_weight = global_weight[:, :mlp_dynamic_expert_num]
    global_fixed_weight = global_weight[:, mlp_dynamic_expert_num:]
    global_dynamic_weight = routing_weights * global_dynamic_weight.sum(-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1])  # 计算dynamic的weight缩放因数
    global_weight = torch.cat((global_dynamic_weight, global_fixed_weight), dim=-1)
    return global_weight


class GRINMoESparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.mlp_dynamic_expert_num = config.mlp_dynamic_expert_num + config.mlp_dynamic_null_expert_num
        self.mlp_dynamic_real_expert_num = config.mlp_dynamic_expert_num
        self.mlp_dynamic_null_expert_num = config.mlp_dynamic_null_expert_num
        self.mlp_dynamic_top_p = config.mlp_dynamic_top_p
        self.mlp_dynamic_top_k = config.mlp_dynamic_top_k
        self.mlp_fixed_expert_num = config.mlp_fixed_expert_num
        self.num_experts = self.mlp_dynamic_expert_num + self.mlp_fixed_expert_num

        if self.mlp_dynamic_top_p == 0:
            print(f"mlp_dynamic_top_p is 0, will use mlp_dynamic_top_k={self.mlp_dynamic_top_k} instead !!!")

        self.ignore_differentiable_router = config.ignore_differentiable_router
        if self.ignore_differentiable_router:
            print("ignore_differentiable_router is True, will not use router_logits !!!")

        # gating & experts
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        # self.dynamic_null_experts = nn.ModuleList([NULLExpertMLP(config) for _ in range(self.mlp_dynamic_null_expert_num)]) # 没有用的专家
        self.fixed_real_moe = nn.ModuleList([SharedExpertMLP(config) for _ in range(self.mlp_fixed_expert_num)])
        # deepspeed moe for dynamic real expert
        self.dynamic_real_moe = MoE(config, DynamicExpertMLP(config), self.mlp_dynamic_real_expert_num, config.ep_size)

        # Jitter parameters
        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise

        self.min_capacity = config.min_capacity
        self.capacity_factor = config.capacity_factor
        self.token_drop = config.token_drop
        self.drop_policy = config.drop_policy

        self.avg_hidden_states_last = config.avg_hidden_states_last
        self.drop_token_num_print = config.drop_token_num_print
        self.fp32_gate = config.fp32_gate

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, aux_balance_weight: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        original_hidden_states = hidden_states

        if self.training and self.fp32_gate:
            hidden_states = hidden_states.float()

        # input jitter_noise
        # Grin moe和deepspeed moe都会添加jitter_noise
        
        assert torch.isnan(hidden_states).sum() == 0, f"before hidden_states has nan: {hidden_states}"
        assert torch.isinf(hidden_states).sum() == 0, f"before hidden_states has inf: {hidden_states}"
        
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise)

        hidden_states = hidden_states.view(-1, hidden_dim)

        # full_router_logits: (batch * sequence_length, n_experts)
        assert torch.isnan(hidden_states).sum() == 0, f"hidden_states has nan: {hidden_states}"
        assert torch.isinf(hidden_states).sum() == 0, f"hidden_states has inf: {hidden_states}"
        
        if self.training and self.fp32_gate:
            full_router_logits = torch.nn.functional.linear(hidden_states, weight=self.gate.weight.float(), bias=None)
        else:
            full_router_logits = self.gate(hidden_states)
            
        assert torch.isnan(full_router_logits).sum() == 0, f"full_router_logits has nan: {full_router_logits}"
        assert torch.isinf(full_router_logits).sum() == 0, f"full_router_logits has inf: {full_router_logits}"
        
        
        dynamic_router_logits = full_router_logits[:, : self.mlp_dynamic_expert_num]

        # 获得动态top_k
        if self.mlp_dynamic_top_p != 0:
            dynamic_top_k = dynamic_expert_selection(dynamic_router_logits, self.mlp_dynamic_top_p)
        else:
            dynamic_top_k = torch.full((dynamic_router_logits.shape[0],), self.mlp_dynamic_top_k, dtype=torch.int, device=dynamic_router_logits.device)

        # expert_mask:moe的路由情况 (batch * sequence_length, expert_num)
        expert_mask = torch.zeros((batch_size * sequence_length, self.num_experts), dtype=torch.int, device=hidden_states.device)

        #
        # ---------------- dynamic top_p experts ----------------
        #

        # 用来存grin路由的sparsemixer的返回值, 即group_routing_weights
        # Todo: group_routing_weights的sum和大概率和top k相同, 这样导致hidden state的权重和不等于1, 需要思考下
        routing_weights = torch.zeros((batch_size * sequence_length, self.mlp_dynamic_expert_num), dtype=hidden_states.dtype, device=hidden_states.device)
        for top_k in range(1, self.mlp_dynamic_expert_num + 1):
            # 获得当前top_k路由模式的token 位置
            group_idx = torch.nonzero(dynamic_top_k == top_k, as_tuple=True)[0]
            if len(group_idx) == 0:
                continue

            dynamic_group_logits = dynamic_router_logits[group_idx]
            # group_selected_experts: (group_batch_size, top_k), 按照优先级排序
            group_routing_weights, group_selected_experts = sparsemixer(
                dynamic_group_logits,
                top_k=top_k,
                jitter_eps=self.router_jitter_noise,
                training=self.training and not self.ignore_differentiable_router,
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            group_expert_mask = torch.nn.functional.one_hot(group_selected_experts, num_classes=self.num_experts)
            group_expert_mask = group_expert_mask.sum(dim=1)

            group_weight = torch.zeros((len(group_idx), self.mlp_dynamic_expert_num), dtype=hidden_states.dtype, device=hidden_states.device)
            group_weight.scatter_(dim=-1, index=group_selected_experts, src=group_routing_weights)
            routing_weights.index_add_(0, group_idx, group_weight)

            # 更新expert_mask
            # 设定0~self.mlp_dynamic_expert_num 为动态专家
            expert_mask.index_add_(0, group_idx, group_expert_mask.to(expert_mask.dtype))

        # 目前决定还是将routing_weights先求和归一化 # Todo: 看看效果
        routing_weights = routing_weights / (routing_weights.sum(dim=-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1]) + 1e-6)

        #
        # ---------------- attention mask ----------------
        #

        if attention_mask is not None:
            # [fix] padding token 的路由会影响aux balance loss的计算, 并且还会占capacity, 这里强制设置为0
            # 看了一下, aux balance loss 在设置aux balance loss的时候融合了attention mask (在GrinQwen2VLForConditionalGeneration的forward中), 这时候不会影响
            # 但是会影响capacity, 这里load_balancing_loss_func 传入capacity_expert_mask可能会导致padding经过softmax后为nan
            assert len(attention_mask.shape) == 2, f"{attention_mask.shape}"  # B, L
            attention_mask = attention_mask.to(expert_mask.dtype).view(-1).unsqueeze(-1).expand(-1, self.num_experts)
            # print(attention_mask.size())
            # print(expert_mask.size())
            # print(hidden_states.size())
            # expert_mask = expert_mask * attention_mask
            expert_mask = expert_mask * attention_mask[-hidden_states.shape[0]:, :]

        #
        # ---------------- fixed top_p experts ----------------
        #

        if self.mlp_dynamic_expert_num < self.num_experts:
            expert_mask[:, self.mlp_dynamic_expert_num :] = 1  # 只需要把expert mask 设为1即可

        #
        # ---------------- aux balance loss ----------------
        #

        # 这里选择在token drop前计算aux balance loss, 更加精确, 但是要重新算一下global weight
        aux_loss = load_balancing_loss_func(
            expert_mask=expert_mask,
            mlp_dynamic_expert_num=self.mlp_dynamic_expert_num,
            global_weight=None,
            full_router_logits=full_router_logits,
            routing_weights=routing_weights,
            aux_balance_weight=aux_balance_weight,
        )
        
        #
        # ---------------- token drop ----------------
        #

        if self.token_drop:  # and self.training:
            expert_mask_dtype = expert_mask.dtype
            capacity = _capacity(batch_size * sequence_length, self.mlp_dynamic_expert_num, torch.tensor(self.capacity_factor), torch.tensor(self.min_capacity))
            if self.drop_policy == "probs":
                if capacity > dynamic_router_logits.shape[0]:
                    print(f"[warning] token capacity({capacity}) > token num({dynamic_router_logits.shape[0]}), setting capacity=token num")
                    capacity = dynamic_router_logits.shape[0]
                dynamic_expert_mask = expert_mask[:, : self.mlp_dynamic_expert_num].bool()
                token_drop_router_logits = torch.masked_fill(dynamic_router_logits, ~dynamic_expert_mask, torch.finfo(dynamic_router_logits.dtype).min)
                capacity_probs, capacity_indices = torch.topk(token_drop_router_logits, k=capacity, dim=0, sorted=False)
                capacity_mask = torch.zeros_like(expert_mask).scatter(0, capacity_indices, 1)
                capacity_mask[:, self.mlp_dynamic_expert_num :] = 1
                expert_mask = torch.logical_and(expert_mask, capacity_mask)

                ori_token_num = dynamic_expert_mask.sum().item()
                cur_token_num = expert_mask[:, : self.mlp_dynamic_expert_num].sum().item()
                # if self.drop_token_num_print and ("RANK" not in os.environ or int(os.environ["RANK"]) == 0):
                    # print(f"drop {ori_token_num - cur_token_num} tokens from total {ori_token_num} tokens")

            elif self.drop_policy == "position":
                locations = torch.cumsum(expert_mask, dim=0) - 1
                expert_mask *= torch.lt(locations, capacity)
            else:
                raise ValueError(f"Invalid drop_policy: {self.drop_policy}")
            expert_mask = expert_mask.to(expert_mask_dtype)

            # V5 tips: 这里修改了expert mask, 所以routing_weights有些有权重的地方需要重新mask, 然后重新unify一次, 相对比例不变
            routing_weights = routing_weights.masked_fill(~(expert_mask[:, : self.mlp_dynamic_expert_num].bool()), 0.0)
            routing_weights = routing_weights / (routing_weights.sum(dim=-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1]) + 1e-6)

        # global_weight 存储全局权重分配
        if self.mlp_dynamic_expert_num < self.num_experts:
            global_weight = cal_global_weight(expert_mask, full_router_logits, self.mlp_dynamic_expert_num, routing_weights)
        else:
            global_weight = routing_weights

        # 懒得看之前的代码了, 这里放一个debug
        if not ((expert_mask == 0) & (attention_mask == 1) & (global_weight != 0)).sum() == 0:
            iindex = ((expert_mask == 0) & (attention_mask == 1) & (global_weight != 0)).nonzero().tolist()[:1]
            xxx  = (expert_mask == 0) & (attention_mask == 1) & (global_weight != 0)
            raise ValueError(f"iindex: {iindex}\nexpert_mask: {[expert_mask[xx[0]] for xx in iindex]}\nattention_mask: {[attention_mask[xx[0]] for xx in iindex]}\nglobal_weight: {[global_weight[xx[0]] for xx in iindex]}\nrouting_weights: {[routing_weights[xx[0]] for xx in iindex]}\nxxx: {[xxx[xx[0]] for xx in iindex]}\nfull_router_logits: {[full_router_logits[xx[0]] for xx in iindex]}")
        assert ((expert_mask == 0) & (attention_mask == 1) & (global_weight != 0)).sum() == 0
        # assert ((expert_mask != 0) & (global_weight <= 0)).sum() == 0 # global_weight有时候还会是0, 主要原因是routing_weights的不确定性
        # assert (global_weight.sum(dim=1) != 1).sum() == 0 # 精度差异

        #
        # ---------------- 路由计算结束, 开始过expert ----------------
        #

        hidden_states = original_hidden_states.view(-1, hidden_dim)

        #  final_hidden_states: moe最终输出的表示; expert_mask:moe的路由情况 (batch * sequence_length, expert_num)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        global_weight = global_weight.to(hidden_states.dtype)

        #
        # ---------------- 空专家 (旧代码) ----------------
        #

        # 空专家不需要专家并行, 对最终结果也没有影响, 这里直接跳过就好了-_-
        # for expert_idx in range(self.mlp_dynamic_real_expert_num, self.mlp_dynamic_expert_num):
        #     expert_layer = self.dynamic_null_experts[expert_idx - self.mlp_dynamic_real_expert_num]
        #     top_x = torch.nonzero(expert_mask[:, expert_idx], as_tuple=True)[0]

        #     if top_x.shape[0] == 0:
        #         continue

        #     # in torch it is faster to index using lists than torch tensors
        #     top_x_list = top_x.tolist()

        #     # Index the correct hidden states and compute the expert hidden state for
        #     # the current expert. We need to make sure to multiply the output hidden
        #     # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        #     current_state = hidden_states[top_x_list]
        #     current_hidden_states = expert_layer(current_state) * global_weight[top_x_list, None, expert_idx]

        #     # However `index_add_` only support torch tensors for indexing so we'll use
        #     # the `top_x` tensor here.
        #     final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        #
        # ---------------- 动态实专家 ----------------
        #

        current_hidden_states = self.dynamic_real_moe(hidden_states, expert_mask=expert_mask[:, : self.mlp_dynamic_real_expert_num], router_weight=global_weight[:, : self.mlp_dynamic_real_expert_num])
        
        # print(f"{os.environ['RANK']} 44444444444444444444")
        
        try: 
            assert torch.isnan(current_hidden_states).sum() == 0
        except AssertionError as e:
            print(f"current_hidden_states has nan or inf: {current_hidden_states} !!!!!!!!!!!!!!!!! ")
        
        final_hidden_states = final_hidden_states + current_hidden_states

        #
        # ---------------- 固定专家 ----------------
        #

        for expert_idx in range(self.mlp_fixed_expert_num):
            expert_layer = self.fixed_real_moe[expert_idx]

            current_state = hidden_states
            current_global_weight = global_weight[:, self.mlp_dynamic_expert_num + expert_idx].unsqueeze(-1)  # fixed expert的weight
            current_hidden_states = expert_layer(current_state) * current_global_weight

            final_hidden_states = final_hidden_states + current_hidden_states

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        # print(f"{os.environ['RANK']} 333333333333333333333333, {not self.training and self.avg_hidden_states_last}")
        if not self.training and self.avg_hidden_states_last:
            # Todo: 不知道为什么多卡之间会出现hidden state不同步的情况
            # 目前是在推理的时候发现的, 相同的数据在EP4的情况下, 第18层hidden state会出现第一次不同, 个人认为是精度和卡不同导致的小误差, 但是在之后的层数会被放大
            # https://docs.pytorch.org/docs/stable/distrisbuted.html#torch.distributed.ReduceOp
            # print(f"{os.environ['RANK']} 1111111111111111111")
            dist.all_reduce(final_hidden_states, op=dist.ReduceOp.AVG, group=self.dynamic_real_moe.deepspeed_moe.ep_group)
            # print(f"{os.environ['RANK']} 222222222222222222")

        return final_hidden_states, full_router_logits, dynamic_top_k, expert_mask, global_weight, aux_loss


def load_balancing_loss_func(
    expert_mask: torch.Tensor,
    mlp_dynamic_expert_num: int,
    global_weight: Optional[torch.Tensor] = None,
    full_router_logits: Optional[torch.Tensor] = None,
    routing_weights: Optional[torch.Tensor] = None,
    aux_balance_weight: Optional[torch.Tensor] = None,
    version=2,
) -> float:
    # Todo: top K的专家数目K会影响aux balance loss吗? aux balance loss会不会影响top P的路由的选择数目的情况?
    if version == 1:
        assert False
        # 这个函数认为前 mlp_dynamic_expert_num 是dynamic expert, 后面都是shared expert
        # 并且将固定计算所有token的路由, 尝试解决shared expert的tokens_per_expert固定为1的缺点 (solution flag)

        # 根据full_router_logits计算global weight
        if global_weight is None:
            global_weight = cal_global_weight(expert_mask, full_router_logits, mlp_dynamic_expert_num, routing_weights)

        num_experts = expert_mask.shape[1]
        if aux_balance_weight is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
            # solution flag
            tokens_per_expert[mlp_dynamic_expert_num:] = torch.mean(tokens_per_expert[:mlp_dynamic_expert_num])
            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(global_weight, dim=0)
        else:
            batch_size, sequence_length = aux_balance_weight.shape
            num_hidden_layers = global_weight.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = aux_balance_weight[None, :, :, None].expand((num_hidden_layers, batch_size, sequence_length, num_experts)).reshape(-1, num_experts).to(global_weight.device)

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
            # solution flag
            tokens_per_expert[mlp_dynamic_expert_num:] = torch.mean(tokens_per_expert[:mlp_dynamic_expert_num])

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(global_weight * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)

        return overall_loss * num_experts

    elif version == 2:
        # 这个函数认为前 mlp_dynamic_expert_num 是dynamic expert, 后面都是shared expert
        # 并且将固定计算dynamic_expert token的路由

        min_dtype = torch.finfo(full_router_logits.dtype).min  # 防止expert_mask 全0的时候出现nan, 所以不用-inf
        global_weight = full_router_logits.masked_fill(expert_mask == 0, min_dtype)
        global_weight = global_weight[:, :mlp_dynamic_expert_num]
        global_weight = torch.softmax(global_weight, dim=-1)
        expert_mask = expert_mask[:, :mlp_dynamic_expert_num]

        num_experts = expert_mask.shape[-1]
        if aux_balance_weight is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(global_weight, dim=0)
        else:
            batch_size, sequence_length = aux_balance_weight.shape
            num_hidden_layers = global_weight.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = aux_balance_weight[None, :, :, None].expand((num_hidden_layers, batch_size, sequence_length, num_experts)).reshape(-1, num_experts).to(global_weight.device)

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(global_weight * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)

        return overall_loss * num_experts

    else:
        raise KeyError


#
# ---------------------------- deepspeed moe ----------------------------
#


# 没有改
class Experts(deepspeed.moe.experts.Experts):
    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(deepspeed.moe.experts.Experts, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.deepspeed_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output


class MOELayer(deepspeed.moe.sharded_moe.MOELayer):
    def __init__(
        self,
        experts: nn.Module,
        ep_group_name,
        ep_size,
        num_local_experts: int,
        # use_tutel: bool = False # 强制False, 后面的逻辑都删了
    ) -> None:
        super(deepspeed.moe.sharded_moe.MOELayer, self).__init__()

        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

    # copy
    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def forward(self, hidden_states: Tensor, expert_mask: Tensor, router_weight: Tensor) -> Tensor:
        # hidden_states: [B * S, d_model]
        # expert_mask: [B * S, expert_num]
        # router_weight: [B * S, expert_num], 注意这里router_weight和不一定是0, 空专家和fix专家的没放上去

        router_weight = router_weight * expert_mask

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).start()

        # Implement Algorithm 2 from GShard paper.
        d_model = hidden_states.shape[-1]
        seq_len = hidden_states.shape[0]
        expert_num = expert_mask.shape[-1]

        # motified from deepspeed topK gating, 最长capacity的同步, 为了通讯能够shape对齐
        # Communicate across expert processes to pick the maximum capacity.
        capacity = expert_mask.sum(dim=0).max()
        # capacity = torch.tensor(8, device=capacity.device, dtype=capacity.dtype)
        # print("!!!!!!!!!!!!!!!!!!!")
        # print(f"capacity: {capacity}")
        if self.ep_group is not None:
            dist.all_reduce(capacity, op=dist.ReduceOp.MAX, group=self.ep_group)
        # print("*******************")
        # # 这里还没看, 但是暂时用不上, 应该和token drop相关
        # if groups._get_expert_model_parallel_world_size() == 1:
        #     # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
        #     # This is since we are going to activate drop_tokens() to drop duplicate tokens.
        #     tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
        #     new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)

        # # Todo 两次compress_matrix可加速
        compres_hidden_states = hidden_states.unsqueeze(1).expand(seq_len, expert_num, d_model)  # [B * S, expert_num, d_model]
        # print(f"compres_hidden_states: {compres_hidden_states.shape}, expert_mask: {expert_mask.shape}, capacity: {capacity}")
        compres_hidden_states = compress_matrix(compres_hidden_states, expert_mask, force_dim=capacity, allow_larger_dim=True)  # [C, expert_num, d_model]
        compres_expert_mask = compress_matrix(expert_mask, expert_mask, force_dim=capacity, allow_larger_dim=True)
        dispatched_input = einsum("ce,cem->ecm", compres_expert_mask, compres_hidden_states)

        # # 这里先不考虑capacity, 将所有序列的token都all reduce, 这样通讯量会变大, 后续改良,
        # # 此外, 即使没有路由的token的hidden_state也会作为全0向量输入到expert中, 输出的时候记得再过一遍 expert mask (不能保证expert的输入为0时输出一定为0)
        # dispatched_input = einsum("se,sm->esm", expert_mask, hidden_states)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).start()
        # print(f"{os.environ['RANK']} ckpt1 {self.ep_group}")

        # 不考虑 tensor-parallel 的情况: Todo
        assert deepspeed.utils.groups.mpu is None
        # if groups._get_expert_model_parallel_world_size() == 1:
        #     # If the non-expert is tensor-parallel, it will create
        #     # duplicate tokens on the tensor-parallel ranks.
        #     # Since our experts are not tensor-parallel, these duplicates
        #     # need to be dropped to ensure correctness.
        #     # this also doubles up as a communication optimization as we are
        #     # reducing the all-to-all communication volume.
        #     # raise NotImplementedError
        #     dispatched_input = drop_tokens(dispatched_input, dim=1)
        # print(f"ep_group: {self.ep_group}")
        # print(f"ep_group: {self.ep_group}")
        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        # print(f"{os.environ['RANK']} ckpt2, {self.wall_clock_breakdown}")
        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).stop()
            self.time_falltoall = self.timers(FIRST_ALLTOALL_TIMER).elapsed(reset=False)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        expert_output = self.experts(dispatched_input)

        # print(f"{os.environ['RANK']} ckpt2.1")
        
        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).start()
            
        # if self.ep_group is not None:
        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        # print(f"{os.environ['RANK']} ckpt2.2")
        
        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(SECOND_ALLTOALL_TIMER).elapsed(reset=False)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        # 不考虑 tensor-parallel 的情况
        # if groups._get_expert_model_parallel_world_size() == 1:
        #     # the dropped duplicate tokens need to be gathered on each
        #     # tensor parallel rank again for the tensor-parallel
        #     # non-expert of the next layer.
        #     # raise NotImplementedError
        #     expert_output = gather_tokens(expert_output, dim=1)

        # print(f"{os.environ['RANK']} ckpt2.3, {expert_output.transpose(0, 1).shape}, {expert_mask.shape} {expert_mask.max()} {expert_mask.min()} {expert_mask.sum()}")
        
        # 使用压缩 ecm -> sem
        expert_output = decompress_matrix(expert_output.transpose(0, 1), expert_mask, allow_larger_dim=True)  # [B * S, expert_num, d_model]
        
        # print(f"{os.environ['RANK']} ckpt2.3.1, {router_weight.shape}, {expert_output.shape}")
        
        combined_output = einsum("se,sem->sm", router_weight, expert_output)
        
        # print(f"{os.environ['RANK']} ckpt2.3.2, {router_weight.shape}, {expert_output.shape}")

        # # 这里router_weight在之前用点乘融合了expert_mask, 确保了输入为0的地方输出也为0了
        # combined_output = einsum("se,esm->sm", router_weight, expert_output)

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).stop()
            self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)

        # print("ckpt3")
        return combined_output


class MoE(deepspeed.moe.layer.MoE):
    def __init__(self, config, expert, num_experts, ep_size, moe_name_prefix="ep_size"):
        super(deepspeed.moe.layer.MoE, self).__init__()

        # self.use_residual = use_residual # 强制False, 后面的逻辑都删了

        self.enable_expert_tensor_parallelism = config.enable_expert_tensor_parallelism
        self.ep_size = ep_size
        self.num_experts = num_experts

        assert self.num_experts % self.ep_size == 0, f"Number of experts ({self.num_experts}) should be divisible by expert parallel size ({self.ep_size})"

        self.expert_group_name = f"{moe_name_prefix}_{self.ep_size}"
        self.num_local_experts = self.num_experts // self.ep_size

        log_dist(f"Creating MoE layer with num_experts: {self.num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}", [0])

        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = MOELayer(experts, self.expert_group_name, self.ep_size, self.num_local_experts)

    # copy
    def set_deepspeed_parallelism(self, use_data_before_expert_parallel_=False):
        self._create_process_groups(use_data_before_expert_parallel_=use_data_before_expert_parallel_)

    # copy
    def _create_process_groups(self, use_data_before_expert_parallel_=False):
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(self.ep_size, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(self.ep_size, mpu=groups.mpu, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.deepspeed_moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))

    def forward(self, *input_args, **input_kwargs):
        return self.deepspeed_moe(*input_args, **input_kwargs)


#
# ---------------------------- moe done ----------------------------
#


class GrinQwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; unexpected results may be encountered.")
        # self.self_attn = Qwen2VLAttention(config, layer_idx)
        self.self_attn = QWEN2_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        # MoE Flag
        self.mlp = GRINMoESparseMoeBlock(config)
        # self.mlp = Qwen2MLP(config)

        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        aux_balance_weight: Optional[torch.Tensor] = None,
        padding_token_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits_and_topk: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # print("moe start!!!!!")
        hidden_states, router_logits, router_top_k, router_expert_mask, router_weight, aux_loss = self.mlp(hidden_states, padding_token_mask, aux_balance_weight)
        # print("moe end!!!!!")
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # MoE Flag - return
        if output_router_logits_and_topk:
            outputs += (router_logits,)
            outputs += (router_top_k,)
        outputs += (router_expert_mask,)
        outputs += (router_weight,)
        outputs += (aux_loss,)

        return outputs



class GrinQwen2VLPreTrainedModel(PreTrainedModel):
    config_class = GrinQwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GrinQwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if FAST_INIT:
            """
            only initialize the gate linear since we will load state dict for other parameters
            """
            if isinstance(module, GRINMoESparseMoeBlock):
                module.gate.weight.data.normal_(mean=0.0, std=std)
                if module.gate.bias is not None:
                    module.gate.bias.data.zero_()
        else:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()


class GrinQwen2VLModel(GrinQwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GrinQwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        aux_balance_weight: Optional[torch.Tensor] = None,
        padding_token_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits_and_topk: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # MoE Flag
        all_router_logits = () if output_router_logits_and_topk else None
        all_router_top_k = () if output_router_logits_and_topk else None
        all_router_expert_mask = ()
        all_router_weight = ()
        all_aux_loss = ()
        next_decoder_cache = None

        for index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    aux_balance_weight,
                    padding_token_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits_and_topk,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    aux_balance_weight=aux_balance_weight,
                    padding_token_mask=padding_token_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits_and_topk=output_router_logits_and_topk,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                # print("decoder layer end !!!!!!!!")

            hidden_states = layer_outputs[0]
            # if torch.distributed.is_initialized():
            #     local_rank = torch.distributed.get_rank()
            #     print(f"local_rank: {local_rank} layer_index: {index} [EP4 DEBUG] HIDDEN_STATES: {hidden_states}")

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # MoE Flag - return
            if output_router_logits_and_topk:
                all_router_logits += (layer_outputs[-5],)
                all_router_top_k += (layer_outputs[-4],)
            all_router_expert_mask += (layer_outputs[-3],)
            all_router_weight += (layer_outputs[-2],)
            all_aux_loss += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # MoE Flag - return
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits, all_router_top_k, all_router_expert_mask, all_router_weight, all_aux_loss] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            all_router_logits=all_router_logits,
            all_router_top_k=all_router_top_k,
            all_router_expert_mask=all_router_expert_mask,
            all_router_weight=all_router_weight,
            all_aux_loss=all_aux_loss,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if self.config._attn_implementation == "sdpa" and attention_mask is not None and attention_mask.device.type == "cuda" and not output_attentions:
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class GrinQwen2VLForConditionalGeneration(GrinQwen2VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _supports_attention_backend = True
    def __init__(self, config):
        super().__init__(config)
        # visual part
        if config.vision_config:
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.vision_aligner = build_vision_aligner(config)
        # audio part
        self.audio_tower = None
        self.audio_aligner = None
        if config.audio_config:
            self.audio_tower = build_whisper_tower(config, delay_load=False)###  need change delay_load
            self.audio_aligner = build_whisper_aligner(config)
        
        if config.tune_image_generator:
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
            
        
        self.model = GrinQwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"
        # MoE Flag
        self.l_aux_weight = config.l_aux_weight
        self.min_l_aux_weight = config.min_l_aux_weight
        self.l_aux_weight_decay_steps = max(1, config.l_aux_weight_decay_steps)
        self.mlp_dynamic_expert_num = config.mlp_dynamic_expert_num
        self.mlp_fixed_expert_num = config.mlp_fixed_expert_num
        self.mlp_dynamic_null_expert_num = config.mlp_dynamic_null_expert_num
        self.input_max_length = 0
        self.training_steps = 0
        
        self.data_type_count = {}
        self.layer_num = config.num_hidden_layers
        self.mlp_expert_num = self.mlp_dynamic_expert_num + self.mlp_dynamic_null_expert_num
        self.atten_expert_num = 0  # GrinQwen2VL doesn't have attention experts

        # Initialize weights and apply final processing
        self.post_init()

    # MoE Flag
    @property
    def cur_aux_weight(self):
        if self.training_steps >= self.l_aux_weight_decay_steps:
            return self.min_l_aux_weight
        return self.l_aux_weight - (self.l_aux_weight - self.min_l_aux_weight) / self.l_aux_weight_decay_steps * self.training_steps

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def initialize_audio_modules(self, model_args, fsdp=None):
        if hasattr(model_args, "audio_tower") and model_args.audio_tower is not None:
            whisper_tower = model_args.audio_tower
            pretrain_audio_aligner = model_args.pretrain_whisper_aligner
            whisper_query_tokens_size = model_args.whisper_query_tokens_size
            self.config.whisper_tower = whisper_tower

            if self.audio_tower is None:
                audio_tower = build_whisper_tower(model_args)
                if fsdp is not None and len(fsdp) > 0:self.audio_tower = [audio_tower]
                else:self.audio_tower = audio_tower
            else:
                if fsdp is not None and len(fsdp) > 0:audio_tower = self.audio_tower[0]
                else:audio_tower = self.audio_tower
                audio_tower.load_model()

            self.config.whisper_projector_type = getattr(model_args, 'whisper_projector_type', 'linear')
            self.config.whisper_hidden_size = audio_tower.hidden_size
            self.config.whisper_query_tokens_size = whisper_query_tokens_size

            if getattr(self, 'audio_aligner', None) is None:
                self.audio_aligner = build_whisper_aligner(self.config)
            else:
                # In case it is frozen by LoRA
                for p in self.audio_aligner.parameters():
                    p.requires_grad = True

            if pretrain_audio_aligner is not None:
                print("whisper aligner: ",pretrain_audio_aligner)
                whisper_aligner_weights = torch.load(pretrain_audio_aligner, map_location='cpu')
                # for k,v in self.audio_aligner.named_parameters():
                #     if "layers.31.self_attn.k_proj.weight" in k:
                #         print(k,v)
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self.audio_aligner.load_state_dict(get_w(whisper_aligner_weights, 'audio_aligner'))
                # for k,v in self.audio_aligner.named_parameters():
                #     if "layers.31.self_attn.k_proj.weight" in k:
                #         print(k,v)

    def initialize_image_generator(self, model_args):
        self.tune_image_generator = getattr(model_args, 'tune_image_generator', False)
        if getattr(self, "image_generator", None) is None:
            self.image_generator = image_generator() 
        
    
    def extract_tts_pos(self,input_ids):
        # <sosp> 32000 <eop> 32001 <eosp> 32002
        sosp_pos = []
        eop_pos = []
        eosp_pos = []
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


    def extract_speech_hidden_state(self,sosp_pos, eop_pos, eosp_pos, speech_splits, last_hidden):
        all_prompt = []
        all_prefix = []
        all_text = []
        for b_sosp_pos, b_eop_pos, b_eosp_pos, b_speech_splits, b_last_hidden in zip(sosp_pos, eop_pos, eosp_pos, speech_splits, last_hidden):
            assert len(b_sosp_pos) == len(b_eop_pos) == len(b_eosp_pos)
            speech_idx = 0
            split_idx = 0
            for speech_idx in range(len(b_sosp_pos)):
                prompt = b_last_hidden[b_sosp_pos[speech_idx]+1:b_eop_pos[speech_idx]]
                split_chunk = []
                split_pos = []
                while len(b_speech_splits)>0 and split_idx<len(b_speech_splits) and b_speech_splits[split_idx] > b_eop_pos[speech_idx] and b_speech_splits[split_idx] < b_eosp_pos[speech_idx]:
                    split_pos.append(b_speech_splits[split_idx])
                    split_idx+=1
                    if split_idx>=len(b_speech_splits):
                        break
                # first cut
                all_prompt.append(prompt)
                all_prefix.append(None)
                if len(split_pos):
                    all_text.append(b_last_hidden[b_eop_pos[speech_idx]+1:split_pos[0]])
                else:
                    all_text.append(b_last_hidden[b_eop_pos[speech_idx]+1:b_eosp_pos[speech_idx]])
                # other cuts
                for cut_idx,_ in enumerate(split_pos):
                    all_prompt.append(prompt)
                    all_prefix.append(b_last_hidden[split_pos[cut_idx]-self.prefix_len:split_pos[cut_idx]])
                    if cut_idx==len(split_pos)-1:
                        all_text.append(b_last_hidden[split_pos[cut_idx]:b_eosp_pos[speech_idx]])
                    else:
                        all_text.append(b_last_hidden[split_pos[cut_idx]:split_pos[cut_idx+1]])
        batch = {}
        # pad prompt
        prompt,prompt_len = self.pad_hidden(all_prompt,last_hidden)
        batch["prompt"] = prompt
        batch["prompt_lens"] = torch.LongTensor(prompt_len).to(device=prompt.device)
        # pad prefix
        if sum([pr!=None for pr in all_prefix]):
            prefix,prefix_len = self.pad_hidden(all_prefix,last_hidden)
            batch["prefix"] = prefix
            batch["prefix_lens"] = torch.LongTensor(prefix_len).to(device=prefix.device)
            # print(batch["prefix"].shape,batch["prefix_lens"])
        # pad text
        text,text_len = self.pad_hidden(all_text,last_hidden)
        batch["x"] = text
        batch["x_lens"] = torch.LongTensor(text_len).to(device=text.device)
        # print(batch["x"].shape,batch["x_lens"])
        
        # print(batch["prompt"].shape,batch["prompt_lens"])

        return batch

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        vision_vision_select_layer = model_args.vision_vision_select_layer
        vision_spatial_pool_stride = model_args.vision_spatial_pool_stride
        # vision_vision_select_feature = model_args.vision_vision_select_feature
        pretrain_vision_aligner_adapter = model_args.pretrain_vision_aligner_adapter
        # vision_patch_merge_type = model_args.vision_patch_merge_type
        
        self.config.vision_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if getattr(self, 'vision_tower', None) is None:
            vision_tower = build_vision_tower(model_args)
            
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.visual_aligner_type = getattr(model_args, "visual_aligner_type", "linear")
        self.config.vision_hidden_size = vision_tower.hidden_size
        self.config.vision_vision_select_layer = vision_vision_select_layer
        self.config.vision_spatial_pool_stride = vision_spatial_pool_stride
        # self.config.vision_vision_select_feature = vision_vision_select_feature
        # self.config.vision_patch_merge_type = vision_patch_merge_type
        if getattr(self, "vision_aligner", None) is None:
            self.vision_aligner = build_vision_aligner(self.config, vision_cfg=vision_tower.config)
        else:
            # In case it is frozen by LoRA
            for p in self.vision_aligner.parameters():
                p.requires_grad = True
        
        if pretrain_vision_aligner_adapter is not None:
            vision_aligner_weights = torch.load(pretrain_vision_aligner_adapter, map_location="cpu")
            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
            
            incompatible_keys = self.vision_aligner.load_state_dict(get_w(vision_aligner_weights, "vision_aligner"))
            rank0_print(f"Loaded vision_aligner weights from {pretrain_vision_aligner_adapter}. Incompatible keys: {incompatible_keys}")
    
    def initialize_image_generator(self, model_args):
        self.tune_image_generator = getattr(model_args, 'tune_image_generator', False)
        if getattr(self, "image_generator", None) is None:
            self.image_generator = image_generator() 
    
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        # spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = torch.tensor(0, dtype=torch.float32, device=input_ids.device)
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = torch.tensor(1.0, dtype=torch.float32, device=input_ids.device)
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item(),
                        w.item(),
                    )
                    text_len = ed - st

                    base_llm_grid_h, base_llm_grid_w, base_llm_grid_t = self.config.vision_spatial_pool_stride, self.config.vision_spatial_pool_stride, 1
                    
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)              

                    t_index = torch.arange(base_llm_grid_t).view(-1, 1).expand(-1, base_llm_grid_h * base_llm_grid_w).flatten()
                    # print(f"t_index: {t_index}")
                    h_index = torch.arange(base_llm_grid_h).view(1, -1, 1).expand(base_llm_grid_t, -1, base_llm_grid_w).flatten()
                    w_index = torch.arange(base_llm_grid_w).view(1, 1, -1).expand(base_llm_grid_t, base_llm_grid_h, -1).flatten()
                    base_position_ids = torch.stack([t_index, h_index, w_index])
                    
                    # print(f"second_per_grid_t: {second_per_grid_t}")
                    
                    range_tensor = torch.arange(llm_grid_t, device=base_position_ids.device).view(-1, 1)
                    second_per_grid_t = second_per_grid_t.to(range_tensor.device)
                    # second_per_grid_t = torch.as_tensor(
                    #     second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    # )
                    
                    tokens_per_second = 2.0 # following qwen2.5 vl setting
                    range_tensor = range_tensor.float() * second_per_grid_t * tokens_per_second
                    range_tensor = torch.round(range_tensor).to(base_position_ids.device).to(base_position_ids.dtype)
                    # print(range_tensor)
                    
                    # second_per_grid_t = second_per_grid_t.to(base_position_ids.device).to(base_position_ids.dtype)
                    
                    vision_position_ids = []
                    for t in range(llm_grid_t):
                        for h in range(llm_grid_h):
                            for w in range(llm_grid_w):
                                c_position_ids = base_position_ids.clone()
                                c_position_ids[0] += range_tensor[t]
                                c_position_ids[1] += h * base_llm_grid_h
                                c_position_ids[2] += w * base_llm_grid_w
                                vision_position_ids.append(c_position_ids)
                        
                    vision_position_ids = torch.cat(vision_position_ids, dim=1)
                    
                    llm_pos_ids_list.append(vision_position_ids + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w * base_llm_grid_h * base_llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_rope_index_new(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        # spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index, audio_index = 0, 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums, audio_nums= 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                audio_start_indices = torch.argwhere(input_ids == audio_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_tokens = input_ids[audio_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                audio_nums = (audio_tokens == audio_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                for _ in range(image_nums + video_nums + audio_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    
                    audio_mode = False
                    if ed_image < ed_video and ed_image < ed_audio:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = torch.tensor(0, dtype=torch.float32, device=input_ids.device)
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    elif ed_video < ed_image and ed_video < ed_audio:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = torch.tensor(1.0, dtype=torch.float32, device=input_ids.device)
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    elif ed_audio < ed_image and ed_audio < ed_video:
                        t = audio_grid_thw[audio_index]*10
                        h = t
                        w = t
                        audio_hw = 20
                        audio_index += 1
                        remain_audios -= 1
                        ed = ed_audio
                        audio_mode = True
                    else:
                        raise ValueError("position error of special tokens.")
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item(),
                        w.item(),
                    )
                    text_len = ed - st

                    base_llm_grid_h, base_llm_grid_w, base_llm_grid_t = self.config.vision_spatial_pool_stride, self.config.vision_spatial_pool_stride, 1
                    
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    # print("adding text:",text_len)
                    
                    # range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    # expanded_range = range_tensor.expand(-1, base_llm_grid_h * base_llm_grid_w)
                    # second_per_grid_t = torch.as_tensor(
                    #     second_per_grid_t, 
                    # )
                    # time_tensor = expanded_range * second_per_grid_t
                    # time_tensor_long = time_tensor.long()
                    # t_index = time_tensor_long.flatten()
                    
                    
                    if not audio_mode:
                        t_index = torch.arange(base_llm_grid_t).view(-1, 1).expand(-1, base_llm_grid_h * base_llm_grid_w).flatten()
                        # print(f"t_index: {t_index}")
                        h_index = torch.arange(base_llm_grid_h).view(1, -1, 1).expand(base_llm_grid_t, -1, base_llm_grid_w).flatten()
                        w_index = torch.arange(base_llm_grid_w).view(1, 1, -1).expand(base_llm_grid_t, base_llm_grid_h, -1).flatten()
                        base_position_ids = torch.stack([t_index, h_index, w_index])
                        
                        # print(f"second_per_grid_t: {second_per_grid_t}")
                        
                        range_tensor = torch.arange(llm_grid_t, device=base_position_ids.device).view(-1, 1)
                        second_per_grid_t = second_per_grid_t.to(range_tensor.device)
                        # second_per_grid_t = torch.as_tensor(
                        #     second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                        # )
                        
                        tokens_per_second = 2.0 # following qwen2.5 vl setting
                        range_tensor = range_tensor.float() * second_per_grid_t * tokens_per_second
                        range_tensor = torch.round(range_tensor).to(base_position_ids.device).to(base_position_ids.dtype)
                        # print(range_tensor)
                        
                        # second_per_grid_t = second_per_grid_t.to(base_position_ids.device).to(base_position_ids.dtype)
                        
                        vision_position_ids = []
                        for t in range(llm_grid_t):
                            for h in range(llm_grid_h):
                                for w in range(llm_grid_w):
                                    c_position_ids = base_position_ids.clone()
                                    c_position_ids[0] += range_tensor[t]
                                    c_position_ids[1] += h * base_llm_grid_h
                                    c_position_ids[2] += w * base_llm_grid_w
                                    vision_position_ids.append(c_position_ids)
                            
                        vision_position_ids = torch.cat(vision_position_ids, dim=1)
                        # print("vision len",vision_position_ids.shape)

                        llm_pos_ids_list.append(vision_position_ids + text_len + st_idx)
                        st = ed + llm_grid_t * llm_grid_h * llm_grid_w * base_llm_grid_h * base_llm_grid_w
                    else:
                        t_index = torch.arange(base_llm_grid_t).view(-1, 1).expand(-1, audio_hw).flatten()
                        h_index = torch.arange(base_llm_grid_t).view(-1, 1).expand(-1, audio_hw).flatten()#torch.arange(audio_hw).view(-1, 1).flatten()
                        w_index = torch.arange(base_llm_grid_t).view(-1, 1).expand(-1, audio_hw).flatten()#torch.arange(audio_hw).view(-1, 1).flatten()
                        base_position_ids = torch.stack([t_index, h_index, w_index])

                        range_tensor = torch.arange(llm_grid_t, device=base_position_ids.device).view(-1, 1)
                        
                        tokens_per_second = 2.0 # following qwen2.5 vl setting
                        whisper_second = 3.0
                        range_tensor = range_tensor.float() * tokens_per_second * whisper_second
                        range_tensor = torch.round(range_tensor).to(base_position_ids.device).to(base_position_ids.dtype)

                        vision_position_ids = []
                        for t in range(llm_grid_t):
                            c_position_ids = base_position_ids.clone()
                            c_position_ids[0] += range_tensor[t]
                            c_position_ids[1] += range_tensor[t]#t * audio_hw
                            c_position_ids[2] += range_tensor[t]#t * audio_hw
                            vision_position_ids.append(c_position_ids)
                            
                        vision_position_ids = torch.cat(vision_position_ids, dim=1)
                        # print("audio len",vision_position_ids.shape)
                        llm_pos_ids_list.append(vision_position_ids + text_len + st_idx)
                        st = ed + llm_grid_t * audio_hw
                    # print(vision_position_ids[0])
                    
                    # t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    # h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    # w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    # print("adding text:",text_len)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                # print("idsshape",input_ids.shape,llm_positions.shape)
                # print("ids",input_ids.tolist())
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            # print("mrope",position_ids.tolist(),mrope_position_deltas)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def l2_loss(aelf, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (N, T_I_V_A.txt, D) tensor.
            v: (N, T_I_V_A.txt, D) tensor.
        Returns:
            l1_loss: (N,) tensor of summed L1 loss.
        """
        assert u.shape == v.shape, (u.shape, v.shape)
        return ((u - v) ** 2).mean(dim=-1) ** 0.5

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        aux_balance_weight: Optional[torch.LongTensor] = None,
        padding_token_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits_and_topk: Optional[bool] = None,
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
        
        
    ) -> Union[Tuple, MoEQwen2VLCausalLMOutputWithPast]:
        
        if torch.distributed.is_initialized(): 
            local_rank = torch.distributed.get_rank()
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        device = torch.device(f"cuda:{local_rank}")
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if position_ids is None:
            if audio_features is not None:
                position_ids, rope_deltas = self.get_rope_index_new(input_ids, image_grid_thw, video_grid_thw, audio_grid_thw, second_per_grid_ts, attention_mask)
            else:
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask)
            
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
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
                
                if golden_visual_embedding is not None:
                    for idx, visual_fc_layer in zip([-1], self.visual_hidden_fcs):
                        visual_hidden_embeddings = visual_fc_layer(image_embeds.unsqueeze(0)).to(device)
                
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.vision_tower(pixel_values_videos)
                video_embeds = self.vision_aligner(video_embeds)
                
                height = width = self.vision_tower.num_patches_per_side
                num_frames, num_tokens, num_dim = video_embeds.shape
                video_embeds = video_embeds.view(num_frames, height, width, -1)
                video_embeds = video_embeds.permute(0, 3, 1, 2).contiguous()
                
                scaled_shape = (self.config.vision_spatial_pool_stride, self.config.vision_spatial_pool_stride)
                adaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(scaled_shape)
                video_embeds = adaptiveAvgPool2d(video_embeds)
                video_embeds = video_embeds.permute(0, 2, 3, 1)
                video_embeds = video_embeds.view(num_frames, -1, num_dim).reshape(-1, num_dim)
                
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            if audio_features is not None:
                audio_features = audio_features.type(self.audio_tower.dtype)
                audio_output = self.audio_tower(audio_features.unsqueeze(dim = 0))
                audio_embeds = self.audio_aligner(encoder_output=audio_output)
                audio_embeds = audio_embeds.view(-1,audio_embeds.shape[-1])
                n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
                n_audio_features = audio_embeds.shape[0]
                if n_audio_tokens != n_audio_features:
                    raise ValueError(
                        f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                    )
                audio_mask = (
                    (input_ids == self.config.audio_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if attention_mask is not None and aux_balance_weight is not None:
        if aux_balance_weight is not None:
            aux_balance_weight = attention_mask * aux_balance_weight

        # if attention_mask is not None and padding_token_mask is None:
        if padding_token_mask is None:
            assert len(attention_mask.shape) == 2, f"{attention_mask.shape}"  # B, L
            padding_token_mask = attention_mask.bool()[:,-inputs_embeds.shape[1]:]
        
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_token_mask=padding_token_mask,
            aux_balance_weight=aux_balance_weight,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_router_logits_and_topk=True,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        all_aux_loss = outputs.all_aux_loss if return_dict else outputs[-1]
        # Fix for multi-GPU inference
        first_device = all_aux_loss[0].device
        all_aux_loss = [l.to(first_device) if l.device != first_device else l for l in all_aux_loss]
        all_aux_loss = torch.mean(torch.cat([l.unsqueeze(0) for l in all_aux_loss], dim=0))
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # MoE Flag - return
            # adding aux_balance_loss
            aux_loss = self.cur_aux_weight * all_aux_loss
            self.training_steps += 1
            self.input_max_length = max(self.input_max_length, input_ids.shape[1])
            loss = loss + aux_loss  # make sure to reside in the same device
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return MoEQwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
            all_router_logits=outputs.all_router_logits,
            all_router_top_k=outputs.all_router_top_k,
            all_router_expert_mask=outputs.all_router_expert_mask,
            all_router_weight=outputs.all_router_weight,
            aux_balance_loss=all_aux_loss,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        audio_features= None,
        image_grid_thw=None,
        video_grid_thw=None,
        audio_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]


        rope_deltas = kwargs.get("rope_deltas", None)
        position_ids = None
        if attention_mask is not None and position_ids is None:
            # print(attention_mask)
            # print(cache_position)
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
            # if cache_position is None:
                if audio_features is not None:
                    position_ids, rope_deltas = self.get_rope_index_new(input_ids, image_grid_thw, video_grid_thw, audio_grid_thw, second_per_grid_ts, attention_mask)
                else:
                    position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask)
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None
            audio_features = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "audio_features": audio_features,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
                "second_per_grid_ts": second_per_grid_ts,
                
            }
        )
        return model_inputs


AutoConfig.register("grin_qwen2_vl", GrinQwen2VLConfig)
AutoModelForCausalLM.register(GrinQwen2VLConfig, GrinQwen2VLForConditionalGeneration)
