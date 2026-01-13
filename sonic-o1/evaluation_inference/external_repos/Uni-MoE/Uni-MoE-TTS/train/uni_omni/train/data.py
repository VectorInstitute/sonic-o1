

import os
import io
from dataclasses import dataclass, field
import json
import zipfile
from typing import Dict, Optional, Sequence, List, Any, Union
import random

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

import torch

import transformers

from uni_omni.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, AUDIO_TOKEN_INDEX
from torch.utils.data import Dataset

from uni_omni import conversation as conversation_lib
from uni_omni.model import *
from uni_omni.mm_utils import tokenizer_image_audio_video_token

from PIL import Image
import time


local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    model_eval_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    local_files_only: Optional[str] = field(default=False)
    lora_path: Optional[str] = field(default=None)
    pretrain_mlp_gate: Optional[str] = field(default=None)
    expert_dir: Optional[str] = field(default=None)
    num_experts: Optional[int] = field(default=4)
    num_experts_per_tok: Optional[int] = field(default=2)
    eval_ep_size: Optional[int] = field(default=4)

    tune_speech_generator: Optional[str] = field(default=False)
    tune_speech_generator_only: Optional[str] = field(default=False)
    speech_generator_type: Optional[str] = field(default='ar')
    load_weight_from_qwen : Optional[str] = field(default=None)

    idim : Optional[int] = field(default=4096) # 896 输入的维度 llama decoder的特征维度
    odim : Optional[int] = field(default=4096) # 1024 词表维度 tokenizer的词表大小
    encoder_pre_norm_type : Optional[str] = field(default="ln") # "ln" 似乎没用到
    encoder_drop_rate : Optional[float] = field(default=0.1 ) # 0.1 drop out的程度 因为没有NAR所以用不到
    encoder_criterion : Optional[str] = field(default="ce") # "ce" 损失函数类型，不改动
    encoder_upsample_rate : Optional[int] = field(default=9) # 9 没有做upsample所以也用不到
    transformer_attention_dim : Optional[int] = field(default=896) # 896 llama hidden_size 4096 
    transformer_linear_units : Optional[int] = field(default=4864) # 4864 llama intermediate_size 11008
    transformer_num_blocks : Optional[int] = field(default=12) # 4 llama num_hidden_layers 32 
    transformer_attention_heads : Optional[int] = field(default=14) # 14 llama num_attention_heads 32
    transformer_kv_heads : Optional[int] = field(default=2) 
    transformer_dropout_rate : Optional[float] = field(default=0.1 ) # 0.1 
    encoder_output_dim : Optional[int] = field(default=896) # 896 输出词表大小
    do_cross_attention : Optional[bool] = field(default=True)
    cross_attention_layer_num : Optional[int] = field(default=6)
    
    tune_qwen25vl_proj_only : Optional[bool] = field(default=False)
    expert_dir : Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    codes_folder: Optional[str] = field(default=None)
    pad_audio: bool = True
    mix_va: bool = False
    eval_data_type: str = field(default=None, metadata={"help": "Eval only. Data type of the dataset."})
    eval_output: Optional[str] = field(default=None, metadata={"help": "Eval only. Output path of the eval result."})

    audio_mode: Optional[str] = field(default="tts-pretrain")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    llm_lora_enable: bool = False
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    use_pretrain_lora: bool = False # need change
    pretrain_lora_enable: bool = False # need change
    dataloader_pin_memory: bool = False # need change
    enable_deepspeed_moe: bool = False # need change
    lora_only_mlp: bool = False # need change
    from_lora_path: Optional[str] = field(default=None)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 augment_config_path=None,
                 ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.data_path = data_path
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.local_rank = data_args.local_rank

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths_type(self):
        length_list = []
        # data_type = ["video", "voice", "image"] 
        # use 3 bits to represent the modality
        # only text is 000, only image is 001, video is 010, voice is 100
        # video and voice is 110, video and image is 011, voice and image is 101
        # video, voice and image is 111
        for sample in self.list_data_dict:
            cur_len = len(sample["codes"])+len(sample["input_ids"])+len(sample["prompt_ids"]) if "input_ids" in sample else len(sample["codes"])+len(sample["prompt_ids"])
            cur_type = 0
            if "prefix_ids" in sample:
                cur_type += 1
            length_list.append((cur_len, cur_type))
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if True:
            # TODO: define number of retries somewhere else
            num_base_retries = 30
            num_final_retries = 300

            # try the current sample first
            # for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

            # try other samples, in case it is file corruption issue
            for attempt_idx in range(num_base_retries):
                next_index = (i + 1 + attempt_idx)%(len(self.list_data_dict))
                try:
                    # sample_idx = random.choice(range(len(self)))
                    sample = self._get_item(next_index)
                    return sample
                except Exception as e:
                    # no need to sleep
                    print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                    pass

            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                raise e
        else:
            return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        # print("rank[",self.local_rank,"]indice:",i)
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        text_len = 200
        bos_token="<s>"
        eos_token="</s>"
        speech_num = 0
        data_dict = {}
        # 传递给模型的内容包括：需要转化的文本，切分后的prefix文本，提示词prompt文本，均以id的形式
        # 用相同的IGNORE作为pad，最终根据这些pad输入到模型
        # target为语音token(不用变)
        # batch["input_ids"]
        # batch["attention_mask"]
        # batch["prompt_ids"]
        # batch["prefix_ids"]
        # batch["codes"]
        codes = sources[0]["codes"]
        if type(codes) == str:
            codes = json.load(open(os.path.join(self.data_args.codes_folder,codes),"r"))
        if len(codes)>1800:
            print("code too long!")
            assert 1==2
        # pref_len = len(sources[0]["prefix_ids"]) if "prefix_ids" in sources[0] else 0
        # if len(codes)+len(sources[0]["input_ids"])+pref_len>1100:
        #     print("len too long!")
        #     assert 1==2
        if len(sources[0]["prompt_ids"])>1000:
            print("prompt too long!")
            assert 1==2
        data_dict["input_ids"] = torch.LongTensor(sources[0]["input_ids"]) if "input_ids" in sources[0] else None
        data_dict["prompt_ids"] = torch.LongTensor(sources[0]["prompt_ids"])
        data_dict["prefix_ids"] = torch.LongTensor(sources[0]["prefix_ids"]) if "prefix_ids" in sources[0] else None
        data_dict["codes"] = torch.LongTensor(codes)
        data_dict["labels"] = None
        return data_dict

    


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    processor: Any

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, codes, prompt_ids, prefix_ids = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "codes", "prompt_ids", "prefix_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id) if input_ids[0] is not None else None
        # print(input_ids)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX) if labels[0] is not None else None

        codes = torch.nn.utils.rnn.pad_sequence(codes,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX) if codes[0] is not None else None
        
        prompt_ids = torch.nn.utils.rnn.pad_sequence(prompt_ids,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX) if prompt_ids[0] is not None else None
        
        
        prefix_ids = torch.nn.utils.rnn.pad_sequence(prefix_ids,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX) if sum([1 if pid==None else 0 for pid in prefix_ids])==0 else None
        
        if input_ids is None:
            input_ids = prompt_ids

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]  if labels is not None else None
        
        batch = {}
        
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["attention_mask"] =input_ids.ne(self.tokenizer.pad_token_id) # [:,2:] # ATTN !!!!!!!!!!!!!!!!!!!!!!!!!!!!

        batch["codes"] = codes
        batch["prefix_ids"] = prefix_ids
        batch["prompt_ids"] = prompt_ids

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,processor=None)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
