from .ar_decoder import LLM2TTSCodecAR
# from ar_decoder import LLM2TTSCodecAR

def build_ar_ori_v2_speech_generator(config):
    generator_type = getattr(config, 'speech_generator_type', 'ar_ori_v2')
    if generator_type == 'ar_ori_v2':
        return LLM2TTSCodecAR(config)

    raise ValueError(f'Unknown generator type: {generator_type}')


def try_gen():
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='4,5'
    import torch
    from dataclasses import dataclass, field
    from typing import Dict, Optional, Sequence, List, Any, Union
    from ar_decoder import LLM2TTSCodecAR
    idim = 4
    @dataclass
    class ModelArguments:
        idim : Optional[int] = field(default=4) # 896 输入的维度 llama decoder的特征维度
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
        
    device = torch.device("cuda:0")
    batch = {}
    batch["x"] = torch.rand(3, 4, idim).to(device)
    batch['x_lens'] = torch.LongTensor([3,4,2]).to(device)
    batch['y'] = torch.LongTensor([[11,230,55,33,77,-100,-100,-100],[11,230,55,33,77,666,-100,-100],[11,230,55,33,457,717,100,102]]).to(device)
    batch['y_lens'] = torch.sum(batch['y']!=-100,dim=-1).to(device)
    batch["prompt"] = torch.rand(3, 3, idim).to(device)
    batch['prompt_lens'] = torch.LongTensor([1, 2, 3]).to(device)
    # batch["prefix"] = torch.rand(3, 3, idim).to(device)
    # batch['prefix_lens'] = torch.LongTensor([2, 3, 1]).to(device)
    # print(batch['y_lens'])
    # return
    config = ModelArguments()
    model = LLM2TTSCodecAR(config)
    
    model = model.to(device)
    
    
    # print(batch["x"].device)

    model(batch)

# try_gen()