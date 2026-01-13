import os 
import sys 
import torch
from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
from uni_moe.qwen_vl_utils import process_mm_info
from uni_moe.model import deepspeed_moe_inference_utils

def initial_input(processor, messages):
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    texts = texts.replace("<image>","<|vision_start|><|image_pad|><|vision_end|>").replace("<audio>","<|audio_start|><|audio_pad|><|audio_end|>").replace("<video>","<|vision_start|><|video_pad|><|vision_end|>")
    image_inputs, video_inputs, audio_inputs = process_mm_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
    return inputs

def infer_single(model, processor, messages, kwargs):
    inputs = initial_input(processor, messages).to(device=model.device)
    for k, v in inputs.items():
        if k in ["pixel_values", "pixel_values_videos", "audio_features"]:
            inputs[k] = v.to(dtype=torch.bfloat16)
    output_ids = model.generate(
        **inputs,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
        **kwargs
    )

    text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
    print(text)

# Model Initialization
model_path = "HIT-TMG/Uni-MoE-2.0-Omni"

processor = Qwen2VLProcessor.from_pretrained(model_path)

model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).cuda()

processor.data_args = model.config

kwargs = dict(do_sample=False, num_beams=1, temperature=0.0, max_new_tokens=2048)

# Image Understanding
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "examples/assets/image/general.jpg"},
            {"type": "text", "text": "<image>\nDescribe this image."},
        ],
    }
]
infer_single(model,processor,messages,kwargs)

# Multi-Image Understanding
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "examples/assets/image/multi_1.png"},
            {"type": "image", "image": "examples/assets/image/multi_2.png"},
            {"type": "text", "text": "Figure 1: <image>\nFigure 2: <image>\nBriefly compare Figure 1 and Figure 2."},
        ],
    }
]
infer_single(model,processor,messages,kwargs)