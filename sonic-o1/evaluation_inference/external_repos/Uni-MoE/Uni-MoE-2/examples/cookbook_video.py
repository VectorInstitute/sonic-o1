import os 
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
    inputs["second_per_grid_ts"] = inputs["second_grid_ts"]
    del inputs["second_grid_ts"]
    inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
    
    return inputs

def infer_single(model, processor, messages, kwargs):
    inputs = initial_input(processor, messages).to(device=model.device)
    for k, v in inputs.items():
        if k in ["pixel_values", "pixel_values_videos", "audio_features"]:
            inputs[k] = v.to(dtype=torch.bfloat16)
    print(inputs.keys())
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


# Video Understanding
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "examples/assets/video/demo.mp4"},
            {"type": "text", "text": "<video>\nWhat is the genre of this video?"},
        ],
    }
]
infer_single(model,processor,messages,kwargs)


# Video Omni Understanding
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "examples/assets/video/omni.mp4"},
            {"type": "video", "video": "examples/assets/video/omni.mp4"},
            {"type": "text", "text": "<video>\n<audio>\nThe man recommend to refer to the document on github, what he first says the recipient can use?\nA.Shift_Key\nB.F_Keys\nC.Delete_Key\nD.Number_Keys."},
        ],
    }
]
infer_single(model,processor,messages,kwargs)