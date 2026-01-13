import os 
import torch
from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
from uni_moe.qwen_vl_utils import process_mm_info
from uni_moe.model import deepspeed_moe_inference_utils

model_path = "HIT-TMG/Uni-MoE-2.0-Omni"
processor = Qwen2VLProcessor.from_pretrained(model_path)

model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).cuda()

processor.data_args = model.config

messages = [{
    "role": "user", 
    "content": [
            {"type": "text", "text": "<audio>\n<image>\nAnswer the question in the audio."},
            {"type": "audio", "audio": "examples/assets/audio/quick_start.mp3"},
            {"type": "image", "image": "examples/assets/image/quick_start.jpg"}
        ]
}]

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

inputs = inputs.to(device=model.device)

output_ids = model.generate(
    **inputs,
    use_cache=True,
    pad_token_id=processor.tokenizer.eos_token_id,
    max_new_tokens=4096,
    temperature=1.0,
    do_sample=True
)

text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
print(text)