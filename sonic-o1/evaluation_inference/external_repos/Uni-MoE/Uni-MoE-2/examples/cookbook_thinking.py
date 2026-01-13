import os 
import torch
from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
from uni_moe.model.modeling_qwen_grin_moe import GrinQwen2VLForConditionalGeneration
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
            v = v.to(dtype=torch.bfloat16)
    output_ids = model.generate(
        **inputs,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
        **kwargs
    )

    text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
    print(text)

# Model Initialization
model_path = "HIT-TMG/Uni-MoE-2.0-Thinking"

processor = Qwen2VLProcessor.from_pretrained(model_path)

model = GrinQwen2VLForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).cuda()

processor.data_args = model.config

kwargs = dict(do_sample=True, num_beams=1, temperature=1.0, max_new_tokens=4096)

# General Understanding
messages = [
    {
       "role": "system",
       "content": "You are Uni-MoE-v2, a helpful multi-modal model. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines." 
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "examples/assets/image/thinking.jpg"},
            {"type": "text", "text": "<image>\nHint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Several people compared how many Web pages they had visited. What is the mean of the numbers?'"},
        ],
    }
]
infer_single(model,processor,messages,kwargs)
