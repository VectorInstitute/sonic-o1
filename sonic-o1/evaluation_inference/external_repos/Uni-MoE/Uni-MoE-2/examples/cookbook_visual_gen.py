import os 
import sys 
from typing import Dict, Optional, Sequence, List, Any, Union

import torch, torchaudio
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
from uni_moe.qwen_vl_utils import process_mm_info
from PIL import Image
from uni_moe.model import deepspeed_moe_inference_utils
import torch.distributed as dist


def load_unimoe(model_path: str):
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )
    model.cuda()

    # sync processors
    processor.data_args = model.config

    return model, processor


EXAMPLES = [
    # generation
    {
        "prompt": "<image>\nImage generation: In the art piece, a realistically depicted young girl with flowing blonde hair gazes intently into the distance, her eyes reflecting the vibrant hues of a spring forest. The verdant greens and soft pastels of the budding trees are captured in subtle brushstrokes, giving the scene a serene and tranquil atmosphere. The minimalist composition focuses on the girl's expression of wonder and the lush woodland background, while the texture of the oil paint adds depth and richness to the canvas.",
        "input_image": None,
        "out_name": "genarate.png",
    },
    # thinking
    {
        "prompt": "You should first think step by step about how to construct the image, including background, objects, colors, lighting, and style. \nThe reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, \ni.e., <think>1. Position a man seated indoors, capturing him from the upper chest to just above the chin, focusing on ... </think> <answer> Here is the image: [TASK0][TASK1][TASK2][IMG0][IMG1][IMG2][IMG3][IMG4][IMG5][IMG6][IMG7][IMG8][IMG9][IMG10][IMG11][IMG12][IMG13][IMG14][IMG15][IMG16][IMG17][IMG18][IMG19][IMG20][IMG21][IMG22][IMG23][IMG24][IMG25][IMG26][IMG27][IMG28][IMG29][IMG30][IMG31] </answer>,\nwhich means your output should start with <think> and end with </answer>\n\n\n<image>\nImage generation: An apple orchard during the winter.",
        "input_image": None,
        "out_name": "thinking.png",
    },
    # edition
    {
        "prompt": "<image>\nAdd a dog standing near the fence in the foreground, close to the road.",
        "input_image": "examples/assets/visual_gen/input_images/edit.jpg",
        "out_name": "edit.png",
    },
    # low-level
    {
        "prompt": "<image>\nRemove the rain from this image.",
        "input_image": "examples/assets/visual_gen/input_images/derain.png",
        "out_name": "derain.png",
    },
    # multigen
    {
        "prompt": "<image>\nCanny edge to image: Bachalpsee Lake and Wetterhorn.",
        "input_image": "examples/assets/visual_gen/input_images/multigen.png",
        "out_name": "multigen.png",
    },
]


def make_message(prompt: str, image_path: str = None) -> List[Dict[str, Any]]:
    """Return messages list compatible with the processor.apply_chat_template
    If image_path is provided, include it as first message of type image.
    """
    user_items = []
    if image_path is not None:
        user_items.append({"type": "image", "image": image_path})
    else: 
        user_items.append({"type": "image", "image": "examples/assets/visual_gen/input_images/white.png"})
    user_items.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": user_items}]


def run_batch(model_path: str, examples: List[Dict[str, Any]], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model, processor = load_unimoe(model_path)
    
    import json
    name_list = []
    for name, params in model.named_parameters():
        name_list.append(name)
    json.dump(name_list, open("model_param_names.json","w"), indent=4)
    
    for i, ex in enumerate(examples, start=1):
        print(f"\n=== [{i}/{len(examples)}]  prompt={ex['prompt']}")
        messages = make_message(ex['prompt'], ex.get('input_image'))
        print(messages)

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

        # ensure batch dim
        if inputs.get("input_ids") is None:
            print("Warning: input_ids missing, skipping example")
            continue
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)

        # prepare save path
        base_out = os.path.splitext(ex['out_name'])[0]
        save_name = f"{base_out}.png"
        save_path = os.path.join(save_dir, save_name)

        # call generate_visualgen
        output_ids = model.generate_visualgen(
            model_path=model_path,
            input_ids=inputs["input_ids"].to(device=model.device),
            pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16,device=model.device) if "pixel_values" in inputs else None,
            image_grid_thw=inputs.get("image_grid_thw", None),
            pixel_values_videos=inputs.get("pixel_values_videos", None),
            video_grid_thw=inputs.get("video_grid_thw", None),
            audio_features=inputs.get("audio_features", None),
            audio_grid_thw=inputs.get("audio_grid_thw", None),
            use_cache=True,
            attention_mask=inputs["input_ids"].ne(processor.tokenizer.pad_token_id),
            pad_token_id=processor.tokenizer.eos_token_id,
            golden_caption_emb=None,
            golden_task_emb=None,
            golden_visual_emb=None,
            image_path=ex.get("input_image", None),
            save_path=save_path,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            max_new_tokens=4096,
        )

        decoded = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        print("Generated text output:\n", decoded)
        print("Saved image to:", save_path)


if __name__ == "__main__":
    MODEL_PATH = "HIT-TMG/Uni-MoE-2.0-Image"
    SAVE_DIR = "examples/assets/visual_gen/generated_images"
    run_batch(MODEL_PATH, EXAMPLES, SAVE_DIR)
