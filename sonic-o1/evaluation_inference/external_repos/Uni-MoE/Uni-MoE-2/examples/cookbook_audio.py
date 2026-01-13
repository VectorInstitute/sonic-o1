import os 
import torch, torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
from uni_moe.qwen_vl_utils import process_mm_info
from uni_moe.model import deepspeed_moe_inference_utils
from tqdm import *


def initial_input(processor, messages):
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, no=True
    )
    image_inputs, video_inputs, audio_inputs = process_mm_info(messages)
    for pi,pr in enumerate(prompt):
        prompt[pi] = prompt[pi].replace("<image>","<|vision_start|><|image_pad|><|vision_end|>").replace("<audio>","<|audio_start|><|audio_pad|><|audio_end|>").replace("<video>","<|vision_start|><|video_pad|><|vision_end|>")
    inputs = processor(
            text=prompt,
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            padding=True,
            return_tensors="pt",
    )
    if "second_grid_ts" in inputs:
        inputs["second_per_grid_ts"] = inputs["second_grid_ts"]
        del inputs["second_grid_ts"]
    inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
    return inputs


def load_unimoe(model_path,do_audio_out=False):
    pretrain_speech_gen=model_path+"/speech_gen/speech_gen_ep2.bin"
    pretrain_speech_gen_type="ar_ori_v2_new"
    wavtokenizer = None
    if do_audio_out:
        from decoder.pretrained import WavTokenizer
        wav_device=torch.device('cuda')
        wav_config_path = model_path+"/speech_gen//wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        wav_model_path = model_path+"/speech_gen/wavtokenizer_large_unify_600_24k.ckpt"
        wavtokenizer = WavTokenizer.from_pretrained0802(wav_config_path, wav_model_path)
        wavtokenizer = wavtokenizer.to(wav_device)

    # audio gen
    class aud_args: 
        pretrain_speech_gen = None
        pretrain_speech_gen_type = None
    audio_args = aud_args()
    audio_args.pretrain_speech_gen = pretrain_speech_gen
    audio_args.pretrain_speech_gen_type = pretrain_speech_gen_type

    processor = Qwen2VLProcessor.from_pretrained(model_path)
    model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model.speech_generator.train()
    model.cuda()
    
    processor.data_args = model.config
        
    return model, processor, wavtokenizer

def infer_single(model,processor,messages,kwargs,wavtokenizer=None,outf=None):
    inputs = initial_input(processor, messages).to(device=model.device)
    for k, v in inputs.items():
        if k in ["pixel_values", "pixel_values_videos", "audio_features"]:
            inputs[k] = v.to(dtype=torch.bfloat16)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            use_cache=True,
            **kwargs
        )
        text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(">>>>Prediction: ", text[0])
    if wavtokenizer:
        out_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        split_tags = ["yes. yes","yes, yes","yes? yes","yes! yes","yes; yes","yes.\nyes","yes.\n\nyes","我。我","我，我","我？我","我！我","我：我","我；我","我。\n我","我。\n\n我"]
        number_tags = ["0","1","2","3","4","5","6","7","8","9"]
        split_tokens = [processor.tokenizer(st,return_tensors="pt",padding="longest").input_ids[0][1] for st in split_tags]
        number_tokens = [processor.tokenizer(nt,return_tensors="pt",padding="longest").input_ids[0][0] for nt in number_tags]
        out_codes,splt = model.generate_from_tokens(
                    out_seq = out_ids.to(device=model.device),
                    split_tokens = split_tokens,
                    number_tokens = number_tokens,
                    maxtoklen = 2048,
                    maxcutlen = 16,
                    )
        splt = splt[0]
        out_ids = out_ids[0]
        out_codes = torch.cat(out_codes,dim=0)
        out_codes = [tok%4096 for tok in out_codes.tolist()]
        out_codes = torch.LongTensor(out_codes)
        out_codes = out_codes.to(device=model.device)
        features = wavtokenizer.codes_to_features(out_codes.unsqueeze(0).unsqueeze(0))
        bandwidth_id = torch.tensor([0]).to(device=model.device)
        audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id) 
        torchaudio.save(outf, audio_out.detach().cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
        print(">>>>Output speech saving to: ", outf)


# init args
model_path = "HIT-TMG//Uni-MoE-2.0-Omni"
kwargs = dict(do_sample=False, num_beams=1, temperature=0.0, max_new_tokens=2048)

model, processor, wavtokenizer = load_unimoe(model_path,do_audio_out=True)
# asr task
print("Testing ASR...")
messages = [[{
        "role": "user", 
        "content": [
            {"type": "text", "text": "<audio>\nTranscribe this audio to text."},
            {"type": "audio", "audio": "examples/assets/audio/7902-96592-0012.flac"}
            ]
        }]]
infer_single(model,processor,messages,kwargs)
# audio cap task
print("Testing audio captioning...")
messages = [[{
        "role": "user", 
        "content": [
            {"type": "text", "text": "<audio>\nGenerate audio caption."},
            {"type": "audio", "audio": "examples/assets/audio/audio_cap.wav"}
            ]
        }]]
infer_single(model,processor,messages,kwargs)
# music cap task
print("Testing music captioning...")
messages = [[{
        "role": "user", 
        "content": [
            {"type": "text", "text": "<audio>\n music caption."},
            {"type": "audio", "audio": "examples/assets/audio/_-kssA-FOzU.wav"}
            ]
        }]]
infer_single(model,processor,messages,kwargs)
# audio qa task
print("Testing audio QA...")
messages = [[{
        "role": "user", 
        "content": [
            {"type": "text", "text": "<audio>\nWhat can you hear from this audio?"},
            {"type": "audio", "audio": "examples/assets/audio/audio_cap.wav"}
            ]
        }]]

infer_single(model,processor,messages,kwargs)
# tts task
print("Testing TTS...")
tts_text = "Greetings my friend, Welcome to try out our Uni MOE Text to Speech model!"
# tts_text = "We present Uni MOE 2 from the Lychee family. As a fully open-source omni modal model, it substantially advances the capabilities of Lychee's Uni MOE series in language-centric multi-modal understanding, reasoning, and generating. Based on the qianwen 2.5 dense architecture, we train Uni MOE 2 from scratch through three core contributions: dynamic capacity Mixture of Experts design, a progressive training strategy enhanced with reinforcement strategy, and a carefully curated multimodal data matching technique. It is capable of cross and tri modality understanding, as well as generating images, text, and speech. Architecturally, our new MOE framework balances computational efficiency and capability for 10 cross-modal inputs using shared, routed, and null experts, while our Omni-Modality 3D RoPE ensures spatio temporal cross modality alignment in the self-attention layer. For training, following cross-modal pretraining, we use a progressive SFT strategy that activates modality specific experts and is enhanced by balanced data composition and an iterative GSPO DPO method to stabilize RL training. Data-wise, the base model, trained on approximately 50B tokens of open-source multimodal data, is equipped with special speech and image generation tokens, allowing it to learn these generative tasks by conditioning its outputs on linguistic cues."
messages = [[{
        "role": "user", 
        "content": [
            {"type": "text", "text": tts_text+"\nYou should read user's query in Jenny's voice.\nGenerate in the following format: <speech start> token indicates the beginning of the speech response, followed by the text prompt. <prompt end> token indicates the end of the text prompt and the start of the speech output. <speech end> token indicates the end of the speech output."},
            ]
        }]]
infer_single(model,processor,messages,kwargs,wavtokenizer=wavtokenizer,outf="examples/assets/results/tts_long.wav")
# a2a task
print("Testing speech conversation...")
messages = [[{
        "role": "user", 
        "content": [
            # {"type": "text", "text": "<audio>\nYou should listen to the user's question and respond in Jenny's voice.\nAnswer in the following format: <speech start> token indicates the beginning of the speech response, followed by the text prompt. <prompt end> token indicates the end of the text prompt and the start of the speech output. <speech end> token indicates the end of the speech output."},
            {"type": "text", "text": "<audio>\n请理解用户给出的语音问题并用Brian音色进行回答。\n回答的格式如下：<speech start>字符表示语音回复的起始，文本提示词开始，<prompt end>字符表示文本提示词结束语音输出开始，<speech end>字符表示语音输出结束。"},
            {"type": "audio", "audio": "examples/assets/audio/5.wav"}
            ]
        }]]

infer_single(model,processor,messages,kwargs,wavtokenizer=wavtokenizer,outf="examples/assets/results/a2a.wav")

# va2a task
print("Testing visual speech conversation...")
messages = [[{
        "role": "user", 
        "content": [
            {"type": "text", "text": "<audio>\n<image>\nYou should listen to the user's question and respond in Jenny's voice.\nAnswer in the following format: <speech start> token indicates the beginning of the speech response, followed by the text prompt. <prompt end> token indicates the end of the text prompt and the start of the speech output. <speech end> token indicates the end of the speech output."},
            {"type": "audio", "audio": "examples/assets/audio/quick_start.mp3"},
            {"type": "image", "image": "examples/assets/image/quick_start.jpg"}
            ]
        }]]
infer_single(model,processor,messages,kwargs,wavtokenizer=wavtokenizer,outf="examples/assets/results/va2a.wav")

# va2a task
print("Testing visual speech conversation...")
messages = [[{
        "role": "user", 
        "content": [
            {"type": "text", "text": "<audio>\n<video>\nYou should listen to the user's question and respond in Jenny's voice.\nAnswer in the following format: <speech start> token indicates the beginning of the speech response, followed by the text prompt. <prompt end> token indicates the end of the text prompt and the start of the speech output. <speech end> token indicates the end of the speech output."},
            {"type": "audio", "audio": "examples/assets/audio/va2a.mp3"},
            {"type": "video", "video": "examples/assets/video/va2a.mp4"}
            ]
        }]]
infer_single(model,processor,messages,kwargs,wavtokenizer=wavtokenizer,outf="examples/assets/results/va2a2.wav")