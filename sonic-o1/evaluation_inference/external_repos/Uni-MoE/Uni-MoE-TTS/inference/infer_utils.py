import os 
import torch
import torchaudio
from speech_generator_AR_MoE.ar_decoder import load_moetts
from transformers import Qwen2VLProcessor
from decoder.pretrained import WavTokenizer

def load_all_models(model_path):
    model = load_moetts(os.path.join(model_path,"speech_gen_ep2.bin"))
    processor = Qwen2VLProcessor.from_pretrained(os.path.join(model_path,"qwen_pp"))
    config_path = os.path.join(model_path,"wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
    model_path = os.path.join(model_path,"wavtokenizer_large_unify_600_24k.ckpt")
    wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
    return model,processor,wavtokenizer

def infer_tts(model,processor,wavtokenizer,wavpath,tts_text,prompt=None,speaker="Brian",Language="English"):
    if prompt == None or type(prompt)!=str:
        text = "<|speech_start|>Language: "+Language+"; Speaker: "+speaker+";<|speech_prompt|>"+tts_text+"<|speech_end|><|im_end|>"
    else:
        text = "<|speech_start|>"+prompt+"<|speech_prompt|>"+tts_text+"<|speech_end|><|im_end|>"
    print("MoE-TTS Transcribing: ",text)
    text_inputs = processor.tokenizer(text, padding=False)
    text_inputs = torch.LongTensor([text_inputs.input_ids])
    split_tags = ["yes. yes","yes, yes","yes? yes","yes! yes","yes; yes","yes.\nyes","yes.\n\nyes","我。我","我，我","我？我","我！我","我：我","我；我","我。\n我","我。\n\n我"]
    number_tags = ["0","1","2","3","4","5","6","7","8","9"]
    # [tensor(29892), tensor(29991), tensor(29973), tensor(29936), tensor(29901), tensor(29889)]
    split_tokens = [processor.tokenizer(st,return_tensors="pt",padding="longest").input_ids[0][1] for st in split_tags]
    number_tokens = [processor.tokenizer(nt,return_tensors="pt",padding="longest").input_ids[0][0] for nt in number_tags]
    out_codes,splt = model.generate_from_tokens(
                out_seq = text_inputs.to(device=model.in_fnn.weight.device),
                split_tokens = split_tokens,
                number_tokens = number_tokens,
                maxtoklen = 2048,
                maxcutlen = 16,
                )
    splt = splt[0]
    out_ids = text_inputs[0]
    codes = [out_ids[:splt[0]]]+[out_ids[splt[k]:splt[k+1]] for k in range(len(splt)-1)]+[out_ids[splt[-1]:]] if len(splt)>0 else [out_ids]
    # for code in codes:
    #     print(processor.batch_decode(code.unsqueeze(0))[0])
    out_codes = torch.cat(out_codes,dim=0)
    out_codes = [tok%4096 for tok in out_codes.tolist()]
    out_codes = torch.LongTensor(out_codes)
    out_codes = out_codes.to(device=model.in_fnn.weight.device)
    features = wavtokenizer.codes_to_features(out_codes.unsqueeze(0).unsqueeze(0))
    bandwidth_id = torch.tensor([0]).to(device=model.in_fnn.weight.device)
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id) 
    torchaudio.save(wavpath, audio_out.detach().cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)