from infer_utils import load_all_models,infer_tts
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for all modality")
    parser.add_argument("--model_dir", type=str, default="path/to/MoE_TTS", help="Path to the Uni-MoE-TTS model")
    parser.add_argument("--wav_path", type=str, default="path/to/output_wav_file.wav", help="Path to the output_wav_file")
    parser.add_argument("--text", type=str, default="Greetings, Welcome to try out our Uni MOE Text to Speech model!", help="The text content to be transform into speech.")
    parser.add_argument("--speaker", type=str, default="Brian", help="English TTS can be performed using Brian's or Jenny's voice. 中文TTS可以使用Brian和Xiaoxiao的音色。")
    parser.add_argument("--language", type=str, default="English", help="English/Chinese")
    parser.add_argument("--prompt", type=str, default=None, help="Stlye prompt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    moe_tts_dir = args.model_dir
    model,processor,wavtokenizer = load_all_models(moe_tts_dir)
    model.cuda()
    wavtokenizer = wavtokenizer.to(model.in_fnn.weight.device)
    infer_tts(model=model,
            processor=processor,
            wavtokenizer=wavtokenizer,
            wavpath=args.wav_path,
            tts_text=args.text,
            prompt=args.prompt,
            speaker=args.speaker, # Eglish: Brian/Xiaoxiao; Chiinese: Brian/Jenny
            Language=args.language # English/Chinese
            )
