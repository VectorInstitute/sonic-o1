import os
import sys
import math
import time
import tempfile
import shutil
from typing import List, Optional, Union
from pathlib import Path

import torch
import torchaudio
import torchvision
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from qwen_vl_utils import smart_resize

# Import from new utils modules, please input the Model path
import sys
import os
sys.path.append('path/to/UniMoE-Audio-preview')

import deepspeed_utils  # This line is important, do not delete it
from utils import (
    Dac,
    preprocess_codec,
    DecoderOutput,
    tts_preprocess,
    t2m_preprocess,
    v2m_preprocess,
    prepare_audio_prompt,
    generate_output,
    frame_process
)

class UniMoEAudio: 
    def __init__(self, model_path: str, device_id: int = 0):
        # Configuration parameters
        self.TORCH_DTYPE = torch.bfloat16
        # Create a temporary directory
        self.TEMP_DIR = tempfile.mkdtemp()
        # Initialize model components
        self._initialize_model(model_path, device_id)
    
    def _initialize_model(self, model_path: str, device_id: int):

        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading UniMoE Audio model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.TORCH_DTYPE,
                attn_implementation="sdpa",
                trust_remote_code=True
            ).eval().to(self.device)
            print("Using SDPA attention implementation")
        except Exception as e:
            print(f"SDPA failed, falling back to eager attention: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.TORCH_DTYPE,
                attn_implementation="eager",
                trust_remote_code=True
            ).eval().to(self.device)
            print("Using eager attention implementation")
        
        print("Loading DAC...")
        self.dac = Dac()
        
        print("Loading Processor...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        
        print("Model initialization complete!")
    
    def __del__(self):
        """Clean up the temporary directory"""
        try:
            if hasattr(self, 'TEMP_DIR') and self.TEMP_DIR and os.path.exists(self.TEMP_DIR):
                shutil.rmtree(self.TEMP_DIR, ignore_errors=True)
        except (AttributeError, TypeError):
            pass
    
    def _getList(self, obj: Union[str, List[str]]) -> List[str]:
        if isinstance(obj, str):
            if not obj.strip():
                raise ValueError("Please enter valid target texts.")
            obj = [obj]
        else:
            obj = [c for c in obj if c.strip()]
            if not obj:
                raise ValueError("Please enter valid target texts.")
        return obj

    def text_to_speech(
            self, 
            transcription: Union[str, List[str]], 
            prompt_transcription: str, 
            prompt_wav: str, 
            output_dir: str = "./",
            max_audio_seconds: int = 10,
            min_audio_seconds: int = 2,  
            temperature: float = 1.0,
            top_p: float = 1.0,
            cfg_filter_top_k: int = 45,
    ) -> List[str]:

        transcription = self._getList(transcription)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        output_paths = []

        prompt_codec = preprocess_codec(self.model, self.dac.encode(prompt_wav))
        text_input, tts_generation_kwargs = tts_preprocess(transcription, prompt_codec, prompt_transcription, self.device)
        source_input = self.tokenizer(text_input, add_special_tokens=False, return_tensors="pt", padding=True).to(self.device)

        prefill, prefill_steps = prepare_audio_prompt(self.model, audio_prompts=[None] * len(transcription))
        dec_output = DecoderOutput(prefill, prefill_steps, self.device)
                
        with torch.no_grad():
            generated_codes, lengths_Bx = self.model.generate(
                input_ids=source_input.input_ids,
                attention_mask=source_input.attention_mask,
                dec_output=dec_output,
                max_tokens= max_audio_seconds * 50, 
                min_tokens= min_audio_seconds * 50, 
                temperature= temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                do_sample=True,
                use_cache=True,
                **tts_generation_kwargs
            )

        audios = generate_output(self.model, generated_codes, lengths_Bx)
        for i in range(len(audios)):
            output_path = os.path.join(output_dir, f"generated_speech_{i}.wav")
            self.dac.decode(audios[i].transpose(0, 1).unsqueeze(0), save_path=output_path, min_duration=1)
            output_paths.append(output_path)

        return output_paths


    def text_to_music(
            self, 
            caption: Union[str, List[str]], 
            output_dir: str = "./",
            max_audio_seconds: int = 20, 
            min_audio_seconds: int = 8,  
            temperature: float = 1.0,
            top_p: float = 1.0,
            cfg_filter_top_k: int = 45,
    ) -> List[str]:
       
        caption = self._getList(caption)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        text_input, t2m_generation_kwargs = t2m_preprocess(caption)
        source_input = self.tokenizer(text_input, add_special_tokens=False, return_tensors="pt", padding=True).to(self.device)

        prefill, prefill_steps = prepare_audio_prompt(self.model, audio_prompts=[None] * len(caption))
        dec_output = DecoderOutput(prefill, prefill_steps, self.device)
                
        with torch.no_grad():
            generated_codes, lengths_Bx = self.model.generate(
                input_ids=source_input.input_ids,
                attention_mask=source_input.attention_mask,
                dec_output=dec_output,
                max_tokens=max_audio_seconds * 50, 
                min_tokens=min_audio_seconds * 50, 
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                do_sample=True,
                use_cache=True,
                **t2m_generation_kwargs
            )
                
        audios = generate_output(self.model, generated_codes, lengths_Bx)
        for i in range(len(audios)):
            output_path = os.path.join(output_dir, f"generated_music_{i}.wav")
            self.dac.decode(audios[i].transpose(0, 1).unsqueeze(0), save_path=output_path, min_duration=1)
            output_paths.append(output_path)
        
        return output_paths
        
        
    def video_text_to_music(
            self, 
            video: Union[str, List[str]], 
            caption: Union[str, List[str]], 
            output_dir: str = "./",
            max_audio_seconds: int = 20, 
            min_audio_seconds: int = 8,  
            temperature: float = 1.0,
            top_p: float = 1.0,
            cfg_filter_top_k: int = 45,
    ) -> List[str]:
        video = self._getList(video)
        caption = self._getList(caption)

        if len(video) != len(caption):
            print("The number of videos and captions must be the same.")
            return []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        output_paths = []

        text_input,  video_inputs, fps_inputs, v2m_generation_kwargs = v2m_preprocess(caption, video)
        source_input = self.processor(text=text_input, images=None, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt", do_resize=False)
        source_input = source_input.to(self.device)

        prefill, prefill_steps = prepare_audio_prompt(self.model, audio_prompts=[None] * len(caption))
        dec_output = DecoderOutput(prefill, prefill_steps, self.device)
                
        with torch.no_grad():
            generated_codes, lengths_Bx = self.model.generate(
                input_ids=source_input.input_ids,
                pixel_values_videos=source_input.pixel_values_videos,
                video_grid_thw=source_input.video_grid_thw,
                second_per_grid_ts=source_input.second_per_grid_ts,
                attention_mask=source_input.attention_mask,
                dec_output=dec_output,
                max_tokens=max_audio_seconds * 50, 
                min_tokens=min_audio_seconds * 50, 
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                do_sample=True,
                use_cache=True,
                **v2m_generation_kwargs
            )
                
        audios = generate_output(self.model, generated_codes, lengths_Bx)
        for i in range(len(audios)):
            output_path = os.path.join(output_dir, f"generated_video_music_{i}.wav")
            self.dac.decode(audios[i].transpose(0, 1).unsqueeze(0), save_path=output_path, min_duration=1)
            output_paths.append(output_path)
        
        return output_paths

# Convenience function
def create_unimoe_audio(model_path: str, device_id: int = 0) -> UniMoEAudio:
    return UniMoEAudio(model_path, device_id)
