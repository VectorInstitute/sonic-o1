"""
models/vita.py

VITA-1.5 implementation following BaseModel pattern.
Self-contained with all VITA-specific logic.
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, Literal
from PIL import Image
from decord import VideoReader, cpu
from transformers import LogitsProcessor
from .base_model import BaseModel
from utils.audio_processor import sample_audio_chunks

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_TOKEN = "<audio>"
DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200


class ForceFirstToken(LogitsProcessor):
    def __init__(self, token_id: int):
        self.token_id = token_id
        self.used = False

    def __call__(self, input_ids, scores):
        if not self.used:
            scores[:] = -float("inf")
            scores[:, self.token_id] = 0
            self.used = True
        return scores

class VITA(BaseModel):
    """VITA-1.5 wrapper following BaseModel pattern"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        self.model_path = config.get('model_path', 'VITA-MLLM/VITA-1.5')
        self.vita_repo_path = config.get('vita_repo_path', None)
        self.model_type = config.get('model_type', 'qwen2p5_instruct')
        self.conv_mode = config.get('conv_mode', 'qwen2p5_instruct')
        
        # Frame config
        self.default_max_frames = config.get('max_frames', 256)
        self.default_min_frames = config.get('min_frames', 4)
        self.video_framerate = config.get('video_framerate', 1)
        self.image_aspect_ratio = config.get('image_aspect_ratio', 'pad')
        
        # Audio config
        self.audio_feature_rate = 12.5  # VITA audio encoder output rate
        
        # Generation config
        gen_config = config.get('generation_config', {})
        self.temperature = gen_config.get('temperature', 0.01)
        self.top_p = gen_config.get('top_p', None)
        self.num_beams = gen_config.get('num_beams', 1)
        self.max_new_tokens = gen_config.get('max_new_tokens', 1024)
        
        # Model components (loaded in load())
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.audio_processor = None
        self.context_len = None
        self.conv_templates = None
        
        # Stats
        self.stats = {
            'total_samples': 0,
            'audio_chunks_sampled': 0,
        }

    def convert_av1_to_h264(self, video_path: Path, output_dir: Optional[Path] = None) -> Path:
            """
            Convert AV1 video to H.264 for Decord compatibility.
            
            Args:
                video_path: Input video path
                output_dir: Output directory (default: creates 'converted' subdir)
            
            Returns:
                Path to converted video
            """
            import subprocess
            
            if output_dir is None:
                output_dir = video_path.parent / "converted"
                output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / f"{video_path.stem}_h264{video_path.suffix}"
            
            # Check if already converted
            if output_path.exists():
                logger.info(f"Using cached converted video: {output_path.name}")
                return output_path
            
            logger.info(f"Converting AV1 to H.264: {video_path.name}")
            
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'copy',
                '-y',
                str(output_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"✓ Conversion successful: {output_path.name}")
                return output_path
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Conversion failed: {e.stderr}")
                raise RuntimeError(f"Failed to convert AV1 video: {e.stderr}")
            
    def _check_video_compatibility(self, video_path: Path) -> Optional[Path]:
        """
        Check if video is compatible with Decord.
        If AV1, automatically convert to H.264.
        
        Args:
            video_path: Original video path
            
        Returns:
            Path to compatible video (original or converted)
        """
        from decord import VideoReader, cpu
        import subprocess
        
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
            frame_count = len(vr)
            del vr
            logger.info(f"✓ Video compatible: {video_path.name} ({frame_count} frames)")
            return video_path
        except Exception as e:
            logger.warning(f"⚠ Video incompatible with Decord: {video_path.name}")
            
            # Detect codec
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name', 
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(video_path)],
                    capture_output=True, text=True, check=True
                )
                codec = result.stdout.strip()
                
                if codec == 'av1':
                    logger.info(f"  → Detected AV1 codec - converting to H.264...")
                    try:
                        converted_path = self.convert_av1_to_h264(video_path)
                        return converted_path
                    except Exception as conv_error:
                        logger.error(f"  ✗ Conversion failed: {conv_error}")
                        return None
                else:
                    logger.warning(f"  ✗ Unsupported codec '{codec}' - skipping")
                    return None
            except Exception as probe_error:
                logger.error(f"  ✗ Failed to detect codec: {probe_error}")
                return None
                
    def load(self):
        """Load VITA model"""
        # Add VITA repo to path if specified
        if self.vita_repo_path:
            vita_path = os.path.expanduser(self.vita_repo_path)
            if vita_path not in sys.path:
                sys.path.insert(0, vita_path)
                logger.info(f"Added VITA repo to path: {vita_path}")
        
        # Import VITA modules
        try:
            from vita.conversation import conv_templates
            from vita.model.builder import load_pretrained_model
            from vita.util.mm_utils import get_model_name_from_path
            from vita.util.utils import disable_torch_init
        except ImportError as e:
            raise ImportError(
                f"Failed to import VITA modules. Set 'vita_repo_path' in config.\n"
                f"Error: {e}"
            )
        
        self.conv_templates = conv_templates
        
        logger.info(f"Loading VITA from {self.model_path}")
        disable_torch_init()
        
        model_path = os.path.expanduser(self.model_path)
        model_name = get_model_name_from_path(model_path)
        
        # Load model
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, self.model_type
        )
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Load vision tower
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        self.image_processor = vision_tower.image_processor
        
        # Load audio encoder
        audio_encoder = self.model.get_audio_encoder()
        audio_encoder.to(dtype=torch.float16)
        self.audio_processor = audio_encoder.audio_processor
        
        self.model.eval()
        
        logger.info(f"VITA loaded successfully ({self.model_type})")
    
    def _get_rawvideo_dec(
        self,
        video_path: str,
        max_frames: int,
        min_frames: Optional[int] = None,
        s: Optional[float] = None,
        e: Optional[float] = None,
    ):
        """Extract video frames using decord (VITA-specific)"""
        if min_frames is None:
            min_frames = self.default_min_frames
        
        # Handle segment times
        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0.0 else 0.0
            end_time = end_time if end_time >= 0.0 else 0.0
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        vreader = VideoReader(video_path, ctx=cpu(0))
        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        
        if num_frames > 0:
            sample_fps = int(self.video_framerate)
            t_stride = int(round(float(fps) / sample_fps))
            all_pos = list(range(f_start, f_end + 1, t_stride))
            
            # Sample frames based on max_frames
            if len(all_pos) > max_frames:
                sample_pos = [
                    all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
                ]
            elif len(all_pos) < min_frames:
                sample_pos = [
                    all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
                ]
            else:
                sample_pos = all_pos
            
            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            
            # Apply padding if needed
            if self.image_aspect_ratio == "pad":
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                
                patch_images = [
                    expand2square(i, tuple(int(x * 255) for x in self.image_processor.image_mean))
                    for i in patch_images
                ]
            
            # Preprocess
            patch_images = [
                self.image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
            
            patch_images = torch.stack(patch_images)
            slice_len = patch_images.shape[0]
            return patch_images, slice_len
        else:
            raise ValueError(f"video path: {video_path} error.")
    
    def generate(
        self,
        frames: Union[str, Path],
        audio: Optional[Union[str, Path]] = None,
        prompt: str = "Describe what you see and hear.",
        fps: Optional[float] = None,
        video_category: Optional[Literal['short', 'medium', 'long']] = None,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response from video and audio."""
        from vita.conversation import SeparatorStyle
        from vita.util.mm_utils import (
            KeywordsStoppingCriteria,
            tokenizer_image_audio_token,
            tokenizer_image_token,
        )
        
        actual_max_frames = max_frames if max_frames is not None else self.default_max_frames
        
        temperature = kwargs.get('temperature', self.temperature)
        top_p = kwargs.get('top_p', self.top_p)
        num_beams = kwargs.get('num_beams', self.num_beams)
        max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
        
        try:
            self.stats['total_samples'] += 1
            
            # Process audio with error handling
            has_audio = audio is not None and os.path.exists(str(audio))
            if has_audio:
                logger.info(f"Loading audio: {audio}")
                try:
                    audio_features, audio_for_llm_lens = self.audio_processor.process(str(audio))
                    logger.info(f"Original audio: {audio_features.shape[0]} frames")
                    
                    # Validate audio features
                    if audio_features.shape[0] == 0:
                        logger.warning("Audio features are empty, falling back to dummy audio")
                        has_audio = False
                    else:
                        # Apply audio chunking if max_chunks specified
                        if max_audio_chunks is not None:
                            chunk_duration = kwargs.get('audio_chunk_duration_sec', 10.0)
                            
                            audio_features, audio_for_llm_lens = sample_audio_chunks(
                                audio_features,
                                audio_for_llm_lens,
                                chunk_duration_sec=chunk_duration,
                                feature_rate=self.audio_feature_rate,
                                max_chunks=max_audio_chunks
                            )
                            self.stats['audio_chunks_sampled'] += 1
                        
                except Exception as e:
                    logger.error(f"Audio processing failed: {e}")
                    logger.warning("Falling back to dummy audio")
                    has_audio = False
            
            # Prepare audio tensors (either real or dummy)
            if has_audio:
                audio_length = audio_features.shape[0]
                audio_tensor = torch.unsqueeze(audio_features, dim=0)
                audio_length_tensor = torch.unsqueeze(torch.tensor(audio_length), dim=0)
                audio_for_llm_lens_tensor = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
                
                audios = {
                    "audios": audio_tensor.half().cuda(),
                    "lengths": audio_length_tensor.half().cuda(),
                    "lengths_for_llm": audio_for_llm_lens_tensor.cuda()
                }
            else:
                # Dummy audio
                dummy_audio = torch.zeros(400, 80)
                audio_length = dummy_audio.shape[0]
                audio_for_llm_lens = 60
                dummy_audio = torch.unsqueeze(dummy_audio, dim=0)
                audio_length_tensor = torch.unsqueeze(torch.tensor(audio_length), dim=0)
                audio_for_llm_lens_tensor = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
                
                audios = {
                    "audios": dummy_audio.half().cuda(),
                    "lengths": audio_length_tensor.half().cuda(),
                    "lengths_for_llm": audio_for_llm_lens_tensor.cuda()
                }
            
            # Process video (rest of your code remains the same)
            has_video = frames is not None and os.path.exists(str(frames))
            if has_video:
                from pathlib import Path
                video_path = Path(frames)
                
                compatible_video_path = self._check_video_compatibility(video_path)
                
                if compatible_video_path is None:
                    raise RuntimeError(
                        f"Video codec incompatible with Decord (likely AV1). "
                        f"Skipping this video: {video_path}"
                    )
                
                logger.info(f"Loading video: {compatible_video_path}")
                video_frames, slice_len = self._get_rawvideo_dec(
                    str(compatible_video_path),
                    max_frames=actual_max_frames,
                    min_frames=self.default_min_frames,
                )
                image_tensor = video_frames.half().cuda()
                
                sf_masks = torch.ones(slice_len, dtype=torch.long, device=image_tensor.device)
                if has_audio:
                    qs = ((DEFAULT_IMAGE_TOKEN + "\n") * slice_len) + DEFAULT_AUDIO_TOKEN + "\n" + prompt
                else:
                    qs = ((DEFAULT_IMAGE_TOKEN + "\n") * slice_len) + "\n" + prompt
                        
                modality = "video"
                logger.info(f"Video frames: {slice_len}")
            else:
                image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=self.model.dtype, device="cuda")
                slice_len = 0
                
                if has_audio:
                    qs = prompt + DEFAULT_AUDIO_TOKEN
                else:
                    qs = prompt
                
                modality = "lang"
            
            # Rest of generation code unchanged...
            conv = self.conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt_formatted = conv.get_prompt(modality)
            
            if has_audio:
                input_ids = (
                    tokenizer_image_audio_token(
                        prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )
            else:
                input_ids = (
                    tokenizer_image_token(
                        prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )
            
            if conv.sep_style == SeparatorStyle.Qwen2p5Instruct:
                stop_str = "<|im_end|>"
            elif conv.sep_style == SeparatorStyle.TWO:
                stop_str = conv.sep2
            else:
                stop_str = conv.sep
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            logger.info("Generating response...")
            lbrace_id = self.tokenizer.encode("{", add_special_tokens=False)[0]
            force_json_start = ForceFirstToken(lbrace_id)
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    audios=audios,
                    sf_masks=sf_masks,
                    do_sample=False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    logits_processor=[force_json_start],
                    eos_token_id=151645,
                    pad_token_id=151643,
                )
            
            output_ids = output_ids.sequences
            input_token_len = input_ids.shape[1]
            
            if self.model_type == "mixtral-8x7b":
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    logger.warning(f"{n_diff_input_output} output_ids differ from input_ids")
                    output_ids = output_ids[:, input_token_len:]
            
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            
            logger.info(f"Generated {len(outputs)} characters")
            
            return self.postprocess_output(outputs)
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e)
            
            if "out of memory" in error_msg.lower() or "size of tensor" in error_msg.lower():
                logger.error(f"OOM error: {error_msg[:200]}...")
                torch.cuda.empty_cache()
                raise RuntimeError(f"Out of memory: {e}")
            else:
                logger.error(f"Generation failed: {e}")
                raise RuntimeError(f"Generation failed: {e}")
    
    def unload(self):
        """Unload model and free memory"""
        logger.info("Unloading VITA model...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.image_processor is not None:
            del self.image_processor
            self.image_processor = None
        
        if self.audio_processor is not None:
            del self.audio_processor
            self.audio_processor = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("VITA model unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            'model_path': self.model_path,
            'model_type': self.model_type,
            'backend': 'HuggingFace Transformers',
            'native_video': True,
            'native_audio': True,
            'default_max_frames': self.default_max_frames,
            'default_min_frames': self.default_min_frames,
            'video_framerate': self.video_framerate,
            'context_length': self.context_len,
            'statistics': self.stats,
        })
        return info