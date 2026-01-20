"""
models/unimoe.py
Uni-MoE-2.0-Omni implementation with video and audio support.
Supports both single-GPU and multi-GPU inference.
"""
import os
import sys
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

try:
    import torch
    import deepspeed
    import torch.distributed as dist
except ImportError as e:
    raise ImportError(
        f"Please install required packages: {e}\n"
        "pip install torch deepspeed"
    )

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class UniMoe(BaseModel):
    """
    Uni-MoE-2.0-Omni wrapper with native video and audio support.
    Supports both single-GPU and multi-GPU inference modes.
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Model configuration
        self.model_path = config.get('model_path', 'HIT-TMG/Uni-MoE-2.0-Omni')
        
        # Handle Uni-MoE package path
        self.unimoe_package_path = config.get('unimoe_package_path', None)
        if self.unimoe_package_path:
            unimoe_path = str(Path(self.unimoe_package_path).resolve())
            if unimoe_path not in sys.path:
                sys.path.insert(0, unimoe_path)
                logger.info(f"Added Uni-MoE package path: {unimoe_path}")
        
        # Import Uni-MoE components after path is set
        try:
            from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
            from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
            from uni_moe.qwen_vl_utils import process_mm_info
            # Import inference utils to patch DeepSpeed MoE for single-machine inference
            from uni_moe.model import deepspeed_moe_inference_utils
            
            self.Qwen2VLProcessor = Qwen2VLProcessor
            self.GrinQwen2VLOutForConditionalGeneration = GrinQwen2VLOutForConditionalGeneration
            self.process_mm_info = process_mm_info
        except ImportError as e:
            raise ImportError(
                f"Failed to import Uni-MoE components: {e}\n"
                "Please ensure 'unimoe_package_path' is set in config or Uni-MoE is in PYTHONPATH"
            )
        
        # Device configuration
        # Options: 'cuda:0' (single GPU), 'auto' (multi-GPU), or specific device
        self.device = config.get('device', 'cuda:0')
        self.multi_gpu = self.device == 'auto'
        
        self.dtype = config.get('dtype', 'bfloat16')
        
        # Parse dtype
        if self.dtype == 'bfloat16':
            self.torch_dtype = torch.bfloat16
        elif self.dtype == 'float16':
            self.torch_dtype = torch.float16
        elif self.dtype == 'float32':
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.bfloat16
        
        # Generation config
        gen_config = config.get('generation_config', {})
        self.temperature = gen_config.get('temperature', 0.7)
        self.top_p = gen_config.get('top_p', 0.95)
        self.max_new_tokens = gen_config.get('max_new_tokens', 2048)
        
        # Video processing config
        self.default_max_frames = config.get('max_frames', 480)
        self.default_min_frames = config.get('min_frames', 64)
                
        # DeepSpeed initialization flag (only for single-GPU mode)
        self._deepspeed_initialized = False
        
        self.model = None
        self.processor = None
    
    def _init_deepspeed_single_gpu(self):
        """Initialize DeepSpeed for single-GPU mode (not needed for multi-GPU)"""
        if self.multi_gpu:
            logger.info("Multi-GPU mode - skipping DeepSpeed initialization")
            return
            
        if self._deepspeed_initialized or dist.is_initialized():
            logger.info("DeepSpeed already initialized, skipping...")
            return
        
        try:
            logger.info("Initializing DeepSpeed for single-GPU mode...")
            
            # Set environment variables for single-GPU distributed setup
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["LOCAL_RANK"] = "0"
            
            # Initialize DeepSpeed distributed backend
            deepspeed.init_distributed(dist_backend="nccl")
            
            self._deepspeed_initialized = True
            logger.info("DeepSpeed initialized successfully")
            
        except Exception as e:
            logger.warning(f"DeepSpeed initialization warning: {e}")
            # Continue anyway - might still work
    
    def load(self):
        """Load the Uni-MoE model and processor"""
        try:
            logger.info(f"Loading Uni-MoE model from {self.model_path}")
            
            if self.multi_gpu:
                num_gpus = torch.cuda.device_count()
                logger.info(f"Multi-GPU mode enabled - using {num_gpus} GPUs with device_map='auto'")
            else:
                logger.info(f"Single-GPU mode - using device: {self.device}")
                # Initialize DeepSpeed for single-GPU
                self._init_deepspeed_single_gpu()
            
            # Load processor
            self.processor = self.Qwen2VLProcessor.from_pretrained(self.model_path)
            
            # Load model with appropriate device mapping
            if self.multi_gpu:
                # Multi-GPU: use device_map="auto" for automatic layer distribution
                self.model = self.GrinQwen2VLOutForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    device_map="auto",  # Automatically split across GPUs
                    low_cpu_mem_usage=True,
                )
                
                # Print device distribution
                if hasattr(self.model, 'hf_device_map'):
                    logger.info("=== Device Map ===")
                    device_distribution = {}
                    for name, device in self.model.hf_device_map.items():
                        device_distribution[device] = device_distribution.get(device, 0) + 1
                    for device, count in sorted(device_distribution.items()):
                        logger.info(f"  {device}: {count} modules")
                    logger.info("==================")
            else:
                # Single-GPU: load normally and move to specified device
                self.model = self.GrinQwen2VLOutForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                )
                self.model.to(self.device)
            
            # Set processor data args from model config
            self.processor.data_args = self.model.config
            
            if self.multi_gpu:
                logger.info(f"Successfully loaded Uni-MoE across {torch.cuda.device_count()} GPUs")
            else:
                logger.info(f"Successfully loaded Uni-MoE on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Uni-MoE model: {e}")
        
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
                    
    def generate(
        self,
        frames: Optional[str] = None,  # Video file path (now optional)
        audio: Optional[str] = None,   # Audio file path (now optional)
        prompt: str = "",
        fps: Optional[float] = None,
        video_category: Optional[str] = None,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response from video and/or audio.
        
        Args:
            frames: Path to video file (optional)
            audio: Path to audio file (optional)
            prompt: Text prompt for generation
            fps: Ignored - Uni-MoE samples based on max_frames limit
            video_category: Video length category (unused, for API consistency)
            max_frames: Maximum frames for video processing
            max_audio_chunks: Unused for Uni-MoE
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Determine modality mode
        if frames is None and audio is None:
            raise ValueError("At least one of 'frames' or 'audio' must be provided")
        
        has_video = frames is not None
        has_audio = audio is not None
        
        if has_video and has_audio:
            modality_mode = "video+audio"
            modal_type = 'video'
        elif has_video:
            modality_mode = "video-only"
            modal_type = 'video'
        else:
            modality_mode = "audio-only"
            modal_type = 'audio'
        
        logger.info(f"Modality mode: {modality_mode}")
        
        # Use max_frames from parameter if provided, otherwise use default
        actual_max_frames = max_frames if max_frames is not None else self.default_max_frames
        actual_min_frames = self.default_min_frames
        
        try:
            # Handle video path if present
            video_path = None
            if has_video:
                if not isinstance(frames, str):
                    raise ValueError(
                        f"Uni-MoE requires video file path (str), got {type(frames)}"
                    )
                video_path = Path(frames)
                if not video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                video_path = self._check_video_compatibility(video_path)
                if video_path is None:
                    raise RuntimeError(
                        f"Video codec incompatible with Decord (likely AV1). "
                        f"Skipping this video."
                    )
                logger.info(f"Processing video: {video_path.name}")
            
            # Handle audio path if present
            audio_path = None
            if has_audio:
                if not isinstance(audio, str):
                    raise ValueError(
                        f"Uni-MoE requires audio file path (str), got {type(audio)}"
                    )
                audio_path = Path(audio)
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                logger.info(f"Processing audio: {audio_path.name}")
            
            # Build text prompt with appropriate modality tokens
            text_prompt_parts = []
            if has_video:
                text_prompt_parts.append("<video>")
            if has_audio:
                text_prompt_parts.append("<audio>")
            text_prompt_parts.append(prompt)
            text_prompt = "\n".join(text_prompt_parts)
            
            # Build content list with appropriate modalities
            content = [{"type": "text", "text": text_prompt}]
            
            if has_video:
                content.append({
                    "type": "video",
                    "video": str(video_path),
                    "max_frames": actual_max_frames,
                    "min_frames": actual_min_frames
                })
            
            if has_audio:
                content.append({
                    "type": "audio",
                    "audio": str(audio_path)
                })
            
            messages = [{"role": "user", "content": content}]
            
            logger.info(f"Text prompt tokens: {text_prompt_parts[:-1]}")  # Show modality tokens
            if has_video:
                logger.info(f"Using max_frames: {actual_max_frames}, min_frames: {actual_min_frames}")
            
            # Apply chat template
            texts = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Token replacements (required for Uni-MoE)
            texts = texts.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            texts = texts.replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>")
            texts = texts.replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
            
            # Process multimodal inputs
            image_inputs, video_inputs, audio_inputs = self.process_mm_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                audios=audio_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Apply Uni-MoE specific fixes
            if "second_grid_ts" in inputs:
                inputs["second_per_grid_ts"] = inputs["second_grid_ts"]
                del inputs["second_grid_ts"]
            
            if inputs["input_ids"].dim() == 1:
                inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
            
            # Move inputs to appropriate device
            if self.multi_gpu:
                first_device = next(self.model.parameters()).device
                inputs = {k: v.to(first_device) if isinstance(v, torch.Tensor) else v 
                        for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)
            
            # Convert visual/audio features to model dtype
            for k, v in inputs.items():
                if k in ["pixel_values", "pixel_values_videos", "audio_features"]:
                    inputs[k] = v.to(dtype=self.torch_dtype)
            
            logger.info(f"Input shape: {inputs['input_ids'].shape}")
            
            # Get generation parameters
            temperature = kwargs.get('temperature', self.temperature)
            top_p = kwargs.get('top_p', self.top_p)
            max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
            
            logger.info(f"Generating response (temp={temperature}, top_p={top_p}, max_tokens={max_new_tokens})...")
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0
                )
            
            # Decode (skip input tokens)
            input_token_len = inputs["input_ids"].shape[-1]
            generated_tokens = output_ids[:, input_token_len:]
            response_text = self.processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"Generated response ({len(response_text)} chars) - Mode: {modality_mode}")
            
            return self.postprocess_output(response_text)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Generation failed: {e}")
    
    def unload(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded and memory cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            'model_path': self.model_path,
            'model_type': 'Uni-MoE Omni',
            'native_video': True,
            'native_audio': True,
            'device': str(self.device),
            'multi_gpu': self.multi_gpu,
            'dtype': str(self.dtype),
            'default_max_frames': self.default_max_frames,  # UPDATED
            'default_min_frames': self.default_min_frames,  # UPDATED
            'deepspeed_enabled': self._deepspeed_initialized,
        })
        
        if self.multi_gpu and hasattr(self.model, 'hf_device_map'):
            info['num_gpus'] = torch.cuda.device_count()
            
        return info