"""
models/phi4.py

Phi-4 Multimodal implementation following BaseModel pattern.
Self-contained with all Phi-4-specific logic.

Based on microsoft/Phi-4-multimodal-instruct
https://huggingface.co/microsoft/Phi-4-multimodal-instruct
"""

import os
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, Literal, List, Tuple
from PIL import Image

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class Phi4(BaseModel):
    """Phi-4 Multimodal wrapper following BaseModel pattern"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        self.model_path = config.get('model_path', 'microsoft/Phi-4-multimodal-instruct')
        
        # Device config
        self.device_map = config.get('device_map', 'auto')
        self.torch_dtype = config.get('torch_dtype', 'bfloat16')
        self.attn_implementation = config.get('attn_implementation', 'eager')
        self.trust_remote_code = config.get('trust_remote_code', True)
        
        # Frame config
        self.default_max_frames = config.get('max_frames', 256)
        self.default_min_frames = config.get('min_frames', 64)
        
        # Audio config
        self.audio_sample_rate = config.get('audio_sample_rate', 16000)
        self.audio_mono = config.get('audio_mono', True)
        
        # Generation config
        gen_config = config.get('generation_config', {})
        self.temperature = gen_config.get('temperature', 0.7)
        self.top_p = gen_config.get('top_p', 0.95)
        self.max_new_tokens = gen_config.get('max_new_tokens', 8192)
        
        # Model components (loaded in load())
        self.processor = None
        self.model = None
        self.generation_config = None
        
        # Stats
        self.stats = {
            'total_samples': 0,
            'audio_chunks_sampled': 0,
            'avg_frames_per_sample': 0,
            'total_frames_processed': 0,
        }
    
    def load(self):
        """Load Phi-4 model and processor"""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers. Please install: pip install transformers\n"
                f"Error: {e}"
            )
        
        logger.info(f"Loading Phi-4 from {self.model_path}")
        logger.info(f"Using device_map={self.device_map}, dtype={self.torch_dtype}")
        logger.info(f"Found {torch.cuda.device_count()} GPUs")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        
        # Determine torch dtype
        if self.torch_dtype == "auto":
            torch_dtype = "auto"
        elif self.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.torch_dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = "auto"
            logger.warning(f"Unknown dtype {self.torch_dtype}, using 'auto'")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
            _attn_implementation=self.attn_implementation,
        )
        
        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        
        # Log device distribution
        if hasattr(self.model, "hf_device_map"):
            logger.info("=== Device Map ===")
            device_counts = {}
            for module, device in self.model.hf_device_map.items():
                device_counts[device] = device_counts.get(device, 0) + 1
            for device, count in sorted(device_counts.items(), key=lambda x: str(x[0])):
                logger.info(f"  {device}: {count} modules")
            logger.info("==================")
        
        self.model.eval()
        logger.info("Phi-4 loaded successfully")
    
    def _extract_video_frames(
        self, 
        video_path: Union[str, Path], 
        max_frames: int
    ) -> List[Image.Image]:
        """
        Extract frames from video using OpenCV with uniform sampling.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of PIL Images
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV required for frame extraction. "
                "Install with: pip install opencv-python"
            )
        
        video_path = str(video_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(
            f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration"
        )
        
        # Calculate frame indices to extract (uniform sampling)
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(
                0, total_frames - 1, max_frames, dtype=int
            ).tolist()
        
        # Ensure minimum frames
        if len(frame_indices) < self.default_min_frames:
            # For very short videos, repeat frames if necessary
            repeat_count = (self.default_min_frames + len(frame_indices) - 1) // len(frame_indices)
            frame_indices = (frame_indices * repeat_count)[:self.default_min_frames]
        
        # Extract frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            else:
                logger.warning(f"Failed to read frame at index {idx}")
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not extract any frames from video: {video_path}")
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def _load_audio(
        self, 
        audio_path: Union[str, Path]
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from file with automatic format detection.
        Supports m4a, mp3, wav, flac, etc.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa required for audio loading. "
                "Install with: pip install librosa"
            )
        
        audio_path = str(audio_path)
        
        logger.info(f"Loading audio: {audio_path}")
        
        # Load audio with librosa (handles all formats via ffmpeg)
        audio, sr = librosa.load(
            audio_path,
            sr=self.audio_sample_rate,
            mono=self.audio_mono
        )
        
        duration = len(audio) / sr
        logger.info(f"Loaded audio: {duration:.2f}s @ {sr}Hz, mono={self.audio_mono}")
        
        return audio, sr
    
    def _chunk_audio(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        max_chunks: Optional[int],
        chunk_duration_sec: float = 10.0
    ) -> Tuple[np.ndarray, int]:
        """
        Chunk audio to maximum duration if max_chunks is specified.
        
        Args:
            audio_array: Audio waveform
            sample_rate: Sample rate
            max_chunks: Maximum number of chunks (None = no chunking)
            chunk_duration_sec: Duration of each chunk in seconds
        
        Returns:
            Tuple of (chunked_audio, sample_rate)
        """
        if max_chunks is None:
            return audio_array, sample_rate
        
        # Calculate max samples
        max_samples = int(max_chunks * chunk_duration_sec * sample_rate)
        
        if len(audio_array) > max_samples:
            original_duration = len(audio_array) / sample_rate
            chunked_duration = max_samples / sample_rate
            
            logger.info(
                f"Chunking audio: {original_duration:.2f}s -> {chunked_duration:.2f}s "
                f"({max_chunks} chunks × {chunk_duration_sec}s)"
            )
            
            self.stats['audio_chunks_sampled'] += 1
            return audio_array[:max_samples], sample_rate
        
        return audio_array, sample_rate
    
    def generate(
        self,
        frames: Union[List[np.ndarray], np.ndarray, str],
        audio: Optional[Union[np.ndarray, str]],
        prompt: str,
        fps: Optional[float] = None,
        video_category: Optional[Literal['short', 'medium', 'long']] = None,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response from video frames and audio.
        
        Args:
            frames: Video file path (str) - we extract frames externally
            audio: Audio file path (str) or None
            prompt: Text prompt for the model
            fps: Ignored (kept for API compatibility)
            video_category: Ignored (kept for API compatibility)
            max_frames: Maximum frames to extract (set by external retry)
            max_audio_chunks: Maximum audio chunks (set by external retry)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text response
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not isinstance(frames, str):
            raise ValueError(
                f"Phi-4 requires video file path (str), got {type(frames)}"
            )
        
        video_path = Path(frames)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Use external max_frames or default
        actual_max_frames = max_frames if max_frames is not None else self.default_max_frames
        
        try:
            logger.info(
                f"Processing: max_frames={actual_max_frames}, "
                f"max_audio_chunks={max_audio_chunks}"
            )
            
            # 1. Extract video frames
            video_frames = self._extract_video_frames(video_path, actual_max_frames)
            num_frames = len(video_frames)
            
            # Track stats
            self.stats['total_samples'] += 1
            self.stats['total_frames_processed'] += num_frames
            self.stats['avg_frames_per_sample'] = (
                self.stats['total_frames_processed'] / self.stats['total_samples']
            )
            
            # 2. Load and chunk audio if provided
            has_audio = audio is not None and isinstance(audio, str) and os.path.exists(audio)
            
            if has_audio:
                audio_array, sr = self._load_audio(audio)
                
                # Chunk audio if needed
                audio_array, sr = self._chunk_audio(
                    audio_array,
                    sr,
                    max_audio_chunks,
                    kwargs.get('audio_chunk_duration_sec', 10.0)
                )
                
                audio_input = [(audio_array, sr)]
            else:
                audio_input = None
                logger.info("No audio provided")
            
            # 3. Build Phi-4 prompt with special tokens
            user_prompt = '<|user|>'
            assistant_prompt = '<|assistant|>'
            prompt_suffix = '<|end|>'
            
            # Build image placeholders: <|image_1|><|image_2|>...<|image_n|>
            image_placeholders = ''.join([f'<|image_{i+1}|>' for i in range(num_frames)])
            
            # Build full prompt based on modality
            if has_audio:
                # Video + Audio format
                full_prompt = (
                    f'{user_prompt}{image_placeholders}<|audio_1|>'
                    f'{prompt}{prompt_suffix}{assistant_prompt}'
                )
            else:
                # Video only format
                full_prompt = (
                    f'{user_prompt}{image_placeholders}'
                    f'{prompt}{prompt_suffix}{assistant_prompt}'
                )
            
            logger.info(f"Prompt: {full_prompt[:200]}...")
            
            # 4. Process inputs
            processor_inputs = {
                'text': full_prompt,
                'images': video_frames,
                'return_tensors': 'pt'
            }
            
            if has_audio:
                processor_inputs['audios'] = audio_input
            
            inputs = self.processor(**processor_inputs)
            device = next(self.model.parameters()).device
            clean = {}
            for k, v in inputs.items():
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    clean[k] = v.to(device)
                else:
                    clean[k] = v
            inputs = clean
            # Get generation parameters
            temperature = kwargs.get('temperature', self.temperature)
            top_p = kwargs.get('top_p', self.top_p)
            max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
            
            logger.info("Generating response...")
            
            # 5. Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=False, 
                    generation_config=self.generation_config,
                )
            
            # 6. Decode (skip input tokens)
            input_length = inputs['input_ids'].shape[1]
            output_ids = output_ids[:, input_length:]
            
            response = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Generated response ({len(response)} chars)")
            
            return self.postprocess_output(response)
        
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e)
            
            # Check for OOM
            if "out of memory" in error_msg.lower() or "CUDA out of memory" in error_msg:
                logger.error(f"OOM error: {error_msg[:200]}...")
                self._clear_memory()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(f"Out of memory: {e}")
            
            # Check for CUDA errors
            elif "CUDA" in error_msg or "device" in error_msg:
                logger.error(f"CUDA error: {error_msg[:200]}...")
                raise RuntimeError(f"CUDA error: {e}")
            
            # Check for context length errors
            elif any(keyword in error_msg.lower() for keyword in [
                'context', 'token', 'length', 'limit', 'maximum', 'exceed'
            ]):
                logger.error(f"Context length error: {error_msg[:200]}...")
                raise RuntimeError(f"Context length exceeded: {e}")
            
            else:
                logger.error(f"Generation failed: {e}", exc_info=True)
                raise RuntimeError(f"Generation failed: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise RuntimeError(f"Generation failed: {e}")
    
    def unload(self):
        """Unload model and free memory"""
        logger.info("Unloading Phi-4 model...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if self.generation_config is not None:
            del self.generation_config
            self.generation_config = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Phi-4 model unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            'model_path': self.model_path,
            'backend': 'HuggingFace Transformers',
            'native_video': False,  # We extract frames externally
            'native_audio': True,
            'device_map': self.device_map,
            'torch_dtype': self.torch_dtype,
            'attn_implementation': self.attn_implementation,
            'default_max_frames': self.default_max_frames,
            'default_min_frames': self.default_min_frames,
            'audio_sample_rate': self.audio_sample_rate,
            'audio_mono': self.audio_mono,
            'statistics': self.stats,
        })
        return info
    def _clear_memory(self):
        """Aggressively clear GPU memory"""
        # Clear gradients
        if self.model is not None:
            self.model.zero_grad(set_to_none=True)
        
        # Clear all cached tensors
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Clear IPC cache
            torch.cuda.ipc_collect()
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats