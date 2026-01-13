"""
models/gpt4o.py
GPT-4o implementation with video frames and caption support.
"""
import os
import time
import base64
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "Please install openai: pip install openai"
    )

from .base_model import BaseModel
from utils.frame_sampler import FrameSampler
from utils.caption_handler import CaptionHandler

logger = logging.getLogger(__name__)


class GPT4o(BaseModel):
    """
    GPT-4o wrapper with support for:
    - Video frames (extracted and encoded as images)
    - Captions (from SRT files)
    - Multimodal (frames + captions)
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Model capabilities
        self.supports_video = config.get('supports_video', True)
        self.supports_audio = False  # GPT-4o doesn't support audio streams
        self.use_captions = config.get('use_captions', False)
        
        # API configuration
        api_key_env = config.get('api_key_env', 'OPENAI_API_KEY')
        self.api_key = os.getenv(api_key_env)
        
        if not self.api_key:
            raise ValueError(
                f"API key not found. Please set {api_key_env} environment variable."
            )
        
        self.model_version = config.get('model_version', 'gpt-4o')
        
        # Generation config
        gen_config = config.get('generation_config', {})
        self.temperature = gen_config.get('temperature', 0.7)
        self.max_tokens = gen_config.get('max_tokens', 4096)
        self.top_p = gen_config.get('top_p', 1.0)
        
        # Frame configuration
        self.max_frames = config.get('max_frames', 128)
        self.image_detail = config.get('image_detail', 'auto')  # 'auto', 'low', 'high'
        
        # Retry configuration
        retry_config = config.get('retry_override', config.get('retry', {}))
        self.frame_count_fallback = retry_config.get('frame_count_fallback', [128, 64, 32, 16])
        self.caption_chunks_fallback = retry_config.get('caption_chunks_fallback', [None, 32, 16, 8])
        
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 2)
        
        # Dataset root for caption discovery
        self.dataset_root = config.get('dataset_root', None)
        
        # Initialize handlers
        self.client = None
        self.frame_sampler = None
        self.caption_handler = None
        
    def load(self):
        """Initialize OpenAI client and handlers"""
        try:
            self.client = OpenAI(api_key=self.api_key)
            
            # Initialize frame sampler if video support enabled
            if self.supports_video:
                self.frame_sampler = FrameSampler()
                logger.info("Frame sampler initialized")
            
            # Initialize caption handler if caption support enabled
            if self.use_captions:
                self.caption_handler = CaptionHandler(
                    caption_chunks_fallback=self.caption_chunks_fallback
                )
                logger.info("Caption handler initialized")
            
            logger.info(f"Loaded GPT-4o ({self.model_version})")
            logger.info(f"Video support: {self.supports_video}, Caption support: {self.use_captions}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load GPT-4o: {e}")
    
    def generate(
        self,
        frames: Union[str, List[Path]],
        audio: Optional[str],
        prompt: str,
        fps: Optional[float] = None,
        video_category: Optional[str] = None,
        max_frames: Optional[int] = None,
        max_caption_chunks: Optional[int] = None,
        caption_path: Optional[str] = None,
        segment: Optional[Dict] = None,  # NEW: {'start': 30.0, 'end': 60.0}
        **kwargs
    ) -> str:
        """
        Generate response using GPT-4o.
        
        Args:
            frames: Video file path (str) for frame extraction
            audio: Ignored (GPT-4o doesn't support audio)
            prompt: Text prompt
            fps: Ignored (kept for API compatibility)
            video_category: Ignored (kept for API compatibility)
            max_frames: Maximum frames to sample (for retry logic)
            max_caption_chunks: Maximum caption chunks (for retry logic)
            caption_path: Optional explicit caption file path
            segment: Optional segment info {'start': float, 'end': float} for caption filtering
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.client is None:
            raise RuntimeError("Client not loaded. Call load() first.")
        
        # Validate inputs based on configuration
        if self.supports_video and not self.use_captions:
            # Video-only mode
            if not isinstance(frames, str):
                raise ValueError("GPT-4o requires video file path (str) for video-only mode")
        elif not self.supports_video and self.use_captions:
            # Text-only mode - captions required
            pass  # Will auto-discover or use provided caption_path
        elif self.supports_video and self.use_captions:
            # Multimodal mode
            if not isinstance(frames, str):
                raise ValueError("GPT-4o requires video file path (str) for multimodal mode")
        else:
            raise ValueError("GPT-4o must have either video or caption support enabled")
        
        try:
            # Determine actual max frames and caption chunks
            actual_max_frames = max_frames or self.max_frames
            actual_max_caption_chunks = max_caption_chunks  # None or int
            
            # Process inputs
            video_path = Path(frames) if isinstance(frames, str) else None
            
            # Auto-discover caption path if needed and not provided
            if self.use_captions and caption_path is None and video_path is not None:
                caption_path = self.caption_handler.auto_discover_caption_path(
                    video_path, 
                    dataset_root=Path(self.dataset_root) if self.dataset_root else None
                )
            
            # Extract segment info only if caption_handler is available
            if segment is None and video_path is not None and self.caption_handler is not None:
                segment = self.caption_handler.extract_segment_info(video_path)
                if segment:
                    logger.info(f"Auto-extracted segment info: {segment}")
            
            # Build message content
            content_parts = []
            
            # Add frames if video mode
            if self.supports_video and video_path is not None:
                frame_paths = self._extract_frames(video_path, actual_max_frames)
                
                for frame_path in frame_paths:
                    base64_image = self._encode_image_to_base64(frame_path)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": self.image_detail
                        }
                    })
                
                logger.info(f"Added {len(frame_paths)} frames to request")
            
            # Add captions if caption mode
            if self.use_captions and caption_path is not None:
                # Check if we need segment-based caption extraction
                if segment is not None:
                    # Task 2/3: Extract captions for specific time segment
                    start_time = segment.get('start', 0.0)
                    end_time = segment.get('end', None)
                    
                    if end_time is not None:
                        caption_text = self.caption_handler.get_caption_text_for_segment(
                            Path(caption_path),
                            start_time=start_time,
                            end_time=end_time,
                            num_chunks=actual_max_caption_chunks
                        )
                        logger.info(f"Extracted caption for segment [{start_time:.1f}s - {end_time:.1f}s]")
                    else:
                        # No end time provided, fall back to full caption
                        caption_text = self.caption_handler.get_caption_text(
                            Path(caption_path),
                            num_chunks=actual_max_caption_chunks
                        )
                else:
                    # Task 1: Full video caption
                    caption_text = self.caption_handler.get_caption_text(
                        Path(caption_path),
                        num_chunks=actual_max_caption_chunks
                    )
                
                if caption_text:
                    # Add captions as a separate text block
                    content_parts.append({
                        "type": "text",
                        "text": f"Transcript:\n{caption_text}"
                    })
                    logger.info(f"Added caption text ({len(caption_text)} chars, chunks={actual_max_caption_chunks})")
                else:
                    logger.warning("No caption text extracted")
            
            # Add prompt
            content_parts.append({
                "type": "text",
                "text": prompt
            })
            
            # Generate with retries
            for attempt in range(self.retry_attempts):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_version,
                        messages=[
                            {
                                "role": "user",
                                "content": content_parts
                            }
                        ],
                        temperature=kwargs.get('temperature', self.temperature),
                        max_tokens=kwargs.get('max_tokens', self.max_tokens),
                        top_p=kwargs.get('top_p', self.top_p),
                    )
                    
                    response_text = response.choices[0].message.content
                    logger.info(f"Generated response ({len(response_text)} chars)")
                    
                    return self.postprocess_output(response_text)
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Handle rate limits
                    if "429" in error_str or "rate_limit" in error_str.lower():
                        if attempt < self.retry_attempts - 1:
                            wait_time = (attempt + 1) * 5
                            logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                    
                    # Handle context length errors
                    if "context_length" in error_str.lower() or "maximum context" in error_str.lower():
                        logger.error(f"Context length exceeded: {e}")
                        raise RuntimeError(f"Context length exceeded: {e}")
                    
                    logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
                        
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
    
    def _extract_frames(self, video_path: Path, num_frames: int) -> List[Path]:
        """
        Extract frames from video using FrameSampler.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of paths to extracted frame images
        """
        if self.frame_sampler is None:
            raise RuntimeError("Frame sampler not initialized")
        
        logger.info(f"Extracting {num_frames} frames from {video_path.name}")
        
        frame_paths = self.frame_sampler.sample_frames_uniform(
            video_path=video_path,
            num_frames=num_frames
        )
        
        if not frame_paths:
            raise RuntimeError(f"Failed to extract frames from {video_path}")
        
        return frame_paths
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to encode image {image_path}: {e}")
    
    def unload(self):
        """Clean up resources"""
        # Cleanup frame sampler
        if self.frame_sampler is not None:
            self.frame_sampler.cleanup()
            self.frame_sampler = None
        
        # Cleanup caption handler
        if self.caption_handler is not None:
            self.caption_handler.cleanup()
            self.caption_handler = None
        
        self.client = None
        logger.info("GPT-4o unloaded and resources cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            'model_version': self.model_version,
            'api_based': True,
            'supports_video': self.supports_video,
            'supports_audio': False,
            'use_captions': self.use_captions,
            'max_frames': self.max_frames,
            'image_detail': self.image_detail,
            'frame_count_fallback': self.frame_count_fallback,
            'caption_chunks_fallback': self.caption_chunks_fallback,
        })
        return info