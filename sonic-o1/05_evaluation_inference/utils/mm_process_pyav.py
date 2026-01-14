"""
Production-grade PyAV-based multimodal processing for Qwen3-Omni
Drop-in replacement for qwen_omni_utils.process_mm_info
Features:
- Memory-efficient video frame sampling (streams frames, no OOM)
- Fast audio loading with automatic truncation for context limits
- Metadata caching for repeated operations
- Optimized resampling and resizing
Usage:
    from mm_process_pyav import process_mm_info_pyav
    
    audios, images, videos = process_mm_info_pyav(
        conversation,
        use_audio_in_video=False,
        max_audio_duration=None  # Auto-truncate if needed
    )
"""
import os
import math
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from functools import lru_cache

import av
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ============================================================================
# Constants
# ============================================================================
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
FRAME_FACTOR = 2
FPS = 1.0  # Default FPS for sampling
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
SAMPLE_RATE = 16000

# ============================================================================
# Optimized Math Helpers
# ============================================================================
@lru_cache(maxsize=1024)
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

@lru_cache(maxsize=1024)
def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

@lru_cache(maxsize=1024)
def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

@lru_cache(maxsize=2048)
def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> Tuple[int, int]:
    """
    Calculate optimal resize dimensions maintaining aspect ratio.
    Results are cached for performance.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(int(height / beta), factor))
        w_bar = max(factor, floor_by_factor(int(width / beta), factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
    
    return h_bar, w_bar

def smart_nframes(ele: dict, total_frames: int, video_fps: float) -> int:
    """
    Calculate optimal number of frames to sample from video.
    
    Args:
        ele: Dict with optional 'nframes', 'fps', 'min_frames', 'max_frames'
        total_frames: Total frames available in video
        video_fps: Original video FPS
        
    Returns:
        Number of frames to sample (divisible by FRAME_FACTOR)
    """
    if "nframes" in ele:
        return round_by_factor(ele["nframes"], FRAME_FACTOR)
    
    fps = ele.get("fps", FPS)
    min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
    max_frames = floor_by_factor(
        ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), 
        FRAME_FACTOR
    )
    
    nframes = int(total_frames * fps / video_fps)
    nframes = max(min_frames, min(nframes, max_frames, total_frames))
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should be in [{FRAME_FACTOR}, {total_frames}], got {nframes}")
    
    return nframes

# ============================================================================
# Video Processing
# ============================================================================
@dataclass
class VideoMetadata:
    """Cached video metadata to avoid repeated file opens."""
    total_frames: int
    fps: float
    width: int
    height: int

_video_metadata_cache: Dict[str, VideoMetadata] = {}

def get_video_metadata(video_path: str) -> VideoMetadata:
    """Extract and cache video metadata."""
    if video_path in _video_metadata_cache:
        return _video_metadata_cache[video_path]
    
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    
    metadata = VideoMetadata(
        total_frames=video_stream.frames or sum(1 for _ in container.decode(video=0)),
        fps=float(video_stream.average_rate),
        width=video_stream.width,
        height=video_stream.height
    )
    
    container.close()
    _video_metadata_cache[video_path] = metadata
    return metadata

def fetch_video_pyav(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor:
    """
    Memory-efficient video loading using PyAV.
    
    Streams frames during decode instead of loading entire video.
    Frames are resized inline to minimize memory footprint.
    
    Args:
        ele: Dict with 'video' path and optional frame sampling params
        image_factor: Resize factor (default: 28)
        
    Returns:
        Video tensor of shape (nframes, C, H, W) in float32
    """
    video_path = ele["video"]
    st = time.time()
    
    try:
        metadata = get_video_metadata(video_path)
        total_frames = metadata.total_frames
        video_fps = metadata.fps
        height = metadata.height
        width = metadata.width
    except Exception:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        total_frames = video_stream.frames or sum(1 for _ in container.decode(video=0))
        video_fps = float(video_stream.average_rate)
        height = video_stream.height
        width = video_stream.width
        container.close()
    
    video_start = ele.get("video_start", 0.0)
    video_end = ele.get("video_end", None)
    
    start_frame = max(0, int(video_start * video_fps)) if video_start > 0 else 0
    if video_end is not None:
        end_frame = min(int(video_end * video_fps), total_frames - 1)
        total_frames = end_frame - start_frame + 1
    else:
        end_frame = total_frames - 1
    
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    
    if nframes == total_frames:
        indices = list(range(start_frame, end_frame + 1))
    else:
        indices = np.linspace(start_frame, end_frame, nframes, dtype=np.int32).tolist()
    
    indices_set = set(indices)
    
    container = av.open(video_path, options={
        'threads': 'auto',
        'thread_queue_size': '512',
    })
    video_stream = container.streams.video[0]
    video_stream.thread_type = 'AUTO'
    
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), 
        int(min_pixels * 1.05)
    )
    max_pixels = min(ele.get("max_pixels", max_pixels), max_pixels)
    
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=image_factor
        )
    else:
        resized_height, resized_width = smart_resize(
            height, width, factor=image_factor,
            min_pixels=min_pixels, max_pixels=max_pixels
        )
    
    frames_array = np.empty((nframes, resized_height, resized_width, 3), dtype=np.uint8)
    
    if start_frame > 10:
        try:
            container.seek(int(start_frame / video_fps * av.time_base), stream=video_stream)
            frame_idx = start_frame
        except Exception:
            frame_idx = 0
    else:
        frame_idx = 0
    
    collected = 0
    resize_transform = transforms.Resize(
        (resized_height, resized_width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    )
    
    for frame in container.decode(video=0):
        if frame_idx in indices_set:
            frame_np = frame.to_ndarray(format='rgb24')
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)
            frame_resized = resize_transform(frame_tensor)
            frames_array[collected] = frame_resized.permute(1, 2, 0).numpy()
            
            collected += 1
            if collected >= nframes:
                break
        
        frame_idx += 1
        if frame_idx > end_frame:
            break
    
    container.close()
    
    video = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()
    
    print(f"  Video: {nframes} frames ({height}x{width} -> {resized_height}x{resized_width}) in {time.time()-st:.3f}s")
    return video

# ============================================================================
# Audio Processing
# ============================================================================
def load_audio_pyav(
    audio_path: str, 
    sr: int = SAMPLE_RATE,
    offset: float = 0.0, 
    duration: float = None,
    max_duration: float = None,
    max_chunks: int = None,
    chunk_duration_sec: float = 10.0
) -> np.ndarray:
    """
    Memory-efficient audio loading using PyAV with optional chunking.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate (default: 16000)
        offset: Start time in seconds
        duration: Duration to load (None = full audio)
        max_duration: Maximum duration to load (TRUNCATES - cuts end)
        max_chunks: Maximum number of chunks (SAMPLES uniformly across audio)
        chunk_duration_sec: Duration of each chunk in seconds
        
    Returns:
        Mono audio array at target sample rate (float32)
    """
    st = time.time()
    
    # Open container and check for audio stream
    container = av.open(audio_path)
    
    # Handle missing audio stream
    if len(container.streams.audio) == 0:
        container.close()
        print(f"  Audio: No audio stream found in {audio_path}, returning empty array")
        return np.array([], dtype=np.float32)
    
    audio_stream = container.streams.audio[0]
    original_sr = audio_stream.sample_rate
    
    start_sample = int(offset * original_sr)
    end_sample = int((offset + duration) * original_sr) if duration else None
    
    est_samples = int((duration or 60) * original_sr)
    audio_buffer = np.zeros(est_samples, dtype=np.float32)
    write_pos = 0
    
    current_sample = 0
    for frame in container.decode(audio=0):
        frame_data = frame.to_ndarray()
        
        if frame_data.ndim > 1:
            frame_data = frame_data.mean(axis=0, dtype=np.float32)
        else:
            frame_data = frame_data.astype(np.float32)
        
        frame_samples = len(frame_data)
        
        if current_sample + frame_samples < start_sample:
            current_sample += frame_samples
            continue
        
        if end_sample and current_sample >= end_sample:
            break
        
        frame_start = max(0, start_sample - current_sample)
        frame_end = min(frame_samples, end_sample - current_sample) if end_sample else frame_samples
        chunk = frame_data[frame_start:frame_end]
        
        chunk_len = len(chunk)
        if write_pos + chunk_len > len(audio_buffer):
            audio_buffer = np.resize(audio_buffer, write_pos + chunk_len + est_samples)
        
        audio_buffer[write_pos:write_pos + chunk_len] = chunk
        write_pos += chunk_len
        current_sample += frame_samples
    
    container.close()
    
    audio = audio_buffer[:write_pos]
    
    # Handle empty audio (no frames decoded)
    if len(audio) == 0:
        print(f"  Audio: No audio data decoded from {audio_path}, returning empty array")
        return np.array([], dtype=np.float32)
    
    # Resample if needed
    if original_sr != sr:
        ratio = sr / original_sr
        new_length = int(len(audio) * ratio)
        
        if 0.9 < ratio < 1.1:
            indices = (np.arange(new_length) / ratio).astype(np.int32)
            indices = np.clip(indices, 0, len(audio) - 1)
            audio = audio[indices]
        else:
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)
    
    # Apply chunking if requested (UNIFORM SAMPLING)
    if max_chunks is not None:
        samples_per_chunk = int(chunk_duration_sec * sr)
        total_samples = len(audio)
        num_chunks = int(np.ceil(total_samples / samples_per_chunk))
        
        # Only chunk if we exceed max_chunks, otherwise return full audio
        if num_chunks > max_chunks:
            print(f"  Audio chunking: {total_samples} samples -> {num_chunks} chunks, sampling {max_chunks}")
            
            # Create chunks
            chunks = []
            for i in range(num_chunks):
                start_idx = i * samples_per_chunk
                end_idx = min((i + 1) * samples_per_chunk, total_samples)
                chunks.append(audio[start_idx:end_idx])
            
            # Uniformly sample max_chunks
            sample_indices = np.linspace(0, num_chunks - 1, max_chunks, dtype=int)
            sampled_chunks = [chunks[i] for i in sample_indices]
            audio = np.concatenate(sampled_chunks, axis=0)
            
            print(f"  Final audio: {len(audio)} samples (from {total_samples})")
    
    # Apply truncation if requested (CUTS END - fallback)
    elif max_duration is not None:
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
    
    actual_duration = len(audio) / sr
    
    # Determine status for logging
    was_chunked = False
    was_truncated = False
    if max_chunks is not None:
        samples_per_chunk = int(chunk_duration_sec * sr)
        num_chunks = int(np.ceil(len(audio_buffer[:write_pos]) / samples_per_chunk))
        was_chunked = num_chunks > max_chunks
    elif max_duration is not None:
        was_truncated = actual_duration >= max_duration * 0.99
    
    status = ' (chunked)' if was_chunked else (' (truncated)' if was_truncated else '')
    print(f"  Audio: {actual_duration:.2f}s{status} in {time.time()-st:.3f}s")
    
    return audio


def process_audio_info_pyav(
    conversations, 
    use_audio_in_video: bool, 
    max_audio_duration: float = None,
    max_audio_chunks: int = None,  # NEW
    audio_chunk_duration_sec: float = 10.0  # NEW
) -> Optional[List[np.ndarray]]:
    """
    Process audio from conversation structure.
    
    Args:
        conversations: Conversation structure
        use_audio_in_video: Extract audio from video (not implemented)
        max_audio_duration: Maximum audio duration in seconds (truncates if exceeded)
        max_audio_chunks: Maximum number of audio chunks (uniform sampling)
        audio_chunk_duration_sec: Duration of each chunk in seconds
        
    Returns:
        List of mono audio arrays or None
    """
    audios = []
    
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            
            for ele in message["content"]:
                if ele["type"] != "audio":
                    continue
                
                if "audio" not in ele and "audio_url" not in ele:
                    continue
                
                path = ele.get("audio", ele.get("audio_url"))
                audio_start = ele.get("audio_start", 0.0)
                audio_end = ele.get("audio_end", None)
                
                # Handle numpy arrays
                if isinstance(path, np.ndarray):
                    if path.ndim > 1:
                        raise ValueError("Only mono audio supported")
                    
                    start_idx = int(SAMPLE_RATE * audio_start)
                    end_idx = None if audio_end is None else int(SAMPLE_RATE * audio_end)
                    audio = path[start_idx:end_idx]
                    
                    # Apply chunking or truncation
                    if max_audio_chunks is not None:
                        # Apply chunking logic on numpy array
                        samples_per_chunk = int(audio_chunk_duration_sec * SAMPLE_RATE)
                        total_samples = len(audio)
                        num_chunks = int(np.ceil(total_samples / samples_per_chunk))
                        
                        if num_chunks > max_audio_chunks:
                            chunks = []
                            for i in range(num_chunks):
                                start_idx = i * samples_per_chunk
                                end_idx = min((i + 1) * samples_per_chunk, total_samples)
                                chunks.append(audio[start_idx:end_idx])
                            
                            sample_indices = np.linspace(0, num_chunks - 1, max_audio_chunks, dtype=int)
                            sampled_chunks = [chunks[i] for i in sample_indices]
                            audio = np.concatenate(sampled_chunks, axis=0)
                    elif max_audio_duration is not None:
                        max_samples = int(SAMPLE_RATE * max_audio_duration)
                        if len(audio) > max_samples:
                            audio = audio[:max_samples]
                    
                    audios.append(audio)
                    continue
                
                # File path
                if path.startswith("file://"):
                    path = path[7:]
                
                duration = None if audio_end is None else (audio_end - audio_start)
                
                # Load with chunking or truncation
                audio = load_audio_pyav(
                    path, 
                    sr=SAMPLE_RATE, 
                    offset=audio_start, 
                    duration=duration,
                    max_duration=max_audio_duration,
                    max_chunks=max_audio_chunks,
                    chunk_duration_sec=audio_chunk_duration_sec
                )
                audios.append(audio)
    
    return audios if audios else None

def process_vision_info_pyav(conversations) -> Tuple[Optional[List[Image.Image]], Optional[List[torch.Tensor]]]:
    """Process images and videos from conversation structure."""
    videos = []
    images = []
    
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            
            for ele in message["content"]:
                if ele["type"] == "video" and ("video" in ele or "video_url" in ele):
                    videos.append(fetch_video_pyav(ele))
                elif ele["type"] == "image" and "image" in ele:
                    img_path = ele["image"]
                    if img_path.startswith("file://"):
                        img_path = img_path[7:]
                    images.append(Image.open(img_path).convert("RGB"))
    
    return (images if images else None, videos if videos else None)

def process_mm_info_pyav(
    conversations, 
    use_audio_in_video: bool = False,
    max_audio_duration: float = None,
    max_audio_chunks: int = None,  
    audio_chunk_duration_sec: float = 10.0  
) -> Tuple[Optional[List[np.ndarray]], Optional[List[Image.Image]], Optional[List[torch.Tensor]]]:
    """
    Process multimodal information from conversations using PyAV.
    
    Drop-in replacement for qwen_omni_utils.process_mm_info with:
    - Memory-efficient video loading (no OOM on long videos)
    - Fast audio processing with chunking OR truncation
    - Optimized resampling and resizing
    
    Args:
        conversations: Conversation structure with multimodal content
        use_audio_in_video: Extract audio from video files (not implemented)
        max_audio_duration: Maximum audio duration in seconds (truncates if exceeded)
        max_audio_chunks: Maximum number of audio chunks (uniform sampling, preferred over truncation)
        audio_chunk_duration_sec: Duration of each chunk in seconds
        
    Returns:
        Tuple of (audios, images, videos) where:
        - audios: List[np.ndarray] of mono audio at SAMPLE_RATE
        - images: List[PIL.Image] 
        - videos: List[torch.Tensor] of shape (T, C, H, W)
    """
    audios = process_audio_info_pyav(
        conversations, 
        use_audio_in_video, 
        max_audio_duration,
        max_audio_chunks,
        audio_chunk_duration_sec
    )
    images, videos = process_vision_info_pyav(conversations)
    return audios, images, videos





# Alias for compatibility
process_mm_info = process_mm_info_pyav