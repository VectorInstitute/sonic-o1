"""
Frame Sampler Utility

Extracts sample frames from video segments for GPT-4V validation.
Uses PyAV for direct video frame extraction (faster and more reliable than FFmpeg subprocess).
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
import shutil

logger = logging.getLogger(__name__)

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    logger.error("PyAV not installed. Install with: pip install av")


class FrameSampler:
    """Sample frames from video segments for visual validation"""
    
    def __init__(self, config=None):
        """
        Initialize frame sampler.
        
        Args:
            config: Optional configuration object
        """
        if not PYAV_AVAILABLE:
            raise ImportError("PyAV is required. Install with: pip install av")
        
        self.config = config
        # Use scratch directory like video segmenter
        scratch_base = Path.home() / 'scratch' / 'frame_sampler'
        scratch_base.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix='frames_', dir=scratch_base))
        self._cleaned_up = False
        logger.info(f"Frame sampler temporary directory: {self.temp_dir}")
    
    def sample_frames_from_segment(
        self,
        video_path: Path,
        segment_start: float,
        segment_end: float,
        num_frames: int = 8,
        strategy: str = 'uniform'
    ) -> List[Path]:
        """
        Sample frames from a video segment.
        
        Args:
            video_path: Path to video file
            segment_start: Start time of segment in seconds
            segment_end: End time of segment in seconds
            num_frames: Number of frames to sample
            strategy: Sampling strategy ('uniform', 'keyframes', or 'adaptive')
            
        Returns:
            List of paths to extracted frame images
        """
        try:
            if strategy == 'uniform':
                return self._sample_uniform_frames(
                    video_path, segment_start, segment_end, num_frames
                )
            elif strategy == 'keyframes':
                return self._sample_keyframes(
                    video_path, segment_start, segment_end, num_frames
                )
            elif strategy == 'adaptive':
                return self._sample_adaptive_frames(
                    video_path, segment_start, segment_end, num_frames
                )
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Error sampling frames: {e}")
            return []
    
    def _sample_uniform_frames(
        self,
        video_path: Path,
        segment_start: float,
        segment_end: float,
        num_frames: int
    ) -> List[Path]:
        """
        Sample frames uniformly across the segment using PyAV.
        """
        frame_paths = []
        
        # Ensure temp directory exists
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Temp dir didn't exist, recreated: {self.temp_dir}")
        
        # Add epsilon buffer
        epsilon = 0.1
        safe_end = max(segment_start, segment_end - epsilon)
        safe_duration = safe_end - segment_start
        
        # Calculate timestamps
        if num_frames == 1:
            timestamps = [segment_start + safe_duration / 2]
        else:
            interval = safe_duration / (num_frames - 1)
            timestamps = [segment_start + i * interval for i in range(num_frames)]
        
        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            time_base = video_stream.time_base
            
            for i, timestamp in enumerate(timestamps):
                try:
                    frame_path = self.temp_dir / f"frame_{i:03d}_t{timestamp:.2f}s.jpg"
                    
                    # Convert timestamp to PTS
                    pts = int(timestamp / float(time_base))
                    
                    # Seek to timestamp
                    container.seek(pts, stream=video_stream)
                    
                    # Decode next frame
                    frame_found = False
                    for frame in container.decode(video=0):
                        img = frame.to_image()
                        img.save(str(frame_path), 'JPEG', quality=95)
                        if frame_path.exists():
                            frame_paths.append(frame_path)
                            logger.debug(f"Extracted frame at {timestamp:.2f}s")
                            frame_found = True
                        else:
                            logger.warning(f"Frame saved but file doesn't exist: {frame_path}")
                        break
                    
                    if not frame_found:
                        logger.warning(f"No frame decoded at {timestamp:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"Failed to extract frame at {timestamp:.2f}s: {e}")
                    import traceback
                    logger.warning(f"Traceback: {traceback.format_exc()}")
                    continue
            
            container.close()
            
        except Exception as e:
            logger.error(f"Error opening video: {e}")
            return []
        
        logger.info(f"Extracted {len(frame_paths)}/{num_frames} uniform frames")
        return frame_paths
    
    def _sample_keyframes(
        self,
        video_path: Path,
        segment_start: float,
        segment_end: float,
        num_frames: int
    ) -> List[Path]:
        """
        Sample keyframes (I-frames) from the segment using PyAV.
        """
        frame_paths = []
        
        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            time_base = float(video_stream.time_base)
            
            # Convert to PTS
            start_pts = int(segment_start / time_base)
            end_pts = int(segment_end / time_base)
            
            container.seek(start_pts, stream=video_stream)
            
            keyframe_count = 0
            for frame in container.decode(video=0):
                frame_time = frame.pts * time_base
                
                if frame_time > segment_end:
                    break
                if frame_time < segment_start:
                    continue
                
                # Only keyframes
                if frame.key_frame:
                    frame_path = self.temp_dir / f"keyframe_{keyframe_count:03d}_t{frame_time:.2f}s.jpg"
                    img = frame.to_image()
                    img.save(str(frame_path), quality=95)
                    frame_paths.append(frame_path)
                    keyframe_count += 1
                    
                    if keyframe_count >= num_frames:
                        break
            
            container.close()
            
            logger.info(f"Extracted {len(frame_paths)} keyframes")
            
            # Supplement if needed
            if len(frame_paths) < num_frames:
                logger.info(f"Supplementing with uniform frames")
                uniform_frames = self._sample_uniform_frames(
                    video_path, segment_start, segment_end, num_frames - len(frame_paths)
                )
                frame_paths.extend(uniform_frames)
            
        except Exception as e:
            logger.error(f"Error extracting keyframes: {e}")
            return self._sample_uniform_frames(video_path, segment_start, segment_end, num_frames)
        
        return frame_paths[:num_frames]
    
    def _sample_adaptive_frames(
        self,
        video_path: Path,
        segment_start: float,
        segment_end: float,
        num_frames: int
    ) -> List[Path]:
        """
        Adaptive sampling: denser at start/end, sparse in middle.
        """
        epsilon = 0.1
        safe_end = max(segment_start, segment_end - epsilon)
        
        # Adaptive distribution
        num_start = max(2, int(num_frames * 0.3))
        num_end = max(2, int(num_frames * 0.3))
        num_middle = num_frames - num_start - num_end
        
        timestamps = []
        
        # Start frames
        start_zone = (safe_end - segment_start) * 0.2
        for i in range(num_start):
            t = segment_start + (i / (num_start - 1) if num_start > 1 else 0.5) * start_zone
            timestamps.append(t)
        
        # Middle frames
        middle_start = segment_start + start_zone
        middle_end = safe_end - start_zone
        middle_duration = middle_end - middle_start
        for i in range(num_middle):
            t = middle_start + (i / (num_middle - 1) if num_middle > 1 else 0.5) * middle_duration
            timestamps.append(t)
        
        # End frames
        end_zone_start = safe_end - start_zone
        for i in range(num_end):
            t = end_zone_start + (i / (num_end - 1) if num_end > 1 else 0.5) * start_zone
            timestamps.append(t)
        
        # Extract frames
        frame_paths = []
        
        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            time_base = video_stream.time_base
            
            for i, timestamp in enumerate(sorted(timestamps)):
                try:
                    frame_path = self.temp_dir / f"adaptive_frame_{i:03d}_t{timestamp:.2f}s.jpg"
                    
                    pts = int(timestamp / float(time_base))
                    container.seek(pts, stream=video_stream)
                    
                    for frame in container.decode(video=0):
                        img = frame.to_image()
                        img.save(str(frame_path), quality=95)
                        frame_paths.append(frame_path)
                        break
                        
                except Exception as e:
                    logger.error(f"Error at {timestamp:.2f}s: {e}")
                    continue
            
            container.close()
            
        except Exception as e:
            logger.error(f"Error with adaptive sampling: {e}")
            return []
        
        logger.info(f"Extracted {len(frame_paths)} adaptive frames")
        return frame_paths
    
    def sample_frames_at_timestamps(
        self,
        video_path: Path,
        timestamps: List[float],
        segment_start: float = 0.0
    ) -> List[Tuple[float, Path]]:
        """
        Sample frames at specific timestamps using PyAV.
        """
        frame_data = []
        
        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            time_base = video_stream.time_base
            
            for timestamp in timestamps:
                if timestamp is None:
                    continue
                
                try:
                    relative_time = timestamp - segment_start
                    frame_path = self.temp_dir / f"verify_t{timestamp:.2f}s_rel{relative_time:.2f}s.jpg"
                    
                    pts = int(timestamp / float(time_base))
                    container.seek(pts, stream=video_stream)
                    
                    for frame in container.decode(video=0):
                        img = frame.to_image()
                        img.save(str(frame_path), quality=95)
                        frame_data.append((timestamp, frame_path))
                        logger.debug(f"Extracted frame at {timestamp:.2f}s")
                        break
                        
                except Exception as e:
                    logger.error(f"Error at {timestamp:.2f}s: {e}")
                    continue
            
            container.close()
            
        except Exception as e:
            logger.error(f"Error sampling timestamps: {e}")
            return []
        
        logger.info(f"Extracted {len(frame_data)} verification frames")
        return frame_data
    
    def cleanup(self):
        """Clean up temporary frame files"""
        if self._cleaned_up:
            return
        
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up frame sampler temp directory: {self.temp_dir}")
                self._cleaned_up = True
        except Exception as e:
            logger.warning(f"Failed to cleanup frame sampler temp dir: {e}")
