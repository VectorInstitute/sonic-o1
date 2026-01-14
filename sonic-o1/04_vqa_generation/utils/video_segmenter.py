"""
Video segmentation utility using FFmpeg
"""
import subprocess
import tempfile
import shutil
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)


class VideoSegmenter:
    """Handle video segmentation using FFmpeg"""
    
    def __init__(self, config):
        """
        Initialize segmenter with configuration.
        
        Args:
            config: Configuration object with video settings
        """
        self.summarization_segment_duration = int(config.video.summarization_segment_duration)
        self.mcq_segment_duration = int(config.video.mcq_segment_duration)
        self.temporal_localization_segment_duration = int(config.video.temporal_localization_segment_duration) 
        self.segment_overlap = int(config.video.segment_overlap)
    
    @staticmethod
    def get_actual_duration(video_path: Path) -> float:
        """
        Get actual video duration using multiple methods.
        
        FIXED: Try stream duration first, then format duration.
        Stream duration is more reliable for videos with metadata issues.
        """
        # Method 1: Try stream duration (more reliable)
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=nw=1:nk=1",
            str(video_path),
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output and output != 'N/A':
                    try:
                        duration = float(output)
                        if duration > 0:
                            logger.debug(f"Got stream duration: {duration:.3f}s")
                            return duration
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug(f"Stream duration failed: {e}")
        
        # Method 2: Try format duration (fallback)
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(video_path),
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    try:
                        duration = float(output)
                        if duration > 0:
                            logger.debug(f"Got format duration: {duration:.3f}s")
                            return duration
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug(f"Format duration failed: {e}")
        
        raise Exception(f"Could not get duration from {video_path}")

    def segment_video(self, 
                     video_path: Path,
                     duration_seconds: float,
                     task_type: str = 'summarization',
                     output_dir: Optional[Path] = None) -> List[Dict]:
        """
        Segment video into chunks based on task type.
        """
    
        try:
            actual_duration = self.get_actual_duration(video_path)
            
            # FIXED: Detect severe duration mismatch (likely corrupted metadata)
            duration_ratio = actual_duration / duration_seconds if duration_seconds > 0 else 999
            
            if duration_ratio > 5 or duration_ratio < 0.2:
                logger.error(
                    f"SEVERE duration mismatch for {video_path}: "
                    f"metadata={duration_seconds:.1f}s, ffprobe={actual_duration:.1f}s (ratio={duration_ratio:.1f}x). "
                    f"Video metadata is likely corrupted. Using metadata value to be safe."
                )
                # Don't trust ffprobe when ratio is extreme
                actual_duration = duration_seconds
            elif abs(actual_duration - duration_seconds) > 0.5:
                logger.warning(
                    f"Duration mismatch for {video_path}: "
                    f"metadata={duration_seconds:.3f}s, ffprobe={actual_duration:.3f}s. "
                    f"Using ffprobe value."
                )
            
            duration_seconds = actual_duration
        except Exception as e:
            logger.warning(
                f"Could not get actual duration with ffprobe for {video_path}, "
                f"falling back to provided {duration_seconds:.3f}s. Error: {e}"
            )

        # Small epsilon to avoid sampling exactly at the end
        epsilon = 0.05
        duration_seconds = max(0.0, duration_seconds - epsilon)

        if task_type == 'summarization':
            max_segment_duration = self.summarization_segment_duration
        elif task_type == 'mcq':
            max_segment_duration = self.mcq_segment_duration
        elif task_type == 'temporal_localization':
            max_segment_duration = self.temporal_localization_segment_duration
        else:
            max_segment_duration = self.mcq_segment_duration
            logger.warning(f"Unknown task_type '{task_type}', defaulting to MCQ segment duration")
        
        if duration_seconds <= max_segment_duration:
            logger.info(
                f"Video duration ({duration_seconds:.3f}s) <= max segment ({max_segment_duration}s) "
                f"for {task_type}, returning as single segment"
            )
            return [{
                'segment_path': video_path,
                'start': 0.0,
                'end': duration_seconds,
                'duration': duration_seconds,
                'segment_number': 0
            }]
                
        if output_dir is None:
            output_dir = Path.home() / 'scratch' / 'video_segments' / f"{task_type}_{int(time.time())}"
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = output_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = None
        
        num_segments = int(duration_seconds / max_segment_duration) + 1
        logger.info(
            f"Segmenting {duration_seconds:.3f}s video for {task_type} into {num_segments} chunks "
            f"(max {max_segment_duration}s each with {self.segment_overlap}s overlap)"
        )
        
        segments = []
        
        try:
            for i in range(num_segments):
                start_time = max(0, i * max_segment_duration - (self.segment_overlap if i > 0 else 0))
                segment_duration = min(
                    max_segment_duration + self.segment_overlap,
                    duration_seconds - start_time
                )
                if segment_duration <= 0:
                    break

                end_time = start_time + segment_duration
                
                segment_path = output_dir / f"segment_{i:03d}{video_path.suffix}"
                
                logger.info(f"Creating segment {i+1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s")
                
                # FIXED: Use copy codec when possible (much faster)
                # Calculate timeout based on segment duration (2x for safety)
                timeout = max(300, int(segment_duration * 2))
                
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-i', str(video_path),
                    '-t', str(segment_duration),
                    '-c', 'copy',  # FIXED: Copy codec (no re-encoding)
                    '-avoid_negative_ts', 'make_zero',  # FIXED: Better timestamp handling
                    str(segment_path)
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    raise Exception(f"Failed to create segment {i}: {result.stderr}")
                
                if not segment_path.exists():
                    raise Exception(f"Segment file not created: {segment_path}")
                
                segments.append({
                    'segment_path': segment_path,
                    'start': start_time,
                    'end': end_time,
                    'duration': segment_duration,
                    'segment_number': i,
                    'is_temp': temp_dir is not None
                })
            
            logger.info(f"Successfully created {len(segments)} segments for {task_type}")
            return segments
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise

    
    def segment_audio(self,
                     audio_path: Path,
                     duration_seconds: float,
                     task_type: str = 'summarization',
                     output_dir: Optional[Path] = None) -> List[Dict]:
        """
        Segment audio file into chunks based on task type.
        """
        if task_type == 'summarization':
            max_segment_duration = self.summarization_segment_duration
        elif task_type == 'mcq':
            max_segment_duration = self.mcq_segment_duration
        elif task_type == 'temporal_localization':
            max_segment_duration = self.temporal_localization_segment_duration
        else:
            max_segment_duration = self.mcq_segment_duration
            logger.warning(f"Unknown task_type '{task_type}', defaulting to MCQ segment duration")
        
        
        if duration_seconds <= max_segment_duration:
            return [{
                'segment_path': audio_path,
                'start': 0,
                'end': duration_seconds,
                'duration': duration_seconds,
                'segment_number': 0
            }]
        
        if output_dir is None:
            output_dir = Path.home() / 'scratch' / 'audio_segments' / f"{task_type}_{int(time.time())}"
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = output_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = None
        num_segments = int(duration_seconds / max_segment_duration) + 1
        segments = []
        
        try:
            for i in range(num_segments):
                start_time = max(0, i * max_segment_duration - (self.segment_overlap if i > 0 else 0))
                segment_duration = min(
                    max_segment_duration + self.segment_overlap,
                    duration_seconds - start_time
                )
                
                segment_path = output_dir / f"segment_{i:03d}{audio_path.suffix}"
                
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-i', str(audio_path),
                    '-t', str(segment_duration),
                    '-c', 'copy',
                    str(segment_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg audio segment error: {result.stderr}")
                    raise Exception(f"Failed to create audio segment {i}")
                
                segments.append({
                    'segment_path': segment_path,
                    'start': start_time,
                    'end': start_time + segment_duration,
                    'duration': segment_duration,
                    'segment_number': i,
                    'is_temp': temp_dir is not None
                })
            
            logger.info(f"Successfully created {len(segments)} audio segments for {task_type}")
            return segments
            
        except Exception as e:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
    
    def extract_transcript_segment(self, 
                                transcript_path: Path,
                                start_time: float,
                                end_time: float,
                                strip_timestamps: bool = False) -> str:
        """
        Extract portion of SRT transcript for a time segment.
        """
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            segments = content.strip().split('\n\n')
            extracted = []
            
            for segment in segments:
                lines = segment.split('\n')
                if len(lines) < 3:
                    continue
                
                timestamp_pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})'
                match = re.search(timestamp_pattern, lines[1])
                
                if match:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
                    seg_start = h1*3600 + m1*60 + s1 + ms1/1000
                    seg_end = h2*3600 + m2*60 + s2 + ms2/1000
                    
                    if seg_start < end_time and seg_end > start_time:
                        if strip_timestamps:
                            text_lines = lines[2:]
                            extracted.append(' '.join(text_lines))
                        else:
                            extracted.append(segment)
            
            if strip_timestamps:
                return ' '.join(extracted)
            else:
                return '\n\n'.join(extracted)
            
        except Exception as e:
            logger.warning(f"Could not extract transcript segment: {e}")
            return ""
    
    def cleanup_segments(self, segments: List[Dict]):
        """
        Cleanup temporary segment files.
        """
        for seg in segments:
            if seg.get('is_temp', False):
                try:
                    seg_path = seg['segment_path']
                    if seg_path.exists():
                        temp_dir = seg_path.parent
                        if temp_dir.exists() and 'segments' in temp_dir.name:
                            shutil.rmtree(temp_dir)
                            logger.info(f"Cleaned up temp directory: {temp_dir}")
                            break
                except Exception as e:
                    logger.warning(f"Failed to cleanup segment: {e}")