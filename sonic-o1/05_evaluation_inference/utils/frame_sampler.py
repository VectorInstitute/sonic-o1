"""frame_sampler.py.

Extract frames from videos for image-based models using PyAV.

Author: SONIC-O1 Team
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger(__name__)

try:
    import av

    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    logger.error("PyAV not installed. Install with: pip install av")


class FrameSampler:
    """Sample frames from videos for image-based models."""

    def __init__(self):
        if not PYAV_AVAILABLE:
            raise ImportError("PyAV is required. Install with: pip install av")

        # Use SCRATCH_DIR or TMPDIR environment variable, fallback to home
        scratch_base = os.environ.get("SCRATCH_DIR") or os.environ.get("TMPDIR")
        if scratch_base:
            scratch_base = Path(scratch_base) / "frame_sampler"
        else:
            scratch_base = Path.home() / "scratch" / "frame_sampler"

        scratch_base.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="frames_", dir=scratch_base))
        self._cleaned_up = False
        logger.info(f"Frame sampler temp directory: {self.temp_dir}")

    def sample_frames_fps(
        self,
        video_path: Path,
        fps: float = 1.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> List[Path]:
        """
        Sample frames at specified FPS from video using PyAV.

        Args:
            video_path: Path to video file (.mp4)
            fps: Frames per second to sample
            start_time: Start time in seconds
            end_time: End time in seconds (None = entire video)

        Returns
        -------
            List of paths to extracted frame images
        """
        frame_paths = []

        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]

            if end_time is None:
                duration = float(video_stream.duration * video_stream.time_base)
                end_time = duration

            time_base = float(video_stream.time_base)
            interval = 1.0 / fps

            current_time = start_time
            frame_idx = 0

            start_pts = int(start_time / time_base)
            container.seek(start_pts, stream=video_stream)

            for frame in container.decode(video=0):
                frame_time = float(frame.pts * time_base)

                if frame_time < start_time:
                    continue
                if frame_time > end_time:
                    break

                if frame_time >= current_time:
                    frame_path = (
                        self.temp_dir / f"frame_{frame_idx:04d}_t{frame_time:.2f}s.jpg"
                    )
                    img = frame.to_image()
                    img.save(str(frame_path), "JPEG", quality=95)
                    frame_paths.append(frame_path)

                    frame_idx += 1
                    current_time += interval

            container.close()
            logger.info(f"Extracted {len(frame_paths)} frames at {fps} FPS")

        except Exception as e:
            logger.error(f"Error sampling frames: {e}")
            return []

        return frame_paths

    def sample_frames_uniform(
        self,
        video_path: Path,
        num_frames: int,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> List[Path]:
        """
        Sample N frames uniformly from video using PyAV.

        Args:
            video_path: Path to video file (.mp4)
            num_frames: Number of frames to sample
            start_time: Start time in seconds
            end_time: End time in seconds (None = entire video)

        Returns
        -------
            List of paths to extracted frame images
        """
        frame_paths = []

        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]

            if end_time is None:
                duration = float(video_stream.duration * video_stream.time_base)
                end_time = duration

            epsilon = 0.1
            safe_end = max(start_time, end_time - epsilon)
            duration = safe_end - start_time

            if num_frames == 1:
                timestamps = [start_time + duration / 2]
            else:
                interval = duration / (num_frames - 1)
                timestamps = [start_time + i * interval for i in range(num_frames)]

            time_base = float(video_stream.time_base)

            for i, timestamp in enumerate(timestamps):
                try:
                    frame_path = self.temp_dir / f"frame_{i:03d}_t{timestamp:.2f}s.jpg"

                    pts = int(timestamp / time_base)
                    container.seek(pts, stream=video_stream)

                    for frame in container.decode(video=0):
                        img = frame.to_image()
                        img.save(str(frame_path), "JPEG", quality=95)
                        frame_paths.append(frame_path)
                        break

                except Exception as e:
                    logger.warning(f"Failed to extract frame at {timestamp:.2f}s: {e}")
                    continue

            container.close()
            logger.info(f"Extracted {len(frame_paths)}/{num_frames} uniform frames")

        except Exception as e:
            logger.error(f"Error sampling frames: {e}")
            return []

        return frame_paths

    def cleanup(self):
        """Clean up temporary frame files."""
        if self._cleaned_up:
            return

        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up frame sampler temp directory")
                self._cleaned_up = True
        except Exception as e:
            logger.warning(f"Failed to cleanup frame sampler temp dir: {e}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()
