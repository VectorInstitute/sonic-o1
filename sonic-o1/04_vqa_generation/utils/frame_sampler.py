"""frame_sampler.py.

Extract sample frames from video segments for GPT-4V validation.
Uses PyAV for frame extraction (faster than FFmpeg subprocess).

Author: SONIC-O1 Team
"""

import logging
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import List, Tuple


logger = logging.getLogger(__name__)

try:
    import av

    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    logger.error("PyAV not installed. Install with: pip install av")


class FrameSampler:
    """Sample frames from video segments for visual validation."""

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
        scratch_base = Path.home() / "scratch" / "frame_sampler"
        scratch_base.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="frames_", dir=scratch_base))
        self._cleaned_up = False
        logger.info("Frame sampler temp dir: %s", self.temp_dir)

    def sample_frames_from_segment(
        self,
        video_path: Path,
        segment_start: float,
        segment_end: float,
        num_frames: int = 8,
        strategy: str = "uniform",
    ) -> List[Path]:
        """
        Sample frames from a video segment.

        Args:
            video_path: Path to video file
            segment_start: Start time of segment in seconds
            segment_end: End time of segment in seconds
            num_frames: Number of frames to sample
            strategy: Sampling strategy ('uniform', 'keyframes', or 'adaptive')

        Returns
        -------
        list of Path
            Paths to extracted frame images.
        """
        try:
            if strategy == "uniform":
                return self._sample_uniform_frames(
                    video_path, segment_start, segment_end, num_frames
                )
            if strategy == "keyframes":
                return self._sample_keyframes(
                    video_path, segment_start, segment_end, num_frames
                )
            if strategy == "adaptive":
                return self._sample_adaptive_frames(
                    video_path, segment_start, segment_end, num_frames
                )
            raise ValueError("Unknown sampling strategy: %s" % strategy)

        except Exception as e:
            logger.error("Error sampling frames: %s", e)
            return []

    def _sample_uniform_frames(
        self,
        video_path: Path,
        segment_start: float,
        segment_end: float,
        num_frames: int,
    ) -> List[Path]:
        """Sample frames uniformly across the segment using PyAV."""
        frame_paths = []

        # Ensure temp directory exists
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.warning("Temp dir recreated: %s", self.temp_dir)

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
                    frame_path = (
                        self.temp_dir / "frame_%03d_t%.2fs.jpg" % (i, timestamp)
                    )

                    # Convert timestamp to PTS
                    pts = int(timestamp / float(time_base))

                    # Seek to timestamp
                    container.seek(pts, stream=video_stream)

                    # Decode next frame
                    frame_found = False
                    for frame in container.decode(video=0):
                        img = frame.to_image()
                        img.save(str(frame_path), "JPEG", quality=95)
                        if frame_path.exists():
                            frame_paths.append(frame_path)
                            logger.debug("Extracted frame at %.2fs", timestamp)
                            frame_found = True
                        else:
                            logger.warning("Frame saved but missing: %s", frame_path)
                        break

                    if not frame_found:
                        logger.warning("No frame at %.2fs", timestamp)

                except Exception as e:
                    logger.warning("Failed frame at %.2fs: %s", timestamp, e)
                    logger.warning("Traceback: %s", traceback.format_exc())
                    continue

            container.close()

        except Exception as e:
            logger.error("Error opening video: %s", e)
            return []

        logger.info(
            "Extracted %d/%d uniform frames",
            len(frame_paths),
            num_frames,
        )
        return frame_paths

    def _sample_keyframes(
        self,
        video_path: Path,
        segment_start: float,
        segment_end: float,
        num_frames: int,
    ) -> List[Path]:
        """Sample keyframes (I-frames) from the segment using PyAV."""
        frame_paths = []

        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            time_base = float(video_stream.time_base)

            # Convert to PTS
            start_pts = int(segment_start / time_base)
            int(segment_end / time_base)

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
                    name = "keyframe_%03d_t%.2fs.jpg" % (keyframe_count, frame_time)
                    frame_path = self.temp_dir / name
                    img = frame.to_image()
                    img.save(str(frame_path), quality=95)
                    frame_paths.append(frame_path)
                    keyframe_count += 1

                    if keyframe_count >= num_frames:
                        break

            container.close()

            logger.info("Extracted %d keyframes", len(frame_paths))

            # Supplement if needed
            if len(frame_paths) < num_frames:
                logger.info("Supplementing with uniform frames")
                uniform_frames = self._sample_uniform_frames(
                    video_path,
                    segment_start,
                    segment_end,
                    num_frames - len(frame_paths),
                )
                frame_paths.extend(uniform_frames)

        except Exception as e:
            logger.error("Error extracting keyframes: %s", e)
            return self._sample_uniform_frames(
                video_path, segment_start, segment_end, num_frames
            )

        return frame_paths[:num_frames]

    def _sample_adaptive_frames(
        self,
        video_path: Path,
        segment_start: float,
        segment_end: float,
        num_frames: int,
    ) -> List[Path]:
        """Adaptive sampling: denser at start/end, sparse in middle."""
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
            t = (
                segment_start
                + (i / (num_start - 1) if num_start > 1 else 0.5) * start_zone
            )
            timestamps.append(t)

        # Middle frames
        middle_start = segment_start + start_zone
        middle_end = safe_end - start_zone
        middle_duration = middle_end - middle_start
        for i in range(num_middle):
            t = (
                middle_start
                + (i / (num_middle - 1) if num_middle > 1 else 0.5) * middle_duration
            )
            timestamps.append(t)

        # End frames
        end_zone_start = safe_end - start_zone
        for i in range(num_end):
            t = (
                end_zone_start
                + (i / (num_end - 1) if num_end > 1 else 0.5) * start_zone
            )
            timestamps.append(t)

        # Extract frames
        frame_paths = []

        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            time_base = video_stream.time_base

            for i, timestamp in enumerate(sorted(timestamps)):
                try:
                    frame_path = (
                        self.temp_dir
                        / "adaptive_frame_%03d_t%.2fs.jpg"
                        % (i, timestamp)
                    )

                    pts = int(timestamp / float(time_base))
                    container.seek(pts, stream=video_stream)

                    for frame in container.decode(video=0):
                        img = frame.to_image()
                        img.save(str(frame_path), quality=95)
                        frame_paths.append(frame_path)
                        break

                except Exception as e:
                    logger.error("Error at %.2fs: %s", timestamp, e)
                    continue

            container.close()

        except Exception as e:
            logger.error("Error adaptive sampling: %s", e)
            return []

        logger.info("Extracted %d adaptive frames", len(frame_paths))
        return frame_paths

    def sample_frames_at_timestamps(
        self,
        video_path: Path,
        timestamps: List[float],
        segment_start: float = 0.0,
    ) -> List[Tuple[float, Path]]:
        """
        Sample frames at specific timestamps using PyAV.

        Args
        ----
        video_path : Path
            Path to video file.
        timestamps : list of float
            Timestamps in seconds.
        segment_start : float
            Segment start (for relative naming).

        Returns
        -------
        list of (float, Path)
            (timestamp, frame_path) pairs.
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
                    frame_path = (
                        self.temp_dir
                        / "verify_t%.2fs_rel%.2fs.jpg"
                        % (timestamp, relative_time)
                    )

                    pts = int(timestamp / float(time_base))
                    container.seek(pts, stream=video_stream)

                    for frame in container.decode(video=0):
                        img = frame.to_image()
                        img.save(str(frame_path), quality=95)
                        frame_data.append((timestamp, frame_path))
                        logger.debug("Extracted frame at %.2fs", timestamp)
                        break

                except Exception as e:
                    logger.error("Error at %.2fs: %s", timestamp, e)
                    continue

            container.close()

        except Exception as e:
            logger.error("Error sampling timestamps: %s", e)
            return []

        logger.info("Extracted %d verification frames", len(frame_data))
        return frame_data

    def cleanup(self):
        """Clean up temporary frame files."""
        if self._cleaned_up:
            return

        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up frame sampler temp: %s", self.temp_dir)
                self._cleaned_up = True
        except Exception as e:
            logger.warning("Failed to cleanup frame sampler: %s", e)
