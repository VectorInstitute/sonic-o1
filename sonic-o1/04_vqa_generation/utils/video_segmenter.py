"""
video_segmenter.py.

Video segmentation utility using FFmpeg.

Author: SONIC-O1 Team
"""

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


class VideoSegmenter:
    """Handle video segmentation using FFmpeg."""

    def __init__(self, config):
        """
        Initialize segmenter with configuration.

        Args:
            config: Configuration object with video settings
        """
        self.summarization_segment_duration = int(
            config.video.summarization_segment_duration
        )
        self.mcq_segment_duration = int(config.video.mcq_segment_duration)
        self.temporal_localization_segment_duration = int(
            config.video.temporal_localization_segment_duration
        )
        self.segment_overlap = int(config.video.segment_overlap)

    @staticmethod
    def get_actual_duration(video_path: Path) -> float:
        """
        Get actual video duration using multiple methods.

        Tries stream duration first, then format duration.
        Stream duration is more reliable for videos with metadata issues.

        Returns
        -------
        float
            Duration in seconds.

        Raises
        ------
        Exception
            If duration cannot be obtained from the video.
        """
        # Method 1: Try stream duration (more reliable)
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if output and output != "N/A":
                    try:
                        duration = float(output)
                        if duration > 0:
                            logger.debug("Got stream duration: %.3fs", duration)
                            return duration
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug("Stream duration failed: %s", e)

        # Method 2: Try format duration (fallback)
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    try:
                        duration = float(output)
                        if duration > 0:
                            logger.debug("Got format duration: %.3fs", duration)
                            return duration
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug("Format duration failed: %s", e)

        raise Exception("Could not get duration from %s" % video_path)

    def segment_video(
        self,
        video_path: Path,
        duration_seconds: float,
        task_type: str = "summarization",
        output_dir: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Segment video into chunks based on task type.

        Args
        ----
        video_path : Path
            Path to the video file.
        duration_seconds : float
            Duration in seconds (from metadata; ffprobe may override).
        task_type : str
            One of "summarization", "mcq", "temporal_localization".
        output_dir : Path, optional
            Directory for segment files; created if None.

        Returns
        -------
        list of dict
            Each dict has segment_path, start, end, duration,
            segment_number, and optionally is_temp.
        """
        try:
            actual_duration = self.get_actual_duration(video_path)

            # Detect severe duration mismatch (likely corrupted metadata)
            duration_ratio = (
                actual_duration / duration_seconds if duration_seconds > 0 else 999
            )

            if duration_ratio > 5 or duration_ratio < 0.2:
                logger.error(
                    "SEVERE duration mismatch for %s: metadata=%.1fs, "
                    "ffprobe=%.1fs (ratio=%.1fx). Video metadata likely "
                    "corrupted. Using metadata value.",
                    video_path,
                    duration_seconds,
                    actual_duration,
                    duration_ratio,
                )
                actual_duration = duration_seconds
            elif abs(actual_duration - duration_seconds) > 0.5:
                logger.warning(
                    "Duration mismatch for %s: metadata=%.3fs, "
                    "ffprobe=%.3fs. Using ffprobe value.",
                    video_path,
                    duration_seconds,
                    actual_duration,
                )

            duration_seconds = actual_duration
        except Exception as e:
            logger.warning(
                "Could not get actual duration with ffprobe for %s, "
                "falling back to %.3fs. Error: %s",
                video_path,
                duration_seconds,
                e,
            )

        # Small epsilon to avoid sampling exactly at the end
        epsilon = 0.05
        duration_seconds = max(0.0, duration_seconds - epsilon)

        if task_type == "summarization":
            max_segment_duration = self.summarization_segment_duration
        elif task_type == "mcq":
            max_segment_duration = self.mcq_segment_duration
        elif task_type == "temporal_localization":
            max_segment_duration = self.temporal_localization_segment_duration
        else:
            max_segment_duration = self.mcq_segment_duration
            logger.warning(
                "Unknown task_type '%s', defaulting to MCQ segment duration",
                task_type,
            )

        if duration_seconds <= max_segment_duration:
            logger.info(
                "Video duration (%.3fs) <= max segment (%s)s for %s, "
                "returning as single segment",
                duration_seconds,
                max_segment_duration,
                task_type,
            )
            return [
                {
                    "segment_path": video_path,
                    "start": 0.0,
                    "end": duration_seconds,
                    "duration": duration_seconds,
                    "segment_number": 0,
                }
            ]

        if output_dir is None:
            output_dir = (
                Path.home()
                / "scratch"
                / "video_segments"
                / f"{task_type}_{int(time.time())}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = output_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = None

        num_segments = int(duration_seconds / max_segment_duration) + 1
        logger.info(
            "Segmenting %.3fs video for %s into %d chunks "
            "(max %ss each with %ss overlap)",
            duration_seconds,
            task_type,
            num_segments,
            max_segment_duration,
            self.segment_overlap,
        )

        segments = []

        try:
            for i in range(num_segments):
                start_time = max(
                    0,
                    i * max_segment_duration - (self.segment_overlap if i > 0 else 0),
                )
                segment_duration = min(
                    max_segment_duration + self.segment_overlap,
                    duration_seconds - start_time,
                )
                if segment_duration <= 0:
                    break

                end_time = start_time + segment_duration

                segment_path = output_dir / ("segment_%03d%s" % (i, video_path.suffix))

                logger.info(
                    "Creating segment %d/%d: %.1fs - %.1fs",
                    i + 1,
                    num_segments,
                    start_time,
                    end_time,
                )

                # FIXED: Use copy codec when possible (much faster)
                # Calculate timeout based on segment duration (2x for safety)
                timeout = max(300, int(segment_duration * 2))

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-i",
                    str(video_path),
                    "-t",
                    str(segment_duration),
                    "-c",
                    "copy",  # FIXED: Copy codec (no re-encoding)
                    "-avoid_negative_ts",
                    "make_zero",  # FIXED: Better timestamp handling
                    str(segment_path),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )

                if result.returncode != 0:
                    logger.error("FFmpeg error: %s", result.stderr)
                    raise Exception(
                        "Failed to create segment %d: %s" % (i, result.stderr)
                    )

                if not segment_path.exists():
                    raise Exception("Segment file not created: %s" % segment_path)

                segments.append(
                    {
                        "segment_path": segment_path,
                        "start": start_time,
                        "end": end_time,
                        "duration": segment_duration,
                        "segment_number": i,
                        "is_temp": temp_dir is not None,
                    }
                )

            logger.info(
                "Successfully created %d segments for %s",
                len(segments),
                task_type,
            )
            return segments

        except Exception as e:
            logger.error("Error during segmentation: %s", e)
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise

    def segment_audio(
        self,
        audio_path: Path,
        duration_seconds: float,
        task_type: str = "summarization",
        output_dir: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Segment audio file into chunks based on task type.

        Args
        ----
        audio_path : Path
            Path to the audio file.
        duration_seconds : float
            Duration in seconds.
        task_type : str
            One of "summarization", "mcq", "temporal_localization".
        output_dir : Path, optional
            Directory for segment files; created if None.

        Returns
        -------
        list of dict
            Each dict has segment_path, start, end, duration,
            segment_number, and optionally is_temp.
        """
        if task_type == "summarization":
            max_segment_duration = self.summarization_segment_duration
        elif task_type == "mcq":
            max_segment_duration = self.mcq_segment_duration
        elif task_type == "temporal_localization":
            max_segment_duration = self.temporal_localization_segment_duration
        else:
            max_segment_duration = self.mcq_segment_duration
            logger.warning(
                "Unknown task_type '%s', defaulting to MCQ segment duration",
                task_type,
            )

        if duration_seconds <= max_segment_duration:
            return [
                {
                    "segment_path": audio_path,
                    "start": 0,
                    "end": duration_seconds,
                    "duration": duration_seconds,
                    "segment_number": 0,
                }
            ]

        if output_dir is None:
            output_dir = (
                Path.home()
                / "scratch"
                / "audio_segments"
                / f"{task_type}_{int(time.time())}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = output_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = None
        num_segments = int(duration_seconds / max_segment_duration) + 1
        segments = []

        try:
            for i in range(num_segments):
                start_time = max(
                    0,
                    i * max_segment_duration - (self.segment_overlap if i > 0 else 0),
                )
                segment_duration = min(
                    max_segment_duration + self.segment_overlap,
                    duration_seconds - start_time,
                )

                segment_path = output_dir / ("segment_%03d%s" % (i, audio_path.suffix))

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-i",
                    str(audio_path),
                    "-t",
                    str(segment_duration),
                    "-c",
                    "copy",
                    str(segment_path),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )

                if result.returncode != 0:
                    logger.error("FFmpeg audio segment error: %s", result.stderr)
                    raise Exception("Failed to create audio segment %d" % i)

                segments.append(
                    {
                        "segment_path": segment_path,
                        "start": start_time,
                        "end": start_time + segment_duration,
                        "duration": segment_duration,
                        "segment_number": i,
                        "is_temp": temp_dir is not None,
                    }
                )

            logger.info(
                "Successfully created %d audio segments for %s",
                len(segments),
                task_type,
            )
            return segments

        except Exception:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise

    def extract_transcript_segment(
        self,
        transcript_path: Path,
        start_time: float,
        end_time: float,
        strip_timestamps: bool = False,
    ) -> str:
        """
        Extract portion of SRT transcript for a time segment.

        Args
        ----
        transcript_path : Path
            Path to SRT file.
        start_time : float
            Segment start time in seconds.
        end_time : float
            Segment end time in seconds.
        strip_timestamps : bool
            If True, return plain text; else keep SRT blocks.

        Returns
        -------
        str
            Extracted transcript text or empty string on error.
        """
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()

            segments = content.strip().split("\n\n")
            extracted = []

            for segment in segments:
                lines = segment.split("\n")
                if len(lines) < 3:
                    continue

                timestamp_pattern = (
                    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
                    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
                )
                match = re.search(timestamp_pattern, lines[1])

                if match:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
                    seg_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
                    seg_end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000

                    if seg_start < end_time and seg_end > start_time:
                        if strip_timestamps:
                            text_lines = lines[2:]
                            extracted.append(" ".join(text_lines))
                        else:
                            extracted.append(segment)

            if strip_timestamps:
                return " ".join(extracted)
            return "\n\n".join(extracted)

        except Exception as e:
            logger.warning("Could not extract transcript segment: %s", e)
            return ""

    def cleanup_segments(self, segments: List[Dict]):
        """
        Clean up temporary segment files.

        Args
        ----
        segments : list of dict
            List of segment dicts with segment_path and is_temp.
        """
        for seg in segments:
            if seg.get("is_temp", False):
                try:
                    seg_path = seg["segment_path"]
                    if seg_path.exists():
                        temp_dir = seg_path.parent
                        if temp_dir.exists() and "segments" in temp_dir.name:
                            shutil.rmtree(temp_dir)
                            logger.info(
                                "Cleaned up temp directory: %s",
                                temp_dir,
                            )
                            break
                except Exception as e:
                    logger.warning(f"Failed to cleanup segment: {e}")
