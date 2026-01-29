"""Segment videos and audio using ffmpeg/ffprobe."""

import logging
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)


def _run_cmd(cmd, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a subprocess command with logging and error handling."""
    logger.debug("Running command: %s", " ".join(map(str, cmd)))
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise EnvironmentError(
            "ffmpeg/ffprobe not found. Make sure FFmpeg is installed "
            "and available on PATH."
        )

    if result.returncode != 0:
        logger.error("Command failed: %s", result.stderr)
    else:
        logger.debug("Command stdout: %s", result.stdout)
    return result


class VideoSegmenter:
    """Segment videos and audio using FFmpeg."""

    def __init__(self):
        # Optional sanity check so failures happen early
        for bin_name in ("ffmpeg", "ffprobe"):
            try:
                subprocess.run(
                    [bin_name, "-version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except FileNotFoundError:
                raise EnvironmentError(
                    f"{bin_name} not found. Install FFmpeg and ensure it is on PATH."
                )

    # -------------------------------------------------------------------------
    # Duration helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def get_video_duration(video_path: Path) -> float:
        """
        Get video duration using ffprobe.

        Tries stream duration first (more reliable), then falls back
        to container/format duration.

        Args:
            video_path: Path to video file

        Returns
        -------
            Duration in seconds (float)
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 1) Try video stream duration
        stream_cmd = [
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

        result = _run_cmd(stream_cmd, timeout=10)
        if result.returncode == 0:
            out = result.stdout.strip()
            if out and out != "N/A":
                try:
                    duration = float(out)
                    if duration > 0:
                        logger.debug("Got stream duration: %.3fs", duration)
                        return duration
                except ValueError:
                    pass

        # 2) Fallback: format duration
        format_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video_path),
        ]
        result = _run_cmd(format_cmd, timeout=10)
        if result.returncode == 0:
            out = result.stdout.strip()
            if out and out != "N/A":
                try:
                    duration = float(out)
                    if duration > 0:
                        logger.debug("Got format duration: %.3fs", duration)
                        return duration
                except ValueError:
                    pass

        raise RuntimeError(f"Could not determine duration for {video_path}")

    # -------------------------------------------------------------------------
    # Video segment extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_video_segment(
        video_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
    ) -> Path:
        """
        Extract video segment using FFmpeg (video + audio, stream copy if possible).

        Args:
            video_path: Path to source video (.mp4, etc.)
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path for output segment

        Returns
        -------
            Path to extracted segment
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        segment_duration = end_time - start_time
        # Safety timeout: 2x segment duration, minimum 60s
        timeout = max(60, int(segment_duration * 2))

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{segment_duration:.3f}",
            "-c",
            "copy",  # stream copy: fast, no re-encode
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]

        result = _run_cmd(cmd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to extract video segment: {result.stderr.strip()}"
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Segment file not created: {output_path}")

        logger.info(
            "Extracted video segment: %.1fs - %.1fs -> %s",
            start_time,
            end_time,
            output_path,
        )
        return output_path

    # -------------------------------------------------------------------------
    # Audio segment extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_audio_segment(
        audio_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
        output_format: str = "m4a",
    ) -> Path:
        """
        Extract audio segment using FFmpeg.

        Args:
            audio_path: Path to source audio
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path for output segment
            output_format: 'm4a' (AAC) or 'wav' (PCM S16LE 16kHz)

        Returns
        -------
            Path to extracted segment
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        segment_duration = end_time - start_time
        timeout = max(60, int(segment_duration * 2))

        output_format = output_format.lower()

        # Base command
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            str(audio_path),
            "-t",
            f"{segment_duration:.3f}",
        ]

        if output_format == "wav":
            # PCM 16-bit, 16kHz — good for ASR pipelines
            cmd += [
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
            ]
        elif output_format == "m4a":
            # AAC in an m4a container
            cmd += [
                "-c:a",
                "aac",
                "-b:a",
                "128k",
            ]
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")

        cmd.append(str(output_path))

        result = _run_cmd(cmd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to extract audio segment: {result.stderr.strip()}"
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Audio segment not created: {output_path}")

        logger.info(
            "Extracted audio segment: %.1fs - %.1fs (%s) -> %s",
            start_time,
            end_time,
            output_format,
            output_path,
        )
        return output_path

    # -------------------------------------------------------------------------
    # Audio format conversion
    # -------------------------------------------------------------------------
    @staticmethod
    def convert_audio_format(
        input_path: Path,
        output_path: Path,
        output_format: str = "wav",
    ) -> Path:
        """
        Convert audio to a different format using FFmpeg.

        Args:
            input_path: Path to input audio
            output_path: Path for output audio
            output_format: Target format ('wav' or 'm4a')

        Returns
        -------
            Path to converted audio
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio not found: {input_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_format = output_format.lower()

        cmd = ["ffmpeg", "-y", "-i", str(input_path)]

        if output_format == "wav":
            cmd += [
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
            ]
        elif output_format == "m4a":
            cmd += [
                "-c:a",
                "aac",
                "-b:a",
                "128k",
            ]
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")

        cmd.append(str(output_path))

        result = _run_cmd(cmd, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert audio: {result.stderr.strip()}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Converted audio not created: {output_path}")

        logger.info("Converted audio to %s -> %s", output_format, output_path)
        return output_path
