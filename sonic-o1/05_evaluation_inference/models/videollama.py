"""
models/videollama.py
VideoLLaMA2 implementation following BaseModel pattern.
Self-contained with all VideoLLaMA2-specific logic.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class VideoLLaMA2(BaseModel):
    """VideoLLaMA2 wrapper following BaseModel pattern."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)

        self.model_path = config.get("model_path", "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV")
        self.videollama_repo_path = config.get("videollama_repo_path")

        self.default_max_frames = config.get("max_frames", 32)
        self.default_min_frames = config.get("min_frames", 8)

        self.device_map = config.get("device_map", "auto")
        self.dtype = config.get("dtype", "bfloat16")

        gen_config = config.get("generation_config", {})
        self.temperature = gen_config.get("temperature", 0.7)
        self.top_p = gen_config.get("top_p", 0.95)
        self.max_new_tokens = gen_config.get("max_new_tokens", 2048)

        self.model = None
        self.raw_processor = None
        self.tokenizer = None
        self._original_max_frames = None

    def load(self):
        """Load VideoLLaMA2 model."""
        if self.videollama_repo_path:
            repo_path = os.path.expanduser(self.videollama_repo_path)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
                logger.info(f"Added VideoLLaMA2 repo to path: {repo_path}")

        try:
            from videollama2 import mm_utils, model_init
            from videollama2.utils import disable_torch_init
        except ImportError as e:
            raise ImportError(
                f"Failed to import VideoLLaMA2 modules. Set 'videollama_repo_path' in config.\n"
                f"Error: {e}"
            )

        self._original_max_frames = mm_utils.MAX_FRAMES
        mm_utils.MAX_FRAMES = self.default_max_frames
        logger.info(
            f"Patched MAX_FRAMES: {self._original_max_frames} -> {self.default_max_frames}"
        )

        logger.info(f"Loading VideoLLaMA2 from {self.model_path}")
        disable_torch_init()

        self.model, processor_dict, self.tokenizer = model_init(
            self.model_path, device_map=self.device_map
        )

        # Store ALL processors (not just video)
        self.processor_dict = processor_dict
        self.raw_processor = processor_dict["video"].keywords[
            "processor"
        ]  # Keep this for compatibility

        logger.info("VideoLLaMA2 model loaded successfully")

        if self.device_map == "auto" and hasattr(self.model, "hf_device_map"):
            logger.info("\n=== Device Map ===")
            device_distribution = {}
            for _name, device in self.model.hf_device_map.items():
                device_distribution[device] = device_distribution.get(device, 0) + 1
            for device, count in sorted(device_distribution.items()):
                logger.info(f"{device}: {count} modules")
            logger.info("==================\n")

    def convert_av1_to_h264(
        self, video_path: Path, output_dir: Optional[Path] = None
    ) -> Path:
        """
        Convert AV1 video to H.264 for compatibility.

        Args:
            video_path: Input video path
            output_dir: Output directory (default: creates 'converted' subdir)

        Returns
        -------
            Path to converted video
        """
        if output_dir is None:
            output_dir = video_path.parent / "converted"
            output_dir.mkdir(exist_ok=True)

        output_path = output_dir / f"{video_path.stem}_h264{video_path.suffix}"

        if output_path.exists():
            logger.info(f"Using cached converted video: {output_path.name}")
            return output_path

        logger.info(f"Converting AV1 to H.264: {video_path.name}")

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Conversion successful: {output_path.name}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e.stderr}")
            raise RuntimeError(f"Failed to convert AV1 video: {e.stderr}")

    def _check_video_compatibility(self, video_path: Path) -> Optional[Path]:
        """
        Check if video is compatible with VideoLLaMA2's video processor.
        If AV1, automatically convert to H.264.

        Args:
            video_path: Original video path

        Returns
        -------
            Path to compatible video (original or converted), or None if incompatible
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            codec = result.stdout.strip()

            if codec == "av1":
                logger.info(
                    f"Detected AV1 codec in {video_path.name} - converting to H.264..."
                )
                try:
                    return self.convert_av1_to_h264(video_path)
                except Exception as conv_error:
                    logger.error(f"Conversion failed: {conv_error}")
                    return None
            else:
                logger.info(f"Video codec '{codec}' - proceeding without conversion")
                return video_path

        except subprocess.TimeoutExpired:
            logger.warning(
                f"ffprobe timeout for {video_path.name} - assuming compatible"
            )
            return video_path
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to detect codec for {video_path.name}: {e.stderr}")
            return video_path
        except FileNotFoundError:
            logger.warning("ffprobe not found - skipping AV1 detection")
            return video_path

    def _merge_video_audio(self, video_path: Path, audio_path: Path) -> Path:
        """Merge separate video and audio files into a single video file."""
        temp_output = Path(tempfile.mktemp(suffix=".mp4"))

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-shortest",
            "-y",
            str(temp_output),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Merged video and audio to {temp_output}")
            return temp_output
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg merge failed: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to merge video and audio: {e}")

    def generate(
        self,
        frames: Optional[str],
        audio: Optional[str],
        prompt: str,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> str:
        from videollama2 import mm_infer
        from videollama2.mm_utils import process_video as pv

        if frames is None and audio is None:
            raise ValueError("At least one of 'frames' or 'audio' must be provided")

        has_video = frames is not None
        has_audio = audio is not None

        if has_video and has_audio:
            modality_mode = "video+audio"
            modal_type = "video"
        elif has_video:
            modality_mode = "video-only"
            modal_type = "video"
        else:
            modality_mode = "audio-only"
            modal_type = "audio"

        logger.info(f"Modality mode: {modality_mode}")

        actual_max_frames = (
            max_frames if max_frames is not None else self.default_max_frames
        )

        if modality_mode == "audio-only":
            if hasattr(self.model.model, "vision_tower"):
                self.model.model.vision_tower = None
        elif modality_mode == "video-only":
            if hasattr(self.model.model, "audio_tower"):
                self.model.model.audio_tower = None

        if has_video:
            video_path = Path(frames)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            compatible_video_path = self._check_video_compatibility(video_path)
            if compatible_video_path is None:
                raise RuntimeError(
                    f"Video codec incompatible with VideoLLaMA2. "
                    f"Skipping this video: {video_path}"
                )

        if has_audio:
            audio_path = Path(audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

        temp_merged_file = None

        try:
            if modality_mode == "audio-only":
                logger.info("Processing audio-only")
                audio_processor = self.processor_dict["audio"]
                tensor = audio_processor(str(audio_path))

            elif modality_mode == "video-only":
                logger.info(
                    f"Processing video-only with max_frames={actual_max_frames}"
                )
                tensor = pv(
                    str(compatible_video_path),
                    processor=self.raw_processor,
                    aspect_ratio=None,
                    num_frames=actual_max_frames,
                    va=False,
                )

            else:
                logger.info("Merging video and audio")
                merged_video = self._merge_video_audio(
                    compatible_video_path, audio_path
                )
                temp_merged_file = merged_video

                logger.info(
                    f"Processing video+audio with max_frames={actual_max_frames}"
                )
                tensor = pv(
                    str(merged_video),
                    processor=self.raw_processor,
                    aspect_ratio=None,
                    num_frames=actual_max_frames,
                    va=True,
                )

            output = mm_infer(
                tensor,
                prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                modal=modal_type,
                do_sample=False,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
            )

            logger.info(f"Generated {len(output)} characters")

            return self.postprocess_output(output)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e)

            if (
                "out of memory" in error_msg.lower()
                or "size of tensor" in error_msg.lower()
            ):
                logger.error(f"OOM error: {error_msg[:200]}...")
                torch.cuda.empty_cache()
                raise RuntimeError(f"Out of memory: {e}")
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")

        finally:
            if temp_merged_file and temp_merged_file.exists():
                temp_merged_file.unlink()
                logger.info(f"Cleaned up temporary file: {temp_merged_file}")

    def unload(self):
        """Unload model and restore MAX_FRAMES."""
        logger.info("Unloading VideoLLaMA2 model...")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.raw_processor is not None:
            del self.raw_processor
            self.raw_processor = None

        try:
            from videollama2 import mm_utils

            mm_utils.MAX_FRAMES = self._original_max_frames
            logger.info(f"Restored MAX_FRAMES to {self._original_max_frames}")
        except:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("VideoLLaMA2 model unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                "model_path": self.model_path,
                "backend": "HuggingFace Transformers",
                "native_video": True,
                "native_audio": True,
                "default_max_frames": self.default_max_frames,
                "default_min_frames": self.default_min_frames,
            }
        )
        return info
