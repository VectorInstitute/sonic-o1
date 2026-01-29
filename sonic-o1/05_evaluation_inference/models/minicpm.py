"""minicpm.py
MiniCPM-o-2.6 implementation with omni multimodal support.

Author: SONIC-O1 Team
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


try:
    import av
    import librosa
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:
    raise ImportError(
        f"Please install required packages: {e}\n"
        "pip install torch numpy transformers pillow librosa av"
    )

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class MiniCPM(BaseModel):
    """
    MiniCPM-o-2.6 wrapper with omni multimodal support.

    Processes video frames and audio chunks in a specialized format.
    Automatically calculates optimal FPS based on max_frames limit.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]) -> None:
        super().__init__(model_name, config)

        # Model configuration
        self.model_path = config.get("model_path", "openbmb/MiniCPM-o-2_6")

        # Device configuration
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = config.get("dtype", torch.bfloat16)
        self.attn_implementation = config.get("attn_implementation", "sdpa")

        # Generation config
        gen_config = config.get("generation_config", {})
        self.temperature = gen_config.get("temperature", 0.7)
        self.top_p = gen_config.get("top_p", 0.95)
        self.max_new_tokens = gen_config.get("max_new_tokens", 2048)

        # Audio processing config
        self.audio_sr = config.get("audio_sample_rate", 16000)
        self.audio_mono = config.get("audio_mono", True)

        # Frame limits
        self.default_min_frames = config.get("min_frames", 64)
        self.default_max_frames = config.get("max_frames", 256)

        logger.info(
            f"MiniCPM frame limits: {self.default_min_frames}-{self.default_max_frames}"
        )
        # Model settings
        self.init_vision = config.get("init_vision", True)
        self.init_audio = config.get("init_audio", True)
        self.init_tts = config.get("init_tts", False)
        self.language = config.get("language", "en")

        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load the MiniCPM-o-2.6 model and tokenizer."""
        try:
            logger.info(f"Loading MiniCPM-o-2.6 model from {self.model_path}")

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
                torch_dtype=self.dtype,
                init_vision=self.init_vision,
                init_audio=self.init_audio,
                init_tts=self.init_tts,
            )
            self.model = self.model.eval().to(self.device)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            logger.info(f"Successfully loaded MiniCPM-o-2.6 on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load MiniCPM-o-2.6 model: {e}")

    def _calculate_optimal_fps(self, duration: float) -> float:
        """
        Calculate optimal FPS to stay within max_frames limit.

        Always uses 1.0 fps unless it would exceed max_frames.

        Args:
            duration: Video duration in seconds.

        Returns:
            Optimal FPS value (frames per second).
        """
        # Start with 1 fps (1 frame per second)
        target_fps = 1.0

        # Calculate how many frames we'd get at 1 fps
        frames_at_1fps = int(duration * 1.0)

        # If 1 fps exceeds max_frames, reduce fps to hit max_frames exactly
        if frames_at_1fps > self.default_max_frames:
            target_fps = self.default_max_frames / duration
            logger.info(
                f"Duration: {duration:.1f}s @ 1fps would give {frames_at_1fps} frames "
                f"(exceeds max_frames={self.default_max_frames}) -> reducing to {target_fps:.4f} fps"
            )
        else:
            logger.info(
                f"Duration: {duration:.1f}s @ 1fps = {frames_at_1fps} frames "
                f"(within max_frames={self.default_max_frames})"
            )

        # Check min_frames
        estimated_frames = int(duration * target_fps)
        if estimated_frames < self.default_min_frames:
            logger.warning(
                f"Estimated {estimated_frames} frames is below min_frames={self.default_min_frames}. "
                f"Video may be too short for meaningful analysis."
            )

        logger.info(f"Using FPS: {target_fps:.4f} -> ~{estimated_frames} frames")
        return target_fps

    def _extract_video_audio_chunks(
        self,
        video_path: str,
        audio_path: str,
        target_num_frames: int,  # max_frames from config
        flatten: bool = True,
    ) -> List:
        """
        Extract video frames and audio chunks, then uniformly subsample.

        Strategy:
        1. Extract ALL frames at 1 fps (proven working approach)
        2. Extract ALL 1-second audio chunks
        3. Uniformly subsample both to target_num_frames
        """
        logger.info(
            f"Extracting frames and audio at 1 fps, then subsampling to {target_num_frames} frames..."
        )

        # Load audio with librosa
        audio_np, sr = librosa.load(audio_path, sr=self.audio_sr, mono=self.audio_mono)
        audio_duration = len(audio_np) / sr

        # Load video with PyAV
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        video_fps = float(video_stream.average_rate)
        total_frames = video_stream.frames
        video_duration = total_frames / video_fps

        logger.info(f"  Video: {video_duration:.1f}s @ {video_fps:.1f}fps")
        logger.info(f"  Audio: {audio_duration:.1f}s @ {sr}Hz")

        # Use the shorter duration
        duration = min(audio_duration, video_duration)
        num_units = math.ceil(duration)  # 1 fps = ceil(duration) frames

        logger.info(f"  Step 1: Extracting {num_units} units at 1 fps...")

        # Lists to collect all frames and audio chunks
        frames_list = []
        audio_list = []

        # Extract at 1 fps (matches working example)
        for i in range(num_units):
            # Frame at second i+1 (working example logic)
            target_time = min(i + 1, duration)

            # Skip if exceeds duration
            if target_time > duration:
                break

            # Calculate PTS for seeking
            target_pts = int(
                target_time
                * video_stream.time_base.denominator
                / video_stream.time_base.numerator
            )

            try:
                # Seek and extract frame
                container.seek(target_pts, stream=video_stream)

                frame = None
                for packet in container.demux(video_stream):
                    for frame_obj in packet.decode():
                        frame = frame_obj
                        break
                    if frame is not None:
                        break

                if frame is not None:
                    # Convert to PIL Image
                    image = frame.to_image()

                    # Get 1 second of audio (working example logic)
                    audio_chunk = audio_np[sr * i : sr * (i + 1)]

                    # Verify audio chunk
                    if len(audio_chunk) == 0:
                        logger.warning(f"  Empty audio chunk at unit {i}, skipping")
                        continue

                    # Add to lists
                    frames_list.append(image)
                    audio_list.append(audio_chunk)

            except Exception as e:
                logger.warning(f"  Error at unit {i}: {e}")
                continue

        container.close()

        total_extracted = len(frames_list)
        logger.info(f"  Step 1 complete: Extracted {total_extracted} frame-audio pairs")

        # Step 2: Uniform subsampling if needed
        if total_extracted <= target_num_frames:
            # No subsampling needed
            logger.info(
                f"  Step 2: No subsampling needed ({total_extracted} <= {target_num_frames})"
            )
            selected_frames = frames_list
            selected_audio = audio_list
        else:
            # Uniformly subsample
            logger.info(
                f"  Step 2: Subsampling {total_extracted} -> {target_num_frames} frames..."
            )

            # Calculate indices for uniform sampling
            indices = np.linspace(0, total_extracted - 1, target_num_frames, dtype=int)

            selected_frames = [frames_list[i] for i in indices]
            selected_audio = [audio_list[i] for i in indices]

            logger.info(f"  Subsampling indices: {indices[:5]}...{indices[-5:]}")

        final_count = len(selected_frames)
        logger.info(f"  Final: {final_count} frame-audio pairs ready")

        # Build contents in MiniCPM format
        contents = []
        for i in range(final_count):
            if flatten:
                contents.extend(["<unit>", selected_frames[i], selected_audio[i]])
            else:
                contents.append(["<unit>", selected_frames[i], selected_audio[i]])

        # Validate
        if flatten:
            expected_elements = final_count * 3
            if len(contents) != expected_elements:
                raise RuntimeError(
                    f"Content structure error: {len(contents)} elements != {expected_elements} expected"
                )

        logger.info(f"  ✓ Built {final_count} units for model input")

        return contents

    def generate(
        self,
        frames: Union[List[np.ndarray], np.ndarray, str],
        audio: Optional[Union[np.ndarray, str]],
        prompt: str,
        fps: Optional[float] = None,  # Ignored - kept for API compatibility
        video_category: Optional[Literal["short", "medium", "long"]] = None,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate response from video and audio.

        Note: fps parameter is ignored. Frame extraction is always based on max_frames.

        Args:
            frames: Video file path (str)
            audio: Audio file path (str)
            prompt: Text prompt for generation
            fps: Ignored - kept for API compatibility
            video_category: Unused
            **kwargs: Additional generation parameters.

        Returns:
            Generated text response.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Use max_frames from parameter if provided, otherwise use default
        actual_max_frames = (
            max_frames if max_frames is not None else self.default_max_frames
        )

        # Validate inputs
        if not isinstance(frames, str):
            raise ValueError(
                f"MiniCPM requires video file path (str), got {type(frames)}. "
                f"Ensure 'supports_video: true' in config."
            )

        if not isinstance(audio, str):
            raise ValueError(
                f"MiniCPM requires audio file path (str), got {type(audio)}"
            )

        try:
            video_path = Path(frames)
            audio_path = Path(audio)

            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            logger.info(f"Processing video: {video_path.name}")
            logger.info(f"Processing audio: {audio_path.name}")
            logger.info(
                f"Using max_frames: {actual_max_frames}"
            )  # LOG THE ACTUAL VALUE

            # Extract frames and audio chunks
            contents = self._extract_video_audio_chunks(
                str(video_path),
                str(audio_path),
                target_num_frames=actual_max_frames,
                flatten=True,
            )

            # Validate
            num_units = len(contents) // 3
            logger.info(f"Total units ready for model: {num_units}")

            if num_units > actual_max_frames:
                raise RuntimeError(
                    f"BUG: Extracted {num_units} frames exceeds max_frames ({actual_max_frames})"
                )

            # Build conversation
            sys_msg = self.model.get_sys_prompt(mode="omni", language=self.language)
            msg = {"role": "user", "content": contents + [prompt]}
            msgs = [sys_msg, msg]

            # Generation parameters
            temperature = kwargs.get("temperature", self.temperature)
            kwargs.get("top_p", self.top_p)
            max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)

            logger.info(
                f"Generating (temp={temperature}, max_tokens={max_new_tokens})..."
            )

            # Generate
            res = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                omni_input=True,
                use_tts_template=False,
                generate_audio=False,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
            )

            response_text = res["text"]
            logger.info(f"Generated response ({len(response_text)} chars)")

            return self.postprocess_output(response_text)

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Generation failed: {e}")

    def unload(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded and memory cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                "model_path": self.model_path,
                "model_type": "Omni Multimodal",
                "device": str(self.device),
                "dtype": str(self.dtype),
                "audio_sample_rate": self.audio_sr,
                "default_frame_limits": f"{self.default_min_frames}-{self.default_max_frames}",
                "fps_strategy": "Adaptive (always respects max_frames)",
                "input_format": "Interleaved frames and audio chunks",
            }
        )
        return info
