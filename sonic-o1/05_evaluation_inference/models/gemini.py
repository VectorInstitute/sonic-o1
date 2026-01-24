"""
models/gemini.py
Gemini 3.0 Pro implementation adapted from working Gemini 2.5 code.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional


try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Please install google-genai: pip install google-genai")

from .base_model import BaseModel


logger = logging.getLogger(__name__)


class Gemini(BaseModel):
    """Gemini 3.0 Pro Preview wrapper with native video and audio support."""

    # Timeout configurations for different video lengths
    # short: < 5 minutes
    # medium: 5-20 minutes
    # long: > 20 minutes
    TIMEOUT_CONFIG = {
        "short": 180,  # 3 minutes - videos under 5 minutes
        "medium": 600,  # 10 minutes - videos 5-20 minutes
        "long": 1800,  # 30 minutes - videos over 20 minutes
    }

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)

        self.supports_video = True
        self.supports_audio = True

        api_key_env = config.get("api_key_env", "GEMINI_API_KEY")
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(
                f"API key not found. Please set {api_key_env} environment variable."
            )

        self.model_version = config.get("model_version", "gemini-3-pro-preview")

        gen_config = config.get("generation_config", {})
        self.generation_config = types.GenerateContentConfig(
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.95),
            top_k=gen_config.get("top_k", 40),
            max_output_tokens=gen_config.get("max_output_tokens", 8192),
        )

        # Retry configuration
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 2)

        # Default timeout (can be overridden per request)
        self.default_timeout = config.get(
            "file_processing_timeout", self.TIMEOUT_CONFIG["medium"]
        )

        self.client = None

    def _get_timeout_for_category(
        self, category: Optional[Literal["short", "medium", "long"]] = None
    ) -> int:
        """
        Get appropriate timeout based on video category.

        Args:
            category: Video length category
                - 'short': < 5 minutes
                - 'medium': 5-20 minutes
                - 'long': > 20 minutes
                - None: uses default timeout

        Returns
        -------
            Timeout in seconds
        """
        if category is None:
            return self.default_timeout

        timeout = self.TIMEOUT_CONFIG.get(category)
        if timeout is None:
            logger.warning(f"Unknown category '{category}', using default timeout")
            return self.default_timeout

        logger.info(f"Using {timeout}s timeout for '{category}' video")
        return timeout

    def _estimate_timeout_from_file(self, video_path: Path) -> int:
        """
        Estimate timeout based on file size as a fallback.
        Rule of thumb: ~1 second per MB + base timeout of 180s.

        Args:
            video_path: Path to video file

        Returns
        -------
            Estimated timeout in seconds
        """
        try:
            file_size_mb = video_path.stat().st_size / (1024 * 1024)

            # Base timeout + file size factor
            base_timeout = 180
            size_factor = int(file_size_mb)  # 1 second per MB

            estimated = base_timeout + size_factor

            # Cap at 'long' timeout maximum
            max_timeout = self.TIMEOUT_CONFIG["long"]
            timeout = min(estimated, max_timeout)

            logger.info(
                f"Estimated timeout from file size ({file_size_mb:.1f}MB): {timeout}s"
            )
            return timeout

        except Exception as e:
            logger.warning(f"Could not estimate timeout from file: {e}")
            return self.default_timeout

    def load(self):
        try:
            os.environ["GEMINI_API_KEY"] = self.api_key
            self.client = genai.Client()
            logger.info("Loaded Gemini 3.0 Pro Preview")
        except Exception as e:
            raise RuntimeError(f"Failed to load Gemini client: {e}")

    def generate(
        self,
        frames: str,
        audio: Optional[str],
        prompt: str,
        video_category: Optional[Literal["short", "medium", "long"]] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate content from video/audio with Gemini.

        Args:
            frames: Path to video file
            audio: Optional path to audio file
            prompt: Text prompt for generation
            video_category: Optional category hint for timeout selection
                - 'short': < 5 minutes (180s timeout)
                - 'medium': 5-20 minutes (600s timeout)
                - 'long': > 20 minutes (1800s timeout)
            **kwargs: Additional generation parameters

        Returns
        -------
            Generated text response
        """
        if self.client is None:
            raise RuntimeError("Client not loaded. Call load() first.")

        if not isinstance(frames, str):
            raise ValueError(
                f"Gemini requires video file path (str), got {type(frames)}"
            )

        try:
            media_files = [("video", Path(frames))]

            if audio is not None and isinstance(audio, str) and os.path.exists(audio):
                media_files.append(("audio", Path(audio)))

            config = self.generation_config
            if "temperature" in kwargs:
                config.temperature = kwargs["temperature"]
            if "max_output_tokens" in kwargs:
                config.max_output_tokens = kwargs["max_output_tokens"]

            # Determine timeout
            if video_category:
                timeout = self._get_timeout_for_category(video_category)
            else:
                # Fallback: estimate from file size
                timeout = self._estimate_timeout_from_file(Path(frames))

            response_text = self._process_with_file_api(
                media_files, prompt, config, timeout, fps
            )

            return self.postprocess_output(response_text)

        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def _process_with_file_api(
        self,
        media_files: list,
        prompt: str,
        config: types.GenerateContentConfig,
        timeout: int,
        fps: Optional[float] = None,
    ) -> str:
        """
        Process files using Gemini File API.

        Args:
            media_files: List of (media_type, Path) tuples
            prompt: Text prompt
            config: Generation config
            timeout: Processing timeout in seconds
        """
        uploaded_files = []

        try:
            # Upload all files
            for media_type, media_path in media_files:
                logger.info(f"Uploading {media_type}: {media_path.name}")
                uploaded_file = self.client.files.upload(file=str(media_path))
                logger.info(f"Uploaded {media_type}: {uploaded_file.name}")
                uploaded_files.append((media_type, uploaded_file))

            # Wait for processing with configurable timeout
            check_interval = 2  # Check every 2 seconds
            wait_time = 0
            all_processed = False

            logger.info(f"Waiting for file processing (timeout: {timeout}s)...")

            while not all_processed and wait_time < timeout:
                all_processed = True
                for i, (media_type, uploaded_file) in enumerate(uploaded_files):
                    updated_file = self.client.files.get(name=uploaded_file.name)
                    uploaded_files[i] = (media_type, updated_file)

                    if updated_file.state == "PROCESSING":
                        all_processed = False
                        if wait_time % 10 == 0:  # Log every 10 seconds
                            logger.info(
                                f"Still processing {media_type} ({wait_time}s elapsed)..."
                            )
                    elif updated_file.state == "FAILED":
                        error_msg = getattr(updated_file, "error", "Unknown error")
                        raise Exception(f"File processing failed: {error_msg}")

                if not all_processed:
                    time.sleep(check_interval)
                    wait_time += check_interval

            if not all_processed:
                raise Exception(f"File processing timeout after {timeout}s")

            logger.info(f"All files processed successfully in {wait_time}s")

            # Generate content with retries
            for attempt in range(self.retry_attempts):
                try:
                    content_parts = []

                    # Add all media files
                    for media_type, uploaded_file in uploaded_files:
                        if media_type == "video" and fps is not None:
                            # Add video with FPS metadata
                            content_parts.append(
                                types.Part(
                                    file_data=types.FileData(
                                        file_uri=uploaded_file.uri,
                                        mime_type=uploaded_file.mime_type,
                                    ),
                                    video_metadata=types.VideoMetadata(fps=fps),
                                )
                            )
                        else:
                            # Add without metadata (audio or video without FPS)
                            content_parts.append(
                                types.Part(
                                    file_data=types.FileData(
                                        file_uri=uploaded_file.uri,
                                        mime_type=uploaded_file.mime_type,
                                    )
                                )
                            )

                    # Add prompt
                    content_parts.append(types.Part(text=prompt))

                    response = self.client.models.generate_content(
                        model=self.model_version,
                        contents=types.Content(parts=content_parts),
                        config=config,
                    )

                    return response.text

                except Exception as e:
                    error_str = str(e)
                    if (
                        "429" in error_str
                        or "quota" in error_str.lower()
                        or "resource_exhausted" in error_str.lower()
                    ) and attempt < self.retry_attempts - 1:
                        wait_time = (attempt + 1) * 5
                        logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise

        finally:
            # Cleanup uploaded files
            for media_type, uploaded_file in uploaded_files:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.debug(f"Deleted uploaded file: {uploaded_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {uploaded_file.name}: {e}")

    def unload(self):
        self.client = None

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "model_version": self.model_version,
                "api_based": True,
                "native_video": True,
                "native_audio": True,
                "sdk": "google-genai",
                "timeout_config": self.TIMEOUT_CONFIG,
            }
        )
        return info
