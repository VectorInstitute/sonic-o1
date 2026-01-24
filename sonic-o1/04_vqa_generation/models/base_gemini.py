"""Base Gemini API client - reusable logic for all VQA tasks."""

import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)


class BaseGeminiClient:
    """Base class for Gemini API interactions."""

    def __init__(self, config):
        """
        Initialize Gemini client with configuration.

        Args:
            config: Configuration object with API settings
        """
        self.config = config
        self.model_name = config.gemini.model_name
        self.retry_attempts = int(config.gemini.retry_attempts)
        self.retry_delay = int(config.gemini.retry_delay)
        self.file_processing_timeout = int(config.gemini.file_processing_timeout)
        self.inline_threshold = (
            int(config.file_processing.inline_threshold_mb) * 1024 * 1024
        )

        # Rate limiting settings (with type conversion)
        self.rate_limit_delay = int(
            getattr(config.rate_limit, "delay_after_api_call", 2)
        )
        self.rate_limit_max_retries = int(
            getattr(config.rate_limit, "max_retries_on_rate_limit", 5)
        )
        self.rate_limit_backoff = int(
            getattr(config.rate_limit, "rate_limit_backoff_multiplier", 2)
        )

        self.setup_client()

    def setup_client(self):
        """Initialize the Gemini client."""
        api_key = self.config.gemini.api_key
        if api_key.startswith("${") and api_key.endswith("}"):
            # Extract environment variable name
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"Environment variable {env_var} not set")

        os.environ["GEMINI_API_KEY"] = api_key
        self.client = genai.Client()
        logger.info(f"Initialized Gemini client with model: {self.model_name}")

    def generate_content(
        self, media_files: List[Tuple[str, Path]], prompt: str, video_fps: float = 1.0
    ) -> str:
        """
        Generate content using Gemini with multimodal inputs.

        Args:
            media_files: List of tuples (media_type, Path) - e.g., [('video', path), ('audio', path)]
            prompt: Text prompt for generation
            video_fps: FPS for video sampling (default: 1.0)

        Returns
        -------
            Generated text response
        """
        # Calculate total size to determine processing method
        total_size = sum(os.path.getsize(path) for _, path in media_files)

        if total_size > self.inline_threshold:
            logger.info(
                f"Using File API for large media (size: {total_size / (1024 * 1024):.2f}MB)"
            )
            return self._process_with_file_api(media_files, prompt, video_fps)
        logger.info(
            f"Using inline processing (size: {total_size / (1024 * 1024):.2f}MB)"
        )
        return self._process_inline(media_files, prompt, video_fps)

    def _process_with_file_api(
        self, media_files: List[Tuple[str, Path]], prompt: str, video_fps: float = 1.0
    ) -> str:
        """Process large files using Gemini File API."""
        uploaded_files = []
        try:
            # Upload all media files
            for media_type, media_path in media_files:
                uploaded_file = self.client.files.upload(file=str(media_path))
                logger.info(f"Uploaded {media_type}: {uploaded_file.name}")
                uploaded_files.append(
                    (media_type, uploaded_file)
                )  # Store type with file

            # Wait for all files to process
            max_wait = self.file_processing_timeout
            wait_time = 0
            all_processed = False

            while not all_processed and wait_time < max_wait:
                all_processed = True
                for i, (media_type, uploaded_file) in enumerate(uploaded_files):
                    updated_file = self.client.files.get(name=uploaded_file.name)
                    uploaded_files[i] = (media_type, updated_file)
                    if updated_file.state == "PROCESSING":
                        all_processed = False
                    elif updated_file.state == "FAILED":
                        error_msg = getattr(updated_file, "error", "Unknown error")
                        raise Exception(f"File processing failed: {error_msg}")

                if not all_processed:
                    time.sleep(10)
                    wait_time += 10
                    if wait_time % 60 == 0:  # Log every minute
                        logger.info(
                            f"Still waiting for file processing... ({wait_time}s elapsed)"
                        )

            if not all_processed:
                raise Exception(f"File processing timeout after {max_wait}s")

            # Generate content with uploaded files + prompt
            for attempt in range(self.retry_attempts):
                try:
                    # Build content parts with video_metadata for video files
                    content_parts = []
                    for media_type, uploaded_file in uploaded_files:
                        if media_type == "video":
                            content_parts.append(
                                types.Part(
                                    file_data=types.FileData(
                                        file_uri=uploaded_file.uri,
                                        mime_type=uploaded_file.mime_type,
                                    ),
                                    video_metadata=types.VideoMetadata(fps=video_fps),
                                )
                            )
                        else:
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
                        model=self.model_name,
                        contents=types.Content(parts=content_parts),
                    )
                    return response.text
                except Exception as e:
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

    def _process_inline(
        self, media_files: List[Tuple[str, Path]], prompt: str, video_fps: float = 1.0
    ) -> str:
        """Process small files using inline data."""
        parts = []

        # Add all media files as inline data
        for media_type, media_path in media_files:
            with open(media_path, "rb") as f:
                media_bytes = f.read()

            mime_type = self._get_mime_type(media_path)

            # Add video metadata only for video files
            if media_type == "video":
                parts.append(
                    types.Part(
                        inline_data=types.Blob(data=media_bytes, mime_type=mime_type),
                        video_metadata=types.VideoMetadata(fps=video_fps),
                    )
                )
                logger.info(
                    f"Added {media_type} ({mime_type}) as inline data with fps={video_fps}"
                )
            else:
                parts.append(
                    types.Part(
                        inline_data=types.Blob(data=media_bytes, mime_type=mime_type)
                    )
                )
                logger.info(f"Added {media_type} ({mime_type}) as inline data")

        # Add text prompt
        parts.append(types.Part(text=prompt))

        # Generate with retries
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=types.Content(parts=parts)
                )
                return response.text
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        return None

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for media file."""
        extension_map = {
            # Video
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
            ".mkv": "video/x-matroska",
            ".m4v": "video/x-m4v",
            # Audio
            ".m4a": "audio/m4a",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
        }

        ext = file_path.suffix.lower()
        return extension_map.get(ext, "application/octet-stream")
