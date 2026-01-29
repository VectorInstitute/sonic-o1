"""base_model.py

Abstract base class for all multimodal models in the evaluation framework.

Author: SONIC-O1 Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all multimodal models.

    All model implementations must inherit from this class and implement
    the abstract methods.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]) -> None:
        """
        Initialize the base model.

        Args:
            model_name: Name of the model.
            config: Configuration dict from models_config.yaml.
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.supports_video = config.get("supports_video", True)
        self.supports_audio = config.get("supports_audio", True)

    @abstractmethod
    def load(self) -> None:
        """
        Initialize and load the model.

        Load model weights, processors/tokenizers, and move to device.

        Raises:
            Exception: If model loading fails.
        """
        pass

    @abstractmethod
    def generate(
        self,
        frames: Union[List[np.ndarray], np.ndarray, str],
        audio: Optional[Union[np.ndarray, str]],
        prompt: str,
        fps: Optional[float] = None,
        video_category: Optional[Literal["short", "medium", "long"]] = None,
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate response from video frames and audio.

        Args:
            frames: Either:
                - List of video frames (for image models)
                - Video file path (for video models)
                - Numpy array of frames
            audio: Either:
                - Audio data as numpy array
                - Audio file path
                - None if audio not available
            prompt: Text prompt for the model
            fps: Optional FPS for video (used by video models for memory).
            video_category: Optional video length category for timeout/memory:
                - 'short': < 5 minutes
                - 'medium': 5-20 minutes
                - 'long': > 20 minutes
            **kwargs: Additional model-specific parameters (e.g. temperature,
                max_tokens, top_p).

        Returns:
            Model's text response.

        Raises:
            Exception: If generation fails.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Clean up model resources.

        Clear model from memory, release GPU memory, close file handles.
        """
        pass

    def preprocess_frames(
        self, frames: Union[List[np.ndarray], np.ndarray], **kwargs
    ) -> Any:
        """
        Preprocess frames for model input.

        This is an optional method that can be overridden for custom preprocessing.

        Args:
            frames: Input frames
            **kwargs: Additional preprocessing parameters.

        Returns:
            Preprocessed frames in model-specific format.
        """
        return frames

    def preprocess_audio(self, audio: Union[np.ndarray, str], **kwargs) -> Any:
        """
        Preprocess audio for model input.

        This is an optional method that can be overridden for custom preprocessing.

        Args:
            audio: Input audio
            **kwargs: Additional preprocessing parameters.

        Returns:
            Preprocessed audio in model-specific format.
        """
        return audio

    def postprocess_output(self, output: Any) -> str:
        """
        Postprocess model output.

        This is an optional method that can be overridden for custom postprocessing.

        Args:
            output: Raw model output.

        Returns:
            Cleaned and formatted output text.
        """
        if isinstance(output, str):
            return output.strip()
        return str(output).strip()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dict with name, supports_video, supports_audio, config.
        """
        return {
            "name": self.model_name,
            "supports_video": self.supports_video,
            "supports_audio": self.supports_audio,
            "config": self.config,
        }

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
