"""
Base model class for evaluation framework.
All model implementations should inherit from this class.
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

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the base model.

        Args:
            model_name: Name of the model
            config: Configuration dictionary from models_config.yaml
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.supports_video = config.get("supports_video", True)
        self.supports_audio = config.get("supports_audio", True)

    @abstractmethod
    def load(self):
        """
        Initialize and load the model.

        This method should:
        - Load model weights
        - Initialize processors/tokenizers
        - Set up any required configurations
        - Move model to appropriate device

        Raises
        ------
            Exception: If model loading fails
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
            fps: Optional FPS for video processing (used by video models for memory optimization)
            video_category: Optional video length category for timeout/memory optimization:
                - 'short': < 5 minutes
                - 'medium': 5-20 minutes
                - 'long': > 20 minutes
            **kwargs: Additional model-specific parameters such as:
                - temperature: Sampling temperature
                - max_tokens: Maximum generation length
                - top_p: Nucleus sampling parameter

        Returns
        -------
            str: Model's text response

        Raises
        ------
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def unload(self):
        """
        Clean up model resources.

        This method should:
        - Clear model from memory
        - Release GPU memory
        - Close any open file handles
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
            **kwargs: Additional preprocessing parameters

        Returns
        -------
            Preprocessed frames in model-specific format
        """
        return frames

    def preprocess_audio(self, audio: Union[np.ndarray, str], **kwargs) -> Any:
        """
        Preprocess audio for model input.

        This is an optional method that can be overridden for custom preprocessing.

        Args:
            audio: Input audio
            **kwargs: Additional preprocessing parameters

        Returns
        -------
            Preprocessed audio in model-specific format
        """
        return audio

    def postprocess_output(self, output: Any) -> str:
        """
        Postprocess model output.

        This is an optional method that can be overridden for custom postprocessing.

        Args:
            output: Raw model output

        Returns
        -------
            str: Cleaned and formatted output text
        """
        if isinstance(output, str):
            return output.strip()
        return str(output).strip()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns
        -------
            Dictionary containing model metadata
        """
        return {
            "name": self.model_name,
            "supports_video": self.supports_video,
            "supports_audio": self.supports_audio,
            "config": self.config,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
