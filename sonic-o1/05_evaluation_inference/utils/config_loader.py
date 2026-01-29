"""
Configuration Loader Utility

Loads and provides access to configuration from YAML files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML."""

    def __init__(self, config_path: str = "models_config.yaml"):
        """
        Initialize config loader.

        Args:
            config_path: Path to models configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "dataset_path": "../dataset",
            "vqa_path": "../vqa",
            "tasks": ["t1_summarization", "t2_mcq", "t3_temporal"],
            "topics": [
                "01_Patient-Doctor_Consultations",
                "02_Job_Interviews",
                "03_News_Broadcasts",
                "04_Educational_Lectures",
                "05_Sports_Commentary",
                "06_Cooking_Shows",
                "07_Political_Debates",
                "08_Product_Reviews",
                "09_Travel_Vlogs",
                "10_Gaming_Streams",
                "11_Music_Performances",
                "12_Tech_Tutorials",
                "13_Documentary_Excerpts",
            ],
            "preprocessing": {
                "t1": {
                    "short_mid_fps": 1,
                    "long_default_fps": 1,
                    "long_fallback_fps": 0.5,
                },
                "t2_t3": {
                    "segment_max_duration": 180,
                    "image_model_frames": 128,
                    "video_model_fps": 1,
                },
            },
            "retry": {
                "max_attempts": 3,
                "fps_fallback": [1, 0.5, 0.25],
                "frame_count_fallback": [128, 64, 32, 16],
            },
            "metrics": {
                "llm_judge_model": "gpt-5-mini",
                "iou_thresholds": [0.3, 0.5, 0.7],
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g. 'metrics.llm_judge_model')
            default: Default value if key not found

        Returns
        -------
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_dataset_path(self) -> str:
        """Get dataset path."""
        return self.get("dataset_path", "../dataset")

    def get_vqa_path(self) -> str:
        """Get VQA ground truth path."""
        return self.get("vqa_path", "../vqa")

    def get_topics(self) -> list:
        """Get list of topics."""
        return self.get("topics", self._get_default_config()["topics"])

    def get_tasks(self) -> list:
        """Get list of tasks."""
        return self.get("tasks", ["t1_summarization", "t2_mcq", "t3_temporal"])

    def get_llm_judge_model(self) -> str:
        """Get LLM judge model name."""
        return self.get("metrics.llm_judge_model", "gpt-5-mini")

    def get_iou_thresholds(self) -> list:
        """Get IoU thresholds for T3 evaluation."""
        return self.get("metrics.iou_thresholds", [0.3, 0.5, 0.7])

    def get_models(self) -> list:
        """Get list of configured models."""
        models = self.get("models", [])
        return [m["name"] for m in models if isinstance(m, dict) and "name" in m]


# Global config instance
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global config instance (singleton pattern).

    Args:
        config_path: Path to config file (only used for first call)

    Returns
    -------
        ConfigLoader instance
    """
    global _config_instance

    if _config_instance is None:
        if config_path is None:
            config_path = "models_config.yaml"
        _config_instance = ConfigLoader(config_path)

    return _config_instance


def reset_config():
    """Reset global config instance (useful for testing)."""
    global _config_instance
    _config_instance = None
