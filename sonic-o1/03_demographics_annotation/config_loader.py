"""
Configuration loader for YAML-based config with .env support
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# Load .env file from the same directory as this script
load_dotenv(Path(__file__).parent / '.env')

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and validate configuration from YAML file with environment variable support"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = Path(__file__).parent / config_path
        self.config = self.load_config()
        self._resolve_environment_variables()
        self._resolve_paths()
    
    def load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _resolve_environment_variables(self):
        """Resolve environment variables in config"""
        # Handle API key - check multiple sources in order of priority
        if 'model' in self.config:
            # Priority 1: Environment variable from .env or system
            api_key_from_env = os.getenv('GEMINI_API_KEY')
            
            # Priority 2: Config file (if not using ${} syntax)
            api_key_from_config = self.config['model'].get('api_key', '')
            
            # If config uses ${VAR} syntax, replace with env var
            if api_key_from_config.startswith('${') and api_key_from_config.endswith('}'):
                env_var = api_key_from_config[2:-1]
                self.config['model']['api_key'] = os.getenv(env_var, '')
            elif api_key_from_env:
                # Use environment variable if available
                self.config['model']['api_key'] = api_key_from_env
            elif api_key_from_config and not api_key_from_config.startswith('${'):
                # Use config value if it's not a variable reference
                self.config['model']['api_key'] = api_key_from_config
            else:
                # No API key found
                self.config['model']['api_key'] = ''
                logger.warning("No API key found in environment or config")
    
    def _resolve_paths(self):
        """Resolve relative paths to absolute"""
        if 'dataset' in self.config and 'base_path' in self.config['dataset']:
            base_path = Path(self.config['dataset']['base_path'])
            if not base_path.is_absolute():
                # Make relative to config file location
                base_path = (self.config_path.parent / base_path).resolve()
            self.config['dataset']['base_path'] = str(base_path)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return self.config.get('dataset', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.config.get('processing', {})
    
    def get_demographics_config(self) -> Dict[str, Any]:
        """Get demographics configuration"""
        return self.config.get('demographics', {})

@dataclass
class Config:
    """Unified configuration class with .env support"""
    
    def __init__(self, config_path: str = "config.yaml"):
        loader = ConfigLoader(config_path)
        # Store the raw config from loader
        self.raw_config = loader.config
        
        # Model settings
        model_cfg = loader.get_model_config()
        self.model_name = model_cfg.get('name', 'gemini-2.5-flash')
        self.api_key = model_cfg.get('api_key', '')
        
        # Log API key status (not the key itself)
        if self.api_key:
            logger.info(f"API key loaded (length: {len(self.api_key)})")
        else:
            logger.warning("No API key loaded")
        
        self.temperature = model_cfg.get('temperature', 0.3)
        self.max_output_tokens = model_cfg.get('max_output_tokens', 1024)
        self.timeout = model_cfg.get('timeout', 60)
        self.retry_attempts = model_cfg.get('retry_attempts', 3)
        self.retry_delay = model_cfg.get('retry_delay', 5)
        
        # Dataset settings
        dataset_cfg = loader.get_dataset_config()
        self.base_path = Path(dataset_cfg.get('base_path', '../dataset'))
        self.topics = dataset_cfg.get('topics', [])
        
        # Demographics settings
        demo_cfg = loader.get_demographics_config()
        self.races = demo_cfg.get('races', [])
        self.genders = demo_cfg.get('genders', [])
        self.age_groups = demo_cfg.get('age_groups', [])
        self.languages = demo_cfg.get('languages', [])
        
        # Processing settings
        proc_cfg = loader.get_processing_config()
        self.batch_size = proc_cfg.get('batch_size', 5)
        self.save_interval = proc_cfg.get('save_interval', 10)
        self.use_cache = proc_cfg.get('use_cache', True)
        self.output_format = proc_cfg.get('output_format', 'metadata_enhanced.json')
        
        # File patterns
        self.video_pattern = proc_cfg.get('video_pattern', 'video_{number}.mp4')
        self.audio_pattern = proc_cfg.get('audio_pattern', 'audio_{number}.m4a')
        self.caption_pattern = proc_cfg.get('caption_pattern', 'caption_{number}.srt')
        
        # Multimodal processing settings
        self.file_processing_timeout = proc_cfg.get('file_processing_timeout', 7200)
        self.max_video_duration = proc_cfg.get('max_video_duration', 3300)
        self.max_transcript_length = proc_cfg.get('max_transcript_length', 50000)
        self.prefer_video_with_audio = proc_cfg.get('prefer_video_with_audio', True)
        
        # Output settings
        self.save_raw_responses = proc_cfg.get('save_raw_responses', True)
        self.create_backup = proc_cfg.get('create_backup', True)
        
        rate_limit_cfg = self.raw_config.get('rate_limit', {})
        self.delay_between_videos = rate_limit_cfg.get('delay_between_videos', 10)
        self.delay_after_long_video = rate_limit_cfg.get('delay_after_long_video', 30)
        self.long_video_threshold = rate_limit_cfg.get('long_video_threshold', 1800)


        # Quality settings
        quality_cfg = self.raw_config.get('quality', {})
        self.min_confidence = quality_cfg.get('min_confidence', 0.5)
        self.require_explanation = quality_cfg.get('require_explanation', True)
        self.validate_json = quality_cfg.get('validate_json', True)
        
        # Logging settings
        log_cfg = self.raw_config.get('logging', {})
        self.log_file = log_cfg.get('log_file', 'demographics_annotation.log')
        self.log_format = log_cfg.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.console_output = log_cfg.get('console_output', True)
        self.file_output = log_cfg.get('file_output', True)
    
    def get_topic_paths(self, topic: str) -> Dict[str, Path]:
        """Get all relevant paths for a topic"""
        return {
            "videos": self.base_path / "videos" / topic,
            "audios": self.base_path / "audios" / topic,
            "captions": self.base_path / "captions" / topic,
            "metadata": self.base_path / "videos" / topic / "metadata.json"
        }
    
    def get_file_path(self, topic: str, file_type: str, number: str) -> Path:
        """Get specific file path based on pattern"""
        paths = self.get_topic_paths(topic)
        
        if file_type == "video":
            return paths["videos"] / self.video_pattern.format(number=number)
        elif file_type == "audio":
            return paths["audios"] / self.audio_pattern.format(number=number)
        elif file_type == "caption":
            return paths["captions"] / self.caption_pattern.format(number=number)
        else:
            raise ValueError(f"Unknown file type: {file_type}")