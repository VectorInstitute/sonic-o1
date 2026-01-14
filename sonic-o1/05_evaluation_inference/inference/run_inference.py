"""
inference/run_inference.py
Main inference pipeline for model evaluation with resume capability.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import yaml
from tqdm import tqdm
import os
import shutil
import re
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

from prompts.t1_prompts import get_t1_prompt, get_t1_empathy_prompt
from prompts.t2_prompts import get_t2_prompt
from prompts.t3_prompts import get_t3_prompt
from utils.frame_sampler import FrameSampler
from utils.segmenter import VideoSegmenter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceRunner:
    """Run inference for model evaluation with resume capability"""
    
    def __init__(self, config_path: str, experiment_name: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_path = Path(self.config['dataset_path'])
        self.vqa_path = Path(self.config['vqa_path'])
        
        self.model = None
        self.model_name = None
        self.model_config = None
        self.experiment_name = experiment_name  
        
        self.frame_sampler = None
        self.video_segmenter = VideoSegmenter()
        
        self.retry_config = self.config['retry']
        self.preprocessing_config = self.config['preprocessing']
        
        self.video_metadata = {}  
        self.failed_entries = []
    
    def _get_temp_dir(self) -> Path:
        """Get temporary directory for video segments"""
        unique_suffix = f"temp_segments_{os.getpid()}"
        temp_base = os.environ.get('SCRATCH_DIR') or os.environ.get('TMPDIR')
        
        if temp_base:
            temp_dir = Path(temp_base) / unique_suffix
        else:
            temp_dir = Path.home() / 'scratch' / unique_suffix
        
        return temp_dir
    
    def _load_video_metadata(self, topic_name: str) -> Dict:
        """Load metadata_enhanced.json for specific topic"""
        metadata_path = self.dataset_path / 'videos' / topic_name / 'metadata_enhanced.json'
        
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
        
        metadata_dict = {item['video_id']: item for item in metadata_list}
        logger.info(f"Loaded metadata for {len(metadata_dict)} videos")
        return metadata_dict
        
    def get_video_category(self, video_id: str) -> str:
        """Get duration category (short/medium/long) for video"""
        if video_id in self.video_metadata:
            return self.video_metadata[video_id].get('duration_category', 'medium')
        return 'medium'
    
    def load_model(self, model_name: str):
        """Load specified model"""
        model_config = None
        for m in self.config['models']:
            if m['name'] == model_name:
                model_config = m
                break
        
        if model_config is None:
            raise ValueError(f"Model {model_name} not found in config")
        
        self.model_config = model_config
        model_class = model_config['class']
            
        if model_class == 'Gemini':
            from models.gemini import Gemini
            self.model = Gemini(model_name, model_config)
        elif model_class == 'Qwen3Omni':
            from models.qwen3 import Qwen3Omni
            self.model = Qwen3Omni(model_name, model_config)
        elif model_class == 'MiniCPM':
            from models.minicpm import MiniCPM
            self.model = MiniCPM(model_name, model_config)
        elif model_class == 'UniMoe':
            from models.unimoe import UniMoe
            self.model = UniMoe(model_name, model_config)
        elif model_class == 'VITA':
            from models.vita import VITA
            self.model = VITA(model_name, model_config)
        elif model_class == 'VideoLLaMA2':
            from models.videollama import VideoLLaMA2
            self.model = VideoLLaMA2(model_name, model_config)
        elif model_class == 'Phi4':
            from models.phi4 import Phi4
            self.model = Phi4(model_name, model_config)
        elif model_class == 'GPT4o':
            from models.gpt4o import GPT4o
            self.model = GPT4o(model_name, model_config)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
            
        self.model.load()
        self.model_name = model_name
        
        #if not model_config.get('supports_video', True):
        #    self.frame_sampler = FrameSampler()
        
        logger.info(f"Loaded model: {model_name}")
    
    def get_video_path(self, topic_name: str, video_number: str) -> Path:
        """Build path to video file"""
        video_path = self.dataset_path / 'videos' / topic_name / f'video_{video_number}.mp4'
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return video_path
    
    def get_audio_path(self, topic_name: str, video_number: str) -> Optional[Path]:
        """Build path to audio file"""
        audio_path = self.dataset_path / 'audios' / topic_name / f'audio_{video_number}.m4a'
        if audio_path.exists():
            return audio_path
        return None
    
    def load_ground_truth(self, task: str, topic_name: str) -> Dict:
        """Load ground truth JSON for task and topic"""
        gt_path = self.vqa_path / task / f'{topic_name}.json'
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")
        
        with open(gt_path, 'r') as f:
            return json.load(f)
    
    def get_prediction_path(self, task: str, topic_name: str) -> Path:
        """Get path to prediction file"""
        if self.experiment_name:
            output_dir = Path('results/predictions') / self.experiment_name / self.model_name / task
        else:
            output_dir = Path('results/predictions') / self.model_name / task
        
        return output_dir / f'{topic_name}.json'
    
    def load_existing_predictions(self, task: str, topic_name: str) -> Optional[Dict]:
        """Load existing predictions if they exist"""
        pred_path = self.get_prediction_path(task, topic_name)
        if pred_path.exists():
            try:
                with open(pred_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing predictions: {e}")
        return None
    
    def validate_output(self, output: Dict, task: str) -> Tuple[bool, str]:
        """Validate model output format"""
        try:
            if task == 'task1_summarization':
                required = ['summary_detailed', 'summary_short', 'timeline', 'glossary', 'confidence']
                for field in required:
                    if field not in output:
                        return False, f"Missing field: {field}"
                
                if not isinstance(output['summary_short'], list):
                    return False, "summary_short must be a list"
                if not isinstance(output['timeline'], list):
                    return False, "timeline must be a list"
                if not isinstance(output['glossary'], list):
                    return False, "glossary must be a list"
                    
            elif task == 'task1_empathy':
                required = ['summary_detailed_empathic', 'confidence']
                for field in required:
                    if field not in output:
                        return False, f"Missing field: {field}"
                if not isinstance(output['summary_detailed_empathic'], str):
                    return False, "summary_detailed_empathic must be a string"
            
            elif task == 'task2_mcq':
                required = ['answer_letter', 'answer_index', 'rationale', 'confidence']
                for field in required:
                    if field not in output:
                        return False, f"Missing field: {field}"
                
                if output['answer_letter'] not in ['A', 'B', 'C', 'D', 'E']:
                    return False, f"Invalid answer_letter: {output['answer_letter']}"
                if not (0 <= output['answer_index'] <= 4):
                    return False, f"Invalid answer_index: {output['answer_index']}"
            
            elif task == 'task3_temporal_localization':
                if 'questions' not in output:
                    return False, "Missing field: questions"
                if not isinstance(output['questions'], list):
                    return False, "questions must be a list"
                
                for q in output['questions']:
                    required = ['question_id', 'start_s', 'end_s', 'confidence', 'rationale_model']
                    for field in required:
                        if field not in q:
                            return False, f"Missing field in question: {field}"
            
            return True, "Valid"
        
        except Exception as e:
            return False, f"Validation error: {e}"
        
    def _clean_json_response(self, response: str) -> str:
        """Clean markdown code blocks and common JSON errors from response"""
        response = response.strip()
        
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines).strip()
        
        response = response.replace('```json', '').replace('```', '')
        response = re.sub(r':\s*"?(\d+\.?\d*)s"?([,\s\n}])', r': \1\2', response)
        response = re.sub(r',(\s*[}\]])', r'\1', response)
        response = re.sub(r'"\s*\n\s*"', '",\n"', response)
        
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            response = response[start:end+1]
        
        return response
        
    def _generate_with_retry(
            self,
            video_path: Path,
            audio_path: Optional[Path],
            prompt: str,
            video_category: str,
            task_type: str = 't1',
            topic_name: Optional[str] = None,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """Generate response with retry logic, JSON parsing, and validation"""
            if topic_name:
                self.current_topic_name = topic_name

            # Apply model capability filters
            supports_video = self.model_config.get('supports_video', True)
            supports_audio = self.model_config.get('supports_audio', True)
            
            actual_video_path = video_path if supports_video else None
            actual_audio_path = audio_path if supports_audio else None
            
            max_attempts = self.retry_config['max_attempts']
            
            # Determine retry strategy from config
            # Models specify their retry strategy type in config
            retry_strategy = self.model_config.get('retry_strategy', 'auto')
            
            # Auto-detect strategy if not explicitly set
            if retry_strategy == 'auto':
                is_api_model = 'api_key_env' in self.model_config
                # API models default to 'fps', local models default to 'frame_count'
                retry_strategy = 'fps' if is_api_model else 'frame_count'
            
            # Get fallback options based on strategy
            if retry_strategy == 'frame_count_caption':
                # Frame count + caption chunks (e.g., GPT-4o)
                if 'retry_override' in self.model_config:
                    frame_options = self.model_config['retry_override'].get('frame_count_fallback', self.retry_config['frame_count_fallback'])
                    caption_options = self.model_config['retry_override'].get('caption_chunks_fallback', [None, None, None, None])
                else:
                    frame_options = self.retry_config['frame_count_fallback']
                    caption_options = [None, None, None, None]
                    
            elif retry_strategy == 'frame_count':
                # Frame count + audio chunks (e.g., local models)
                if 'retry_override' in self.model_config:
                    frame_options = self.model_config['retry_override'].get('frame_count_fallback', self.retry_config['frame_count_fallback'])
                    audio_options = self.model_config['retry_override'].get('audio_chunks_fallback', self.retry_config['audio_chunks_fallback'])
                else:
                    frame_options = self.retry_config['frame_count_fallback']
                    audio_options = self.retry_config.get('audio_chunks_fallback', [None, None, None, None])
                audio_chunk_duration = self.retry_config.get('audio_chunk_duration_sec', 10.0)
                
            elif retry_strategy == 'fps':
                # FPS-based (e.g., Gemini)
                fps_options = self.retry_config['fps_fallback']
            
            else:
                raise ValueError(f"Unknown retry_strategy: {retry_strategy}")
            
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    if retry_strategy == 'frame_count_caption':
                        # Frame count + caption chunks
                        max_frames = frame_options[min(attempt, len(frame_options) - 1)]
                        max_caption_chunks = caption_options[min(attempt, len(caption_options) - 1)]
                        
                        # Discover caption path for GPT-4o (needs topic context)
                        caption_path = None
                        if self.model_config.get('use_captions', False) and actual_video_path:
                            # topic_name needs to be passed to this method - see Step 3
                            caption_path = self._get_caption_path_for_video(actual_video_path, self.current_topic_name)
                        
                        preprocessing = {
                            'max_frames': max_frames,
                            'max_caption_chunks': max_caption_chunks,
                            'attempt': attempt + 1,
                            'method': 'frame_caption_sampling',
                            'video_category': video_category
                        }
                        
                        logger.info(f"Attempt {attempt + 1}/{max_attempts}: max_frames={max_frames}, max_caption_chunks={max_caption_chunks}")
                        
                        start_time = time.time()
                        response = self.model.generate(
                            frames=str(actual_video_path) if actual_video_path else None,
                            audio=str(actual_audio_path) if actual_audio_path else None,
                            prompt=prompt,
                            max_frames=max_frames,
                            max_caption_chunks=max_caption_chunks,
                            caption_path=str(caption_path) if caption_path else None,
                            video_category=video_category
                        )
                        
                    elif retry_strategy == 'frame_count':
                        # Frame count + audio chunks
                        max_frames = frame_options[min(attempt, len(frame_options) - 1)]
                        max_audio_chunks = audio_options[min(attempt, len(audio_options) - 1)]
                        
                        preprocessing = {
                            'max_frames': max_frames,
                            'attempt': attempt + 1,
                            'method': 'internal_sampling',
                            'video_category': video_category
                        }
                        
                        start_time = time.time()
                        response = self.model.generate(
                            frames=str(actual_video_path) if actual_video_path else None,
                            audio=str(actual_audio_path) if actual_audio_path else None,
                            prompt=prompt,
                            max_frames=max_frames,
                            max_audio_chunks=max_audio_chunks,
                            audio_chunk_duration_sec=audio_chunk_duration, 
                            video_category=video_category
                        )
                        
                    else:  # retry_strategy == 'fps'
                        # FPS-based
                        fps = fps_options[min(attempt, len(fps_options) - 1)]
                        
                        preprocessing = {
                            'fps_used': fps,
                            'attempt': attempt + 1,
                            'method': 'api_fps_sampling',
                            'video_category': video_category
                        }
                        
                        start_time = time.time()
                        response = self.model.generate(
                            frames=str(actual_video_path) if actual_video_path else None,
                            audio=str(actual_audio_path) if actual_audio_path else None,
                            prompt=prompt,
                            fps=fps,
                            video_category=video_category
                        )
                    
                    preprocessing['inference_time'] = time.time() - start_time
                    response_clean = self._clean_json_response(response)
                    
                    try:
                        output = json.loads(response_clean)
                    except json.JSONDecodeError as e:
                        try:
                            from json_repair import repair_json
                            repaired_json = repair_json(response_clean, return_objects=False)
                            output = json.loads(repaired_json)
                        except Exception:
                            raise ValueError(f"JSON parsing failed: {e}")
                    
                    is_valid, validation_msg = self.validate_output(output, task_type)
                    if not is_valid:
                        raise ValueError(f"Validation failed: {validation_msg}")
                    
                    return output, preprocessing
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {last_error}")
                    
                    if attempt < max_attempts - 1:
                        time.sleep(2)
                        continue
            
            raise RuntimeError(f"All {max_attempts} attempts failed. Last error: {last_error}")
                
    def run_task1(
        self,
        topic_name: str,
        enable_empathy: bool = False,
        dry_run: bool = False,
        overwrite: bool = False,
        retry_failed: bool = False
    ) -> List[Dict]:
        """Run Task 1: Video Summarization"""
        logger.info(f"Running Task 1 (Summarization) for {topic_name}")
        self.video_metadata = self._load_video_metadata(topic_name)
        ground_truth = self.load_ground_truth('task1_summarization', topic_name)
        
        existing_predictions = None
        if not overwrite:
            existing_predictions = self.load_existing_predictions('task1_summarization', topic_name)
        
        # Initialize predictions list with None for each GT entry
        num_gt_entries = len(ground_truth['entries'])
        predictions = [None] * num_gt_entries
        
        # Load existing predictions by matching to GT index
        if existing_predictions and not overwrite:
            existing_entries = existing_predictions.get('entries', [])
            
            for pred in existing_entries:
                pred_key = (pred.get('video_id'), pred.get('video_number'))
                
                for gt_idx, gt_entry in enumerate(ground_truth['entries']):
                    if predictions[gt_idx] is not None:
                        continue
                    
                    gt_key = (gt_entry['video_id'], gt_entry['video_number'])
                    
                    if pred_key == gt_key:
                        if 'error' not in pred:
                            predictions[gt_idx] = pred
                        break
        
        num_done = sum(1 for p in predictions if p is not None)
        logger.info(f"Found {num_done}/{num_gt_entries} already processed videos")
        
        indices_to_process = [i for i, p in enumerate(predictions) if p is None]
        
        if not indices_to_process:
            logger.info("No entries to process")
            return [p for p in predictions if p is not None]
        
        logger.info(f"Processing {len(indices_to_process)} videos")
        
        if dry_run:
            logger.info("DRY RUN - No actual inference will be performed")
            return [p for p in predictions if p is not None]
        
        for gt_idx in tqdm(indices_to_process, desc=f"Task 1 - {topic_name}"):
            entry = ground_truth['entries'][gt_idx]
            video_id = entry['video_id']
            video_number = entry['video_number']
            duration = entry['duration_seconds']
            video_category = self.get_video_category(video_id)
            
            converted_audio_path = None
            
            try:
                video_path = self.get_video_path(topic_name, video_number)
                audio_path = self.get_audio_path(topic_name, video_number)
                
                if audio_path and self._get_audio_format() == 'wav':
                    wav_path = audio_path.with_suffix('.wav')
                    converted_audio_path = self.video_segmenter.convert_audio_format(
                        audio_path, wav_path, 'wav'
                    )
                    audio_path = converted_audio_path
                                
                prompt = get_t1_prompt(duration)
                output, preprocessing = self._generate_with_retry(
                    video_path, audio_path, prompt, video_category, 'task1_summarization', topic_name
                )
                
                prediction = {
                    'video_id': video_id,
                    'video_number': video_number,
                    'duration_seconds': duration,
                    'preprocessing': preprocessing,
                    'outputs': output
                }
                
                if enable_empathy:
                    try:
                        empathy_prompt = get_t1_empathy_prompt(duration)
                        empathy_output, _ = self._generate_with_retry(
                            video_path, audio_path, empathy_prompt, video_category, 'task1_empathy',topic_name
                        )
                        prediction['empathy'] = empathy_output
                    except Exception as e:
                        prediction['empathy_error'] = str(e)
                
                predictions[gt_idx] = prediction
                
            except Exception as e:
                logger.error(f"Error processing video {video_number}: {e}")
                predictions[gt_idx] = {
                    'video_id': video_id,
                    'video_number': video_number,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.failed_entries.append({
                    'task': 'task1_summarization',
                    'topic': topic_name,
                    'video_id': video_id,
                    'error': str(e)
                })
            
            finally:
                if converted_audio_path and converted_audio_path.exists():
                    converted_audio_path.unlink()
        
        return [p for p in predictions if p is not None]


    def run_task2(
        self,
        topic_name: str,
        dry_run: bool = False,
        overwrite: bool = False,
        retry_failed: bool = False
    ) -> List[Dict]:
        """Run Task 2: Question Answering"""
        logger.info(f"Running Task 2 (MCQ) for {topic_name}")
        self.video_metadata = self._load_video_metadata(topic_name)
        ground_truth = self.load_ground_truth('task2_mcq', topic_name)
        
        existing_predictions = None
        if not overwrite:
            existing_predictions = self.load_existing_predictions('task2_mcq', topic_name)
        
        # Initialize predictions list with None for each GT entry
        num_gt_entries = len(ground_truth['entries'])
        predictions = [None] * num_gt_entries
        
        # Load existing predictions by matching to GT index
        if existing_predictions and not overwrite:
            existing_entries = existing_predictions.get('entries', [])
            
            for pred in existing_entries:
                pred_seg = pred.get('segment', {})
                pred_key = (
                    pred.get('video_id'),
                    pred.get('video_number'),
                    pred_seg.get('start'),
                    pred_seg.get('end')
                )
                
                for gt_idx, gt_entry in enumerate(ground_truth['entries']):
                    if predictions[gt_idx] is not None:
                        continue
                    
                    gt_key = (
                        gt_entry['video_id'],
                        gt_entry['video_number'],
                        gt_entry['segment']['start'],
                        gt_entry['segment']['end']
                    )
                    
                    if pred_key == gt_key:
                        if 'error' not in pred:
                            predictions[gt_idx] = pred
                        break
        
        num_done = sum(1 for p in predictions if p is not None)
        logger.info(f"Found {num_done}/{num_gt_entries} already processed segments")
        
        indices_to_process = [i for i, p in enumerate(predictions) if p is None]
        
        if not indices_to_process:
            logger.info("No entries to process")
            return [p for p in predictions if p is not None]
        
        logger.info(f"Processing {len(indices_to_process)} segments")
        
        if dry_run:
            logger.info("DRY RUN - No actual inference will be performed")
            return [p for p in predictions if p is not None]
        
        segment_dir = self._get_temp_dir()
        segment_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for gt_idx in tqdm(indices_to_process, desc=f"Task 2 - {topic_name}"):
                entry = ground_truth['entries'][gt_idx]
                video_id = entry['video_id']
                video_number = entry['video_number']
                segment = entry['segment']
                question = entry['question']
                options = entry['options']
                video_category = self.get_video_category(video_id)
                
                video_segment_path = None
                audio_segment_path = None
                
                try:
                    video_path = self.get_video_path(topic_name, video_number)
                    audio_path = self.get_audio_path(topic_name, video_number)
                    
                    video_segment_path = segment_dir / f'seg_{video_number}_{segment["start"]}_{segment["end"]}_{gt_idx}.mp4'
                    self.video_segmenter.extract_video_segment(
                        video_path, segment['start'], segment['end'], video_segment_path
                    )
                    
                    if audio_path:
                        audio_format = self._get_audio_format()
                        audio_segment_path = segment_dir / f'seg_{video_number}_{segment["start"]}_{segment["end"]}_{gt_idx}.{audio_format}'
                        self.video_segmenter.extract_audio_segment(
                            audio_path, segment['start'], segment['end'], 
                            audio_segment_path, output_format=audio_format
                        )
                    
                    prompt = get_t2_prompt(question, options)
                    output, preprocessing = self._generate_with_retry(
                        video_segment_path, audio_segment_path, prompt, video_category, 'task2_mcq', topic_name
                    )
                                        
                    predictions[gt_idx] = {
                        'video_id': video_id,
                        'video_number': video_number,
                        'segment': segment,
                        'question': question,
                        'options': options,
                        'preprocessing': preprocessing,
                        'outputs': output
                    }
        
                except Exception as e:
                    logger.error(f"Error processing segment: {e}")
                    predictions[gt_idx] = {
                        'video_id': video_id,
                        'video_number': video_number,
                        'segment': segment,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.failed_entries.append({
                        'task': 'task2_mcq',
                        'topic': topic_name,
                        'video_id': video_id,
                        'segment': segment,
                        'error': str(e)
                    })
                
                finally:
                    if video_segment_path and video_segment_path.exists():
                        video_segment_path.unlink()
                    if audio_segment_path and audio_segment_path.exists():
                        audio_segment_path.unlink()
                    
                    if video_segment_path:
                        converted_dir = video_segment_path.parent / "converted"
                        potential_converted = converted_dir / f"{video_segment_path.stem}_h264{video_segment_path.suffix}"
                        if potential_converted.exists():
                            potential_converted.unlink()
                        
            return [p for p in predictions if p is not None]
        
        finally:
            shutil.rmtree(segment_dir, ignore_errors=True)
    
    def run_task3(
        self,
        topic_name: str,
        dry_run: bool = False,
        overwrite: bool = False,
        retry_failed: bool = False
    ) -> List[Dict]:
        """Run Task 3: Temporal Localization"""
        logger.info(f"Running Task 3 (Temporal) for {topic_name}")
        self.video_metadata = self._load_video_metadata(topic_name)
        ground_truth = self.load_ground_truth('task3_temporal_localization', topic_name)
        
        existing_predictions = None
        if not overwrite:
            existing_predictions = self.load_existing_predictions('task3_temporal_localization', topic_name)
        
        # Initialize predictions list with None for each GT entry
        num_gt_entries = len(ground_truth['entries'])
        predictions = [None] * num_gt_entries
        
        # Load existing predictions by matching to GT index
        if existing_predictions and not overwrite:
            existing_entries = existing_predictions.get('entries', [])
            
            # Match each existing prediction to its GT index
            for pred in existing_entries:
                pred_seg = pred.get('segment', {})
                pred_key = (
                    pred.get('video_id'),
                    pred.get('video_number'),
                    pred_seg.get('start'),
                    pred_seg.get('end')
                )
                
                # Find matching GT index (first unmatched one with same key)
                for gt_idx, gt_entry in enumerate(ground_truth['entries']):
                    if predictions[gt_idx] is not None:
                        continue  # Already filled
                    
                    gt_key = (
                        gt_entry['video_id'],
                        gt_entry['video_number'],
                        gt_entry['segment']['start'],
                        gt_entry['segment']['end']
                    )
                    
                    if pred_key == gt_key:
                        # Only keep if successful
                        if 'error' not in pred:
                            predictions[gt_idx] = pred
                        break
        
        # Count how many are done
        num_done = sum(1 for p in predictions if p is not None)
        logger.info(f"Found {num_done}/{num_gt_entries} already processed segments")
        
        # Find indices to process
        indices_to_process = [i for i, p in enumerate(predictions) if p is None]
        
        if not indices_to_process:
            logger.info("No entries to process")
            # Convert to list (remove None placeholders - shouldn't be any)
            return [p for p in predictions if p is not None]
        
        logger.info(f"Processing {len(indices_to_process)} segments")
        
        if dry_run:
            logger.info("DRY RUN - No actual inference will be performed")
            return [p for p in predictions if p is not None]
        
        segment_dir = self._get_temp_dir()
        segment_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for gt_idx in tqdm(indices_to_process, desc=f"Task 3 - {topic_name}"):
                entry = ground_truth['entries'][gt_idx]
                video_id = entry['video_id']
                video_number = entry['video_number']
                segment = entry['segment']
                questions = entry['questions']
                video_category = self.get_video_category(video_id)
                
                video_segment_path = None
                audio_segment_path = None
                
                try:
                    video_path = self.get_video_path(topic_name, video_number)
                    audio_path = self.get_audio_path(topic_name, video_number)
                    
                    video_segment_path = segment_dir / f'seg_{video_number}_{segment["start"]}_{segment["end"]}_{gt_idx}.mp4'
                    self.video_segmenter.extract_video_segment(
                        video_path, segment['start'], segment['end'], video_segment_path
                    )
                    
                    if audio_path:
                        audio_format = self._get_audio_format()
                        audio_segment_path = segment_dir / f'seg_{video_number}_{segment["start"]}_{segment["end"]}_{gt_idx}.{audio_format}'
                        self.video_segmenter.extract_audio_segment(
                            audio_path, segment['start'], segment['end'], 
                            audio_segment_path, output_format=audio_format
                        )
                    
                    prompt = get_t3_prompt(questions, segment['start'], segment['end'])
                    output, preprocessing = self._generate_with_retry(
                        video_segment_path, audio_segment_path, prompt, 
                        video_category, 'task3_temporal_localization', topic_name
                    )
                    
                    predictions[gt_idx] = {
                        'video_id': video_id,
                        'video_number': video_number,
                        'segment': segment,
                        'preprocessing': preprocessing,
                        'outputs': output
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing segment: {e}")
                    predictions[gt_idx] = {
                        'video_id': video_id,
                        'video_number': video_number,
                        'segment': segment,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.failed_entries.append({
                        'task': 'task3_temporal_localization',
                        'topic': topic_name,
                        'video_id': video_id,
                        'segment': segment,
                        'error': str(e)
                    })
                            
                finally:
                    if video_segment_path and video_segment_path.exists():
                        video_segment_path.unlink()
                    if audio_segment_path and audio_segment_path.exists():
                        audio_segment_path.unlink()
                    
                    if video_segment_path:
                        converted_dir = video_segment_path.parent / "converted"
                        potential_converted = converted_dir / f"{video_segment_path.stem}_h264{video_segment_path.suffix}"
                        if potential_converted.exists():
                            potential_converted.unlink()
                        
            # Return only non-None entries (all should be filled now)
            return [p for p in predictions if p is not None]
        
        finally:
            shutil.rmtree(segment_dir, ignore_errors=True)
    
    def save_predictions(self, task: str, topic_name: str, predictions: List[Dict]):
        """Save predictions to JSON"""
        if self.experiment_name:
            output_dir = Path('results/predictions') / self.experiment_name / self.model_name / task
        else:
            output_dir = Path('results/predictions') / self.model_name / task
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{topic_name}.json'
        
        successful = len([p for p in predictions if 'error' not in p])
        failed = len([p for p in predictions if 'error' in p])
        
        output_data = {
            'model': self.model_name,
            'topic_name': topic_name,
            'task': task,
            'generated_at': datetime.now().isoformat(),
            'num_entries': len(predictions),
            'num_successful': successful,
            'num_failed': failed,
            'entries': predictions
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved predictions to {output_file} ({successful} successful, {failed} failed)")
    

    def _deduplicate_predictions(self, predictions: List[Dict], task: str) -> List[Dict]:
        """Remove failed entries if successful entry exists for same key"""
        
        def get_key(entry: Dict) -> Tuple:
            if task == 'task1_summarization':
                return (entry.get('video_id'), entry.get('video_number'))
            else:
                seg = entry.get('segment', {})
                return (
                    entry.get('video_id'),
                    entry.get('video_number'),
                    seg.get('start'),
                    seg.get('end')
                )
        
        # First pass: collect all successful keys
        successful_keys = set()
        for p in predictions:
            if 'error' not in p:
                successful_keys.add(get_key(p))
        
        # Second pass: keep only if successful OR no successful version exists
        result = []
        seen_keys = set()
        
        for p in predictions:
            key = get_key(p)
            
            if key in seen_keys:
                continue  # Already added this key
            
            if 'error' in p and key in successful_keys:
                continue  # Skip failed, successful exists
            
            result.append(p)
            seen_keys.add(key)
        
        removed = len(predictions) - len(result)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate/superseded entries")
        
        return result

    def save_failed_log(self):
        """Save log of failed entries"""
        if not self.failed_entries:
            return
        
        if self.experiment_name:
            failed_log_dir = Path('results/failed_logs') / self.experiment_name / self.model_name
        else:
            failed_log_dir = Path('results/failed_logs') / self.model_name
        
        failed_log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        failed_log_file = failed_log_dir / f'failed_{timestamp}.json'
        
        with open(failed_log_file, 'w') as f:
            json.dump(self.failed_entries, f, indent=2)
        
        logger.info(f"Saved failed entries log to {failed_log_file}")
    
    def run(
        self,
        model_name: str,
        tasks: List[str],
        topics: List[str],
        enable_empathy: bool = None,
        dry_run: bool = False,
        overwrite: bool = False,
        retry_failed: bool = False
    ):
        """Run inference for specified model, tasks, and topics"""
        self.model_name = model_name
        if enable_empathy is None:
            enable_empathy = self.config.get('empathy', {}).get('enabled', False)
        
        if not dry_run:
            self.load_model(model_name)
        
        for task in tasks:
            for topic in topics:
                logger.info(f"Processing {task} - {topic}")
                
                try:
                    if task == 'task1_summarization':
                        predictions = self.run_task1(topic, enable_empathy, dry_run, overwrite, retry_failed)
                    elif task == 'task2_mcq':
                        predictions = self.run_task2(topic, dry_run, overwrite, retry_failed)
                    elif task == 'task3_temporal_localization':
                        predictions = self.run_task3(topic, dry_run, overwrite, retry_failed)
                    else:
                        logger.error(f"Unknown task: {task}")
                        continue
                    
                    if not dry_run:
                        self.save_predictions(task, topic, predictions)
                    
                except Exception as e:
                    logger.error(f"Error processing {task} - {topic}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not dry_run:
            if self.frame_sampler:
                self.frame_sampler.cleanup()
            
            if self.failed_entries:
                self.save_failed_log()
                logger.warning(f"Total failed entries: {len(self.failed_entries)}")
            
            self.model.unload()
        
        logger.info("Inference complete")
    
    def _get_audio_format(self) -> str:
        """Get required audio format from model config"""
        return self.model_config.get('audio_format', 'm4a')
    
    def _get_caption_path_for_video(self, video_path: Path, topic_name: str) -> Optional[Path]:
        """
        Get caption path for a video (full or segment).
        Extracts video number from either pattern and builds caption path.
        
        Args:
            video_path: Path to video file (full video or segment)
            topic_name: Topic name for building caption path
            
        Returns:
            Path to caption file if it exists, None otherwise
        """
        import re
        
        video_name = video_path.stem
        
        # Try pattern 1: "video_001"
        match = re.search(r'video_(\d+)', video_name)
        
        # Try pattern 2: "seg_001_30_60_5"
        if not match:
            match = re.search(r'seg_(\d+)_', video_name)
        
        if match:
            video_number = match.group(1)
            caption_path = self.dataset_path / 'captions' / topic_name / f'caption_{video_number}.srt'
            
            if caption_path.exists():
                return caption_path
            else:
                logger.debug(f"Caption file not found: {caption_path}")
        
        return None
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run model inference')
    parser.add_argument('--config', type=str, default='configs/models_config.yaml')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tasks', nargs='+', default=None)
    parser.add_argument('--topics', nargs='+', default=None)
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--empathy', action='store_true', default=None)
    parser.add_argument('--no-empathy', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    
    args = parser.parse_args()
    
    runner = InferenceRunner(args.config, experiment_name=args.experiment_name)
    
    tasks = args.tasks if args.tasks else runner.config['tasks']
    topics = args.topics if args.topics else runner.config['topics']
    
    if args.no_empathy:
        enable_empathy = False
    elif args.empathy:
        enable_empathy = True
    else:
        enable_empathy = None
    
    runner.run(
        args.model,
        tasks,
        topics,
        enable_empathy,
        args.dry_run,
        args.overwrite,
        args.retry_failed
    )