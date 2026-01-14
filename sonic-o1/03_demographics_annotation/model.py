"""
Model interface for demographics annotation using Gemini
"""
from google import genai
from google.genai import types
import os
import json
import time
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)

class DemographicsAnnotator:
    """Handle Gemini API interactions for demographics annotation"""
    
    def __init__(self, config):
        """
        Initialize the Gemini client with configuration.
        
        Args:
            config: Configuration object with model settings
        """
        self.config = config
        self.file_processing_timeout = getattr(config, 'file_processing_timeout', 7200) 
        self.max_video_duration = getattr(config, 'max_video_duration', 3300) 
        
        self.setup_client()
        
    def setup_client(self):
        """Initialize the Gemini client"""
        os.environ['GEMINI_API_KEY'] = self.config.api_key
        self.client = genai.Client()
        logger.info(f"Initialized Gemini client with model: {self.config.model_name}")
        
    def process_media(self,  
                        video_path: Optional[Path],
                        audio_path: Optional[Path],
                        transcript_path: Optional[Path],
                        metadata: Dict[str, Any],
                        config,
                        _is_segment: bool = False) -> Dict[str, Any]:
        """
        Process media files (video, audio, transcript) for demographics annotation.
        Automatically handles videos longer than the limit by segmenting.
        
        Args:
            video_path: Path to video file (optional)
            audio_path: Path to audio file (optional)
            transcript_path: Path to transcript file (optional)
            metadata: Media metadata dictionary
            config: Configuration object
            _is_segment: Internal flag to prevent re-segmentation of segments
        """
        try:
            duration = metadata.get('duration_seconds', 0)
            
            # Validate input
            if not video_path and not audio_path:
                error_msg = "Must provide either video_path or audio_path"
                logger.error(error_msg)
                return self._get_error_response(error_msg)
            
            # Check if video needs segmentation (but NOT if this is already a segment)
            primary_media = video_path if video_path else audio_path
            is_video = primary_media.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.m4v']
            
            # Only segment if: it's a video, it's too long, AND it's not already a segment
            if is_video and duration > self.max_video_duration and not _is_segment:
                logger.warning(
                    f"Video duration ({duration}s) exceeds limit ({self.max_video_duration}s). "
                    f"Will segment and process in chunks."
                )
                return self._process_long_video_segmented(
                    video_path, audio_path, transcript_path, metadata, config
                )
            
            # Normal processing for videos within limit (or segments)
            logger.info(f"Processing media: {primary_media.name}")
            
            # Load transcript if available
            transcript_text = self._load_transcript(transcript_path, config)
            
            # Prepare the prompt with transcript included
            prompt = self._build_prompt(metadata, transcript_text)
            
            # Collect all media files to process
            media_files = []
            if video_path:
                media_files.append(('video', video_path))
            if audio_path:
                media_files.append(('audio', audio_path))
            
            # Determine processing method based on file sizes
            total_size = sum(os.path.getsize(path) for _, path in media_files)
            use_file_api = total_size > 20 * 1024 * 1024  # 20MB threshold
            
            if use_file_api:
                logger.info(f"Using File API for large media (total size: {total_size / (1024*1024):.2f}MB)")
                response_text = self._process_large_media_multimodal(media_files, prompt)
            else:
                logger.info(f"Using inline processing for small media (total size: {total_size / (1024*1024):.2f}MB)")
                response_text = self._process_small_media_multimodal(media_files, prompt)
            
            # Parse JSON response
            demographics_data = self._parse_response(response_text)
            
            # Add raw response if configured
            if config.save_raw_responses:
                demographics_data['raw_response'] = response_text
            
            return demographics_data
            
        except Exception as e:
            logger.error(f"Error processing media: {e}", exc_info=True)
            return self._get_error_response(str(e))
    
    def _process_long_video_segmented(self,
                                       video_path: Optional[Path],
                                       audio_path: Optional[Path],
                                       transcript_path: Optional[Path],
                                       metadata: Dict[str, Any],
                                       config) -> Dict[str, Any]:
        """
        Process videos longer than the limit by segmenting into chunks.
        
        Strategy:
        1. Split video into overlapping segments (e.g., 50min segments with 5min overlap)
        2. Process each segment
        3. Aggregate results with deduplication
        """
        try:
            import subprocess
            import tempfile
            import shutil
            
            duration = metadata.get('duration_seconds', 0)
            segment_duration = self.max_video_duration - 300  # 50 minutes (leaving 5min buffer)
            overlap = 60  # 1 minute overlap to catch people across boundaries
            
            num_segments = int(duration / segment_duration) + 1
            logger.info(f"Splitting {duration}s video into {num_segments} segments of ~{segment_duration}s each")
            
            temp_dir = Path(tempfile.mkdtemp(prefix="video_segments_"))
            all_demographics = []
            
            try:
                for i in range(num_segments):
                    start_time = max(0, i * segment_duration - (overlap if i > 0 else 0))
                    segment_duration_actual = min(segment_duration + overlap, duration - start_time)
                    
                    logger.info(f"Processing segment {i+1}/{num_segments}: {start_time}s to {start_time + segment_duration_actual}s")
                    
                    # Create segment filename
                    segment_video_path = None
                    if video_path:
                        segment_video_path = temp_dir / f"segment_{i:03d}{video_path.suffix}"
                        
                        # Use ffmpeg to extract segment
                        cmd = [
                            'ffmpeg', '-y',
                            '-ss', str(start_time),
                            '-i', str(video_path),
                            '-t', str(segment_duration_actual),
                            '-c', 'copy',  # Fast: just copy streams without re-encoding
                            '-avoid_negative_ts', '1',
                            str(segment_video_path)
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.error(f"FFmpeg error: {result.stderr}")
                            raise Exception(f"Failed to create video segment {i}")
                    
                    # Create audio segment if separate audio exists
                    segment_audio_path = None
                    if audio_path:
                        segment_audio_path = temp_dir / f"segment_{i:03d}{audio_path.suffix}"
                        
                        cmd = [
                            'ffmpeg', '-y',
                            '-ss', str(start_time),
                            '-i', str(audio_path),
                            '-t', str(segment_duration_actual),
                            '-c', 'copy',
                            str(segment_audio_path)
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.warning(f"Could not create audio segment: {result.stderr}")
                            segment_audio_path = None
                    
                    # Extract relevant transcript section
                    segment_transcript_path = None
                    if transcript_path and transcript_path.exists():
                        segment_transcript = self._extract_transcript_segment(
                            transcript_path, start_time, start_time + segment_duration_actual
                        )
                        if segment_transcript:
                            segment_transcript_path = temp_dir / f"segment_{i:03d}.srt"
                            with open(segment_transcript_path, 'w', encoding='utf-8') as f:
                                f.write(segment_transcript)
                    
                    # Process this segment (with _is_segment=True to prevent re-segmentation)
                    segment_metadata = metadata.copy()
                    segment_metadata['duration_seconds'] = segment_duration_actual
                    segment_metadata['segment_info'] = {
                        'segment_number': i + 1,
                        'total_segments': num_segments,
                        'start_time': start_time,
                        'end_time': start_time + segment_duration_actual
                    }
                    
                    segment_demographics = self.process_media(
                        video_path=segment_video_path,
                        audio_path=segment_audio_path,
                        transcript_path=segment_transcript_path,
                        metadata=segment_metadata,
                        config=config,
                        _is_segment=True  # THIS IS THE KEY FIX - prevents re-segmentation
                    )
                    
                    all_demographics.append(segment_demographics)
                
                # Aggregate results
                aggregated = self._aggregate_segment_demographics(all_demographics, num_segments)
                
                # Add segmentation info
                aggregated['demographics_annotation']['segmented'] = True
                aggregated['demographics_annotation']['num_segments'] = num_segments
                aggregated['demographics_annotation']['original_duration'] = duration
                
                return aggregated
                
            finally:
                # Cleanup temporary files
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary segments")
        
        except Exception as e:
            logger.error(f"Error in segmented processing: {e}", exc_info=True)
            return self._get_error_response(f"Segmentation error: {e}")
    
    def _extract_transcript_segment(self, transcript_path: Path, start_time: float, end_time: float) -> str:
        """Extract portion of SRT transcript for a time segment"""
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple SRT parser
            import re
            segments = content.strip().split('\n\n')
            extracted = []
            
            for segment in segments:
                lines = segment.split('\n')
                if len(lines) < 3:
                    continue
                
                # Parse timestamp line (format: 00:00:10,500 --> 00:00:13,000)
                timestamp_match = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
                if timestamp_match:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, timestamp_match.groups())
                    seg_start = h1*3600 + m1*60 + s1 + ms1/1000
                    seg_end = h2*3600 + m2*60 + s2 + ms2/1000
                    
                    # Check if this segment overlaps with our time range
                    if seg_start < end_time and seg_end > start_time:
                        extracted.append(segment)
            
            return '\n\n'.join(extracted)
        
        except Exception as e:
            logger.warning(f"Could not extract transcript segment: {e}")
            return ""
    
    def _aggregate_segment_demographics(self, segment_results: List[Dict], num_segments: int) -> Dict[str, Any]:
        """
        Aggregate demographics from multiple segments with deduplication.
        Uses voting/confidence averaging to merge results.
        """
        # Collect all demographics and confidences
        all_races = {}
        all_genders = {}
        all_ages = {}
        all_languages = {}
        total_individuals = 0
        explanations = []
        
        for seg_result in segment_results:
            if 'error' in seg_result.get('demographics_annotation', {}):
                continue
            
            # Aggregate confidence scores (use maximum confidence seen)
            for race, conf in seg_result.get('demographics_confidence', {}).get('race', {}).items():
                all_races[race] = max(all_races.get(race, 0), conf)
            
            for gender, conf in seg_result.get('demographics_confidence', {}).get('gender', {}).items():
                all_genders[gender] = max(all_genders.get(gender, 0), conf)
            
            for age, conf in seg_result.get('demographics_confidence', {}).get('age', {}).items():
                all_ages[age] = max(all_ages.get(age, 0), conf)
            
            for lang, conf in seg_result.get('demographics_confidence', {}).get('language', {}).items():
                all_languages[lang] = max(all_languages.get(lang, 0), conf)
            
            # Track max individuals seen in any segment - FIX: ensure it's an integer
            individuals_count = seg_result.get('demographics_annotation', {}).get('individuals_count', 0)
            # Convert to int if it's a string
            if isinstance(individuals_count, str):
                try:
                    individuals_count = int(individuals_count)
                except (ValueError, TypeError):
                    individuals_count = 0
            
            total_individuals = max(total_individuals, individuals_count)
            
            explanations.append(seg_result.get('demographics_annotation', {}).get('explanation', ''))
        
        # Build aggregated result
        return {
            "demographics_detailed": {
                "race": list(all_races.keys()),
                "gender": list(all_genders.keys()),
                "age": list(all_ages.keys()),
                "language": list(all_languages.keys())
            },
            "demographics_confidence": {
                "race": all_races,
                "gender": all_genders,
                "age": all_ages,
                "language": all_languages
            },
            "demographics_annotation": {
                "model": self.config.model_name,
                "annotated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "individuals_count": total_individuals,
                "explanation": f"Aggregated from {num_segments} segments. " + " | ".join(explanations[:3])
            }
        }
    def _load_transcript(self, transcript_path: Optional[Path], config) -> str:
        """Load and truncate transcript if needed"""
        transcript_text = ""
        if transcript_path and transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
                max_length = getattr(config, 'max_transcript_length', 50000)
                if len(transcript_text) > max_length:
                    logger.info(f"Truncating transcript from {len(transcript_text)} to {max_length} chars")
                    transcript_text = transcript_text[:max_length] + "\n...[truncated]"
        return transcript_text
    
    def _process_large_media_multimodal(self, media_files: List[tuple], prompt: str) -> str:
        """
        Process large media files using Gemini File API with multimodal support.
        
        Args:
            media_files: List of tuples (media_type, Path)
            prompt: Analysis prompt
            
        Returns:
            Generated response text
        """
        uploaded_files = []
        try:
            # Upload all media files
            for media_type, media_path in media_files:
                uploaded_file = self.client.files.upload(file=str(media_path))
                logger.info(f"Uploaded {media_type} file: {uploaded_file.name}")
                uploaded_files.append(uploaded_file)
            
            # Wait for all files to process
            max_wait = self.file_processing_timeout 
            wait_time = 0
            
            all_processed = False
            while not all_processed and wait_time < max_wait:
                all_processed = True
                for i, uploaded_file in enumerate(uploaded_files):
                    uploaded_files[i] = self.client.files.get(name=uploaded_file.name)
                    if uploaded_files[i].state == "PROCESSING":
                        all_processed = False
                    elif uploaded_files[i].state == "FAILED":
                        raise Exception(f"File processing failed: {getattr(uploaded_files[i], 'error', 'Unknown error')}")
                
                if not all_processed:
                    time.sleep(10)
                    wait_time += 10
                    logger.debug(f"Waiting for file processing... ({wait_time}s)")
            
            if not all_processed:
                raise Exception(f"File processing timeout after {max_wait} seconds")
            
            # Generate content with all uploaded files + prompt
            for attempt in range(self.config.retry_attempts):
                try:
                    # Build content list: [file1, file2, ..., prompt]
                    content_parts = uploaded_files + [prompt]
                    
                    response = self.client.models.generate_content(
                        model=self.config.model_name,
                        contents=content_parts
                    )
                    return response.text
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay)
                    else:
                        raise
                        
        finally:
            # Clean up all uploaded files
            for uploaded_file in uploaded_files:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.info(f"Deleted uploaded file: {uploaded_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete uploaded file: {e}")
    
    
    def _process_small_media_multimodal(self, media_files: List[tuple], prompt: str) -> str:
        """
        Process small media files using inline data with multimodal support.
        
        Args:
            media_files: List of tuples (media_type, Path)
            prompt: Analysis prompt
            
        Returns:
            Generated response text
        """
        # Build content parts
        parts = []
        
        # Add all media files as inline data
        for media_type, media_path in media_files:
            with open(media_path, 'rb') as media_file:
                media_bytes = media_file.read()
            
            mime_type = self._get_media_mime_type(media_path)
            
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        data=media_bytes,
                        mime_type=mime_type
                    )
                )
            )
            logger.info(f"Added {media_type} ({mime_type}) to inline content")
        
        # Add prompt as text
        parts.append(types.Part(text=prompt))
        
        # Generate content with retries
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=types.Content(parts=parts)
                )
                return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise
    
    def _get_media_mime_type(self, media_path: Path) -> str:
        """Get MIME type for media file"""
        extension_map = {
            # Video types
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.wmv': 'video/x-ms-wmv',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.m4v': 'video/x-m4v',
            
            # Audio types
            '.m4a': 'audio/m4a',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac',
            '.aac': 'audio/aac',
        }
        
        extension = media_path.suffix.lower()
        return extension_map.get(extension, 'application/octet-stream')
    
    def _build_prompt(self, metadata: Dict[str, Any], transcript_text: str) -> str:
        """Build the analysis prompt with transcript embedded"""
        from prompts import MAIN_PROMPT_TEMPLATE 
        
        # Prepare transcript section
        if transcript_text:
            transcript_preview = f"TRANSCRIPT/CAPTIONS:\n{transcript_text}"
        else:
            transcript_preview = "No transcript available. Analyze based on visual and audio content only."
        
        prompt = MAIN_PROMPT_TEMPLATE.format(
            title=metadata.get('title', 'Unknown'),
            duration_seconds=metadata.get('duration_seconds', 0),
            topic_name=metadata.get('topic_name', 'Unknown'),
            transcript_preview=transcript_preview,
            model_name=self.config.model_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate JSON response"""
        try:
            # Handle None response
            if response_text is None:
                logger.error("Received None response from API")
                return self._get_error_response("API returned no response")
            
            # Clean response text
            response_text = response_text.strip()
            
            # Check for empty response
            if not response_text:
                logger.error("Received empty response from API")
                return self._get_error_response("API returned empty response")
            
            # Remove markdown code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.rfind("```")
                if end > start:
                    response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.rfind("```")
                if end > start:
                    response_text = response_text[start:end]
            
            # Parse JSON
            data = json.loads(response_text.strip())
            
            # Validate required fields
            required_fields = ["demographics_detailed", "demographics_confidence", "demographics_annotation"]
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field: {field}")
                    # Add default structure if missing
                    if field == "demographics_detailed":
                        data[field] = {"race": [], "gender": [], "age": [], "language": []}
                    elif field == "demographics_confidence":
                        data[field] = {"race": {}, "gender": {}, "age": {}, "language": {}}
                    elif field == "demographics_annotation":
                        data[field] = {
                            "model": self.config.model_name,
                            "annotated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "individuals_count": 0,
                            "explanation": "Partial response"
                        }
            
            # Filter by minimum confidence if configured
            if hasattr(self.config, 'min_confidence'):
                data = self._filter_by_confidence(data, self.config.min_confidence)
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response text: {response_text[:500] if response_text else 'None'}...")
            return self._get_error_response(f"JSON parsing error: {e}")
    
    def _filter_by_confidence(self, data: Dict[str, Any], min_confidence: float) -> Dict[str, Any]:
        """Filter demographics by minimum confidence threshold"""
        if "demographics_confidence" not in data or "demographics_detailed" not in data:
            return data
        
        filtered_data = data.copy()
        
        for category in ["race", "gender", "age", "language"]:
            if category in data["demographics_confidence"]:
                # Filter confidence scores
                filtered_conf = {k: v for k, v in data["demographics_confidence"][category].items()  
                                   if v >= min_confidence}
                filtered_data["demographics_confidence"][category] = filtered_conf
                
                # Update detailed list to match filtered confidence
                if category in data["demographics_detailed"]:
                    filtered_data["demographics_detailed"][category] = list(filtered_conf.keys())
        
        return filtered_data
    
    def _get_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Return error response structure"""
        return {
            "demographics_detailed": {
                "race": [],
                "gender": [],
                "age": [],
                "language": []
            },
            "demographics_confidence": {
                "race": {},
                "gender": {},
                "age": {},
                "language": {}
            },
            "demographics_annotation": {
                "model": self.config.model_name,
                "annotated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "individuals_count": 0,
                "explanation": f"Error: {error_msg}",
                "error": True
            }
        }