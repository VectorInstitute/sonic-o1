"""
Task 1: Video Summarization Model
"""
import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_gemini import BaseGeminiClient
from utils.video_segmenter import VideoSegmenter
from utils.demographics_expander import DemographicsExpander
from prompts.summarization_prompts import (
    get_map_prompt, 
    get_reduce_prompt, 
    get_direct_prompt,
    get_initialize_prompt,
    get_streaming_update_prompt
)

logger = logging.getLogger(__name__)


class SummarizationModel(BaseGeminiClient):
    """Generate video-level summarization VQA entries"""
    
    def __init__(self, config):
        """
        Initialize summarization model.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.config = config
        self.segmenter = VideoSegmenter(config)
        self.demographics_expander = DemographicsExpander(config)
    
    def process_video(self,
                     video_path: Path,
                     audio_path: Optional[Path],
                     transcript_path: Optional[Path],
                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a video and generate summarization VQA entry.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file (optional)
            transcript_path: Path to transcript/caption file (optional)
            metadata: Video metadata from metadata_enhanced.json
            
        Returns:
            VQA entry dict for Task 1
        """
        try:
            video_id = metadata.get('video_id', metadata.get('video_number', 'unknown'))
            duration = metadata.get('duration_seconds', 0)
            category = metadata.get('duration_category', 'short')  
            
            if category not in ['short', 'medium', 'long']:
                if duration <= 300:
                    category = 'short'
                elif duration <= 1800:
                    category = 'medium'
                else:
                    category = 'long'
                logger.warning(
                    f"Video {video_id}: No valid category in metadata, "
                    f"determined as '{category}' based on duration ({duration}s)"
                )
            
            print(f"Processing video {video_id} for summarization (duration: {duration}s, category: {category})")
            
            if category == 'short':
                result = self._process_short_video(video_path, audio_path, transcript_path, metadata)
            else:
                result = self._process_segmented_video(video_path, audio_path, transcript_path, metadata)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
            return self._get_error_entry(metadata, str(e))
    
    def _process_short_video(self,
                            video_path: Path,
                            audio_path: Optional[Path],
                            transcript_path: Optional[Path],
                            metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process short video with direct summarization (no segmentation)"""
        video_id = metadata.get('video_id', metadata.get('video_number', 'unknown'))
        print(f"Direct summarization for short video {video_id}")
        
        transcript_text = self._load_transcript(transcript_path)
        prompt = get_direct_prompt(video_id, metadata, transcript_text, self.config)
        
        media_files = []
        if video_path and video_path.exists():
            media_files.append(('video', video_path))
        if audio_path and audio_path.exists():
            media_files.append(('audio', audio_path))
        
        max_attempts = 3
        summary_data = None
        
        for attempt in range(max_attempts):
            try:
                response_text = self.generate_content(media_files, prompt, video_fps=0.5)
                summary_data = self._parse_summary_response(response_text)
                
                if summary_data.get('confidence', 0) > 0:
                    print(f"Successfully generated summary for {video_id}")
                    break
                else:
                    logger.warning(f"Parse attempt {attempt + 1} failed for {video_id}, retrying...")
                    if attempt < max_attempts - 1:
                        time.sleep(5)
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(5)
        
        if not summary_data or summary_data.get('confidence', 0) == 0:
            logger.error(f"FAILED to generate valid summary for {video_id} after {max_attempts} attempts")
            summary_data = self._get_default_summary()
        
        demographics = self._get_video_demographics(
            video_path, audio_path, transcript_path, metadata, segments_info=None
        )
        
        entry = {
            'video_id': video_id,
            'video_number': metadata.get('video_number', video_id),
            'duration_seconds': metadata.get('duration_seconds', 0),
            'segments_processed': None,
            'summary_short': summary_data.get('summary_short', []),
            'summary_detailed': summary_data.get('summary_detailed', ''),
            'timeline': summary_data.get('timeline', []),
            'glossary': summary_data.get('glossary', []),
            'demographics': demographics.get('demographics', []),
            'confidence': summary_data.get('confidence', 0.0)
        }
        
        return entry
    
    def _process_segmented_video(self,
                                video_path: Path,
                                audio_path: Optional[Path],
                                transcript_path: Optional[Path],
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process medium/long video with MAP-REDUCE approach"""
        video_id = metadata.get('video_id', metadata.get('video_number', 'unknown'))
        duration = metadata.get('duration_seconds', 0)
        
        print(f"MAP-REDUCE processing for {video_id} ({duration}s)")
        
        video_segments = self.segmenter.segment_video(video_path, duration, task_type='summarization')
        audio_segments = None
        if audio_path and audio_path.exists():
            audio_segments = self.segmenter.segment_audio(audio_path, duration, task_type='summarization')
        
        print(f"Created {len(video_segments)} segments")
        
        segment_summaries = []
        for i, seg in enumerate(video_segments):
            try:
                audio_seg_path = audio_segments[i]['segment_path'] if audio_segments else None
                
                transcript_text = ""
                if transcript_path and transcript_path.exists():
                    transcript_text = self.segmenter.extract_transcript_segment(
                        transcript_path, seg['start'], seg['end']
                    )
                
                segment_summary = self._generate_segment_summary(
                    seg, audio_seg_path, transcript_text, metadata
                )
                segment_summaries.append(segment_summary)
                
            except Exception as e:
                logger.error(f"Failed to process segment {i}: {e}")
                continue
        
        merged_summary = self._merge_segment_summaries(video_id, metadata, segment_summaries)
        
        demographics = self._get_video_demographics(
            video_path, audio_path, transcript_path, metadata, 
            segments_info=[{'start': s['start'], 'end': s['end']} for s in video_segments]
        )
        
        try:
            self.segmenter.cleanup_segments(video_segments)
            if audio_segments:
                self.segmenter.cleanup_segments(audio_segments)
        except Exception as e:
            logger.warning(f"Failed to cleanup segments: {e}")
        
        entry = {
            'video_id': video_id,
            'video_number': metadata.get('video_number', video_id),
            'duration_seconds': duration,
            'segments_processed': [{'start': s['start'], 'end': s['end']} for s in video_segments],
            'summary_short': merged_summary.get('summary_short', []),
            'summary_detailed': merged_summary.get('summary_detailed', ''),
            'timeline': merged_summary.get('timeline', []),
            'glossary': merged_summary.get('glossary', []),
            'demographics': demographics.get('demographics', []),
            'confidence': merged_summary.get('confidence', 0.0)
        }
        
        return entry
    
    def _generate_segment_summary(self,
                                segment_info: Dict,
                                audio_path: Optional[Path],
                                transcript_text: str,
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """MAP phase: Generate summary for one segment with retry on JSON parse failure"""
        seg_num = segment_info['segment_number']
        print(f"Generating summary for segment {seg_num} ({segment_info['start']}s-{segment_info['end']}s)")
        
        max_attempts = 2
        
        for attempt in range(max_attempts):
            try:
                prompt = get_map_prompt(segment_info, metadata, transcript_text, self.config)
                
                if attempt > 0:
                    prompt += "\n\nPREVIOUS ATTEMPT RETURNED INVALID JSON. Requirements:\n"
                    prompt += "- Use commas between all object properties except the last\n"
                    prompt += "- Use commas between all array elements except the last\n"
                    prompt += "- Use double quotes for all strings\n"
                    prompt += "- No trailing commas before closing brackets\n"
                    prompt += "- Escape special characters in strings\n"
                
                media_files = []
                seg_path = segment_info['segment_path']
                if seg_path.exists():
                    if seg_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.m4v']:
                        media_files.append(('video', seg_path))
                    else:
                        media_files.append(('audio', seg_path))
                
                if audio_path and audio_path.exists():
                    media_files.append(('audio', audio_path))
                
                total_size = sum(p.stat().st_size for _, p in media_files if p.exists())
                print(f"Calling Gemini with {len(media_files)} files, total {total_size/1024/1024:.1f}MB")
                
                response_text = self.generate_content(media_files, prompt, video_fps=0.5)
                summary_data = self._parse_segment_summary_response(response_text)
                
                if summary_data.get('confidence', 0) > 0:
                    return summary_data
                else:
                    logger.warning(f"Segment {seg_num} parsing failed on attempt {attempt + 1}/{max_attempts}")
                    if attempt < max_attempts - 1:
                        time.sleep(3)
                        
            except Exception as e:
                logger.error(f"Segment {seg_num} generation failed on attempt {attempt + 1}/{max_attempts}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(3)
        
        logger.error(f"Failed to generate valid summary for segment {seg_num} after {max_attempts} attempts")
        return self._get_default_segment_summary()
    
    def _merge_segment_summaries(self,
                                video_id: str,
                                metadata: Dict[str, Any],
                                segment_summaries: List[Dict]) -> Dict[str, Any]:
        """REDUCE phase: Incrementally build summary by adding segments one at a time"""
        print(f"Streaming accumulation for {len(segment_summaries)} segments")
        
        if not segment_summaries:
            return self._get_default_summary()
        
        accumulated_summary = self._initialize_summary_from_segment(
            segment_summaries[0], video_id, metadata
        )
        
        for i, segment in enumerate(segment_summaries[1:], start=2):
            print(f"Adding segment {i}/{len(segment_summaries)} to accumulated summary")
            
            accumulated_summary = self._add_segment_to_summary(
                accumulated_summary, 
                segment, 
                video_id, 
                metadata,
                segment_num=i,
                total_segments=len(segment_summaries)
            )
            
            time.sleep(30)
        
        return accumulated_summary
    
    def _sanitize_metadata_for_prompt(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to avoid triggering safety filters"""
        sanitized = metadata.copy()
        title = sanitized.get('title', '')
        
        if title and re.search(r'\d+\s*year\s*old|minor|child', title, re.IGNORECASE):
            sanitized['title'] = f"[{metadata.get('topic', 'Video')} Incident]"
            logger.info(f"Sanitized title to avoid safety filter: {title[:50]}...")
        
        return sanitized
    
    def _initialize_summary_from_segment(self,
                                        first_segment: Dict,
                                        video_id: str,
                                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert first segment into initial video-level summary structure with retry"""
        
        safe_metadata = self._sanitize_metadata_for_prompt(metadata)
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                prompt = get_initialize_prompt(first_segment, video_id, safe_metadata)
                response_text = self.generate_content([], prompt, video_fps=0.5)
                
                if not response_text or not response_text.strip():
                    logger.warning(f"Empty response on attempt {attempt + 1}, using direct conversion")
                    if attempt == max_attempts - 1:
                        return self._direct_convert_segment_to_summary(first_segment)
                    time.sleep(5)
                    continue
                
                summary_data = self._parse_summary_response(response_text)
                
                if summary_data.get('confidence', 0) > 0:
                    return summary_data
                else:
                    logger.warning(f"Initialize attempt {attempt + 1}/{max_attempts} failed for {video_id}")
                    if attempt < max_attempts - 1:
                        time.sleep(5)
                        
            except Exception as e:
                logger.error(f"Initialize attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(5)
        
        logger.error(f"Failed to initialize summary for {video_id}, using direct conversion")
        return self._direct_convert_segment_to_summary(first_segment)
    
    def _add_segment_to_summary(self,
                            current_summary: Dict[str, Any],
                            new_segment: Dict,
                            video_id: str,
                            metadata: Dict[str, Any],
                            segment_num: int,
                            total_segments: int) -> Dict[str, Any]:
        """Add one new segment to the accumulated summary with retry"""
        
        safe_metadata = self._sanitize_metadata_for_prompt(metadata)
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                prompt = get_streaming_update_prompt(
                    current_summary,
                    new_segment,
                    video_id,
                    safe_metadata,
                    segment_num,
                    total_segments,
                    self.config
                )
                
                response_text = self.generate_content([], prompt, video_fps=0.5)
                
                if not response_text or not response_text.strip():
                    logger.warning(f"Empty response for segment {segment_num}, attempt {attempt + 1}")
                    if attempt == max_attempts - 1:
                        return self._programmatic_merge(current_summary, new_segment)
                    time.sleep(5)
                    continue
                
                summary_data = self._parse_summary_response(response_text)
                
                if summary_data.get('confidence', 0) > 0:
                    return summary_data
                else:
                    logger.warning(f"Merge attempt {attempt + 1}/{max_attempts} failed for segment {segment_num}")
                    if attempt < max_attempts - 1:
                        time.sleep(5)
                        
            except Exception as e:
                logger.error(f"Merge attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(5)
        
        logger.error(f"Failed to merge segment {segment_num}, using programmatic merge")
        return self._programmatic_merge(current_summary, new_segment)
    
    def _direct_convert_segment_to_summary(self, segment: Dict) -> Dict[str, Any]:
        """Directly convert segment format to summary format without LLM"""
        summary_text = segment.get('summary', '')
        
        lines = [l.strip().lstrip('•-* ') for l in summary_text.split('\n') if l.strip()]
        bullet_points = [l for l in lines if len(l) > 10][:5]
        
        if not bullet_points and summary_text:
            bullet_points = [summary_text[:200]]
        
        return {
            'summary_short': bullet_points,
            'summary_detailed': summary_text,
            'timeline': segment.get('mini_timeline', []),
            'glossary': self._entities_to_glossary(segment.get('entities', [])),
            'confidence': segment.get('confidence', 0.5)
        }
    
    def _programmatic_merge(self, current: Dict, new_segment: Dict) -> Dict:
        """Programmatically merge segments without LLM"""
        new_summary = new_segment.get('summary', '')
        if new_summary:
            lines = [l.strip().lstrip('•-* ') for l in new_summary.split('\n') 
                    if l.strip() and len(l) > 10]
            current['summary_short'].extend(lines[:3])
        
        if new_summary:
            current['summary_detailed'] += f"\n\n{new_summary}"
        
        current['timeline'].extend(new_segment.get('mini_timeline', []))
        
        new_terms = self._entities_to_glossary(new_segment.get('entities', []))
        existing = {t['term'].lower() for t in current.get('glossary', [])}
        for term in new_terms:
            if term['term'].lower() not in existing:
                current.setdefault('glossary', []).append(term)
                existing.add(term['term'].lower())
        
        current['confidence'] = min(current.get('confidence', 1.0), 
                                   new_segment.get('confidence', 1.0))
        
        return current
    
    def _entities_to_glossary(self, entities: List) -> List[Dict]:
        """Convert entities to glossary format"""
        glossary = []
        for entity in entities:
            if isinstance(entity, dict):
                glossary.append({
                    'term': entity.get('name', entity.get('term', '')),
                    'definition': entity.get('description', entity.get('definition', '')),
                    'category': entity.get('type', entity.get('category', 'entity'))
                })
        return glossary
    
    def _get_video_demographics(self,
                            video_path: Path,
                            audio_path: Optional[Path],
                            transcript_path: Optional[Path],
                            metadata: Dict[str, Any],
                            segments_info: Optional[List[Dict]]) -> Dict[str, Any]:
        """Get expanded demographics for full video"""
        temp_video = None
        
        try:
            human_demographics = metadata.get('demographics_detailed_reviewed', {})
            if not human_demographics:
                logger.warning(f"No human-reviewed demographics found for {metadata.get('video_id')}")
                return {'demographics': [], 'total_individuals': 0, 'confidence': 0.0}
            
            prompt = self.demographics_expander.build_expansion_prompt(
                human_demographics, 
                segment_info=None
            )
            
            media_files = []
            if video_path and video_path.exists():
                media_files.append(('video', video_path))
            
            if audio_path and audio_path.exists():
                media_files.append(('audio', audio_path))
            
            transcript_text = self._load_transcript(transcript_path)
            if transcript_text:
                prompt += f"\n\nTRANSCRIPT SUMMARY:\n{transcript_text[:2000]}"
            
            response_text = self.generate_content(media_files, prompt, video_fps=0.25)
            demographics_data = self.demographics_expander.parse_demographics_response(response_text)
            
            return demographics_data
            
        except Exception as e:
            logger.error(f"Failed to get video demographics: {e}", exc_info=True)
            return {'demographics': [], 'total_individuals': 0, 'confidence': 0.0}
        
        finally:
            if temp_video and temp_video != video_path and temp_video.exists():
                try:
                    temp_video.unlink()
                    logger.info(f"Cleaned up temporary video: {temp_video.name}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary video: {e}")
    
    def _get_default_summary(self) -> Dict[str, Any]:
        """Return default summary structure when parsing fails"""
        return {
            'summary_short': [],
            'summary_detailed': 'Summary generation failed due to parsing error',
            'timeline': [],
            'glossary': [],
            'confidence': 0.0
        }
    
    def _get_default_segment_summary(self) -> Dict[str, Any]:
        """Return default segment summary structure when parsing fails"""
        return {
            'segment_start': '',
            'segment_end': '',
            'summary': 'Segment summary generation failed',
            'mini_timeline': [],
            'entities': [],
            'confidence': 0.0
        }
    
    def _load_transcript(self, transcript_path: Optional[Path]) -> str:
        """Load and truncate transcript if needed"""
        if not transcript_path or not transcript_path.exists():
            return ""
        
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            max_length = self.config.file_processing.max_transcript_length
            if len(text) > max_length:
                print(f"Truncating transcript from {len(text)} to {max_length} chars")
                text = text[:max_length] + "\n...[truncated]"
            
            return text
        except Exception as e:
            logger.warning(f"Failed to load transcript: {e}")
            return ""
    
    def _fix_common_json_errors(self, text: str) -> str:
        """Attempt to fix common JSON formatting errors from Gemini"""
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        text = re.sub(r'(\"[^\"]*\")\s+(\"\w+\"\s*:)', r'\1,\2', text)
        text = re.sub(r'(\})\s*\n\s*(\{)', r'\1,\2', text)
        text = re.sub(r'(\])\s*\n\s*(\[)', r'\1,\2', text)
        text = re.sub(r'(\})\s*\n\s*(\[)', r'\1,\2', text)
        text = re.sub(r'(\])\s*\n\s*(\{)', r'\1,\2', text)
        text = re.sub(r'("\s*)\n(\s*")', r'\1,\2', text)
        text = re.sub(r'(\":\s*\"[^\"]*\")\s+(\"\w+\":)', r'\1,\2', text)
        text = re.sub(r'(\":\s*\d+\.?\d*)\s+(\"\w+\":)', r'\1,\2', text)
        text = re.sub(r'(\":\s*(?:true|false|null))\s+(\"\w+\":)', r'\1,\2', text)
        text = re.sub(r'(\])\s+(\"\w+\":)', r'\1,\2', text)
        text = re.sub(r'(\})\s+(\"\w+\":)', r'\1,\2', text)
        text = re.sub(r',\s*,', r',', text)
        
        return text
    
    def _parse_summary_response(self, response_text: str) -> Dict[str, Any]:
        """Parse summary response from Gemini with enhanced error handling"""
        try:
            if not response_text or not response_text.strip():
                logger.warning("Empty summary response received")
                return self._get_default_summary()
            
            response_text = response_text.strip()
            
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
            
            response_text = response_text.strip()
            
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error, attempting auto-fix: {e}")
                
                corrected_text = self._fix_common_json_errors(response_text)
                
                try:
                    data = json.loads(corrected_text)
                    logger.info("Successfully fixed JSON formatting!")
                except json.JSONDecodeError as e2:
                    logger.error(f"JSON still invalid after auto-fix: {e2}")
                    
                    error_line = getattr(e2, 'lineno', 0)
                    error_col = getattr(e2, 'colno', 0)
                    
                    if error_line > 0:
                        lines = corrected_text.split('\n')
                        start_line = max(0, error_line - 3)
                        end_line = min(len(lines), error_line + 2)
                        
                        logger.error(f"Error at line {error_line}, column {error_col}:")
                        for i in range(start_line, end_line):
                            if i < len(lines):
                                marker = " >>> " if i == error_line - 1 else "     "
                                logger.error(f"{marker}Line {i+1}: {lines[i][:200]}")
                    
                    debug_file = Path(f"debug_json_error_{int(time.time())}.txt")
                    with open(debug_file, 'w') as f:
                        f.write("=== ORIGINAL ===\n")
                        f.write(response_text)
                        f.write("\n\n=== CORRECTED ===\n")
                        f.write(corrected_text)
                    logger.error(f"Saved problematic JSON to {debug_file}")
                    
                    return self._get_default_summary()
            
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    logger.warning("API returned list, extracting first element")
                    data = data[0]
                else:
                    logger.error("API returned invalid list format")
                    return self._get_default_summary()
            
            if not isinstance(data, dict):
                logger.error(f"Parsed data is not a dict, got {type(data)}")
                return self._get_default_summary()
            
            return {
                'summary_short': data.get('summary_short', []),
                'summary_detailed': data.get('summary_detailed', ''),
                'timeline': data.get('timeline', []),
                'glossary': data.get('glossary', []),
                'confidence': float(data.get('confidence', 0.0))
            }
            
        except Exception as e:
            logger.error(f"Failed to parse summary response: {e}", exc_info=True)
            return self._get_default_summary()
    
    def _parse_segment_summary_response(self, response_text: str) -> Dict[str, Any]:
        """Parse segment summary response from Gemini with enhanced error handling"""
        try:
            if not response_text or not response_text.strip():
                logger.warning("Empty response text received")
                return self._get_default_segment_summary()
            
            response_text = response_text.strip()
            
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
            
            response_text = response_text.strip()
            
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error in segment, attempting auto-fix: {e}")
                
                corrected_text = self._fix_common_json_errors(response_text)
                
                try:
                    data = json.loads(corrected_text)
                    logger.info("Successfully fixed segment JSON formatting!")
                except json.JSONDecodeError as e2:
                    logger.error(f"Segment JSON still invalid after auto-fix: {e2}")
                    
                    error_line = getattr(e2, 'lineno', 0)
                    if error_line > 0:
                        lines = corrected_text.split('\n')
                        start_line = max(0, error_line - 3)
                        end_line = min(len(lines), error_line + 2)
                        
                        logger.error(f"Error at line {error_line}:")
                        for i in range(start_line, end_line):
                            if i < len(lines):
                                marker = " >>> " if i == error_line - 1 else "     "
                                logger.error(f"{marker}Line {i+1}: {lines[i][:200]}")
                    
                    return self._get_default_segment_summary()
            
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    logger.warning("API returned list in segment, extracting first element")
                    data = data[0]
                else:
                    logger.error("API returned invalid list format in segment")
                    return self._get_default_segment_summary()
            
            if not isinstance(data, dict):
                logger.error(f"Segment data is not a dict, got {type(data)}")
                return self._get_default_segment_summary()
            
            return {
                'segment_start': data.get('segment_start', ''),
                'segment_end': data.get('segment_end', ''),
                'summary': data.get('summary', ''),
                'mini_timeline': data.get('mini_timeline', []),
                'entities': data.get('entities', []),
                'confidence': float(data.get('confidence', 0.0))
            }
            
        except Exception as e:
            logger.error(f"Failed to parse segment summary: {e}", exc_info=True)
            return self._get_default_segment_summary()
    
    def _get_error_entry(self, metadata: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Return error entry structure"""
        video_id = metadata.get('video_id', metadata.get('video_number', 'unknown'))
        return {
            'video_id': video_id,
            'video_number': metadata.get('video_number', video_id),
            'duration_seconds': metadata.get('duration_seconds', 0),
            'segments_processed': None,
            'summary_short': [],
            'summary_detailed': f'Error: {error_msg}',
            'timeline': [],
            'glossary': [],
            'demographics': [],
            'confidence': 0.0,
            'error': True
        }