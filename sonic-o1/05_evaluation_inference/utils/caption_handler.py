"""
utils/caption_handler.py
Parse and chunk SRT caption files for models that support text input.
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Optional
import tempfile
import shutil

logger = logging.getLogger(__name__)


class CaptionHandler:
    """
    Handle SRT caption files with sentence-level chunking support.
    Any model with use_captions=True can use this utility.
    """
    
    def __init__(self, caption_chunks_fallback: Optional[List[Optional[int]]] = None):
        """
        Initialize CaptionHandler with chunking fallback strategy.
        
        Args:
            caption_chunks_fallback: Progressive chunking levels
                Example: [None, 32, 16, 8]
                - None: Use all caption text (no chunking)
                - 32: Sample 32 sentence chunks uniformly
                - 16: Sample 16 sentence chunks
                - 8: Sample 8 sentence chunks (most aggressive)
        """
        self.caption_chunks_fallback = caption_chunks_fallback or [None, 32, 16, 8]
        
        # Use SCRATCH_DIR or TMPDIR environment variable, fallback to home
        scratch_base = os.environ.get('SCRATCH_DIR') or os.environ.get('TMPDIR')
        if scratch_base:
            scratch_base = Path(scratch_base) / 'caption_handler'
        else:
            scratch_base = Path.home() / 'scratch' / 'caption_handler'
        
        scratch_base.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix='captions_', dir=scratch_base))
        self._cleaned_up = False
        logger.info(f"Caption handler temp directory: {self.temp_dir}")
    
    def parse_srt_with_timestamps(self, srt_path: Path) -> List[tuple]:
        """
        Parse SRT file preserving timestamp information.
        
        Args:
            srt_path: Path to .srt caption file
            
        Returns:
            List of tuples: [(start_time, end_time, text), ...]
            Times are in seconds (float)
            
        Example:
            [(0.0, 5.0, "Hello, how are you feeling today?"),
             (5.0, 10.0, "I've been experiencing some chest pain.")]
        """
        if not srt_path.exists():
            logger.error(f"SRT file not found: {srt_path}")
            return []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse SRT blocks
            blocks = content.strip().split('\n\n')
            entries = []
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                # Line 0: index (skip)
                # Line 1: timestamp
                # Line 2+: text
                
                timestamp_line = lines[1]
                if '-->' not in timestamp_line:
                    continue
                
                # Parse timestamps: "00:00:05,000 --> 00:00:10,000"
                try:
                    start_str, end_str = timestamp_line.split('-->')
                    start_time = self._parse_srt_timestamp(start_str.strip())
                    end_time = self._parse_srt_timestamp(end_str.strip())
                    
                    # Join text lines
                    text = ' '.join(lines[2:]).strip()
                    
                    if text:
                        entries.append((start_time, end_time, text))
                        
                except Exception as e:
                    logger.warning(f"Failed to parse SRT block: {e}")
                    continue
            
            logger.info(f"Parsed {len(entries)} timestamped entries from {srt_path.name}")
            return entries
            
        except Exception as e:
            logger.error(f"Error parsing SRT file {srt_path}: {e}")
            return []
    
    def _parse_srt_timestamp(self, timestamp_str: str) -> float:
        """
        Parse SRT timestamp to seconds.
        
        Args:
            timestamp_str: Format "HH:MM:SS,mmm" or "HH:MM:SS.mmm"
            
        Returns:
            Time in seconds (float)
        """
        # Replace comma with dot for milliseconds
        timestamp_str = timestamp_str.replace(',', '.')
        
        # Parse HH:MM:SS.mmm
        parts = timestamp_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    
    def parse_srt(self, srt_path: Path) -> List[str]:
        """
        Parse SRT file and extract sentences (removing timestamps and indices).
        
        Args:
            srt_path: Path to .srt caption file
            
        Returns:
            List of sentences/phrases extracted from captions
            
        Example SRT format:
            1
            00:00:00,000 --> 00:00:05,000
            Hello, how are you feeling today?
            
            2
            00:00:05,000 --> 00:00:10,000
            I've been experiencing some chest pain.
        
        Returns: ['Hello, how are you feeling today?', "I've been experiencing some chest pain.", ...]
        """
        # Parse with timestamps then extract just text
        entries = self.parse_srt_with_timestamps(srt_path)
        
        if not entries:
            return []
        
        # Extract text and split into sentences
        text_lines = [text for _, _, text in entries]
        full_text = ' '.join(text_lines)
        sentences = self._split_into_sentences(full_text)
        
        logger.info(f"Parsed {len(sentences)} sentences from {srt_path.name}")
        return sentences
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using common punctuation marks.
        
        Args:
            text: Raw text to split
            
        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation: . ! ?
        # Keep the punctuation with the sentence
        # Handle common abbreviations (Dr. Mr. Mrs. etc.)
        
        # Simple regex for sentence boundaries
        # Match . ! ? followed by space and capital letter, or end of string
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
        
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def get_caption_text_for_segment(
        self,
        srt_path: Path,
        start_time: float,
        end_time: float,
        num_chunks: Optional[int] = None
    ) -> str:
        """
        Get caption text for a specific time segment with optional chunking.
        
        Args:
            srt_path: Path to .srt file
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            num_chunks:
                - None: Return all text in segment (no chunking)
                - int: Sample N chunks uniformly from segment sentences
        
        Returns:
            Processed caption text for the segment as a single string
        """
        # Step 1: Parse SRT with timestamps
        entries = self.parse_srt_with_timestamps(srt_path)
        
        if not entries:
            logger.warning(f"No entries extracted from {srt_path}")
            return ""
        
        # Step 2: Filter entries that overlap with [start_time, end_time]
        filtered_entries = []
        for entry_start, entry_end, text in entries:
            # Check if entry overlaps with segment
            # Overlap if: entry_start < end_time AND entry_end > start_time
            if entry_start < end_time and entry_end > start_time:
                filtered_entries.append((entry_start, entry_end, text))
        
        if not filtered_entries:
            logger.warning(f"No captions found in segment [{start_time:.1f}s - {end_time:.1f}s]")
            return ""
        
        logger.info(f"Found {len(filtered_entries)} caption entries in segment [{start_time:.1f}s - {end_time:.1f}s]")
        
        # Step 3: Extract text (drop timestamps)
        text_lines = [text for _, _, text in filtered_entries]
        full_text = ' '.join(text_lines)
        
        # Step 4: Split into sentences
        sentences = self._split_into_sentences(full_text)
        
        if not sentences:
            logger.warning(f"No sentences extracted from segment")
            return ""
        
        # Step 5: Apply chunking if needed
        if num_chunks is None:
            result = ' '.join(sentences)
            logger.info(f"Using all {len(sentences)} sentences from segment ({len(result)} chars)")
            return result
        else:
            result = self.chunk_sentences(sentences, num_chunks)
            logger.info(f"Sampled {num_chunks} chunks from {len(sentences)} segment sentences ({len(result)} chars)")
            return result
    
    def get_caption_text(
        self, 
        srt_path: Path, 
        num_chunks: Optional[int] = None
    ) -> str:
        """
        Get caption text with optional sentence-level chunking.
        
        Args:
            srt_path: Path to .srt file
            num_chunks: 
                - None: Return all text (no chunking)
                - int: Sample N chunks uniformly from sentences
        
        Returns:
            Processed caption text as a single string
        """
        sentences = self.parse_srt(srt_path)
        
        if not sentences:
            logger.warning(f"No sentences extracted from {srt_path}")
            return ""
        
        if num_chunks is None:
            # Return all text
            full_text = ' '.join(sentences)
            logger.info(f"Using all {len(sentences)} sentences ({len(full_text)} chars)")
            return full_text
        else:
            # Chunk and sample
            sampled_text = self.chunk_sentences(sentences, num_chunks)
            logger.info(f"Sampled {num_chunks} chunks from {len(sentences)} sentences ({len(sampled_text)} chars)")
            return sampled_text
    
    def chunk_sentences(
        self, 
        sentences: List[str], 
        num_chunks: int
    ) -> str:
        """
        Sample sentences uniformly across the list to create chunks.
        
        Strategy: Divide sentences into num_chunks equal groups,
        then sample representative sentences from each group.
        
        Args:
            sentences: Full list of sentences from SRT
            num_chunks: Number of chunks to sample
            
        Returns:
            Concatenated sampled sentences as a single string
        """
        total_sentences = len(sentences)
        
        if num_chunks >= total_sentences:
            # If requesting more chunks than sentences, return all
            logger.info(f"Requested {num_chunks} chunks but only {total_sentences} sentences, using all")
            return ' '.join(sentences)
        
        # Calculate chunk size
        chunk_size = total_sentences / num_chunks
        
        sampled_sentences = []
        
        for i in range(num_chunks):
            # Calculate the center index of this chunk
            chunk_center = int((i + 0.5) * chunk_size)
            
            # Take a small window around the center (1-3 sentences per chunk)
            # For smaller num_chunks, take more sentences per chunk
            window_size = max(1, min(3, total_sentences // (num_chunks * 2)))
            
            start_idx = max(0, chunk_center - window_size // 2)
            end_idx = min(total_sentences, start_idx + window_size)
            
            # Extract sentences from this chunk
            chunk_sentences = sentences[start_idx:end_idx]
            sampled_sentences.extend(chunk_sentences)
        
        # Join all sampled sentences
        result = ' '.join(sampled_sentences)
        
        logger.debug(f"Chunked {total_sentences} sentences → {len(sampled_sentences)} sentences in {num_chunks} chunks")
        
        return result
    
    def extract_segment_info(self, video_path: Path) -> Optional[dict]:
        """
        Extract segment timing information from video filename.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with 'start' and 'end' keys (float seconds), or None if not a segment
            
        Examples:
            seg_001_30_60_5.mp4 → {'start': 30.0, 'end': 60.0}
            seg_003_120.5_180.3_2.mp4 → {'start': 120.5, 'end': 180.3}
            video_001.mp4 → None (not a segment)
        """
        try:
            video_name = video_path.stem
            
            # Pattern: seg_{video_num}_{start}_{end}_{idx}
            # Supports both integer and float timestamps
            match = re.search(r'seg_\d+_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_', video_name)
            
            if match:
                segment_info = {
                    'start': float(match.group(1)),
                    'end': float(match.group(2))
                }
                logger.debug(f"Extracted segment info from {video_name}: {segment_info}")
                return segment_info
            
            # Not a segment file
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting segment info from {video_path}: {e}")
            return None
    
    def auto_discover_caption_path(
        self, 
        video_path: Path,
        dataset_root: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Auto-discover caption file path from video file path.
        Handles both full videos and segments.
        
        Examples:
            Full video:
                /dataset/videos/01_Topic/video_001.mp4
                → /dataset/captions/01_Topic/caption_001.srt
            
            Segment (requires dataset_root and proper parent structure):
                /dataset/videos/01_Topic/video_001.mp4 (original before segmenting)
                Can infer caption path
                
                BUT: /tmp/seg_001_30_60_5.mp4 (after segmenting)
                Cannot infer topic → returns None
                Caller must provide caption_path explicitly
        
        Args:
            video_path: Path to video file (full or segment)
            dataset_root: Optional dataset root path (if None, inferred from video_path)
            
        Returns:
            Path to caption file if it exists, None otherwise
        """
        try:
            video_name = video_path.stem
            topic_name = video_path.parent.name
            
            # Try Pattern 1: Full video "video_001"
            match = re.search(r'video_(\d+)', video_name)
            
            # Try Pattern 2: Segment "seg_001_30_60_5"  
            if not match:
                match = re.search(r'seg_(\d+)_', video_name)
                
                # For segments in temp dirs, we can't infer topic
                # Parent is like "temp_segments_12345", not the topic
                if 'temp' in topic_name.lower() or 'seg' in topic_name.lower():
                    logger.debug(f"Segment in temp dir: {video_path}. Cannot auto-discover caption. Caller must provide caption_path.")
                    return None
            
            if not match:
                logger.warning(f"Could not extract video number from {video_name}")
                return None
            
            video_number = match.group(1)
            
            # Infer dataset root if not provided
            # Assumes structure: dataset_root/videos/topic_name/video_XXX.mp4
            if dataset_root is None:
                # Go up: video_file -> topic_dir -> videos_dir -> dataset_root
                dataset_root = video_path.parent.parent.parent
            
            # Build caption path
            caption_path = dataset_root / 'captions' / topic_name / f'caption_{video_number}.srt'
            
            if caption_path.exists():
                logger.info(f"Auto-discovered caption: {caption_path}")
                return caption_path
            else:
                logger.warning(f"Caption file not found: {caption_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error auto-discovering caption path for {video_path}: {e}")
            return None
    
    def cleanup(self):
        """Clean up temporary caption files"""
        if self._cleaned_up:
            return
        
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up caption handler temp directory")
                self._cleaned_up = True
        except Exception as e:
            logger.warning(f"Failed to cleanup caption handler temp dir: {e}")
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()