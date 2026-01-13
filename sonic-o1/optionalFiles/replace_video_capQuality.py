#!/usr/bin/env python3
"""
Replace videos based on CaptionQuality label in metadata.json
Reads CaptionQuality from dataset/captions/TOPIC/metadata.json
If CaptionQuality == "replace", finds replacement or drops the video
Updates metadata in videos, audios, and captions directories
"""
import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yt_dlp


class VideoReplacementProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "dataset"
        self.videos_dir = self.dataset_dir / "videos"
        self.audios_dir = self.dataset_dir / "audios"
        self.captions_dir = self.dataset_dir / "captions"
        self.quality_annotated_dir = self.base_dir / "videos_QualityAnnotated"
        
    def find_videos_to_replace(self, topic_name: str) -> List[Dict]:
        """
        Find videos marked with CaptionQuality == "replaced" in captions metadata
        Returns list of dicts with video info
        """
        captions_metadata_path = self.captions_dir / topic_name / "metadata.json"
        
        if not captions_metadata_path.exists():
            print(f"  [WARNING] No captions metadata found for {topic_name}")
            return []
        
        with open(captions_metadata_path, 'r') as f:
            captions_metadata = json.load(f)
        
        to_replace = []
        
        for video in captions_metadata:
            caption_quality = video.get('CaptionQuality', '').lower()
            
            if caption_quality == 'replace':
                to_replace.append({
                    'video_number': video['video_number'],
                    'video_id': video['video_id'],
                    'duration_category': video.get('duration_category', 'unknown'),
                    'demographic_label': video.get('demographic_label', 'general'),
                    'duration_seconds': video.get('duration_seconds', 0),
                    'video_data': video
                })
        
        return to_replace
    
    def load_original_metadata(self, topic_name: str) -> List[Dict]:
        """Load original metadata from videos_QualityAnnotated"""
        topic_path = self.quality_annotated_dir / topic_name
        metadata_files = list(topic_path.glob("*_metadata.json"))
        
        if not metadata_files:
            print(f"  [ERROR] No original metadata found for {topic_name}")
            return []
        
        with open(metadata_files[0], 'r') as f:
            return json.load(f)
    
    def filter_candidates(self, all_videos: List[Dict], already_used_ids: List[str]) -> List[Dict]:
        """Filter videos based on quality criteria"""
        filtered = []
        for video in all_videos:
            # Skip if already used
            if video['video_id'] in already_used_ids:
                continue
            
            # Check quality
            if video.get('Qualitylabel') != 'Good':
                continue
            
            # Check copyright
            if video.get('copyright_notice') != 'creativeCommon':
                continue
            
            # Check language
            default_lang = video.get('default_language', '').lower()
            audio_lang = video.get('default_audio_language', '').lower()
            if default_lang != 'en' and audio_lang != 'en':
                continue
            
            filtered.append(video)
        
        return filtered
    
    def find_replacement(self, 
                        video_to_replace: Dict, 
                        candidate_pool: List[Dict],
                        already_used_ids: List[str]) -> Optional[Dict]:
        """
        Find best replacement video matching duration and demographics
        """
        target_duration_cat = video_to_replace['duration_category']
        target_demo = video_to_replace['demographic_label']
        target_duration_sec = video_to_replace['duration_seconds']
        replace_video_id = video_to_replace['video_id']
        
        # Exclude the video being replaced and already used videos
        exclusion_list = already_used_ids + [replace_video_id]
        
        # Filter candidates by criteria
        candidates = self.filter_candidates(candidate_pool, exclusion_list)
        
        if not candidates:
            return None
        
        # First try: exact match on duration category and demographics
        exact_matches = [
            v for v in candidates 
            if v.get('duration_category') == target_duration_cat 
            and v.get('demographic_label') == target_demo
        ]
        
        if exact_matches:
            exact_matches.sort(
                key=lambda v: abs(v.get('duration_seconds', 0) - target_duration_sec)
            )
            return exact_matches[0]
        
        # Second try: match duration category only
        duration_matches = [
            v for v in candidates 
            if v.get('duration_category') == target_duration_cat
        ]
        
        if duration_matches:
            duration_matches.sort(
                key=lambda v: abs(v.get('duration_seconds', 0) - target_duration_sec)
            )
            return duration_matches[0]
        
        # Third try: match demographics only
        demo_matches = [
            v for v in candidates 
            if v.get('demographic_label') == target_demo
        ]
        
        if demo_matches:
            demo_matches.sort(
                key=lambda v: abs(v.get('duration_seconds', 0) - target_duration_sec)
            )
            return demo_matches[0]
        
        # Last resort: closest duration match from all candidates
        candidates.sort(
            key=lambda v: abs(v.get('duration_seconds', 0) - target_duration_sec)
        )
        return candidates[0]
    
    def download_video(self, video_id: str, output_path: str) -> bool:
        """Download video using yt-dlp (max 1080p)"""
        url = f"https://www.youtube.com/watch?v={video_id}"
        cookies_path = self.base_dir / "cookies.txt"
        
        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]',
            'outtmpl': output_path,
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'cookiefile': str(cookies_path) if cookies_path.exists() else None,
            'extractor_args': {
                'youtube': {
                    'player_client': ['default', '-tv']
                }
            },
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to download {video_id}: {e}")
            return False
    
    def validate_audio_size(self, audio_path: str, video_duration_seconds: float) -> Tuple[bool, float, float]:
        """
        Validate that audio file has reasonable size based on video duration
        Uses dynamic threshold: at least 60% of expected size at 192kbps
        Returns: (is_valid, size_in_mb, expected_size_mb)
        """
        if not os.path.exists(audio_path):
            return False, 0.0, 0.0
        
        size_bytes = os.path.getsize(audio_path)
        size_mb = size_bytes / (1024 * 1024)
        
        # Calculate expected audio size at 192kbps
        expected_audio_mb = (video_duration_seconds * 192) / 8 / 1024
        
        # Audio should be at least 60% of expected size
        min_audio_mb = expected_audio_mb * 0.6
        
        # Also set absolute minimum of 0.5 MB for very short videos
        min_audio_mb = max(min_audio_mb, 0.5)
        
        is_valid = size_mb >= min_audio_mb
        
        return is_valid, size_mb, expected_audio_mb
    
    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video using ffmpeg"""
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'aac',
            '-ar', '48000',
            '-ac', '2',
            '-ab', '192k',
            '-y',
            audio_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to extract audio: {e}")
            return False
    
    def download_captions(self, video_id: str, output_path: str) -> bool:
        """Download YouTube captions if available"""
        url = f"https://www.youtube.com/watch?v={video_id}"
        cookies_path = self.base_dir / "cookies.txt"
        output_base = output_path.replace('.srt', '')
        
        ydl_opts = {
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'srt',
            'skip_download': True,
            'outtmpl': output_base,
            'quiet': True,
            'no_warnings': True,
            'cookiefile': str(cookies_path) if cookies_path.exists() else None,
            'extractor_args': {
                'youtube': {
                    'player_client': ['default', '-tv']
                }
            },
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Check for caption file
            possible_paths = [output_path, f"{output_base}.en.srt"]
            
            for path in possible_paths:
                if os.path.exists(path):
                    if path != output_path:
                        os.rename(path, output_path)
                    return True
            
            return False
        except Exception:
            return False
    
    def drop_video(self, topic_name: str, video_number: str) -> bool:
        """
        Drop a video completely when no replacement can be found
        Deletes all files (video, audio, caption)
        """
        print(f"\n  [DROP] Video {video_number} - no replacement available")
        
        # Define paths
        video_path = self.videos_dir / topic_name / f"video_{video_number}.mp4"
        audio_path = self.audios_dir / topic_name / f"audio_{video_number}.m4a"
        caption_path = self.captions_dir / topic_name / f"caption_{video_number}.srt"
        
        # Delete files
        deleted = []
        for path in [video_path, audio_path, caption_path]:
            if path.exists():
                path.unlink()
                deleted.append(path.name)
        
        if deleted:
            print(f"    [DELETE] {', '.join(deleted)}")
        
        return True
    
    def renumber_videos(self, topic_name: str, dropped_numbers: List[str]):
        """
        Renumber videos after dropping some, filling gaps in sequence
        Updates metadata in all three directories
        """
        if not dropped_numbers:
            return
        
        print(f"\n  [RENUMBER] Filling gaps after dropping videos...")
        
        # Load current metadata from captions (source of truth)
        captions_metadata_path = self.captions_dir / topic_name / "metadata.json"
        with open(captions_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        dropped_set = set(dropped_numbers)
        
        # Create mapping: old_num -> new_num
        all_numbers = sorted([v['video_number'] for v in metadata])
        remaining_numbers = [n for n in all_numbers if n not in dropped_set]
        
        # New sequential numbers starting from 001
        new_numbers = [f"{i:03d}" for i in range(1, len(remaining_numbers) + 1)]
        
        # Create renaming map
        rename_map = dict(zip(remaining_numbers, new_numbers))
        
        print(f"    Videos before: {len(all_numbers)}, after: {len(remaining_numbers)}")
        
        # Rename files in all three directories
        for old_num, new_num in rename_map.items():
            if old_num == new_num:
                continue
            
            # Rename video file
            old_video = self.videos_dir / topic_name / f"video_{old_num}.mp4"
            new_video = self.videos_dir / topic_name / f"video_{new_num}.mp4"
            if old_video.exists():
                old_video.rename(new_video)
            
            # Rename audio file
            old_audio = self.audios_dir / topic_name / f"audio_{old_num}.m4a"
            new_audio = self.audios_dir / topic_name / f"audio_{new_num}.m4a"
            if old_audio.exists():
                old_audio.rename(new_audio)
            
            # Rename caption file
            old_caption = self.captions_dir / topic_name / f"caption_{old_num}.srt"
            new_caption = self.captions_dir / topic_name / f"caption_{new_num}.srt"
            if old_caption.exists():
                old_caption.rename(new_caption)
            
            print(f"    {old_num} -> {new_num}")
        
        # Update metadata - remove dropped videos and update video_numbers
        new_metadata = []
        for video in metadata:
            old_num = video['video_number']
            if old_num in dropped_set:
                continue
            
            # Update video_number
            video['video_number'] = rename_map[old_num]
            new_metadata.append(video)
        
        # Save updated metadata to all three directories
        for base_dir in [self.videos_dir, self.audios_dir, self.captions_dir]:
            metadata_path = base_dir / topic_name / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(new_metadata, f, indent=2)
        
        print(f"    ✓ Metadata updated in all directories")
        
        # Update needs_whisper.txt
        needs_whisper_file = self.captions_dir / topic_name / "needs_whisper.txt"
        if needs_whisper_file.exists():
            with open(needs_whisper_file, 'r') as f:
                entries = [line.strip() for line in f if line.strip()]
            
            # Update audio filenames and remove dropped ones
            new_entries = []
            for entry in entries:
                old_num = entry.replace('audio_', '').replace('.m4a', '')
                if old_num in dropped_set:
                    continue
                if old_num in rename_map:
                    new_num = rename_map[old_num]
                    new_entries.append(f"audio_{new_num}.m4a")
            
            # Save updated list
            with open(needs_whisper_file, 'w') as f:
                for entry in sorted(new_entries):
                    f.write(f"{entry}\n")
        
        print(f"    ✓ Renumbering complete")
    
    def replace_video(self, 
                      topic_name: str, 
                      video_to_replace: Dict, 
                      replacement_video: Dict) -> bool:
        """
        Replace a video marked for replacement with a new one
        Downloads video, audio, captions and updates metadata in all directories
        Validates audio size based on video duration
        """
        video_num = video_to_replace['video_number']
        old_video_id = video_to_replace['video_id']
        new_video_id = replacement_video['video_id']
        replacement_duration_sec = replacement_video.get('duration_seconds', 0)
        
        print(f"\n  [REPLACE] Video {video_num}: {old_video_id} → {new_video_id}")
        print(f"    Reason: CaptionQuality = 'replace'")
        print(f"    Match: duration={replacement_video.get('duration_category')}, "
              f"demo={replacement_video.get('demographic_label')}, "
              f"length={replacement_duration_sec}s")
        
        # Define paths
        video_path = self.videos_dir / topic_name / f"video_{video_num}.mp4"
        audio_path = self.audios_dir / topic_name / f"audio_{video_num}.m4a"
        caption_path = self.captions_dir / topic_name / f"caption_{video_num}.srt"
        
        # Use temporary paths for validation
        temp_video_path = video_path.parent / f"temp_video_{video_num}.mp4"
        temp_audio_path = audio_path.parent / f"temp_audio_{video_num}.m4a"
        
        # Download new video to temp location
        print(f"    [DOWNLOAD] Video...")
        if not self.download_video(new_video_id, str(temp_video_path)):
            return False
        print(f"    ✓ Video downloaded")
        
        # Extract audio to temp location
        print(f"    [EXTRACT] Audio...")
        if not self.extract_audio(str(temp_video_path), str(temp_audio_path)):
            temp_video_path.unlink()
            return False
        print(f"    ✓ Audio extracted")
        
        # Validate audio size
        print(f"    [VALIDATE] Audio size...")
        is_valid, audio_size_mb, expected_size_mb = self.validate_audio_size(
            str(temp_audio_path), 
            replacement_duration_sec
        )
        
        if not is_valid:
            min_required = expected_size_mb * 0.6
            print(f"    ✗ Audio too small: {audio_size_mb:.2f} MB "
                  f"(expected ~{expected_size_mb:.2f} MB, min {min_required:.2f} MB)")
            print(f"    [REJECT] This video likely has no speech or is music-only")
            temp_video_path.unlink()
            temp_audio_path.unlink()
            return False
        
        print(f"    ✓ Audio size valid: {audio_size_mb:.2f} MB "
              f"(expected ~{expected_size_mb:.2f} MB)")
        
        # Delete old files
        for path in [video_path, audio_path, caption_path]:
            if path.exists():
                path.unlink()
                print(f"    [DELETE] {path.name}")
        
        # Move temp files to final location
        temp_video_path.rename(video_path)
        temp_audio_path.rename(audio_path)
        
        # Try to download captions
        print(f"    [DOWNLOAD] Captions...")
        has_captions = self.download_captions(new_video_id, str(caption_path))
        
        if has_captions:
            print(f"    ✓ Captions downloaded")
            needs_whisper = False
        else:
            print(f"    [INFO] No captions available, will need Whisper")
            needs_whisper = True
        
        # Update metadata in all three directories
        replacement_video['video_number'] = video_num
        # Set CaptionQuality to empty/good for the replacement
        replacement_video['CaptionQuality'] = ''
        
        for base_dir in [self.videos_dir, self.audios_dir, self.captions_dir]:
            metadata_path = base_dir / topic_name / "metadata.json"
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Find and replace the video entry
            for i, video in enumerate(metadata):
                if video['video_number'] == video_num:
                    metadata[i] = replacement_video.copy()
                    break
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"    ✓ Metadata updated in all directories")
        
        # Update needs_whisper.txt
        needs_whisper_file = self.captions_dir / topic_name / "needs_whisper.txt"
        
        if needs_whisper:
            # Add to needs_whisper.txt
            existing_entries = []
            if needs_whisper_file.exists():
                with open(needs_whisper_file, 'r') as f:
                    existing_entries = [line.strip() for line in f if line.strip()]
            
            audio_filename = f"audio_{video_num}.m4a"
            if audio_filename not in existing_entries:
                existing_entries.append(audio_filename)
                existing_entries.sort()
                
                with open(needs_whisper_file, 'w') as f:
                    for entry in existing_entries:
                        f.write(f"{entry}\n")
                
                print(f"    ✓ Added to needs_whisper.txt")
        else:
            # Remove from needs_whisper.txt if present
            if needs_whisper_file.exists():
                with open(needs_whisper_file, 'r') as f:
                    existing_entries = [line.strip() for line in f if line.strip()]
                
                audio_filename = f"audio_{video_num}.m4a"
                if audio_filename in existing_entries:
                    existing_entries.remove(audio_filename)
                    
                    with open(needs_whisper_file, 'w') as f:
                        for entry in existing_entries:
                            f.write(f"{entry}\n")
                    
                    print(f"    ✓ Removed from needs_whisper.txt")
        
        return True
    
    def process_topic(self, topic_name: str):
        """Process a single topic to replace videos marked as 'replaced'"""
        print(f"\n{'='*70}")
        print(f"Processing Topic: {topic_name}")
        print(f"{'='*70}")
        
        # Find videos marked for replacement
        videos_to_replace = self.find_videos_to_replace(topic_name)
        
        if not videos_to_replace:
            print(f"  ✓ No videos marked for replacement (CaptionQuality != 'replace')")
            return
        
        print(f"  Found {len(videos_to_replace)} video(s) marked for replacement:")
        for v in videos_to_replace:
            print(f"    - Video {v['video_number']}: CaptionQuality = 'replace'")
        
        # Load original metadata pool
        original_metadata = self.load_original_metadata(topic_name)
        
        if not original_metadata:
            print(f"  [ERROR] Cannot load original metadata")
            return
        
        print(f"  Original metadata pool: {len(original_metadata)} videos")
        
        # Get currently used video IDs from captions metadata
        captions_metadata_path = self.captions_dir / topic_name / "metadata.json"
        with open(captions_metadata_path, 'r') as f:
            current_metadata = json.load(f)
        
        used_ids = [v['video_id'] for v in current_metadata]
        print(f"  Currently used videos: {len(used_ids)}")
        
        # Track replacements and drops
        replacements_made = 0
        dropped_videos = []
        
        # Replace each video
        for video_to_replace in videos_to_replace:
            # Remove the video ID from used_ids so we can replace it
            replace_video_id = video_to_replace['video_id']
            if replace_video_id in used_ids:
                used_ids.remove(replace_video_id)
                print(f"\n  Freeing up slot for video {video_to_replace['video_number']} "
                      f"(ID: {replace_video_id})")
            
            # Try up to 5 candidates in case audio validation fails
            max_attempts = 5
            attempt = 0
            success = False
            
            while attempt < max_attempts and not success:
                replacement = self.find_replacement(
                    video_to_replace, 
                    original_metadata, 
                    used_ids
                )
                
                if not replacement:
                    print(f"  [WARNING] No replacement found for video "
                          f"{video_to_replace['video_number']}")
                    self.drop_video(topic_name, video_to_replace['video_number'])
                    dropped_videos.append(video_to_replace['video_number'])
                    break
                
                attempt += 1
                if attempt > 1:
                    print(f"  [RETRY] Attempt {attempt}/{max_attempts}")
                
                success = self.replace_video(topic_name, video_to_replace, replacement)
                
                if success:
                    replacements_made += 1
                    used_ids.append(replacement['video_id'])
                else:
                    # This candidate failed, mark as used and try next
                    used_ids.append(replacement['video_id'])
                    print(f"  [INFO] Trying next candidate...")
            
            if not success and replacement:
                # Exhausted all attempts
                print(f"  [WARNING] Could not replace video "
                      f"{video_to_replace['video_number']}, dropping it")
                self.drop_video(topic_name, video_to_replace['video_number'])
                dropped_videos.append(video_to_replace['video_number'])
        
        # Renumber videos if any were dropped
        if dropped_videos:
            self.renumber_videos(topic_name, dropped_videos)
        
        print(f"\n  ✓ Replacements completed: {replacements_made}/{len(videos_to_replace)}")
        print(f"  ✓ Videos dropped: {len(dropped_videos)}/{len(videos_to_replace)}")


def main():
    """Main function"""
    # Hardcoded configuration
    base_dir = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset"
    
    # Create processor
    processor = VideoReplacementProcessor(base_dir)
    
    # Process all topics in captions directory
    captions_dir = processor.captions_dir
    topics = sorted([d.name for d in captions_dir.iterdir() if d.is_dir()])
    
    print(f"\n{'='*70}")
    print(f"Video Replacement Processor (CaptionQuality-based)")
    print(f"{'='*70}")
    print(f"Base directory: {base_dir}")
    print(f"Topics to process: {len(topics)}")
    print(f"Source: dataset/captions/TOPIC/metadata.json (CaptionQuality field)")
    print(f"Action: Replace videos where CaptionQuality == 'replace'")
    print(f"Audio validation: Dynamic (60% of expected size at 192kbps, min 0.5 MB)")
    print(f"{'='*70}")
    
    # Process each topic
    for topic in topics:
        try:
            processor.process_topic(topic)
        except Exception as e:
            print(f"\n[ERROR] Failed to process {topic}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("All topics processed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()