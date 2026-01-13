#!/usr/bin/env python3
"""
Sync metadata from captions to videos/audios directories
Drop videos marked with CaptionQuality = "replace"
Renumber remaining videos sequentially
Sync metadata_enhanced.json based on video_id (YouTube ID) matching
"""
import json
from pathlib import Path
from typing import Set, List, Dict, Tuple

class MetadataSyncAndDrop:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "dataset"
        self.videos_dir = self.dataset_dir / "videos"
        self.audios_dir = self.dataset_dir / "audios"
        self.captions_dir = self.dataset_dir / "captions"
    
    def load_and_align_metadata(self, topic_name: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load both metadata files and ensure they're aligned.
        Match entries by video_id (YouTube ID), not video_number.
        Remove entries from enhanced that don't exist in base metadata.
        """
        caption_metadata_path = self.captions_dir / topic_name / "metadata.json"
        caption_enhanced_path = self.captions_dir / topic_name / "metadata_enhanced.json"
        
        if not caption_metadata_path.exists():
            print(f"    [ERROR] No metadata.json in captions/{topic_name}")
            return [], None
        
        # Load base metadata
        with open(caption_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get valid video_ids from base metadata (YouTube IDs)
        valid_video_ids = {v['video_id'] for v in metadata}
        print(f"    Base metadata has {len(valid_video_ids)} videos")
        
        # Load and align enhanced metadata if it exists
        enhanced_metadata = None
        if caption_enhanced_path.exists():
            with open(caption_enhanced_path, 'r') as f:
                enhanced_metadata = json.load(f)
            
            original_count = len(enhanced_metadata)
            
            # Filter enhanced metadata to only include video_ids present in base metadata
            enhanced_metadata = [v for v in enhanced_metadata if v['video_id'] in valid_video_ids]
            
            removed_count = original_count - len(enhanced_metadata)
            
            if removed_count > 0:
                print(f"    [ALIGN] Found {removed_count} orphaned entries in metadata_enhanced.json")
                print(f"    [ALIGN] Removing entries with video_ids not in base metadata...")
                
                # Now update video_numbers in enhanced to match base metadata
                # Create a mapping: video_id -> video_number from base metadata
                video_id_to_number = {v['video_id']: v['video_number'] for v in metadata}
                
                for entry in enhanced_metadata:
                    vid_id = entry['video_id']
                    if vid_id in video_id_to_number:
                        old_num = entry['video_number']
                        new_num = video_id_to_number[vid_id]
                        if old_num != new_num:
                            entry['video_number'] = new_num
                            print(f"      Updated video_id {vid_id}: video_number {old_num} -> {new_num}")
                
                # Save the corrected version immediately
                with open(caption_enhanced_path, 'w') as f:
                    json.dump(enhanced_metadata, f, indent=2)
                
                print(f"    ✓ metadata_enhanced.json aligned with metadata.json")
            else:
                print(f"    ✓ metadata_enhanced.json already aligned")
        
        return metadata, enhanced_metadata
    
    def find_videos_to_drop(self, metadata: List[Dict]) -> List[str]:
        """Find video_ids where CaptionQuality == 'replace'"""
        to_drop = []
        for video in metadata:
            caption_quality = video.get('CaptionQuality', '').lower()
            if caption_quality == 'replace':
                to_drop.append({
                    'video_id': video['video_id'],
                    'video_number': video['video_number']
                })
        
        return to_drop
    
    def drop_video_files(self, topic_name: str, video_number: str):
        """Delete video, audio, and caption files for a given video number"""
        video_file = self.videos_dir / topic_name / f"video_{video_number}.mp4"
        audio_file = self.audios_dir / topic_name / f"audio_{video_number}.m4a"
        caption_file = self.captions_dir / topic_name / f"caption_{video_number}.srt"
        
        deleted = []
        for file_path in [video_file, audio_file, caption_file]:
            if file_path.exists():
                file_path.unlink()
                deleted.append(file_path.name)
        
        if deleted:
            print(f"      Deleted: {', '.join(deleted)}")
    
    def remove_from_both_metadata(self, topic_name: str, metadata: List[Dict], 
                                    enhanced_metadata: List[Dict], 
                                    videos_to_drop: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Remove dropped videos from both metadata files using video_id"""
        caption_metadata_path = self.captions_dir / topic_name / "metadata.json"
        caption_enhanced_path = self.captions_dir / topic_name / "metadata_enhanced.json"
        
        # Create set of video_ids to drop
        dropped_video_ids = {v['video_id'] for v in videos_to_drop}
        
        # Filter base metadata by video_id
        new_metadata = [v for v in metadata if v['video_id'] not in dropped_video_ids]
        
        # Save base metadata
        with open(caption_metadata_path, 'w') as f:
            json.dump(new_metadata, f, indent=2)
        print(f"    ✓ Updated metadata.json")
        
        # Filter and save enhanced metadata if it exists
        new_enhanced_metadata = None
        if enhanced_metadata:
            new_enhanced_metadata = [v for v in enhanced_metadata if v['video_id'] not in dropped_video_ids]
            
            with open(caption_enhanced_path, 'w') as f:
                json.dump(new_enhanced_metadata, f, indent=2)
            print(f"    ✓ Updated metadata_enhanced.json")
        
        return new_metadata, new_enhanced_metadata
    
    def renumber_files_and_metadata(self, topic_name: str, metadata: List[Dict], 
                                     enhanced_metadata: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Renumber all files and update metadata sequentially (001, 002, ...)"""
        caption_metadata_path = self.captions_dir / topic_name / "metadata.json"
        caption_enhanced_path = self.captions_dir / topic_name / "metadata_enhanced.json"
        
        # Get current numbers sorted
        current_numbers = sorted([v['video_number'] for v in metadata])
        
        if not current_numbers:
            print(f"    [WARNING] No videos left after dropping!")
            return None, None
        
        # Create mapping: old video_number -> new video_number
        rename_map = {}
        for i, old_num in enumerate(current_numbers, 1):
            new_num = f"{i:03d}"
            if old_num != new_num:
                rename_map[old_num] = new_num
        
        if not rename_map:
            print(f"    No renumbering needed")
            return metadata, enhanced_metadata
        
        print(f"    Renumbering {len(rename_map)} videos...")
        
        # Rename files in all three directories
        for old_num, new_num in rename_map.items():
            # Video
            old_video = self.videos_dir / topic_name / f"video_{old_num}.mp4"
            new_video = self.videos_dir / topic_name / f"video_{new_num}.mp4"
            if old_video.exists():
                old_video.rename(new_video)
            
            # Audio
            old_audio = self.audios_dir / topic_name / f"audio_{old_num}.m4a"
            new_audio = self.audios_dir / topic_name / f"audio_{new_num}.m4a"
            if old_audio.exists():
                old_audio.rename(new_audio)
            
            # Caption
            old_caption = self.captions_dir / topic_name / f"caption_{old_num}.srt"
            new_caption = self.captions_dir / topic_name / f"caption_{new_num}.srt"
            if old_caption.exists():
                old_caption.rename(new_caption)
            
            print(f"      {old_num} -> {new_num}")
        
        # Update base metadata with new video_numbers
        for video in metadata:
            old_num = video['video_number']
            if old_num in rename_map:
                video['video_number'] = rename_map[old_num]
        
        # Update enhanced metadata with new video_numbers
        # Match by video_id to ensure correctness
        if enhanced_metadata:
            # Create mapping: video_id -> new video_number from base metadata
            video_id_to_new_number = {v['video_id']: v['video_number'] for v in metadata}
            
            for video in enhanced_metadata:
                vid_id = video['video_id']
                if vid_id in video_id_to_new_number:
                    video['video_number'] = video_id_to_new_number[vid_id]
        
        # Save updated base metadata
        with open(caption_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save updated enhanced metadata
        if enhanced_metadata:
            with open(caption_enhanced_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2)
        
        # Update needs_whisper.txt
        needs_whisper_file = self.captions_dir / topic_name / "needs_whisper.txt"
        if needs_whisper_file.exists():
            with open(needs_whisper_file, 'r') as f:
                entries = [line.strip() for line in f if line.strip()]
            
            new_entries = []
            for entry in entries:
                old_num = entry.replace('audio_', '').replace('.m4a', '')
                if old_num in rename_map:
                    new_num = rename_map[old_num]
                    new_entries.append(f"audio_{new_num}.m4a")
                else:
                    new_entries.append(entry)
            
            with open(needs_whisper_file, 'w') as f:
                for entry in sorted(new_entries):
                    f.write(f"{entry}\n")
        
        return metadata, enhanced_metadata
    
    def sync_metadata_to_other_dirs(self, topic_name: str, metadata: List[Dict], 
                                     enhanced_metadata: List[Dict] = None):
        """Copy metadata from captions to videos and audios directories"""
        # Sync metadata.json to videos
        video_metadata_path = self.videos_dir / topic_name / "metadata.json"
        with open(video_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"    ✓ Synced metadata.json to videos/")
        
        # Sync metadata.json to audios
        audio_metadata_path = self.audios_dir / topic_name / "metadata.json"
        with open(audio_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"    ✓ Synced metadata.json to audios/")
        
        # Sync metadata_enhanced.json if it exists
        if enhanced_metadata:
            # Sync to videos
            video_enhanced_path = self.videos_dir / topic_name / "metadata_enhanced.json"
            with open(video_enhanced_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2)
            print(f"    ✓ Synced metadata_enhanced.json to videos/")
            
            # Sync to audios
            audio_enhanced_path = self.audios_dir / topic_name / "metadata_enhanced.json"
            with open(audio_enhanced_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2)
            print(f"    ✓ Synced metadata_enhanced.json to audios/")
    
    def process_topic(self, topic_name: str):
        """Process a single topic: align, find replaced videos, drop them, renumber, sync"""
        print(f"\n{'='*70}")
        print(f"Processing: {topic_name}")
        print(f"{'='*70}")
        
        # Load and align both metadata files (match by video_id)
        print(f"  [LOAD] Loading and aligning metadata files by video_id...")
        metadata, enhanced_metadata = self.load_and_align_metadata(topic_name)
        
        if not metadata:
            print(f"  [SKIP] No metadata found")
            return
        
        # Find videos to drop
        to_drop = self.find_videos_to_drop(metadata)
        
        if not to_drop:
            print(f"  ✓ No videos marked for replacement")
            
            # Still sync metadata to ensure consistency across directories
            print(f"\n  [SYNC] Synchronizing metadata to videos/ and audios/...")
            self.sync_metadata_to_other_dirs(topic_name, metadata, enhanced_metadata)
            
            return
        
        print(f"\n  Found {len(to_drop)} video(s) to drop:")
        for v in to_drop:
            print(f"    - Video {v['video_number']} (ID: {v['video_id']}, CaptionQuality = 'replace')")
        
        # Drop the video files
        print(f"\n  [DROP] Deleting files...")
        for v in to_drop:
            print(f"    Video {v['video_number']} (ID: {v['video_id']}):")
            self.drop_video_files(topic_name, v['video_number'])
        
        # Remove from both metadata files (match by video_id)
        print(f"\n  [UPDATE] Removing from metadata files (matching by video_id)...")
        metadata, enhanced_metadata = self.remove_from_both_metadata(
            topic_name, metadata, enhanced_metadata, to_drop
        )
        print(f"    ✓ Removed {len(to_drop)} entries")
        print(f"    Remaining videos: {len(metadata)}")
        
        # Renumber files and metadata
        print(f"\n  [RENUMBER] Renumbering remaining videos...")
        metadata, enhanced_metadata = self.renumber_files_and_metadata(
            topic_name, metadata, enhanced_metadata
        )
        
        if metadata:
            # Sync to other directories
            print(f"\n  [SYNC] Synchronizing metadata to videos/ and audios/...")
            self.sync_metadata_to_other_dirs(topic_name, metadata, enhanced_metadata)
        
        print(f"\n  ✓ Topic processed: {len(to_drop)} dropped, {len(metadata)} remaining")

def main():
    base_dir = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset"
    
    processor = MetadataSyncAndDrop(base_dir)
    
    # Get all topics from captions directory
    captions_dir = processor.captions_dir
    topics = sorted([d.name for d in captions_dir.iterdir() if d.is_dir()])
    
    print(f"\n{'='*70}")
    print(f"Sync Metadata and Drop Replaced Videos")
    print(f"{'='*70}")
    print(f"Base directory: {base_dir}")
    print(f"Topics to process: {len(topics)}")
    print(f"Source of truth: captions/TOPIC/metadata.json")
    print(f"Matching strategy: video_id (YouTube ID)")
    print(f"Also syncing: metadata_enhanced.json")
    print(f"Action: Align by video_id, drop 'replace' videos, renumber, sync")
    print(f"{'='*70}")
    
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