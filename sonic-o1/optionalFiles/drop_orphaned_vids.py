#!/usr/bin/env python3
"""
Cleanup orphaned files across videos, audios, and captions directories
Syncs everything to match the metadata.json files
Removes files that don't have corresponding metadata entries
"""
import json
from pathlib import Path
from typing import Set, List


class DatasetCleaner:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "dataset"
        self.videos_dir = self.dataset_dir / "videos"
        self.audios_dir = self.dataset_dir / "audios"
        self.captions_dir = self.dataset_dir / "captions"
    
    def get_valid_video_numbers(self, topic_name: str) -> Set[str]:
        """Get set of valid video numbers from metadata (source of truth)"""
        # Use videos metadata as source of truth
        metadata_path = self.videos_dir / topic_name / "metadata.json"
        
        if not metadata_path.exists():
            print(f"    [WARNING] No metadata found in videos/{topic_name}")
            return set()
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {video['video_number'] for video in metadata}
    
    def find_orphaned_files(self, topic_name: str, valid_numbers: Set[str]) -> dict:
        """Find files that don't have corresponding metadata entries"""
        orphaned = {
            'videos': [],
            'audios': [],
            'captions': []
        }
        
        # Check videos
        video_dir = self.videos_dir / topic_name
        if video_dir.exists():
            for video_file in sorted(video_dir.glob("video_*.mp4")):
                num = video_file.name.replace('video_', '').replace('.mp4', '')
                if num not in valid_numbers:
                    orphaned['videos'].append(video_file)
        
        # Check audios
        audio_dir = self.audios_dir / topic_name
        if audio_dir.exists():
            for audio_file in sorted(audio_dir.glob("audio_*.m4a")):
                num = audio_file.name.replace('audio_', '').replace('.m4a', '')
                if num not in valid_numbers:
                    orphaned['audios'].append(audio_file)
        
        # Check captions
        caption_dir = self.captions_dir / topic_name
        if caption_dir.exists():
            for caption_file in sorted(caption_dir.glob("caption_*.srt")):
                num = caption_file.name.replace('caption_', '').replace('.srt', '')
                if num not in valid_numbers:
                    orphaned['captions'].append(caption_file)
        
        return orphaned
    
    def find_missing_files(self, topic_name: str, valid_numbers: Set[str]) -> dict:
        """Find metadata entries that don't have corresponding files"""
        missing = {
            'videos': [],
            'audios': [],
            'captions': []
        }
        
        for num in sorted(valid_numbers):
            # Check video
            video_file = self.videos_dir / topic_name / f"video_{num}.mp4"
            if not video_file.exists():
                missing['videos'].append(num)
            
            # Check audio
            audio_file = self.audios_dir / topic_name / f"audio_{num}.m4a"
            if not audio_file.exists():
                missing['audios'].append(num)
            
            # Check caption (optional - might not exist yet)
            caption_file = self.captions_dir / topic_name / f"caption_{num}.srt"
            if not caption_file.exists():
                missing['captions'].append(num)
        
        return missing
    
    def sync_metadata_files(self, topic_name: str):
        """Sync metadata.json across all three directories to match videos/metadata.json
        Preserves CaptionQuality labels from captions metadata"""
        source_metadata_path = self.videos_dir / topic_name / "metadata.json"
        
        if not source_metadata_path.exists():
            print(f"    [ERROR] No source metadata in videos/{topic_name}")
            return
        
        with open(source_metadata_path, 'r') as f:
            source_metadata = json.load(f)
        
        print(f"    Source metadata has {len(source_metadata)} videos")
        
        # Copy to audios (simple copy)
        audio_metadata_path = self.audios_dir / topic_name / "metadata.json"
        if audio_metadata_path.exists():
            with open(audio_metadata_path, 'w') as f:
                json.dump(source_metadata, f, indent=2)
            print(f"    ✓ Synced metadata to audios/")
        
        # Copy to captions (preserve CaptionQuality labels)
        caption_metadata_path = self.captions_dir / topic_name / "metadata.json"
        if caption_metadata_path.exists():
            # Load existing captions metadata to preserve CaptionQuality
            with open(caption_metadata_path, 'r') as f:
                existing_caption_metadata = json.load(f)
            
            # Create mapping: video_number -> CaptionQuality
            caption_quality_map = {}
            for video in existing_caption_metadata:
                video_num = video['video_number']
                if 'CaptionQuality' in video:
                    caption_quality_map[video_num] = video['CaptionQuality']
            
            # Copy source metadata and restore CaptionQuality labels
            new_caption_metadata = []
            for video in source_metadata:
                video_copy = video.copy()
                video_num = video_copy['video_number']
                
                # Restore CaptionQuality if it existed
                if video_num in caption_quality_map:
                    video_copy['CaptionQuality'] = caption_quality_map[video_num]
                
                new_caption_metadata.append(video_copy)
            
            with open(caption_metadata_path, 'w') as f:
                json.dump(new_caption_metadata, f, indent=2)
            
            preserved = len([v for v in new_caption_metadata if 'CaptionQuality' in v])
            print(f"    ✓ Synced metadata to captions/ (preserved {preserved} CaptionQuality labels)")
    
    def clean_orphaned_files(self, orphaned: dict) -> int:
        """Delete orphaned files"""
        total_deleted = 0
        
        for category, files in orphaned.items():
            if files:
                print(f"\n    Deleting {len(files)} orphaned {category}:")
                for file in files:
                    print(f"      - {file.name}")
                    file.unlink()
                    total_deleted += 1
        
        return total_deleted
    
    def update_needs_whisper(self, topic_name: str, valid_numbers: Set[str]):
        """Clean needs_whisper.txt to only include valid video numbers"""
        needs_whisper_file = self.captions_dir / topic_name / "needs_whisper.txt"
        
        if not needs_whisper_file.exists():
            return
        
        with open(needs_whisper_file, 'r') as f:
            entries = [line.strip() for line in f if line.strip()]
        
        # Filter to only valid numbers
        valid_entries = []
        for entry in entries:
            num = entry.replace('audio_', '').replace('.m4a', '')
            if num in valid_numbers:
                valid_entries.append(entry)
        
        if len(valid_entries) != len(entries):
            with open(needs_whisper_file, 'w') as f:
                for entry in sorted(valid_entries):
                    f.write(f"{entry}\n")
            
            removed = len(entries) - len(valid_entries)
            print(f"    ✓ Cleaned needs_whisper.txt (removed {removed} invalid entries)")
    
    def process_topic(self, topic_name: str, delete_orphans: bool = True):
        """Clean up a single topic"""
        print(f"\n{'='*70}")
        print(f"Cleaning: {topic_name}")
        print(f"{'='*70}")
        
        # Get valid video numbers from metadata
        valid_numbers = self.get_valid_video_numbers(topic_name)
        
        if not valid_numbers:
            print(f"  [SKIP] No valid metadata")
            return
        
        print(f"  Valid video numbers from metadata: {sorted(valid_numbers)}")
        print(f"  Total: {len(valid_numbers)} videos")
        
        # Sync metadata files first
        print(f"\n  [SYNC] Synchronizing metadata files...")
        self.sync_metadata_files(topic_name)
        
        # Find orphaned files
        print(f"\n  [CHECK] Finding orphaned files...")
        orphaned = self.find_orphaned_files(topic_name, valid_numbers)
        
        total_orphaned = sum(len(files) for files in orphaned.values())
        
        if total_orphaned == 0:
            print(f"    ✓ No orphaned files found")
        else:
            print(f"    Found {total_orphaned} orphaned files:")
            for category, files in orphaned.items():
                if files:
                    nums = [f.name.split('_')[1].split('.')[0] for f in files]
                    print(f"      {category}: {nums}")
            
            if delete_orphans:
                print(f"\n  [DELETE] Removing orphaned files...")
                deleted = self.clean_orphaned_files(orphaned)
                print(f"    ✓ Deleted {deleted} orphaned files")
        
        # Find missing files
        print(f"\n  [CHECK] Finding missing files...")
        missing = self.find_missing_files(topic_name, valid_numbers)
        
        total_missing = sum(len(nums) for nums in missing.values())
        
        if total_missing == 0:
            print(f"    ✓ No missing files")
        else:
            print(f"    Found {total_missing} missing files:")
            for category, nums in missing.items():
                if nums:
                    print(f"      {category}: {nums}")
            print(f"    [WARNING] These videos are in metadata but files are missing!")
        
        # Clean needs_whisper.txt
        print(f"\n  [CLEAN] Updating needs_whisper.txt...")
        self.update_needs_whisper(topic_name, valid_numbers)
        
        print(f"\n  ✓ Cleanup complete for {topic_name}")


def main():
    base_dir = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset"
    
    cleaner = DatasetCleaner(base_dir)
    
    # Get all topics
    videos_dir = cleaner.videos_dir
    topics = sorted([d.name for d in videos_dir.iterdir() if d.is_dir()])
    
    print(f"\n{'='*70}")
    print(f"Dataset Cleanup - Sync and Remove Orphaned Files")
    print(f"{'='*70}")
    print(f"Base directory: {base_dir}")
    print(f"Topics to clean: {len(topics)}")
    print(f"Source of truth: videos/TOPIC/metadata.json")
    print(f"{'='*70}")
    
    for topic in topics:
        try:
            cleaner.process_topic(topic, delete_orphans=True)
        except Exception as e:
            print(f"\n[ERROR] Failed to clean {topic}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("All topics cleaned!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()