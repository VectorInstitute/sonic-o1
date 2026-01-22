"""Topic Parsing and Video Download Script.

IMPORTANT PREREQUISITE:
Before running this script, you MUST:
1. Create a 'videos_QualityAnnotated' directory in your base directory
2. Structure it following the template in ../huggingface_review_template/
3. Manually review and quality-annotate metadata from
   youtube_metadata_scraper.py output
4. Ensure each video in the metadata has a 'Qualitylabel' field set to
   "Good" for videos that should be included in the final dataset

This script filters videos based on:
- Qualitylabel == "Good"
- copyright_notice == "creativeCommon"
- Language: English (default_language or default_audio_language)

The script will:
- Load metadata from videos_QualityAnnotated/<topic>/<topic>_metadata.json
- Download filtered videos, extract audio, and download captions
- Create a balanced dataset across demographics and durations
- Track videos needing Whisper transcription in needs_whisper.txt files
"""
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import yaml
import yt_dlp


class VideoDatasetProcessor:
    """Process video datasets with quality filtering and downloading."""

    def __init__(self, base_dir: str, max_count: int = 25):
        """Initialize the processor.

        Args:
            base_dir: Base directory for the dataset.
            max_count: Maximum number of videos per topic.
        """
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "dataset"
        self.videos_dir = self.dataset_dir / "videos"
        self.audios_dir = self.dataset_dir / "audios"
        self.captions_dir = self.dataset_dir / "captions"
        self.max_count = max_count

    def create_directories(self, topic_name: str):
        """Create directory structure for a topic.

        Args:
            topic_name: Name of the topic.
        """
        for base in [self.videos_dir, self.audios_dir, self.captions_dir]:
            topic_dir = base / topic_name
            topic_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directories for {topic_name}")

    def get_existing_video_info(
        self, topic_name: str
    ) -> tuple[int, Set[str], int]:
        """Get information about existing videos in a topic.

        Args:
            topic_name: Name of the topic.

        Returns:
            Tuple of (current_count, existing_video_ids, next_video_number).
        """
        metadata_path = self.videos_dir / topic_name / "metadata.json"

        if not metadata_path.exists():
            return 0, set(), 1

        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)

            current_count = len(existing_metadata)
            existing_video_ids = {
                video['video_id'] for video in existing_metadata
            }

            video_numbers = []
            for video in existing_metadata:
                video_num_str = video.get('video_number', '000')
                try:
                    video_numbers.append(int(video_num_str))
                except ValueError:
                    continue

            next_number = (
                max(video_numbers) + 1 if video_numbers
                else current_count + 1
            )

            print(
                f"  Found {current_count} existing videos, "
                f"next number will be {next_number:03d}"
            )
            return current_count, existing_video_ids, next_number

        except Exception as e:
            print(f"  Warning: Could not read existing metadata: {e}")
            return 0, set(), 1

    def load_metadata(self, metadata_path: str) -> List[Dict]:
        """Load metadata JSON file.

        Args:
            metadata_path: Path to metadata file.

        Returns:
            List of video metadata dictionaries.
        """
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def filter_videos(
        self, metadata: List[Dict], existing_video_ids: Set[str]
    ) -> List[Dict]:
        """Filter videos based on criteria and exclude downloaded videos.

        Args:
            metadata: List of video metadata.
            existing_video_ids: Set of already-downloaded video IDs.

        Returns:
            Filtered list of video metadata.
        """
        filtered = []
        excluded_existing = 0

        for video in metadata:
            if video.get('video_id') in existing_video_ids:
                excluded_existing += 1
                continue

            if video.get('Qualitylabel') != 'Good':
                continue

            if video.get('copyright_notice') != 'creativeCommon':
                continue

            default_lang = video.get('default_language', '').lower()
            audio_lang = video.get('default_audio_language', '').lower()
            if default_lang != 'en' and audio_lang != 'en':
                continue

            filtered.append(video)

        print(
            f"✓ Filtered {len(filtered)} new videos from "
            f"{len(metadata)} total"
        )
        print(f"  (Excluded {excluded_existing} already-downloaded videos)")
        return filtered

    def select_videos(
        self, filtered_videos: List[Dict], needed_count: int,
        start_number: int
    ) -> List[Dict]:
        """Select videos with maximum diversity.

        Args:
            filtered_videos: List of filtered video metadata.
            needed_count: Number of videos needed.
            start_number: Starting video number.

        Returns:
            Selected videos with diversity across demographics and duration.
        """
        if len(filtered_videos) == 0:
            print("✓ No new videos to select")
            return []

        if len(filtered_videos) <= needed_count:
            selected = filtered_videos
            print(f"✓ Selected all {len(selected)} available videos")
        else:
            print(
                f"  Selecting {needed_count} from {len(filtered_videos)} "
                f"videos with diversity optimization..."
            )

            duration_demo_groups = defaultdict(lambda: defaultdict(list))

            for video in filtered_videos:
                duration_cat = video.get('duration_category', 'unknown')
                demo_label = video.get('demographic_label', 'general')
                duration_demo_groups[duration_cat][demo_label].append(video)

            target_distribution = {
                'short': int(needed_count * 0.4),
                'medium': int(needed_count * 0.4),
                'long': int(needed_count * 0.2),
            }

            available_counts = {
                cat: sum(len(demos) for demos in groups.values())
                for cat, groups in duration_demo_groups.items()
            }

            print(
                f"  Available: Short={available_counts.get('short', 0)}, "
                f"Medium={available_counts.get('medium', 0)}, "
                f"Long={available_counts.get('long', 0)}"
            )

            adjusted_targets = {}
            for cat in ['short', 'medium', 'long']:
                available = available_counts.get(cat, 0)
                target = target_distribution.get(cat, 0)
                adjusted_targets[cat] = min(target, available)

            total_assigned = sum(adjusted_targets.values())
            remaining = needed_count - total_assigned

            if remaining > 0:
                for cat in ['medium', 'short', 'long']:
                    available = available_counts.get(cat, 0)
                    current = adjusted_targets.get(cat, 0)
                    can_add = min(remaining, available - current)
                    if can_add > 0:
                        adjusted_targets[cat] = (
                            adjusted_targets.get(cat, 0) + can_add
                        )
                        remaining -= can_add
                    if remaining == 0:
                        break

            print(
                f"  Target: Short={adjusted_targets.get('short', 0)}, "
                f"Medium={adjusted_targets.get('medium', 0)}, "
                f"Long={adjusted_targets.get('long', 0)}"
            )

            selected = []

            for duration_cat in ['short', 'medium', 'long']:
                target_count = adjusted_targets.get(duration_cat, 0)
                if target_count == 0:
                    continue

                demo_groups = duration_demo_groups.get(duration_cat, {})
                if not demo_groups:
                    continue

                demo_labels = list(demo_groups.keys())

                videos_selected = 0
                demo_index = 0

                while videos_selected < target_count:
                    demo_label = demo_labels[demo_index % len(demo_labels)]

                    if demo_groups[demo_label]:
                        video = demo_groups[demo_label].pop(0)
                        selected.append(video)
                        videos_selected += 1

                    demo_index += 1

                    if all(
                        len(videos) == 0
                        for videos in demo_groups.values()
                    ):
                        break

            if len(selected) < needed_count:
                remaining_videos = []
                for duration_cat in duration_demo_groups.values():
                    for demo_videos in duration_cat.values():
                        remaining_videos.extend(demo_videos)

                needed_more = needed_count - len(selected)
                selected.extend(remaining_videos[:needed_more])

            print(f"✓ Selected {len(selected)} videos with diversity")

        for idx, video in enumerate(selected):
            video['video_number'] = f"{start_number + idx:03d}"

        duration_counts = defaultdict(int)
        demo_counts = defaultdict(int)

        for video in selected:
            duration_counts[video.get('duration_category', 'unknown')] += 1
            demo_counts[video.get('demographic_label', 'general')] += 1

        print("  Final distribution:")
        print(f"    Duration: {dict(duration_counts)}")
        print(f"    Demographics: {dict(demo_counts)}")

        return selected

    def download_video(self, video_id: str, output_path: str) -> bool:
        """Download video using yt-dlp (max 1080p).

        Args:
            video_id: YouTube video ID.
            output_path: Path to save video.

        Returns:
            True if successful, False otherwise.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        cookies_path = self.base_dir / "cookies.txt"
        ydl_opts = {
            'format': (
                'bestvideo[height<=1080][ext=mp4]+'
                'bestaudio[ext=m4a]/best[height<=1080][ext=mp4]'
            ),
            'outtmpl': output_path,
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'cookiefile': str(cookies_path),
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
            print(f" Failed to download {video_id}: {e}")
            return False

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video using ffmpeg.

        Args:
            video_path: Path to video file.
            audio_path: Path to save audio.

        Returns:
            True if successful, False otherwise.
        """
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
        except FileNotFoundError:
            print(
                "  ✗ ffmpeg not found. "
                "Install with: conda install -c conda-forge ffmpeg"
            )
            return False
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to extract audio: {e}")
            return False

    def download_captions(self, video_id: str, output_path: str) -> bool:
        """Download YouTube captions if available using yt-dlp.

        Args:
            video_id: YouTube video ID.
            output_path: Path to save captions.

        Returns:
            True if successful, False otherwise.
        """
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
            'cookiefile': str(cookies_path),
            'extractor_args': {
                'youtube': {
                    'player_client': ['default', '-tv']
                }
            },
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            possible_paths = [
                output_path,
                f"{output_base}.en.srt",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    if path != output_path:
                        os.rename(path, output_path)
                    return True

            return False
        except Exception:
            return False

    def merge_metadata(
        self, topic_name: str, new_videos: List[Dict]
    ) -> List[Dict]:
        """Merge new videos with existing metadata.

        Args:
            topic_name: Name of the topic.
            new_videos: List of new video metadata.

        Returns:
            Combined metadata list.
        """
        metadata_path = self.videos_dir / topic_name / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)

            combined = existing_metadata + new_videos
            print(
                f"  Merged {len(existing_metadata)} existing + "
                f"{len(new_videos)} new = {len(combined)} total videos"
            )
            return combined
        else:
            return new_videos

    def process_topic(self, topic_metadata_path: str, force: bool = False):
        """Main processing function for a topic.

        Args:
            topic_metadata_path: Path to topic metadata file.
            force: If True, process even if at max count.
        """
        topic_path = Path(topic_metadata_path).parent
        topic_name = topic_path.name

        print(f"\n{'='*60}")
        print(f"Processing Topic: {topic_name}")
        print(f"{'='*60}\n")

        (current_count, existing_video_ids,
         next_number) = self.get_existing_video_info(topic_name)

        if current_count >= self.max_count and not force:
            print(
                f"Topic already has {current_count}/{self.max_count} "
                f"videos - SKIPPING"
            )
            return

        needed_count = self.max_count - current_count
        print(f"  Current: {current_count}/{self.max_count} videos")
        print(f"  Need to add: {needed_count} videos\n")

        self.create_directories(topic_name)

        metadata = self.load_metadata(topic_metadata_path)
        filtered = self.filter_videos(metadata, existing_video_ids)

        if len(filtered) == 0:
            print("No new videos meet the criteria!")
            return

        selected = self.select_videos(filtered, needed_count, next_number)

        if len(selected) == 0:
            print("No videos were selected!")
            return

        needs_whisper = []

        whisper_file = self.captions_dir / topic_name / "needs_whisper.txt"
        if whisper_file.exists():
            with open(whisper_file, 'r') as f:
                needs_whisper = [line.strip() for line in f.readlines()]

        for video in selected:
            video_id = video['video_id']
            video_num = video['video_number']

            print(
                f"\n--- Processing video {video_num}/{self.max_count}: "
                f"{video_id} ---"
            )

            video_path = (
                self.videos_dir / topic_name / f"video_{video_num}.mp4"
            )
            audio_path = (
                self.audios_dir / topic_name / f"audio_{video_num}.m4a"
            )
            caption_path = (
                self.captions_dir / topic_name / f"caption_{video_num}.srt"
            )

            print("Downloading video...")
            if self.download_video(video_id, str(video_path)):
                print(f"Video downloaded: {video_path.name}")

                print("Extracting audio...")
                if self.extract_audio(str(video_path), str(audio_path)):
                    print(f"Audio extracted: {audio_path.name}")
                else:
                    print("Audio extraction failed")

                print("  Downloading captions...")
                if self.download_captions(video_id, str(caption_path)):
                    print(f"Captions downloaded: {caption_path.name}")
                else:
                    print("No captions available, adding to Whisper queue")
                    needs_whisper.append(f"audio_{video_num}.m4a")
            else:
                print("Video download failed, skipping")

        merged_metadata = self.merge_metadata(topic_name, selected)

        print(f"\n{'='*60}")
        print("Saving metadata files...")
        for base_dir in [
            self.videos_dir, self.audios_dir, self.captions_dir
        ]:
            metadata_path = base_dir / topic_name / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(merged_metadata, f, indent=2)
            print(f"✓ Saved: {metadata_path}")

        if needs_whisper:
            with open(whisper_file, 'w') as f:
                for audio_file in needs_whisper:
                    f.write(f"{audio_file}\n")
            print(f"✓ Saved Whisper queue: {whisper_file}")

        print(f"\n{'='*60}")
        print(f"✓ Topic {topic_name} processing complete!")
        print(f"  Total videos now: {len(merged_metadata)}/{self.max_count}")
        print(f"  New videos added: {len(selected)}")
        print(f"  Needs Whisper: {len(needs_whisper)}")
        print(f"{'='*60}\n")

    def generate_summary_from_metadata(self, topic_name: str):
        """Generate summary.json from existing metadata.json file.

        Args:
            topic_name: Name of the topic.

        Returns:
            Summary dictionary or None if error.
        """
        print(f"\nGenerating summary for: {topic_name}")

        metadata_path = self.videos_dir / topic_name / "metadata.json"

        if not metadata_path.exists():
            print(f"✗ Metadata file not found: {metadata_path}")
            return None

        with open(metadata_path, 'r') as f:
            selected_videos = json.load(f)

        print(f"✓ Loaded {len(selected_videos)} videos from metadata")

        whisper_file = self.captions_dir / topic_name / "needs_whisper.txt"
        needs_whisper = []
        if whisper_file.exists():
            with open(whisper_file, 'r') as f:
                needs_whisper = [
                    line.strip().replace('audio_', '').replace('.m4a', '')
                    for line in f.readlines()
                ]

        duration_counts = defaultdict(int)
        demo_counts = defaultdict(int)
        total_duration_seconds = 0

        for video in selected_videos:
            duration_counts[video.get('duration_category', 'unknown')] += 1
            demo_counts[video.get('demographic_label', 'general')] += 1
            total_duration_seconds += video.get('duration_seconds', 0)

        avg_duration_seconds = (
            total_duration_seconds / len(selected_videos)
            if selected_videos else 0
        )

        summary = {
            "topic_name": topic_name,
            "processing_timestamp": datetime.now().isoformat(),
            "statistics": {
                "selected_videos_count": len(selected_videos),
                "videos_with_captions": (
                    len(selected_videos) - len(needs_whisper)
                ),
                "videos_needing_whisper": len(needs_whisper),
                "total_duration_seconds": total_duration_seconds,
                "total_duration_minutes": round(
                    total_duration_seconds / 60, 2
                ),
                "average_duration_seconds": round(avg_duration_seconds, 2),
                "average_duration_minutes": round(
                    avg_duration_seconds / 60, 2
                )
            },
            "distribution": {
                "by_duration": dict(duration_counts),
                "by_demographics": dict(demo_counts)
            },
            "duration_percentages": {
                cat: round((count / len(selected_videos)) * 100, 1)
                for cat, count in duration_counts.items()
            },
            "demographic_percentages": {
                demo: round((count / len(selected_videos)) * 100, 1)
                for demo, count in demo_counts.items()
            },
            "needs_whisper_list": needs_whisper,
            "video_ids": [video['video_id'] for video in selected_videos]
        }

        for base_dir in [
            self.videos_dir, self.audios_dir, self.captions_dir
        ]:
            summary_path = base_dir / topic_name / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Saved: {summary_path}")

        print("✓ Summary generated successfully")
        return summary


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_dir = config['directories']['base_dir']

    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.path.dirname(__file__), base_dir)

    MAX_COUNT = config['max_videos_per_topic']['max_videos_per_topic']

    processor = VideoDatasetProcessor(base_dir, max_count=MAX_COUNT)

    quality_annotated_dir = os.path.join(
        base_dir, "videos_QualityAnnotated"
    )

    if not os.path.exists(quality_annotated_dir):
        print(f"\n{'='*60}")
        print("ERROR: videos_QualityAnnotated directory not found!")
        print(f"{'='*60}")
        print(f"\nExpected location: {quality_annotated_dir}")
        print("\nBEFORE RUNNING THIS SCRIPT, YOU MUST:")
        print("1. Create the 'videos_QualityAnnotated' directory")
        print("2. Review metadata from youtube_metadata_scraper.py output")
        print("3. Add 'Qualitylabel' field to videos you want to include")
        print("4. Structure it following ../huggingface_review_template/")
        print("\nSee README.md for detailed instructions.")
        print(f"{'='*60}\n")
        sys.exit(1)

    topic_dirs = sorted([
        d for d in os.listdir(quality_annotated_dir)
        if os.path.isdir(os.path.join(quality_annotated_dir, d))
    ])

    print(f"\n{'='*60}")
    print(
        f"Found {len(topic_dirs)} topics in videos_QualityAnnotated"
    )
    print(f"Max videos per topic: {MAX_COUNT}")
    print(f"{'='*60}\n")

    topics_processed = 0
    topics_skipped_full = 0
    topics_skipped_no_metadata = 0
    topics_extended = 0

    for topic_dir in topic_dirs:
        current_count, _, _ = processor.get_existing_video_info(topic_dir)

        if current_count >= MAX_COUNT:
            print(
                f"\n⊘ Skipping {topic_dir} - already at max "
                f"({current_count}/{MAX_COUNT})"
            )
            topics_skipped_full += 1
            continue

        topic_path = os.path.join(quality_annotated_dir, topic_dir)
        metadata_files = [
            f for f in os.listdir(topic_path)
            if f.endswith('_metadata.json')
        ]

        if not metadata_files:
            print(f"\n⊘ Skipping {topic_dir} - no metadata file found")
            topics_skipped_no_metadata += 1
            continue

        metadata_path = os.path.join(topic_path, metadata_files[0])

        try:
            if current_count > 0:
                print(
                    f"\nEXTENDING {topic_dir} "
                    f"(has {current_count}/{MAX_COUNT})"
                )
                topics_extended += 1
            else:
                print(f"\nCREATING {topic_dir}")

            processor.process_topic(metadata_path)
            processor.generate_summary_from_metadata(topic_dir)
            topics_processed += 1

        except Exception as e:
            print(f"\n✗ Error processing {topic_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Topics processed: {topics_processed}")
    print(f"  - New topics created: {topics_processed - topics_extended}")
    print(f"  - Existing topics extended: {topics_extended}")
    print(f"Topics skipped (already full): {topics_skipped_full}")
    print(f"Topics skipped (no metadata): {topics_skipped_no_metadata}")
    print(f"{'='*60}\n")