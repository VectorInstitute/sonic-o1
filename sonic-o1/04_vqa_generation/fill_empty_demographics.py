#!/usr/bin/env python3
"""fill_empty_demographics.py.

Fill empty demographics arrays in task1, task2, and task3 VQA JSON files
using DemographicsExpander with Gemini. Task 3 generates per-segment;
Task 2 reuses Task 3 demographics; Task 1 generates for full videos.

Usage:
    python fill_empty_demographics.py --config vqa_config.yaml
    python fill_empty_demographics.py --config vqa_config.yaml --dry-run
    python fill_empty_demographics.py --topics 10,11 --dry-run

Author: SONIC-O1 Team
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Load environment variables
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
except ImportError:
    print("python-dotenv not installed; set GEMINI_API_KEY in environment")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fill_demographics.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration wrapper."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If relative, resolved relative
            to this script's directory.

    Returns
    -------
        Config object with nested attribute access.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).parent / config_path

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


class DemographicsFiller:
    """Fill empty demographics in VQA files."""

    def __init__(self, config: Config, dry_run: bool = False):
        """
        Initialize demographics filler.

        Args:
            config: Configuration object
            dry_run: If True, only report what would be done (no changes).
        """
        self.config = config
        self.dry_run = dry_run
        self.stats = {
            "task1": {"total": 0, "empty": 0, "filled": 0, "failed": 0, "reused": 0},
            "task2": {"total": 0, "empty": 0, "filled": 0, "failed": 0, "reused": 0},
            "task3": {"total": 0, "empty": 0, "filled": 0, "failed": 0, "reused": 0},
        }

        # Lazy import: only load when not dry-run (env-dependent, avoid heavy deps)
        if not dry_run:
            from models.base_gemini import BaseGeminiClient  # noqa: PLC0415
            from utils.demographics_expander import DemographicsExpander  # noqa: PLC0415
            from utils.video_segmenter import VideoSegmenter  # noqa: PLC0415

            self.demographics_expander = DemographicsExpander(config)
            self.segmenter = VideoSegmenter(config)
            self.gemini_client = BaseGeminiClient(config)

        # Load all metadata
        self.metadata_by_topic = self._load_all_metadata()

        # Cache: video_id -> {segment_key -> demographics}
        self.task3_demographics_cache = {}

    def _load_all_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Load metadata_enhanced.json for all topics."""
        metadata_map = {}
        dataset_root = Path(self.config.paths.dataset_root)
        videos_dir = dataset_root / "videos"

        if not videos_dir.exists():
            logger.error(f"Videos directory not found: {videos_dir}")
            return {}

        for topic_dir in sorted(videos_dir.iterdir()):
            if topic_dir.is_dir() and topic_dir.name[0].isdigit():
                parts = topic_dir.name.split("_", 1)
                if len(parts) == 2:
                    topic_id = int(parts[0])

                    metadata_file = topic_dir / "metadata_enhanced.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, "r") as f:
                                metadata_list = json.load(f)

                            # Create map by video_id
                            topic_metadata = {}
                            for meta in metadata_list:
                                video_id = meta.get(
                                    "video_id", meta.get("video_number")
                                )
                                topic_metadata[video_id] = meta

                            metadata_map[topic_id] = topic_metadata
                            logger.info(
                                "Loaded metadata for topic %s: %d videos",
                                topic_id,
                                len(topic_metadata),
                            )
                        except Exception as e:
                            logger.error(
                                "Failed to load metadata for topic %s: %s",
                                topic_id,
                                e,
                            )

        return metadata_map

    def _load_task3_demographics(self, topic_id: int):
        """Load all Task 3 demographics for a topic into cache."""
        if topic_id in self.task3_demographics_cache:
            return  # Already loaded

        output_dir = Path(self.config.paths.output_dir)
        task3_dir = output_dir / "task3_temporal_localization"

        if not task3_dir.exists():
            logger.warning(f"Task 3 directory not found: {task3_dir}")
            self.task3_demographics_cache[topic_id] = {}
            return

        # Find Task 3 JSON file for this topic
        task3_files = list(task3_dir.glob(f"{topic_id:02d}_*.json"))

        if not task3_files:
            logger.warning(f"No Task 3 file found for topic {topic_id}")
            self.task3_demographics_cache[topic_id] = {}
            return

        task3_path = task3_files[0]

        try:
            with open(task3_path, "r") as f:
                task3_data = json.load(f)

            # Build cache: video_id -> {start_time -> demographics_dict}
            cache = defaultdict(dict)

            for entry in task3_data.get("entries", []):
                video_id = entry.get("video_id")
                segment = entry.get("segment", {})
                start = segment.get("start", 0)

                # Use start time as key (rounded to handle float differences)
                start_key = round(start, 1)  # Round to 0.1s precision

                demographics = entry.get("demographics", [])

                # Only cache if demographics exist
                if demographics:
                    cache[video_id][start_key] = {
                        "demographics": demographics,
                        "demographics_total_individuals": entry.get(
                            "demographics_total_individuals", 0
                        ),
                        "demographics_confidence": entry.get(
                            "demographics_confidence", 0.0
                        ),
                        "demographics_explanation": entry.get(
                            "demographics_explanation", ""
                        ),
                        "segment_end": segment.get("end", 0),
                    }

            self.task3_demographics_cache[topic_id] = dict(cache)
            n_segments = sum(len(v) for v in cache.values())
            logger.info(
                "Loaded Task 3 cache for topic %s: %d videos, %d segments",
                topic_id,
                len(cache),
                n_segments,
            )

        except Exception as e:
            logger.error(
                "Failed to load Task 3 demographics for topic %s: %s",
                topic_id,
                e,
            )
            self.task3_demographics_cache[topic_id] = {}

    def _get_task3_demographics(
        self, video_id: str, start_time: float, end_time: float, topic_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get demographics from Task 3 cache for a specific segment.

        Uses start time as key (ignores end time differences).
        """
        # Ensure cache is loaded
        self._load_task3_demographics(topic_id)

        topic_cache = self.task3_demographics_cache.get(topic_id, {})
        video_cache = topic_cache.get(video_id, {})

        # Use start time as key (rounded to 0.1s precision)
        start_key = round(start_time, 1)

        demographics = video_cache.get(start_key)

        if demographics:
            cached_end = demographics.get("segment_end", end_time)
            logger.debug(
                f"Found Task3 match for {video_id} at start={start_time}s "
                f"(Task2 end={end_time}s, Task3 end={cached_end}s)"
            )

        return demographics

    def _update_task3_cache(
        self,
        topic_id: int,
        video_id: str,
        start_time: float,
        end_time: float,
        demographics_data: Dict[str, Any],
    ):
        """Update Task 3 cache with newly filled demographics."""
        # Ensure cache exists for this topic
        if topic_id not in self.task3_demographics_cache:
            self.task3_demographics_cache[topic_id] = {}

        if video_id not in self.task3_demographics_cache[topic_id]:
            self.task3_demographics_cache[topic_id][video_id] = {}

        # Use start time as key (rounded to 0.1s precision)
        start_key = round(start_time, 1)

        # Add segment_end for logging purposes
        demographics_data["segment_end"] = end_time

        # Update cache
        self.task3_demographics_cache[topic_id][video_id][start_key] = demographics_data
        logger.debug("Updated Task3 cache for %s at start=%s", video_id, start_key)

    def get_file_paths(self, video_id: str, topic_id: int) -> Dict[str, Optional[Path]]:
        """Get paths for video, audio, and transcript files."""
        dataset_root = Path(self.config.paths.dataset_root)
        videos_dir = dataset_root / "videos"

        # Find topic directory
        topic_dir = None
        for d in videos_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"{topic_id:02d}_"):
                topic_dir = d
                break

        if not topic_dir:
            logger.warning(f"Topic directory not found for topic {topic_id}")
            return {
                "video_path": None,
                "audio_path": None,
                "transcript_path": None,
            }

        # Get metadata to find video_number
        metadata = self.metadata_by_topic.get(topic_id, {}).get(video_id, {})
        video_number = metadata.get("video_number", video_id)

        # Construct paths
        video_path = topic_dir / f"video_{video_number}.mp4"
        audio_dir = dataset_root / "audios" / topic_dir.name
        audio_path = audio_dir / f"audio_{video_number}.m4a"
        caption_dir = dataset_root / "captions" / topic_dir.name
        transcript_path = (
            caption_dir / f"caption_{video_number}.srt"
        )  # ✓ Fixed: .srt not .txt

        return {
            "video_path": video_path if video_path.exists() else None,
            "audio_path": audio_path if audio_path.exists() else None,
            "transcript_path": (transcript_path if transcript_path.exists() else None),
        }

    def _load_transcript(self, transcript_path: Optional[Path]) -> str:
        """Load and truncate transcript if needed."""
        if not transcript_path or not transcript_path.exists():
            return ""

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                text = f.read()

            max_length = self.config.file_processing.max_transcript_length
            if len(text) > max_length:
                text = text[:max_length] + "\n...[truncated]"

            return text
        except Exception as e:
            logger.warning(f"Failed to load transcript: {e}")
            return ""

    def _generate_with_retry(
        self,
        media_files: List,
        prompt: str,
        video_id: str,
        context: str,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Generate demographics with retry for empty responses (safety filter).

        Args:
            media_files: List of (type, path) tuples
            prompt: Generation prompt
            video_id: Video ID for logging
            context: Context for logging (e.g. "Task1 video", "Task3 segment")
            max_retries: Maximum number of retry attempts

        Returns
        -------
            Response text or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Generating demographics for %s (attempt %d/%d)",
                    context,
                    attempt + 1,
                    max_retries,
                )

                response_text = self.gemini_client.generate_content(
                    media_files, prompt, video_fps=0.25
                )

                # Check for empty response
                if not response_text or len(response_text.strip()) < 10:
                    logger.warning(
                        "Empty response for %s attempt %d/%d",
                        context,
                        attempt + 1,
                        max_retries,
                    )
                    logger.warning(
                        "This is likely due to safety filters being triggered"
                    )

                    if attempt < max_retries - 1:
                        # Exponential backoff: 10s, 20s, 30s
                        wait_time = 10 * (attempt + 1)
                        logger.info(
                            "Retrying after %ds (safety filters)...",
                            wait_time,
                        )
                        time.sleep(wait_time)
                        continue
                    logger.error(
                        "Failed after %d attempts (empty/safety filter)",
                        max_retries,
                    )
                    return None

                # Got a valid response
                logger.info("Received valid response: %d chars", len(response_text))
                return response_text

            except Exception as e:
                logger.error(
                    "Attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )

                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    logger.info(f"Retrying after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts")
                    return None

        return None

    def fill_task1_demographics(self, entry: Dict[str, Any], topic_id: int) -> bool:
        """Fill demographics for Task 1 (Summarization) - full video."""
        video_id = entry.get("video_id")

        if self.dry_run:
            logger.info(
                "[DRY-RUN] Would fill demographics for Task1 video %s",
                video_id,
            )
            return True

        try:
            # Get metadata
            metadata = self.metadata_by_topic.get(topic_id, {}).get(video_id, {})
            if not metadata:
                logger.error("No metadata found for video %s", video_id)
                return False

            human_demographics = metadata.get("demographics_detailed_reviewed", {})
            if not human_demographics:
                logger.warning("No human-reviewed demographics for %s", video_id)
                return False

            # Get file paths
            paths = self.get_file_paths(video_id, topic_id)

            # Build prompt
            prompt = self.demographics_expander.build_expansion_prompt(
                human_demographics,
                segment_info=None,  # Full video for task1
            )

            # Prepare media files
            media_files = []
            if paths["video_path"]:
                media_files.append(("video", paths["video_path"]))
            if paths["audio_path"]:
                media_files.append(("audio", paths["audio_path"]))

            # Add transcript context
            transcript_text = self._load_transcript(paths["transcript_path"])
            if transcript_text:
                prompt += f"\n\nTRANSCRIPT SUMMARY:\n{transcript_text[:2000]}"

            # Generate demographics with retry
            response_text = self._generate_with_retry(
                media_files,
                prompt,
                video_id,
                f"Task1 video {video_id}",
                max_retries=3,
            )

            if not response_text:
                return False

            demographics_data = self.demographics_expander.parse_demographics_response(
                response_text
            )

            # Update entry
            entry["demographics"] = demographics_data.get("demographics", [])

            n_entries = len(entry["demographics"])
            conf = demographics_data.get("confidence", 0)
            logger.info(
                "Filled Task1 video %s: %d entries conf=%.2f",
                video_id,
                n_entries,
                conf,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to fill Task1 demographics for %s: %s",
                video_id,
                e,
                exc_info=True,
            )
            return False

    def fill_task3_demographics(self, entry: Dict[str, Any], topic_id: int) -> bool:
        """Fill demographics for Task 3 (Temporal Localization) - segment."""
        video_id = entry.get("video_id")
        segment = entry.get("segment", {})
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)

        if self.dry_run:
            logger.info(
                "[DRY-RUN] Would fill Task3 video %s segment %s-%ss",
                video_id,
                start_time,
                end_time,
            )
            return True

        video_segments = None
        audio_segments = None

        try:
            # Get metadata
            metadata = self.metadata_by_topic.get(topic_id, {}).get(video_id, {})
            if not metadata:
                logger.error("No metadata for video %s", video_id)
                return False

            human_demographics = metadata.get("demographics_detailed_reviewed", {})
            if not human_demographics:
                logger.warning("No human-reviewed demographics for %s", video_id)
                return False

            # Get file paths
            paths = self.get_file_paths(video_id, topic_id)
            duration = metadata.get("duration_seconds", end_time)

            # Create video segment
            video_segments = self.segmenter.segment_video(
                paths["video_path"],
                duration,
                task_type="temporal_localization",
            )

            # Find the matching segment
            seg_path = None
            for seg in video_segments:
                if seg["start"] == start_time and seg["end"] == end_time:
                    seg_path = seg["segment_path"]
                    break

            if not seg_path or not seg_path.exists():
                logger.error(
                    "Could not create segment for %s at %s-%ss",
                    video_id,
                    start_time,
                    end_time,
                )
                return False

            # Create audio segment if available
            audio_seg_path = None
            if paths["audio_path"]:
                audio_segments = self.segmenter.segment_audio(
                    paths["audio_path"],
                    duration,
                    task_type="temporal_localization",
                )
                for seg in audio_segments:
                    if seg["start"] == start_time and seg["end"] == end_time:
                        audio_seg_path = seg["segment_path"]
                        break

            # Extract transcript for segment
            transcript_text = ""
            if paths["transcript_path"]:
                transcript_text = self.segmenter.extract_transcript_segment(
                    paths["transcript_path"],
                    start_time,
                    end_time,
                    strip_timestamps=True,
                )

            # Build prompt
            prompt = self.demographics_expander.build_expansion_prompt(
                human_demographics,
                segment_info={"start": start_time, "end": end_time},
            )

            # Prepare media files
            media_files = []
            if seg_path.suffix.lower() in [
                ".mp4",
                ".avi",
                ".mov",
                ".webm",
                ".mkv",
                ".m4v",
            ]:
                media_files.append(("video", seg_path))
            else:
                media_files.append(("audio", seg_path))

            if audio_seg_path:
                media_files.append(("audio", audio_seg_path))

            # Add transcript context
            if transcript_text:
                prompt += f"\n\nSEGMENT TRANSCRIPT:\n{transcript_text[:1000]}"

            # Generate demographics with retry
            response_text = self._generate_with_retry(
                media_files,
                prompt,
                video_id,
                "Task3 video %s segment %s-%ss" % (video_id, start_time, end_time),
                max_retries=3,
            )

            if not response_text:
                return False

            demographics_data = self.demographics_expander.parse_demographics_response(
                response_text
            )

            # Update entry
            entry["demographics"] = demographics_data.get("demographics", [])
            entry["demographics_total_individuals"] = demographics_data.get(
                "total_individuals", 0
            )
            entry["demographics_confidence"] = demographics_data.get("confidence", 0.0)
            entry["demographics_explanation"] = demographics_data.get("explanation", "")

            # Update cache so Task 2 can reuse this immediately
            cache_data = {
                "demographics": entry["demographics"],
                "demographics_total_individuals": entry[
                    "demographics_total_individuals"
                ],
                "demographics_confidence": entry["demographics_confidence"],
                "demographics_explanation": entry["demographics_explanation"],
            }
            self._update_task3_cache(
                topic_id, video_id, start_time, end_time, cache_data
            )

            n_ent = len(entry["demographics"])
            conf = demographics_data.get("confidence", 0)
            logger.info(
                "Filled Task3 %s segment %s-%ss: %d entries conf=%.2f",
                video_id,
                start_time,
                end_time,
                n_ent,
                conf,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed Task3 %s segment %s-%ss: %s",
                video_id,
                start_time,
                end_time,
                e,
                exc_info=True,
            )
            return False

        finally:
            # Always cleanup segments
            if video_segments:
                try:
                    self.segmenter.cleanup_segments(video_segments)
                except Exception as e:
                    logger.warning("Failed to cleanup video segments: %s", e)

            if audio_segments:
                try:
                    self.segmenter.cleanup_segments(audio_segments)
                except Exception as e:
                    logger.warning("Failed to cleanup audio segments: %s", e)

    def process_json_file(self, json_path: Path, task_name: str) -> Dict[str, int]:
        """Process a single VQA JSON file and fill empty demographics.

        Task 2: overwrites demographics from Task 3 when available.
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {json_path.name}")
        logger.info(f"{'=' * 80}")

        # Load JSON
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {json_path}: {e}")
            return {
                "total": 0,
                "empty": 0,
                "filled": 0,
                "failed": 0,
                "reused": 0,
            }

        topic_id = data.get("topic_id")
        data.get("topic_name")
        entries = data.get("entries", [])

        stats = {
            "total": len(entries),
            "empty": 0,
            "filled": 0,
            "failed": 0,
            "reused": 0,
        }

        # For Task 2: check all entries for Task 3 matches
        if task_name == "task2":
            logger.info(
                "Task 2: Checking %d entries for Task 3 demographics",
                len(entries),
            )

            # Pre-load Task 3 cache
            self._load_task3_demographics(topic_id)

            updated_count = 0
            for entry in entries:
                video_id = entry.get("video_id")
                segment = entry.get("segment", {})
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)

                # Try to get Task 3 demographics
                task3_demo = self._get_task3_demographics(
                    video_id, start_time, end_time, topic_id
                )

                if task3_demo and task3_demo.get("demographics"):
                    # ALWAYS overwrite with Task 3 (it's reviewed and correct)
                    entry["demographics"] = task3_demo["demographics"]
                    entry["demographics_total_individuals"] = task3_demo[
                        "demographics_total_individuals"
                    ]
                    entry["demographics_confidence"] = task3_demo[
                        "demographics_confidence"
                    ]
                    entry["demographics_explanation"] = task3_demo[
                        "demographics_explanation"
                    ]

                    updated_count += 1
                    self.stats["task2"]["reused"] += 1

            logger.info(
                "Updated %d Task 2 entries with Task 3 demographics",
                updated_count,
            )
            stats["filled"] = updated_count
            stats["reused"] = updated_count

            # Save updated JSON (if not dry-run)
            if not self.dry_run and updated_count > 0:
                try:
                    # Create backup
                    backup_path = json_path.with_suffix(".json.backup")
                    with open(backup_path, "w") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Created backup: {backup_path.name}")

                    # Save updated file
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logger.info(f"✓ Saved updated file: {json_path.name}")
                except Exception as e:
                    logger.error(f"Failed to save {json_path}: {e}")

            return stats

        # For Task 1 and Task 3: Original logic (only fill empty)
        # Find empty demographics
        empty_indices = []
        for i, entry in enumerate(entries):
            demographics = entry.get("demographics", [])
            if not demographics or demographics == []:
                empty_indices.append(i)
                stats["empty"] += 1

        if not empty_indices:
            logger.info(f"✓ No empty demographics found in {json_path.name}")
            return stats

        logger.info("Found %d entries with empty demographics", len(empty_indices))

        # Process each empty entry
        for i in empty_indices:
            entry = entries[i]

            # Rate limiting before each API call
            if not self.dry_run:
                delay = self.config.rate_limit.delay_after_api_call
                logger.info(f"Rate limiting: waiting {delay}s...")
                time.sleep(delay)

            # Fill demographics based on task
            if task_name == "task1":
                success = self.fill_task1_demographics(entry, topic_id)
            elif task_name == "task3":
                success = self.fill_task3_demographics(entry, topic_id)
            else:
                logger.error(f"Unknown task: {task_name}")
                success = False

            if success:
                stats["filled"] += 1
            else:
                stats["failed"] += 1

        # Save updated JSON (if not dry-run)
        if not self.dry_run and stats["filled"] > 0:
            try:
                # Create backup
                backup_path = json_path.with_suffix(".json.backup")
                with open(backup_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"Created backup: {backup_path.name}")

                # Save updated file
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"✓ Saved updated file: {json_path.name}")
            except Exception as e:
                logger.error(f"Failed to save {json_path}: {e}")

        return stats

    def process_all_tasks(self, topic_filter: Optional[List[int]] = None):
        """Process all VQA task dirs. Order: Task 3, Task 1, Task 2."""
        output_dir = Path(self.config.paths.output_dir)

        if not output_dir.exists():
            logger.error(f"Output directory not found: {output_dir}")
            return

        task_dirs = {
            "task1": output_dir / "task1_summarization",
            "task2": output_dir / "task2_mcq",
            "task3": output_dir / "task3_temporal_localization",
        }

        # Order: Task 3, Task 1, Task 2 (Task 2 reuses Task 3)
        task_order = ["task3", "task1", "task2"]

        for task_name in task_order:
            task_dir = task_dirs[task_name]

            if not task_dir.exists():
                logger.warning(f"Task directory not found: {task_dir}")
                continue

            logger.info(f"\n{'#' * 80}")
            logger.info(f"# Processing {task_name.upper()}")
            logger.info(f"{'#' * 80}")

            # Get all JSON files
            json_files = sorted(task_dir.glob("*.json"))

            # Filter by topic if specified
            if topic_filter:
                json_files = [
                    f
                    for f in json_files
                    if any(f.name.startswith(f"{tid:02d}_") for tid in topic_filter)
                ]

            logger.info(f"Found {len(json_files)} JSON files to process")

            for json_path in json_files:
                stats = self.process_json_file(json_path, task_name)

                # Update global stats
                self.stats[task_name]["total"] += stats["total"]
                self.stats[task_name]["empty"] += stats["empty"]
                self.stats[task_name]["filled"] += stats["filled"]
                self.stats[task_name]["failed"] += stats["failed"]
                # reused is already tracked in self.stats

        # Print final summary
        self.print_summary()

    def print_summary(self):
        """Print final statistics summary."""
        logger.info(f"\n{'=' * 80}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'=' * 80}")

        for task_name in ["task1", "task2", "task3"]:
            stats = self.stats[task_name]
            logger.info(f"\n{task_name.upper()}:")
            logger.info(f"  Total entries:        {stats['total']}")
            logger.info(f"  Empty demographics:   {stats['empty']}")
            logger.info(f"  Successfully filled:  {stats['filled']}")
            logger.info(f"  Reused from Task3:    {stats['reused']}")
            logger.info(f"  Failed:               {stats['failed']}")

            if stats["empty"] > 0:
                success_rate = (stats["filled"] / stats["empty"]) * 100
                logger.info(f"  Success rate:         {success_rate:.1f}%")

                if stats["reused"] > 0:
                    reuse_rate = (stats["reused"] / stats["empty"]) * 100
                    logger.info(f"  Reuse rate:           {reuse_rate:.1f}%")

        total_empty = sum(s["empty"] for s in self.stats.values())
        total_filled = sum(s["filled"] for s in self.stats.values())
        total_reused = sum(s["reused"] for s in self.stats.values())
        total_failed = sum(s["failed"] for s in self.stats.values())

        logger.info("\nOVERALL:")
        logger.info(f"  Total empty:          {total_empty}")
        logger.info(f"  Successfully filled:  {total_filled}")
        logger.info(f"  Reused from Task3:    {total_reused}")
        logger.info(f"  Failed:               {total_failed}")

        if total_empty > 0:
            success_rate = (total_filled / total_empty) * 100
            logger.info(f"  Success rate:         {success_rate:.1f}%")

            if total_reused > 0:
                reuse_rate = (total_reused / total_empty) * 100
                api_savings = (total_reused / total_empty) * 100
                logger.info(f"  Reuse rate:           {reuse_rate:.1f}%")
                logger.info(f"  API calls saved:      {api_savings:.1f}%")


def main():
    """Run main entry point."""
    parser = argparse.ArgumentParser(description="Fill Empty Demographics in VQA Files")
    parser.add_argument(
        "--config",
        type=str,
        default="vqa_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help='Comma-separated topic IDs to process (e.g., "10,11")',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Parse topic filter
    topic_filter = None
    if args.topics:
        try:
            topic_filter = [int(t.strip()) for t in args.topics.split(",")]
            logger.info(f"Processing topics: {topic_filter}")
        except ValueError:
            logger.error(f"Invalid topics format: {args.topics}")
            sys.exit(1)

    # Check API key
    if not args.dry_run and not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not found in environment!")
        sys.exit(1)

    # Create filler and run
    filler = DemographicsFiller(config, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("=" * 80)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("=" * 80)

    logger.info("\nOPTIMIZATION: Task 3 first, then Task 1, then Task 2 reuses Task 3")
    logger.info("This will significantly reduce API calls!\n")

    filler.process_all_tasks(topic_filter)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
