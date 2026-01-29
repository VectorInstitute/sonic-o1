"""main.py.

Main VQA generation script. Generates summarization, MCQ, and temporal
localization tasks using Gemini-based multimodal models.

Usage:
    python main.py --topics 1,2,3
    python main.py --all
    python main.py --topics 1 --task summarization

Author: SONIC-O1 Team
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm


# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logging.info("Loaded environment variables from .env file")
except ImportError:
    pass

from models import MCQModel, SummarizationModel, TemporalLocalizationModel


# Setup logging
logger = logging.getLogger(__name__)


class Config:
    """Configuration wrapper with nested attribute access."""

    def __init__(self, config_dict):
        """Initialize from nested dict. Recursively wraps dicts as Config."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If relative, resolved
            relative to this script's directory.

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


def load_metadata_for_topic(topic_dir: Path) -> List[Dict[str, Any]]:
    """
    Load metadata_enhanced.json for a topic directory.

    Args:
        topic_dir: Path to topic directory (e.g. dataset/videos/01_.../)

    Returns
    -------
        List of video metadata dicts
    """
    metadata_file = topic_dir / "metadata_enhanced.json"

    if not metadata_file.exists():
        logger.warning("No metadata_enhanced.json found in %s", topic_dir)
        return []

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        logger.info("Loaded %d videos from %s", len(metadata_list), topic_dir.name)
        return metadata_list

    except Exception as e:
        logger.error("Failed to load metadata from %s: %s", topic_dir, e)
        return []


def get_file_paths(video_meta: Dict[str, Any], topic_dir: Path) -> Dict[str, Path]:
    """
    Get file paths for video, audio, and transcript.

    Args:
        video_meta: Video metadata dict
        topic_dir: Topic directory path

    Returns
    -------
        Dict with keys: video_path, audio_path, transcript_path
    """
    video_number = video_meta.get("video_number", video_meta.get("video_id", "001"))

    # Video path
    video_filename = f"video_{video_number}.mp4"
    video_path = topic_dir / video_filename

    # Audio path (in parent audios directory)
    audio_dir = topic_dir.parent.parent / "audios" / topic_dir.name
    audio_filename = f"audio_{video_number}.m4a"
    audio_path = audio_dir / audio_filename

    # Transcript path (in parent captions directory)
    captions_dir = topic_dir.parent.parent / "captions" / topic_dir.name
    transcript_filename = f"caption_{video_number}.srt"
    transcript_path = captions_dir / transcript_filename

    return {
        "video_path": video_path if video_path.exists() else None,
        "audio_path": audio_path if audio_path.exists() else None,
        "transcript_path": (transcript_path if transcript_path.exists() else None),
    }


def process_topic(
    topic_id: int,
    topic_name: str,
    topic_dir: Path,
    config: Config,
    output_dir: Path,
    task_filter: str = None,
) -> tuple:
    """
    Process all videos in a topic for VQA generation.

    Args:
        topic_id: Topic ID (1-13)
        topic_name: Topic name (e.g., "Patient-Doctor Consultations")
        topic_dir: Path to topic directory
        config: Configuration object
        task_filter: Optional filter - "summarization" or "mcq" (None = both)

    Returns
    -------
        Tuple of (task1_entries, task2_entries)
    """
    logger.info(f"Processing Topic {topic_id}: {topic_name}")

    # Load metadata
    metadata_list = load_metadata_for_topic(topic_dir)
    if not metadata_list:
        logger.warning(f"No videos found for topic {topic_id}")
        return ([], [], [])

    # Initialize models
    task1_entries = []
    task2_entries = []
    task3_entries = []
    # Load existing results if they exist
    existing_task1 = {}
    existing_task2 = {}
    existing_task3 = {}
    if task_filter is None or task_filter == "summarization":
        summarizer = SummarizationModel(config)
        # Check for existing Task 1 output
        task1_output_file = (
            output_dir
            / "task1_summarization"
            / f"{topic_id:02d}_{topic_name.replace(' ', '_')}.json"
        )
        if task1_output_file.exists():
            with open(task1_output_file, "r") as f:
                task1_data = json.load(f)
                # Index by video_id
                for entry in task1_data.get("entries", []):
                    existing_task1[entry["video_id"]] = entry
            logger.info("Loaded %d existing Task 1 entries", len(existing_task1))

    if task_filter is None or task_filter == "mcq":
        mcq_generator = MCQModel(config)
        # Check for existing Task 2 output
        task2_output_file = (
            output_dir
            / "task2_mcq"
            / f"{topic_id:02d}_{topic_name.replace(' ', '_')}.json"
        )
        if task2_output_file.exists():
            with open(task2_output_file, "r") as f:
                task2_data = json.load(f)
                # Index by video_id
                for entry in task2_data.get("entries", []):
                    vid = entry["video_id"]
                    if vid not in existing_task2:
                        existing_task2[vid] = []
                    existing_task2[vid].append(entry)
            n_task2 = sum(len(v) for v in existing_task2.values())
            logger.info("Loaded %d existing Task 2 entries", n_task2)

    if task_filter is None or task_filter == "temporal":
        temporal_generator = TemporalLocalizationModel(config)
        # Check for existing Task 3 output
        task3_output_file = (
            output_dir
            / "task3_temporal_localization"
            / f"{topic_id:02d}_{topic_name.replace(' ', '_')}.json"
        )
        if task3_output_file.exists():
            with open(task3_output_file, "r") as f:
                task3_data = json.load(f)
                for entry in task3_data.get("entries", []):
                    vid = entry["video_id"]
                    if vid not in existing_task3:
                        existing_task3[vid] = []
                    existing_task3[vid].append(entry)
            n_task3 = sum(len(v) for v in existing_task3.values())
            logger.info("Loaded %d existing Task 3 entries", n_task3)

    # Process each video
    for video_meta in tqdm(metadata_list, desc=f"Processing {topic_name}"):
        video_id = video_meta.get("video_id", video_meta.get("video_number", "unknown"))

        video_category = video_meta.get("duration_category", "short")
        duration = video_meta.get("duration_seconds", 0)
        if video_category not in ["short", "medium", "long"]:
            if duration <= 300:  # <= 5 minutes
                video_category = "short"
            elif duration <= 1800:  # <= 30 minutes
                video_category = "medium"
            else:  # > 30 minutes
                video_category = "long"
        # Check if already successfully processed
        skip_task1 = False
        skip_task2 = False
        skip_task3 = False

        if video_id in existing_task1:
            entry = existing_task1[video_id]

            # Check for failure indicators in summary
            has_summary_failure = False

            # Check summary_short for failures (more specific patterns)
            summary_short = entry.get("summary_short", [])
            if isinstance(summary_short, list):
                for item in summary_short:
                    if isinstance(item, str):
                        lower_item = item.lower()
                        # Look for failure patterns in summary_short
                        fail_short = (
                            "unavailable" in lower_item
                            or "summary generation failed" in lower_item
                            or "could not be generated" in lower_item
                            or "summary failed" in lower_item
                            or (
                                "first segment" in lower_item and "failed" in lower_item
                            )
                        )
                        if fail_short:
                            has_summary_failure = True
                            break

            # Check summary_detailed for failures (more specific patterns)
            summary_detailed = entry.get("summary_detailed", "")
            if isinstance(summary_detailed, str):
                lower_detailed = summary_detailed.lower()
                fail_det = (
                    "could not be generated" in lower_detailed
                    or "summary generation failed" in lower_detailed
                    or "parsing error" in lower_detailed
                    or ("failed to" in lower_detailed and "summary" in lower_detailed)
                    or "explicitly reported a failure" in lower_detailed
                )
                if fail_det:
                    has_summary_failure = True

            # Only skip if no failures detected AND confidence > 0
            if not has_summary_failure and entry.get("confidence", 0) > 0:
                skip_task1 = True
                task1_entries.append(entry)
                logger.info("Skipping Task 1 for %s (already processed)", video_id)
            elif has_summary_failure:
                logger.info("Reprocessing Task 1 for %s (previous failure)", video_id)
            else:
                logger.info("Reprocessing Task 1 for %s (confidence was 0)", video_id)
        if video_id in existing_task2:
            all_good = all(e.get("confidence", 0) > 0 for e in existing_task2[video_id])
            if all_good and len(existing_task2[video_id]) > 0:
                skip_task2 = True
                task2_entries.extend(existing_task2[video_id])
                logger.info("Skipping Task 2 for %s (already processed)", video_id)

        if video_id in existing_task3:
            all_good = all(e.get("confidence", 0) > 0 for e in existing_task3[video_id])
            if all_good and len(existing_task3[video_id]) > 0:
                skip_task3 = True
                task3_entries.extend(existing_task3[video_id])
                logger.info("Skipping Task 3 for %s (already processed)", video_id)

        # If both tasks should be skipped, continue
        if (
            task_filter == "summarization"
            and skip_task1
            or task_filter == "mcq"
            and skip_task2
            or task_filter == "temporal"
            and skip_task3
            or task_filter is None
            and skip_task1
            and skip_task2
            and skip_task3
        ):
            continue

        try:
            # Enhance metadata with topic info
            video_meta["topic_id"] = topic_id
            video_meta["topic_name"] = topic_name

            # Get file paths
            file_paths = get_file_paths(video_meta, topic_dir)

            if not file_paths["video_path"] and not file_paths["audio_path"]:
                logger.warning("No video or audio for %s, skipping", video_id)
                continue

            # Task 1: Summarization
            if (
                task_filter is None or task_filter == "summarization"
            ) and not skip_task1:
                logger.info(f"Generating summarization for {video_id}")
                summary_entry = summarizer.process_video(
                    video_path=file_paths["video_path"],
                    audio_path=file_paths["audio_path"],
                    transcript_path=file_paths["transcript_path"],
                    metadata=video_meta,
                )
                task1_entries.append(summary_entry)

            # Task 2: MCQ
            do_mcq = (task_filter is None or task_filter == "mcq") and not skip_task2
            if do_mcq:
                logger.info(f"Generating MCQs for {video_id}")

                new_mcq_entries = mcq_generator.process_video(
                    video_path=file_paths["video_path"],
                    audio_path=file_paths["audio_path"],
                    transcript_path=file_paths["transcript_path"],
                    metadata=video_meta,
                )

                # If we have existing entries, merge intelligently
                if video_id in existing_task2 and len(existing_task2[video_id]) > 0:
                    merged_entries, kept, replaced = merge_entries_keep_good(
                        existing_task2[video_id], new_mcq_entries
                    )
                    task2_entries.extend(merged_entries)
                    logger.info(
                        "Task 2 for %s: kept %s good, replaced %s failed",
                        video_id,
                        kept,
                        replaced,
                    )
                else:
                    # No existing entries, use all new ones
                    task2_entries.extend(new_mcq_entries)

            # Task 3: Temporal Localization
            do_temporal = (
                task_filter is None or task_filter == "temporal"
            ) and not skip_task3
            if do_temporal:
                logger.info(f"Generating temporal questions for {video_id}")

                new_temporal_entries = temporal_generator.process_video(
                    video_path=file_paths["video_path"],
                    audio_path=file_paths["audio_path"],
                    transcript_path=file_paths["transcript_path"],
                    metadata=video_meta,
                )

                # If we have existing entries, merge intelligently
                if video_id in existing_task3 and len(existing_task3[video_id]) > 0:
                    merged_entries, kept, replaced = merge_entries_keep_good(
                        existing_task3[video_id], new_temporal_entries
                    )
                    task3_entries.extend(merged_entries)
                    logger.info(
                        "Task 3 for %s: kept %s good, replaced %s failed",
                        video_id,
                        kept,
                        replaced,
                    )
                else:
                    # No existing entries, use all new ones
                    task3_entries.extend(new_temporal_entries)

            # Rate limiting: Add delay after processing each video
            if not (skip_task1 and skip_task2 and skip_task3):
                delay_between_videos = int(
                    getattr(config.rate_limit, "delay_between_videos", 15)
                )

                # Extra delay for long videos
                if video_category == "long":
                    extra_delay = int(
                        getattr(
                            config.rate_limit,
                            "delay_after_long_video",
                            60,
                        )
                    )
                    total_delay = delay_between_videos + extra_delay
                    logger.info(
                        "Long video (%s) - waiting %ss before next",
                        video_category,
                        total_delay,
                    )
                    time.sleep(total_delay)
                else:
                    logger.info(
                        "Waiting %ss before next video (rate limit)",
                        delay_between_videos,
                    )
                    time.sleep(delay_between_videos)
            else:
                logger.info("Skipped all tasks for %s - no delay", video_id)

        except Exception as e:
            logger.error("Failed to process video %s: %s", video_id, e, exc_info=True)
            continue

    logger.info(
        "Completed Topic %s: %d summaries, %d MCQs",
        topic_id,
        len(task1_entries),
        len(task2_entries),
    )
    return (task1_entries, task2_entries, task3_entries)


def save_task_results(
    task_name: str,
    topic_id: int,
    topic_name: str,
    entries: List[Dict[str, Any]],
    output_dir: Path,
):
    """
    Save VQA entries to JSON file.

    Args:
        task_name: "task1_summarization" or "task2_mcq"
        topic_id: Topic ID
        topic_name: Topic name
        entries: List of VQA entry dicts
        output_dir: Output directory (e.g., vqa/)
    """
    if not entries:
        logger.warning("No entries to save for %s - %s", task_name, topic_name)
        return

    # Create output directory
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Build output JSON
    output_data = {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "task": task_name.replace("task1_", "").replace("task2_", ""),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_entries": len(entries),
        "entries": entries,
    }

    # Save to file
    out_name = f"{topic_id:02d}_{topic_name.replace(' ', '_')}.json"
    output_file = task_dir / out_name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d entries to %s", len(entries), output_file)


def get_all_topic_dirs(dataset_root: Path) -> List[tuple]:
    """
    Get all topic directories.

    Returns
    -------
        List of tuples: (topic_id, topic_name, topic_dir_path)
    """
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return []

    topics = []
    for topic_dir in sorted(videos_dir.iterdir()):
        if topic_dir.is_dir() and topic_dir.name[0].isdigit():
            # Extract topic ID and name from dir (e.g. 01_Patient-Doctor_...)
            parts = topic_dir.name.split("_", 1)
            if len(parts) == 2:
                topic_id = int(parts[0])
                topic_name = parts[1].replace("_", " ")
                topics.append((topic_id, topic_name, topic_dir))

    return topics


def merge_entries_keep_good(
    existing_entries: List[Dict], new_entries: List[Dict]
) -> List[Dict]:
    """
    Merge existing and new entries.

    - Keeping existing entries with confidence > 0
    - Replacing existing entries with confidence 0.0 with matching new entries
    - Adding new entries for segments that don't exist in existing.

    Args:
        existing_entries: List of existing entries for a video_id
        new_entries: List of newly generated entries

    Returns
    -------
        Merged list with good existing + new replacements for failed
    """
    merged = []

    # Keep all existing entries with confidence > 0
    for existing in existing_entries:
        if existing.get("confidence", 0) > 0:
            merged.append(existing)

    # Replace failed entries (confidence 0) with new if segment matches
    failed_segments = {
        (e.get("segment", {}).get("start"), e.get("segment", {}).get("end")): e
        for e in existing_entries
        if e.get("confidence", 0) == 0.0
    }

    new_segments_used = set()

    # Replace failed segments with new entries
    for new in new_entries:
        new_seg = (
            new.get("segment", {}).get("start"),
            new.get("segment", {}).get("end"),
        )

        if new_seg in failed_segments:
            merged.append(new)
            new_segments_used.add(new_seg)
        else:
            exist_segs = [
                (e.get("segment", {}).get("start"), e.get("segment", {}).get("end"))
                for e in existing_entries
            ]
            if new_seg not in exist_segs:
                merged.append(new)
                new_segments_used.add(new_seg)

    # Log what happened
    replaced = len(set(failed_segments.keys()).intersection(new_segments_used))
    kept_good = sum(1 for e in existing_entries if e.get("confidence", 0) > 0)

    return merged, kept_good, replaced


def main():
    """Run main entry point."""
    parser = argparse.ArgumentParser(description="VQA Generation System")
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
        help='Comma-separated topic IDs (e.g., "1,2,3")',
    )
    parser.add_argument("--all", action="store_true", help="Process all topics")
    parser.add_argument(
        "--task",
        type=str,
        choices=["summarization", "mcq", "temporal"],
        default=None,
        help="Process only specific task (default: both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set output directory
    output_dir = Path(args.output) if args.output else Path(config.paths.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Get dataset root
    dataset_root = Path(config.paths.dataset_root)
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return

    # Get all topics
    all_topics = get_all_topic_dirs(dataset_root)
    logger.info(f"Found {len(all_topics)} topics in dataset")

    # Filter topics if specified
    if args.topics:
        topic_ids = [int(tid.strip()) for tid in args.topics.split(",")]
        topics_to_process = [t for t in all_topics if t[0] in topic_ids]
    elif args.all:
        topics_to_process = all_topics
    else:
        logger.error("Must specify either --topics or --all")
        return

    logger.info(f"Processing {len(topics_to_process)} topics")

    # Process each topic
    total_task1 = 0
    total_task2 = 0
    total_task3 = 0
    for topic_id, topic_name, topic_dir in topics_to_process:
        try:
            task1_entries, task2_entries, task3_entries = process_topic(
                topic_id,
                topic_name,
                topic_dir,
                config,
                output_dir,
                task_filter=args.task,
            )

            # Save results
            if args.task is None or args.task == "summarization":
                save_task_results(
                    "task1_summarization",
                    topic_id,
                    topic_name,
                    task1_entries,
                    output_dir,
                )
                total_task1 += len(task1_entries)

            if args.task is None or args.task == "mcq":
                save_task_results(
                    "task2_mcq",
                    topic_id,
                    topic_name,
                    task2_entries,
                    output_dir,
                )
                total_task2 += len(task2_entries)

            if args.task is None or args.task == "temporal":
                save_task_results(
                    "task3_temporal_localization",
                    topic_id,
                    topic_name,
                    task3_entries,
                    output_dir,
                )
                total_task3 += len(task3_entries)

        except Exception as e:
            logger.error(
                "Failed to process topic %s: %s",
                topic_id,
                e,
                exc_info=True,
            )
            continue

    # Final statistics
    logger.info("=" * 60)
    logger.info("VQA Generation Complete!")
    logger.info(f"Topics processed: {len(topics_to_process)}")
    logger.info(f"Task 1 (Summarization): {total_task1} entries")
    logger.info(f"Task 2 (MCQ): {total_task2} entries")
    logger.info(f"Task 3 (Temporal): {total_task3} entries")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
