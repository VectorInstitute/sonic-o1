"""Main script to run demographics annotation pipeline."""

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config_loader import Config
from model import DemographicsAnnotator


class AnnotationPipeline:
    """Main pipeline for processing videos."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.setup_logging()
        self.annotator = DemographicsAnnotator(self.config)
        self.logger = logging.getLogger(__name__)
        self.checkpoint_file = None

    def _get_checkpoint_path(self, output_dir: Path) -> Path:
        """Get consistent checkpoint file path."""
        return output_dir / "metadata_enhanced_checkpoint.json"

    def _load_checkpoint(self, output_dir: Path) -> Optional[List[Dict]]:
        """Load from checkpoint if exists."""
        checkpoint_path = self._get_checkpoint_path(output_dir)
        if checkpoint_path.exists():
            self.logger.info(f"Found checkpoint file: {checkpoint_path}")
            try:
                with open(checkpoint_path, "r") as f:
                    data = json.load(f)
                self.logger.info(f"Loaded {len(data)} processed videos from checkpoint")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
        return None

    def _save_checkpoint(self, output_dir: Path, metadata: List[Dict]):
        """Save checkpoint (atomic write)."""
        checkpoint_path = self._get_checkpoint_path(output_dir)
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            # Write to temp file first
            with open(temp_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Atomic rename
            temp_path.replace(checkpoint_path)
            self.logger.info(f"Checkpoint saved: {len(metadata)} videos")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def _has_empty_demographics(self, video_metadata: Dict) -> bool:
        """Check if video has empty or missing detailed demographics."""
        # Check if demographics_detailed exists and has content
        demographics_detailed = video_metadata.get("demographics_detailed", {})

        if not demographics_detailed:
            return True

        # Check if it has the expected fields with actual values
        required_fields = ["race", "gender", "age", "language"]
        for field in required_fields:
            value = demographics_detailed.get(field)
            # If field exists and has non-empty list, demographics exist
            if value and isinstance(value, list) and len(value) > 0:
                return False  # Found at least one valid field

        return True  # All fields are empty or missing

    def _get_failed_video_indices(
        self, metadata_list: List[Dict], enhanced_metadata: List[Dict]
    ) -> List[int]:
        """Get indices of videos that need reprocessing."""
        failed_indices = []

        for idx, video_meta in enumerate(enhanced_metadata):
            if self._has_empty_demographics(video_meta):
                failed_indices.append(idx)

        return failed_indices

    def setup_logging(self):
        """Setup logging based on configuration."""
        handlers = []

        if self.config.console_output:
            handlers.append(logging.StreamHandler())

        if self.config.file_output:
            log_path = Path(__file__).parent / self.config.log_file
            handlers.append(logging.FileHandler(log_path))

        logging.basicConfig(
            level=logging.INFO, format=self.config.log_format, handlers=handlers
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("google.generativeai").setLevel(logging.WARNING)

    def process_topic(self, topic: str, retry_failed: bool = False) -> Dict[str, Any]:
        """Process all videos in a topic.

        Args:
            topic: Topic name to process
            retry_failed: If True, only reprocess videos with empty demographics
        """
        self.logger.info(f"Processing topic: {topic}")
        if retry_failed:
            self.logger.info(
                "RETRY MODE: Only reprocessing videos with empty demographics"
            )

        # Get paths
        paths = self.config.get_topic_paths(topic)

        # Check if metadata exists
        if not paths["metadata"].exists():
            self.logger.error(f"Metadata file not found: {paths['metadata']}")
            return {"topic": topic, "error": "Metadata not found"}

        # Load existing metadata
        with open(paths["metadata"], "r") as f:
            metadata_list = json.load(f)

        # Create backup if configured
        if self.config.create_backup:
            backup_path = paths["metadata"].with_suffix(".backup.json")
            if not backup_path.exists():
                shutil.copy(paths["metadata"], backup_path)
                self.logger.info(f"Created backup: {backup_path}")

        # Load existing enhanced metadata
        output_path = paths["videos"] / self.config.output_format
        if output_path.exists():
            with open(output_path, "r") as f:
                enhanced_metadata = json.load(f)
            self.logger.info(
                f"Loaded {len(enhanced_metadata)} existing processed videos"
            )
        else:
            enhanced_metadata = []

        # Determine which videos to process
        if retry_failed:
            if not enhanced_metadata:
                self.logger.error(
                    "No enhanced metadata found. Cannot retry failed videos."
                )
                return {"topic": topic, "error": "No enhanced metadata to retry"}

            # Get indices of failed videos
            failed_indices = self._get_failed_video_indices(
                metadata_list, enhanced_metadata
            )

            if not failed_indices:
                self.logger.info(
                    "No failed videos found. All videos have valid demographics."
                )
                return {
                    "topic": topic,
                    "total_videos": len(metadata_list),
                    "failed_videos": 0,
                    "reprocessed": 0,
                    "output_path": str(output_path),
                }

            self.logger.info(
                f"Found {len(failed_indices)} videos with empty demographics"
            )
            indices_to_process = failed_indices
        else:
            # Normal mode: resume from checkpoint or start fresh
            checkpoint_metadata = self._load_checkpoint(paths["videos"])
            if checkpoint_metadata:
                enhanced_metadata = checkpoint_metadata

            if enhanced_metadata:
                start_idx = len(enhanced_metadata)
                indices_to_process = list(range(start_idx, len(metadata_list)))
                self.logger.info(
                    f"Resuming from video {start_idx + 1}/{len(metadata_list)}"
                )
            else:
                indices_to_process = list(range(len(metadata_list)))
                self.logger.info("Starting from beginning")

        # Create raw responses directory
        if self.config.save_raw_responses:
            raw_dir = paths["videos"] / "raw_responses"
            raw_dir.mkdir(exist_ok=True)

        # Process videos
        processed_count = 0
        for idx in indices_to_process:
            video_metadata = metadata_list[idx]
            video_number = video_metadata.get("video_number", f"{idx + 1:03d}")

            self.logger.info(
                f"Processing video {idx + 1}/{len(metadata_list)} (video_{video_number})"
            )

            # Get file paths using config patterns
            video_path = self.config.get_file_path(topic, "video", video_number)
            audio_path = self.config.get_file_path(topic, "audio", video_number)
            caption_path = self.config.get_file_path(topic, "caption", video_number)

            # Determine available media files
            has_video = video_path.exists()
            has_audio = audio_path.exists()
            has_caption = caption_path.exists()

            # Apply preference logic
            if has_video and has_audio and self.config.prefer_video_with_audio:
                self.logger.info(
                    f"Both video and audio exist for {video_number}. "
                    f"Using video only (with embedded audio) as per config."
                )
                has_audio = False

            # Log what we found
            media_info = []
            if has_video:
                media_info.append(f"video ({video_path.name})")
            if has_audio:
                media_info.append(f"audio ({audio_path.name})")
            if has_caption:
                media_info.append(f"caption ({caption_path.name})")

            if not has_video and not has_audio:
                self.logger.warning(
                    f"No media files found for video {video_number}. "
                    f"Checked: {video_path.name}, {audio_path.name}"
                )
                if not retry_failed:
                    enhanced_metadata.append(video_metadata)
                continue

            self.logger.info(
                f"Processing video {video_number} with: {', '.join(media_info)}"
            )

            # Process media with multimodal support
            demographics = self.annotator.process_media(
                video_path=video_path if has_video else None,
                audio_path=audio_path if has_audio else None,
                transcript_path=caption_path if has_caption else None,
                metadata=video_metadata,
                config=self.config,
            )

            # Save raw response if configured
            if self.config.save_raw_responses and "raw_response" in demographics:
                raw_path = raw_dir / f"video_{video_number}_response.json"
                with open(raw_path, "w") as f:
                    json.dump(
                        {"raw_response": demographics["raw_response"]}, f, indent=2
                    )
                del demographics["raw_response"]

            # Merge demographics with original metadata
            enhanced_video_metadata = video_metadata.copy()
            enhanced_video_metadata.update(demographics)

            # Add processing info
            enhanced_video_metadata["processing_info"] = {
                "had_video": has_video,
                "had_audio": has_audio,
                "had_caption": has_caption,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            if retry_failed:
                # Update existing entry
                enhanced_metadata[idx] = enhanced_video_metadata
            else:
                # Append new entry
                enhanced_metadata.append(enhanced_video_metadata)

            processed_count += 1

            # Save checkpoint periodically
            if processed_count % self.config.save_interval == 0:
                self._save_checkpoint(paths["videos"], enhanced_metadata)

            # Rate limiting between videos
            if idx < indices_to_process[-1]:  # Don't wait after last video
                # Get video duration from metadata
                video_duration = video_metadata.get("duration", 0)

                # Determine wait time based on video length
                if video_duration > self.config.long_video_threshold:
                    wait_time = self.config.delay_after_long_video
                    self.logger.info(
                        f"Long video detected ({video_duration}s). "
                        f"Waiting {wait_time}s to respect rate limits..."
                    )
                else:
                    wait_time = self.config.delay_between_videos
                    self.logger.info(f"Waiting {wait_time}s before next video...")

                time.sleep(wait_time)

        # Save final enhanced metadata (atomic write)
        temp_output = output_path.with_suffix(".tmp")

        with open(temp_output, "w") as f:
            json.dump(enhanced_metadata, f, indent=2)

        temp_output.replace(output_path)

        # Clean up checkpoint
        checkpoint_path = self._get_checkpoint_path(paths["videos"])
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            self.logger.info("Checkpoint cleaned up")

        self.logger.info(
            f"Completed topic {topic}. Enhanced metadata saved to {output_path}"
        )

        result = {
            "topic": topic,
            "total_videos": len(metadata_list),
            "processed": processed_count,
            "output_path": str(output_path),
        }

        if retry_failed:
            result["failed_videos"] = len(indices_to_process)

        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run demographics annotation pipeline")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument("--topic", type=str, help="Process specific topic only")
    parser.add_argument("--api-key", type=str, help="Override Gemini API key")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Reprocess all videos even if already done",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only reprocess videos with empty detailed demographics",
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = AnnotationPipeline(config_path=args.config)

    # Override settings if provided
    if args.api_key:
        pipeline.config.api_key = args.api_key
    if args.no_cache:
        pipeline.config.use_cache = False

    # Check API key
    if not pipeline.config.api_key:
        pipeline.logger.error(
            "API key not provided. Set GEMINI_API_KEY environment variable or update config.yaml"
        )
        sys.exit(1)

    # Process topics
    if args.topic:
        # Process single topic
        if args.topic not in pipeline.config.topics:
            pipeline.logger.error(f"Topic {args.topic} not found in configuration")
            sys.exit(1)
        result = pipeline.process_topic(args.topic, retry_failed=args.retry_failed)
        print(f"Completed processing: {result}")
    else:
        # Process all topics
        results = []
        for topic in pipeline.config.topics:
            try:
                result = pipeline.process_topic(topic, retry_failed=args.retry_failed)
                results.append(result)
            except Exception as e:
                pipeline.logger.error(
                    f"Failed to process topic {topic}: {e}", exc_info=True
                )
                results.append({"topic": topic, "error": str(e)})

        # Save summary report
        report_path = (
            Path(pipeline.config.base_path) / "demographics_annotation_report.json"
        )
        with open(report_path, "w") as f:
            json.dump(
                {
                    "processing_date": datetime.now().isoformat(),
                    "model_used": pipeline.config.model_name,
                    "retry_mode": args.retry_failed,
                    "topics_processed": len(results),
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"Completed all topics. Report saved to {report_path}")


if __name__ == "__main__":
    main()
