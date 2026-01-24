"""Task 3: Temporal Action Localization (Open-Ended) Generation Model."""

import base64
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from prompts.temporal_judge_prompts import (
    build_batch_validation_system_prompt,
    build_validation_prompt,
)
from prompts.temporal_localization_prompts import get_temporal_localization_prompt
from utils.demographics_expander import DemographicsExpander
from utils.frame_sampler import FrameSampler
from utils.video_segmenter import VideoSegmenter

from .base_gemini import BaseGeminiClient


logging.basicConfig(
    level=logging.INFO,  # show info and above
    format="%(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


class TemporalLocalizationModel(BaseGeminiClient):
    """Generate segment-level temporal localization VQA entries."""

    def __init__(self, config):
        """
        Initialize Temporal Localization model.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.config = config
        self.segmenter = VideoSegmenter(config)
        self.demographics_expander = DemographicsExpander(config)
        self.frame_sampler = FrameSampler(config)

        # Get questions per segment from config
        self.questions_per_segment = int(
            config.temporal_localization.questions_per_segment
        )

        # Retry configuration
        self.max_retries = getattr(config.temporal_localization, "max_retries", 3)
        self.retry_delay = getattr(
            config.temporal_localization, "retry_delay", 2
        )  # seconds
        # Initialize GPT-4V judge
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                self.judge_enabled = getattr(
                    config.temporal_localization, "judge_enabled", True
                )
                self.judge_model = getattr(
                    config.temporal_localization, "judge_model", "gpt-4o"
                )
                self.judge_frame_count = getattr(
                    config.temporal_localization, "judge_frame_count", 32
                )
                self.temp_frames_dir = Path(tempfile.mkdtemp(prefix="temporal_judge_"))
                logger.info(f"✓ GPT-4V judge enabled (model: {self.judge_model})")
            else:
                self.judge_enabled = False
                logger.warning("OPENAI_API_KEY not set, GPT-4V judge disabled")
        except Exception as e:
            logger.warning(f"GPT-4V judge initialization failed: {e}")
            self.judge_enabled = False

    def process_video(
        self,
        video_path: Path,
        audio_path: Optional[Path],
        transcript_path: Optional[Path],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Process a video and generate temporal localization VQA entries.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file (optional)
            transcript_path: Path to transcript/caption file (optional)
            metadata: Video metadata from metadata_enhanced.json

        Returns
        -------
            List of VQA entry dicts for Task 3 (one entry per segment, each with multiple questions)
        """
        # Track segments to cleanup AFTER processing completes
        video_segments = None
        audio_segments = None

        try:
            video_id = metadata.get("video_id", metadata.get("video_number", "unknown"))
            duration = metadata.get("duration_seconds", 0)

            logger.info(
                f"Processing video {video_id} for temporal localization (duration: {duration}s)"
            )

            # Always segment videos into 3-minute chunks
            video_segments = self.segmenter.segment_video(
                video_path, duration, task_type="temporal_localization"
            )
            audio_segments = None
            if audio_path and audio_path.exists():
                audio_segments = self.segmenter.segment_audio(
                    audio_path, duration, task_type="temporal_localization"
                )

            logger.info(
                f"Created {len(video_segments)} segments for temporal localization"
            )

            # Generate temporal questions for each segment
            temporal_entries = []
            for i, seg in enumerate(video_segments):
                try:
                    # Get corresponding audio segment
                    audio_seg_path = (
                        audio_segments[i]["segment_path"] if audio_segments else None
                    )

                    # Extract transcript for this segment
                    transcript_text = ""
                    if transcript_path and transcript_path.exists():
                        transcript_text = self.segmenter.extract_transcript_segment(
                            transcript_path, seg["start"], seg["end"]
                        )

                    # Generate temporal questions for this segment with retry
                    segment_entry = self._generate_temporal_questions_with_retry(
                        seg, audio_seg_path, transcript_text, metadata
                    )

                    if segment_entry:
                        temporal_entries.append(segment_entry)

                except Exception as e:
                    logger.error(
                        f"Failed to generate temporal questions for segment {i}: {e}"
                    )
                    continue

            logger.info(
                f"Generated {len(temporal_entries)} segment entries (with {sum(e['num_questions'] for e in temporal_entries)} total questions) for video {video_id}"
            )
            return temporal_entries

        except Exception as e:
            logger.error(
                f"Error processing video {video_id} for temporal localization: {e}",
                exc_info=True,
            )
            return []

        finally:
            # This ensures segments exist during GPT-4V validation
            logger.info("====== CLEANUP FINALLY BLOCK STARTING ======")
            try:
                if video_segments:
                    logger.info(f"Cleaning up {len(video_segments)} video segments")
                    self.segmenter.cleanup_segments(video_segments)
                if audio_segments:
                    logger.info(f"Cleaning up {len(audio_segments)} audio segments")
                    self.segmenter.cleanup_segments(audio_segments)
            except Exception as e:
                logger.warning(f"Failed to cleanup segments: {e}")
            # CLEANUP FRAME SAMPLER DIRECTORY
            try:
                self.frame_sampler.cleanup()
                logger.info("Cleaned up FrameSampler temp directory")
            except Exception as e:
                logger.warning(f"Failed to cleanup frame sampler: {e}")

    def _generate_temporal_questions_with_retry(
        self,
        segment_info: Dict,
        audio_path: Optional[Path],
        transcript_text: str,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate temporal questions with retry logic."""
        video_id = metadata.get("video_id", metadata.get("video_number", "unknown"))
        seg_num = segment_info["segment_number"]

        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Attempt {attempt + 1}/{self.max_retries}: Generating temporal questions for {video_id} segment {seg_num}"
                )

                result = self._generate_temporal_questions_for_segment(
                    segment_info, audio_path, transcript_text, metadata
                )

                if result:
                    logger.info(
                        f"✓ Successfully generated temporal questions for segment {seg_num} on attempt {attempt + 1}"
                    )
                    return result
                last_error = "Empty result"

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Attempt {attempt + 1} failed for segment {seg_num}: {e}"
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (attempt + 1)  # Exponential backoff
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)

        logger.error(
            f"✗ Failed to generate temporal questions for segment {seg_num} after {self.max_retries} attempts. Last error: {last_error}"
        )
        return None

    def _generate_temporal_questions_for_segment(
        self,
        segment_info: Dict,
        audio_path: Optional[Path],
        transcript_text: str,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate multiple temporal questions for a video segment - returns ONE entry with questions list."""
        video_id = metadata.get("video_id", metadata.get("video_number", "unknown"))
        seg_num = segment_info["segment_number"]

        logger.info(
            f"Generating {self.questions_per_segment} temporal questions for {video_id} segment {seg_num}"
        )

        try:
            # Build temporal localization prompt
            prompt = get_temporal_localization_prompt(
                segment_info, metadata, transcript_text, self.config
            )

            # Prepare media files
            media_files = []
            seg_path = segment_info["segment_path"]
            if seg_path.exists():
                # Determine if it's video or audio
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

            if audio_path and audio_path.exists():
                media_files.append(("audio", audio_path))

            # Generate temporal questions
            response_text = self.generate_content(media_files, prompt, video_fps=0.5)
            print(
                f"🔍 Segment {seg_num}: start={segment_info['start']}, end={segment_info['end']}, duration={segment_info['end'] - segment_info['start']}"
            )
            print(f"📹 Segment file path: {seg_path}")
            print(f"📹 Segment file exists: {seg_path.exists()}")
            if seg_path.exists():
                import subprocess

                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(seg_path),
                    ],
                    capture_output=True,
                    text=True,
                )
                actual_duration = float(result.stdout.strip())
                print(f"📹 ACTUAL segment file duration: {actual_duration}s")
                print(f"📹 EXPECTED duration: {segment_info['duration']}s")

            print(f"📝 RAW MODEL RESPONSE:\n{response_text}")

            questions_data = self._parse_temporal_response(response_text)

            for i, q in enumerate(questions_data):
                logger.info(
                    f"Question {i + 1}: start_s={q.get('answer', {}).get('start_s')}, end_s={q.get('answer', {}).get('end_s')}"
                )

            # Get demographics for this segment (shared across all questions)
            demographics_data = self._get_segment_demographics(
                segment_info, seg_path, audio_path, transcript_text, metadata
            )

            # Build questions list with IDs and absolute timestamps
            questions_list = []
            segment_start = segment_info["start"]  # Get segment start time

            for q_idx, q_data in enumerate(questions_data):
                # Generate question ID with zero-padded index
                question_id = f"{(q_idx + 1):03d}"  # 001, 002, 003

                # Get answer times (relative to segment)
                answer_start_relative = q_data.get("answer", {}).get("start_s")
                answer_end_relative = q_data.get("answer", {}).get("end_s")

                # Convert to absolute times (relative to full video)
                answer_start_absolute = None
                answer_end_absolute = None
                if answer_start_relative is not None:
                    answer_start_absolute = round(
                        segment_start + answer_start_relative, 3
                    )
                if answer_end_relative is not None:
                    answer_end_absolute = round(segment_start + answer_end_relative, 3)

                question_entry = {
                    "question_id": question_id,
                    "question": q_data.get("question", ""),
                    "temporal_relation": q_data.get("temporal_relation", "after"),
                    "anchor_event": q_data.get("anchor_event", ""),
                    "target_event": q_data.get("target_event", ""),
                    "answer": {
                        "start_s": answer_start_absolute,  # Absolute timestamp
                        "end_s": answer_end_absolute,  # Absolute timestamp
                    },
                    "requires_audio": q_data.get("requires_audio", False),
                    "confidence": q_data.get("confidence", 0.0),
                    "abstained": q_data.get("abstained", False),
                    "rationale_model": q_data.get("rationale_model", ""),
                }

                questions_list.append(question_entry)

            # Calculate segment-level confidence (average of all questions)
            segment_confidence = 0.0
            if questions_list:
                segment_confidence = sum(q["confidence"] for q in questions_list) / len(
                    questions_list
                )

            # Build single entry for this segment
            entry = {
                "video_id": video_id,
                "video_number": metadata.get("video_number", video_id),
                "segment": {"start": segment_info["start"], "end": segment_info["end"]},
                "questions": questions_list,  # List of questions
                "num_questions": len(questions_list),
                "confidence": round(segment_confidence, 3),  # Segment-level confidence
                # Segment-level demographics (shared across all questions)
                "demographics": demographics_data.get("demographics", []),
                "demographics_total_individuals": demographics_data.get(
                    "total_individuals", 0
                ),
                "demographics_confidence": demographics_data.get("confidence", 0.0),
                "demographics_explanation": demographics_data.get("explanation", ""),
            }

            if self.judge_enabled:
                logger.info(f"[{video_id} seg {seg_num}] Validating with GPT-4V...")
                frame_paths = []
                try:
                    video_path = segment_info.get("segment_path")
                    logger.info(f"🔍 Judge video path: {video_path}")
                    logger.info(
                        f"🔍 Judge video exists: {Path(video_path).exists() if video_path else False}"
                    )
                    if video_path and Path(video_path).exists():
                        logger.info(
                            f"🔍 Judge video size: {Path(video_path).stat().st_size} bytes"
                        )

                    if video_path and Path(video_path).exists():
                        import subprocess

                        result = subprocess.run(
                            [
                                "ffprobe",
                                "-v",
                                "error",
                                "-show_entries",
                                "format=duration",
                                "-of",
                                "default=noprint_wrappers=1:nokey=1",
                                str(video_path),
                            ],
                            capture_output=True,
                            text=True,
                        )
                        actual_duration = float(result.stdout.strip())
                        frame_end = min(
                            actual_duration, segment_info["end"] - segment_info["start"]
                        )

                        frame_paths = self.frame_sampler.sample_frames_from_segment(
                            video_path=Path(video_path),
                            segment_start=0.0,  # Segment file always starts at 0
                            segment_end=frame_end,  # Duration of segment file
                            num_frames=self.judge_frame_count,
                            strategy=self.config.temporal_localization.judge_frame_strategy,
                        )

                        if frame_paths:
                            validated_questions, validation_stats = (
                                self._validate_questions(
                                    entry["questions"],
                                    {
                                        "start": segment_info["start"],
                                        "end": segment_info["end"],
                                    },
                                    frame_paths,
                                    transcript_text,
                                )
                            )
                            entry["questions"] = validated_questions
                            entry["num_questions"] = len(validated_questions)
                            entry["validation"] = validation_stats
                            entry["validation"]["judge_used"] = True

                            if validation_stats["total"] > 0:
                                val_rate = (
                                    validation_stats["valid"]
                                    / validation_stats["total"]
                                )
                                entry["confidence"] = round(
                                    entry["confidence"] * (0.7 + 0.3 * val_rate), 3
                                )

                            logger.info(
                                f"✓ {validation_stats['valid']}/{validation_stats['total']} valid, {validation_stats['fixed']} fixed"
                            )
                except Exception as e:
                    logger.error(f"Validation error: {e}")

                finally:
                    # Clean up ONLY the frames from this segment
                    for fp in frame_paths:
                        try:
                            if fp.exists():
                                fp.unlink()
                        except Exception as e:
                            logger.debug(f"Failed to delete frame {fp}: {e}")

            return entry

        except Exception as e:
            logger.error(
                f"Failed to generate temporal questions for segment {seg_num}: {e}",
                exc_info=True,
            )
            return None  # Return None to trigger retry

    def _get_segment_demographics(
        self,
        segment_info: Dict,
        video_path: Path,
        audio_path: Optional[Path],
        transcript_text: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get expanded demographics for a specific segment (same as MCQ)."""
        try:
            # Get human-reviewed demographics from metadata (video-level)
            human_demographics = metadata.get("demographics_detailed_reviewed", {})
            if not human_demographics:
                logger.warning(
                    f"No human-reviewed demographics found for {metadata.get('video_id')}"
                )
                return {
                    "demographics": [],
                    "total_individuals": 0,
                    "confidence": 0.0,
                    "explanation": "No human-reviewed demographics available",
                }

            # Build expansion prompt (segment-level)
            prompt = self.demographics_expander.build_expansion_prompt(
                human_demographics,
                segment_info={
                    "start": segment_info["start"],
                    "end": segment_info["end"],
                },
            )

            # Prepare media files
            media_files = []
            if video_path and video_path.exists():
                if video_path.suffix.lower() in [
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".webm",
                    ".mkv",
                    ".m4v",
                ]:
                    media_files.append(("video", video_path))
                else:
                    media_files.append(("audio", video_path))

            if audio_path and audio_path.exists():
                media_files.append(("audio", audio_path))

            # Add transcript context to prompt
            if transcript_text:
                prompt += f"\n\nSEGMENT TRANSCRIPT:\n{transcript_text[:1000]}"

            # Generate demographics
            response_text = self.generate_content(media_files, prompt, video_fps=0.25)
            demographics_data = self.demographics_expander.parse_demographics_response(
                response_text
            )

            # Ensure all fields are present
            if "explanation" not in demographics_data:
                demographics_data["explanation"] = "No explanation provided"

            return demographics_data

        except Exception as e:
            logger.error(f"Failed to get segment demographics: {e}")
            return {
                "demographics": [],
                "total_individuals": 0,
                "confidence": 0.0,
                "explanation": f"Error generating demographics: {str(e)}",
            }

    def _validate_with_gpt4v(
        self,
        question: Dict,
        segment_info: Dict,
        frame_paths: List[Path],
        transcript_text: str,
    ) -> Dict:
        """Use GPT-4V to validate question."""
        try:
            # Encode frames
            image_contents = []
            for i, frame_path in enumerate(frame_paths):
                with open(frame_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                image_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high",
                        },
                    }
                )
                image_contents.append(
                    {"type": "text", "text": f"[Frame {i + 1}/{len(frame_paths)}]"}
                )

            # Build prompt using imported function
            prompt = build_validation_prompt(question, segment_info, transcript_text)

            # Call GPT-4V
            response = self.openai_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": build_batch_validation_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}] + image_contents,
                    },
                ],
                max_tokens=800,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"GPT-4V error: {e}")
            return {"valid": True, "message": f"Error: {e}"}

    def _convert_rationale_to_absolute(
        self, rationale: str, segment_start: float
    ) -> str:
        import re

        def replace_timestamp(match):
            timestamp_str = match.group(1)
            try:
                timestamp = float(timestamp_str)
                absolute = segment_start + timestamp

                if absolute == int(absolute):
                    return f"{int(absolute)}.0s"
                formatted = f"{absolute:.3f}".rstrip("0").rstrip(".")
                return f"{formatted}s"
            except:
                return match.group(0)

        pattern = r"(?:^|(?<=[\s=,\(\.]))(\d+\.?\d*)s(?=[\s,\.\)\-–]|$)"
        return re.sub(pattern, replace_timestamp, rationale)

    def _validate_questions(
        self,
        questions: List[Dict],
        segment_info: Dict,
        frame_paths: List[Path],
        transcript_text: str,
    ) -> tuple:
        """Validate questions, fix absolute timestamps, drop invalid."""
        segment_start = segment_info["start"]
        segment_end = segment_info["end"]
        segment_duration = segment_end - segment_start

        valid_questions = []
        stats = {
            "total": len(questions),
            "valid": 0,
            "fixed": 0,
            "dropped": 0,
            "reasons": {},
        }

        for i, q in enumerate(questions):
            try:
                # Skip abstained
                if q.get("abstained", False):
                    stats["dropped"] += 1
                    stats["reasons"]["abstained"] = (
                        stats["reasons"].get("abstained", 0) + 1
                    )
                    continue

                answer = q.get("answer", {})
                start_s = answer.get("start_s")
                end_s = answer.get("end_s")

                if start_s is None or end_s is None:
                    stats["dropped"] += 1
                    stats["reasons"]["missing_timestamps"] = (
                        stats["reasons"].get("missing_timestamps", 0) + 1
                    )
                    continue

                # Detect if Gemini output absolute instead of relative timestamps
                fixed = False

                # Strategy: Timestamps after YOUR conversion should be:
                # - In range [segment_start, segment_end] for absolute
                # - OR in range [0, segment_duration] for relative that wasn't converted yet
                #
                # We detect Gemini mistakes by checking if timestamps look like they're
                # in the segment's absolute range when they should already be absolute
                # (i.e., Gemini gave us absolute, we added segment_start, now they're way off)

                if segment_start > 0:  # Not first segment (can't detect for segment 0)
                    # Check if timestamps are suspiciously high (likely double-converted)
                    # OR if they look like segment-relative absolute times

                    # Case 1: Way too high - definitely double-converted
                    if start_s > segment_end or end_s > segment_end:
                        # These are likely double-converted: Gemini gave absolute, we added segment_start
                        # Try converting back
                        test_start = start_s - segment_start
                        test_end = end_s - segment_start

                        # Check if this makes sense
                        if (
                            0 <= test_start <= segment_duration
                            and 0 <= test_end <= segment_duration
                        ):
                            logger.warning(
                                f"Q{i + 1}: Double-converted timestamps [{start_s}, {end_s}] → fixing to relative [{test_start:.1f}, {test_end:.1f}]"
                            )
                            start_s = test_start
                            end_s = test_end
                            q["answer"]["start_s"] = start_s
                            q["answer"]["end_s"] = end_s
                            q["rationale_model"] += " [Judge: fixed double-conversion]"
                            fixed = True
                        else:
                            # Can't fix, drop it
                            stats["dropped"] += 1
                            stats["reasons"]["out_of_bounds"] = (
                                stats["reasons"].get("out_of_bounds", 0) + 1
                            )
                            logger.warning(
                                f"Q{i + 1}: Unfixable out of bounds [{start_s}, {end_s}]"
                            )
                            continue

                    # Case 2: In segment absolute range - Gemini gave absolute, we converted correctly
                    # BUT need to convert to relative for bounds checking
                    elif (
                        start_s >= segment_start
                        and start_s < segment_end
                        and end_s > segment_start
                        and end_s <= segment_end
                    ):
                        logger.warning(
                            f"Q{i + 1}: Absolute timestamps detected [{start_s}, {end_s}] → converting to relative"
                        )
                        start_s = start_s - segment_start
                        end_s = end_s - segment_start
                        q["answer"]["start_s"] = start_s
                        q["answer"]["end_s"] = end_s
                        q["rationale_model"] += " [Judge: absolute→relative]"
                        fixed = True

                # Now check bounds on what should be RELATIVE timestamps
                if start_s < 0 or end_s > segment_duration or start_s >= end_s:
                    stats["dropped"] += 1
                    stats["reasons"]["out_of_bounds"] = (
                        stats["reasons"].get("out_of_bounds", 0) + 1
                    )
                    logger.warning(
                        f"Q{i + 1}: Out of bounds after fixing [{start_s:.1f}, {end_s:.1f}] (segment: 0-{segment_duration}s)"
                    )
                    continue

                # GPT-4V validation (only if bounds check passed)
                if self.judge_enabled and frame_paths:
                    gpt_result = self._validate_with_gpt4v(
                        q, segment_info, frame_paths, transcript_text
                    )

                    if not gpt_result.get("valid", False):
                        stats["dropped"] += 1
                        reason = gpt_result.get("reason", "gpt4v_rejected")
                        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                        logger.warning(
                            f"Q{i + 1}: GPT-4V rejected - {gpt_result.get('message', '')[:100]}"
                        )
                        continue

                    if gpt_result.get("corrected_timestamps"):
                        corrected = gpt_result["corrected_timestamps"]
                        q["answer"]["start_s"] = corrected["start_s"]
                        q["answer"]["end_s"] = corrected["end_s"]
                        q["rationale_model"] += (
                            f" [Judge: {gpt_result.get('correction_reason', 'adjusted')[:50]}]"
                        )
                        fixed = True
                # After all validation passes, convert back to absolute
                answer_start_absolute = round(segment_start + start_s, 3)
                answer_end_absolute = round(segment_start + end_s, 3)

                q["answer"]["start_s"] = answer_start_absolute
                q["answer"]["end_s"] = answer_end_absolute
                q["rationale_model"] = self._convert_rationale_to_absolute(
                    q["rationale_model"], segment_start
                )
                # Question passed validation
                valid_questions.append(q)
                stats["valid"] += 1
                if fixed:
                    stats["fixed"] += 1
                    stats["reasons"]["fixed_timestamps"] = (
                        stats["reasons"].get("fixed_timestamps", 0) + 1
                    )

            except Exception as e:
                logger.error(f"Q{i + 1} validation error: {e}")
                stats["dropped"] += 1
                stats["reasons"]["validation_error"] = (
                    stats["reasons"].get("validation_error", 0) + 1
                )

        logger.info(
            f"Validation: {stats['valid']} valid, {stats['fixed']} fixed, {stats['dropped']} dropped"
        )
        return valid_questions, stats

    def _parse_temporal_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse temporal localization response from Gemini - ORIGINAL VERSION."""
        try:
            response_text = response_text.strip()

            # Remove markdown code blocks
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

            data = json.loads(response_text.strip())

            # Ensure it's a list
            if isinstance(data, dict):
                data = [data]

            # Validate and clean each question
            validated_questions = []
            for i, q in enumerate(data):
                validated_q = self._validate_temporal_question(q, i)
                validated_questions.append(validated_q)

            # Ensure we have exactly the expected number of questions
            while len(validated_questions) < self.questions_per_segment:
                validated_questions.append(
                    self._get_default_temporal_question(len(validated_questions))
                )

            return validated_questions[: self.questions_per_segment]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse temporal JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}...")
            return [
                self._get_default_temporal_question(i)
                for i in range(self.questions_per_segment)
            ]
        except Exception as e:
            logger.error(f"Error parsing temporal response: {e}")
            return [
                self._get_default_temporal_question(i)
                for i in range(self.questions_per_segment)
            ]

    def _validate_temporal_question(
        self, question_data: Dict, index: int
    ) -> Dict[str, Any]:
        """Validate and clean a temporal question."""
        try:
            # Extract answer times
            answer = question_data.get("answer", {})
            start_s = answer.get("start_s")
            end_s = answer.get("end_s")

            # Determine if abstained
            abstained = question_data.get("abstained", False)
            if start_s is None or end_s is None:
                abstained = True

            # Validate temporal_relation
            valid_relations = ["after", "once_finished", "next", "during", "before"]
            temporal_relation = question_data.get("temporal_relation", "after")
            if temporal_relation not in valid_relations:
                logger.warning(
                    f"Invalid temporal_relation '{temporal_relation}', defaulting to 'after'"
                )
                temporal_relation = "after"

            return {
                "question_index": question_data.get("question_index", index),
                "question": question_data.get("question", "When does an event occur?"),
                "temporal_relation": temporal_relation,
                "anchor_event": question_data.get("anchor_event", "Unknown anchor"),
                "target_event": question_data.get("target_event", "Unknown target"),
                "answer": {"start_s": start_s, "end_s": end_s},
                "requires_audio": bool(question_data.get("requires_audio", False)),
                "confidence": float(question_data.get("confidence", 0.0)),
                "abstained": abstained,
                "rationale_model": question_data.get(
                    "rationale_model", "No rationale provided"
                ),
            }

        except Exception as e:
            logger.error(f"Error validating temporal question: {e}")
            return self._get_default_temporal_question(index)

    def _get_default_temporal_question(self, index: int) -> Dict[str, Any]:
        """Return default temporal question structure when parsing fails."""
        return {
            "question_index": index,
            "question": "When does an event occur in this segment?",
            "temporal_relation": "after",
            "anchor_event": "Unable to identify anchor",
            "target_event": "Unable to identify target",
            "answer": {"start_s": None, "end_s": None},
            "requires_audio": False,
            "confidence": 0.0,
            "abstained": True,
            "rationale_model": "Failed to generate temporal question",
        }
